"""Application service: atomic indexing, dual-level retrieval and grounded answers.

Storage modes share one retrieval path:
  local    — SQLite adapter recalls vectors by exact in-process cosine scan
  services — Qdrant recalls bounded candidate IDs; PostgreSQL hydrates them
The retriever receives recall candidates plus graph neighbors. Semantic and
keyword recall have windows; catalog scans and graph expansion still scale
with the corpus, so this is not a hard bound on all query work.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from PyPDF2 import PdfReader

from rag.text_chunks import split_chunks
from rag.graph_retrieval import (DualLevelRetriever, _name_match, _overlap, canonical,
                                 diverse_contexts, source_rounds, tokens)
from rag.bm25 import idf_for
from rag.cache import cache_key
from rag.knowledge_store import KnowledgeStore

KEYWORD_WINDOW = 500   # candidate window for BM25 pre-recall from the term tables
RECALL_CAP = 1000      # safety ceiling for one semantic recall arm


def source_key(source):
    source = str(source)
    if source.startswith(("https://", "http://")):
        raise ValueError("Knowledge sources must be local .md/.txt/.pdf files")
    return os.path.normcase(str(Path(source).expanduser().resolve()))


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_document(source, limit=16 * 1024 * 1024):
    """Bounded local knowledge-file input; no automatic Internet data collection."""
    path = Path(source)
    if path.suffix.lower() not in (".md", ".txt", ".pdf"):
        raise ValueError("Supported local formats: .md, .txt, .pdf")
    if path.stat().st_size > limit:
        raise ValueError("Document exceeds 16 MiB input limit")
    raw, pdf = path.read_bytes(), path.suffix.lower() == ".pdf"
    if pdf:
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("Document has no extractable text (scanned PDFs require OCR)")
    if len(text) > 500000:
        raise ValueError("Extracted text exceeds 500000-character limit; split this document first")
    return text


def index_terms(text):
    """Exact token stream the persistent BM25 index scores (graph tokens at the BM25 boundary)."""
    from rag.bm25 import tokenize
    return tokenize(" ".join(tokens(text)))


class GraphService:
    def __init__(self, data_dir=None, client=None, *, store=None, qdrant=None, cache=None,
                 chunk_size=900, overlap=120, workers=4):
        if not 128 <= chunk_size <= 6000 or not 0 <= overlap < chunk_size or not 1 <= workers <= 8:
            raise ValueError("chunk_size must be 128..6000, overlap < chunk_size, workers 1..8")
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.client = client
        self.chunk_size, self.overlap, self.workers = chunk_size, overlap, workers
        self.store = store if store is not None else KnowledgeStore(self.data_dir / "knowledge.sqlite3")
        self.qdrant = qdrant
        self.cache = cache
        self._pool = None

    # ── bounded snapshot: what the retriever is allowed to see ──

    def _vector_recall(self, kind, vector, floor):
        if self.qdrant is not None:
            return self.qdrant.query(kind, vector, limit=RECALL_CAP, floor=floor)
        method = {"chunks": "vector_candidates", "entities": "entity_vector_candidates",
                  "relations": "relation_vector_candidates"}[kind]
        return getattr(self.store, method)(vector, floor=floor, limit=RECALL_CAP)

    def _bounded_snapshot(self, question, mode, top_k, threshold, anchor=None):
        """Candidate snapshot = bounded recall windows + one graph hop.

        `anchor` pins per-source versions (an agentic run shares one anchor), so a
        source replaced mid-run stays invisible to later steps instead of mixing
        versions. A single query anchors itself: sources changing during assembly
        are dropped at the end, never half-visible.
        """
        generations = self.store.generations()
        anchor = dict(anchor) if anchor is not None else \
            {row["source"]: row["generation"] for row in generations}
        sources = self.store.list_sources()
        fingerprint = self.store.fingerprint()
        if sources and fingerprint != self.client.fingerprint:
            raise ValueError("Index model fingerprint differs. Select the original model/--offline setting or a fresh --data-dir.")
        snapshot = {"sources": sources, "chunks": [], "entities": [], "relations": [],
                    "fingerprint": fingerprint, "bm25_stats": None}
        graph_mode = mode in ("local", "global", "hybrid_graph")
        keywords = self.client.keywords(question) if graph_mode else {"low_level": [], "high_level": []}
        query_vector = self.client.embed([question])[0] if mode != "keyword" else None

        candidate_ids = set()
        if mode in ("vector", "hybrid", "hybrid_graph") and query_vector is not None:
            candidate_ids.update(cid for cid, _ in self._vector_recall("chunks", query_vector, threshold))
        query_terms = []
        if mode in ("keyword", "hybrid", "hybrid_graph"):
            query_terms = index_terms(question)
            stats_n, _ = self.store.bm25_corpus()
            if stats_n == 0 and self.store.stats()["chunks"]:
                # Store built before term persistence: fall back to the full id list, local window stats.
                candidate_ids.update(self.store.all_chunk_ids())
            else:
                candidate_ids.update(self.store.chunk_ids_with_terms(sorted(set(query_terms)), limit=KEYWORD_WINDOW))

        seed_entities, relation_ids = [], set()
        if graph_mode:
            if mode in ("local", "hybrid_graph") and query_vector is not None:
                names = " ".join([question] + keywords.get("low_level", []))
                matched = {e["id"] for e in self.store.entity_catalog() if _name_match(e["name"], names)}
                semantic = {cid for cid, _ in self._vector_recall("entities", query_vector, max(threshold, .30))}
                seed_entities = self.store.fetch_entities(matched | semantic)
            if mode in ("global", "hybrid_graph"):
                theme = " ".join(keywords.get("high_level", [])) or question
                theme_vector = self.client.embed([theme])[0] if theme != question else query_vector
                overlap_ids = {r["id"] for r in self.store.relation_catalog()
                               if _overlap(theme, r["description"] + " " + " ".join(r["keywords"])
                                           + " " + r["relation"]) > 0}
                semantic_ids = {cid for cid, _ in self._vector_recall("relations", theme_vector, max(threshold, .30))}
                relation_ids |= overlap_ids | semantic_ids
            if seed_entities:
                seed_keys = {e["key"] for e in seed_entities}
                relation_ids |= {r["id"] for r in self.store.fetch_relations_by_keys(seed_keys)}
        relations = self.store.fetch_relations(relation_ids) if relation_ids else []
        neighbor_keys = {e["key"] for e in seed_entities} | \
                        {r[field] for r in relations for field in ("source_key", "target_key")}
        entities = {e["id"]: e for e in seed_entities}
        for entity in self.store.fetch_entities_by_keys(neighbor_keys - {e["key"] for e in entities.values()}):
            entities.setdefault(entity["id"], entity)

        chunk_ids = candidate_ids | {e["chunk_id"] for e in entities.values()} | \
                    {r["chunk_id"] for r in relations}
        # Citation binding runs in every mode: after the graph expansion introduced more
        # chunks, fetch their evidence too, then re-fetch once in stable rowid order.
        relation_ids |= {r["id"] for r in self.store.fetch_relations_by_chunk_ids(chunk_ids)}
        relations = self.store.fetch_relations(relation_ids) if relation_ids else []
        chunks = self.store.fetch_chunks(chunk_ids)

        # Version anchor: sources replaced/deleted during this assembly (or since the
        # run's anchor) are dropped whole — a query never mixes versions of a source.
        final = self.store.generations()
        stable = {row["source"] for row in final if anchor.get(row["source"]) == row["generation"]}
        snapshot["chunks"] = [c for c in chunks if c["source"] in stable]
        snapshot["entities"] = [e for e in entities.values() if e["chunk_id"] in
                                {c["id"] for c in snapshot["chunks"]}]
        snapshot["relations"] = [r for r in relations if r["chunk_id"] in
                                 {c["id"] for c in snapshot["chunks"]}]
        snapshot["sources"] = [s for s in sources if s["source"] in stable]

        stats_n, avgdl = self.store.bm25_corpus()
        if query_terms and stats_n:
            idf = {term: idf_for(stats_n, df) for term, df in
                   ((t, self.store.bm25_df(t)) for t in sorted(set(query_terms))) if df > 0}
            snapshot["bm25_stats"] = {"idf": idf, "avgdl": avgdl}
        return snapshot

    # ── ingest ──

    def ingest(self, source, force=False):
        source = source_key(source)
        return self.ingest_text(source, read_document(source), force=force)

    def ingest_text(self, source, text, force=False):
        source = source_key(source)
        if not text.strip() or len(text) > 500000:
            raise ValueError("Document text must contain 1..500000 characters")
        stored_fp = self.store.fingerprint()
        if stored_fp and stored_fp != self.client.fingerprint and self.store.stats()["sources"]:
            raise ValueError("Index model fingerprint differs. Select the original model/--offline setting or a fresh --data-dir.")
        content_hash = _digest(f"index-v2:{self.chunk_size}:{self.overlap}:" + text)
        source_state = self.store.source_state(source)
        old = source_state["record"]
        if old and old["content_hash"] == content_hash and not force:
            result = {"source": source, "status": "unchanged", "revision": old["revision"]}
            if self.qdrant is not None:
                # Retry idempotent publication even after a process restart. Reuse
                # persisted extraction/vectors rather than paying for model calls.
                records = self.store.source_records(source)
                if self.store.source_state(source)["revision"] != source_state["revision"]:
                    raise ValueError("Source changed during vector repair; retry ingestion")
                self._publish_vectors(source, records["chunks"], records["entities"], records["relations"], result)
                if self.cache is not None:
                    self.cache.clear()
            return result
        chunks = split_chunks(text, source, self.chunk_size, self.overlap)
        if not chunks or len(chunks) > 1024:
            raise ValueError("Document must produce 1..1024 chunks")
        for chunk in chunks:
            chunk["id"] = _digest(f"{source}:{content_hash}:{chunk['chunk_index']}")
        # Complete extraction and embedding generation before publishing the ledger.
        # Qdrant publication follows that transaction and can fail independently.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            extractions = list(pool.map(self.client.extract, [c["text"] for c in chunks]))
        entities, relations = [], []
        for chunk, extracted in zip(chunks, extractions):
            seen_entities = set()
            for item in extracted["entities"]:
                name = item["name"].strip()
                key = canonical(name)
                if not key or key in seen_entities:
                    continue
                seen_entities.add(key)
                entities.append({"id": _digest(chunk["id"] + ":entity:" + key), "chunk_id": chunk["id"],
                                 "key": key, "name": name, "description": item.get("description") or name})
            seen_relations = set()
            for item in extracted["relations"]:
                head, tail, label = canonical(item["source"]), canonical(item["target"]), canonical(item["relation"])
                evidence = item.get("evidence", "")
                if not head or not tail or not label or not evidence.strip() or evidence not in chunk["text"]:
                    raise ValueError("Extracted relation has invalid endpoints or no verbatim evidence")
                if head not in seen_entities or tail not in seen_entities:
                    raise ValueError("Extracted relation endpoints are missing from extracted entities")
                rid = _digest(chunk["id"] + ":relation:" + head + ":" + label + ":" + tail)
                if rid in seen_relations:
                    continue
                seen_relations.add(rid)
                relations.append({"id": rid, "chunk_id": chunk["id"], "source_key": head, "target_key": tail,
                                  "relation": label, "description": item.get("description") or evidence,
                                  "keywords": item.get("keywords", []), "evidence": evidence})
        texts = [c["text"] for c in chunks] + [e["name"] + ": " + e["description"] for e in entities]
        texts += [f"{r['source_key']} {r['relation']} {r['target_key']}: {r['description']}; {' '.join(r['keywords'])}" for r in relations]
        vectors = self.client.embed(texts)
        if len(vectors) != len(texts) or not vectors:
            raise ValueError("Embedding count does not match knowledge records")
        dimensions = {len(v) for v in vectors}
        if len(dimensions) != 1 or 0 in dimensions or not all(math.isfinite(float(x)) for v in vectors for x in v):
            raise ValueError("Invalid or inconsistent knowledge embedding vectors")
        for record, vector in zip(chunks + entities + relations, vectors):
            record["embedding"] = vector
        stored_chunks = [{"id": c["id"], "text": c["text"], "embedding": c["embedding"],
                          "metadata": {key: value for key, value in c.items() if key not in ("id", "text", "embedding")}}
                         for c in chunks]
        terms = []
        for chunk in chunks:
            counts = {}
            for term in index_terms(chunk["text"]):
                counts[term] = counts.get(term, 0) + 1
            terms.extend({"chunk_id": chunk["id"], "term": term, "tf": tf} for term, tf in counts.items())
        self.store.replace_source(source, content_hash, self.client.fingerprint, chunks=stored_chunks,
                                  entities=entities, relations=relations,
                                  expected_revision=source_state["revision"], terms=terms)
        result = {"source": source, "status": "indexed", "chunks": len(chunks), "entities": len(entities),
                  "relations": len(relations), "graph_status": "ready" if relations else "no_relations",
                  "revision": source_state["revision"] + 1, "offline": self.client.offline}
        self._publish_vectors(source, stored_chunks, entities, relations, result)
        if self.cache is not None:
            self.cache.clear()
        return result

    def _publish_vectors(self, source, chunks, entities, relations, result):
        """Ledger is committed; mirror vectors into Qdrant. Orphaned copies stay invisible (PG decides)."""
        if self.qdrant is None:
            return
        try:
            self.qdrant.ensure(len(chunks[0]["embedding"]))
            self.qdrant.upsert("chunks", chunks, lambda row: source)
            self.qdrant.upsert("entities", entities, lambda row: source)
            self.qdrant.upsert("relations", relations, lambda row: source)
            result["vector_status"] = "ready"
        except Exception as exc:
            result["vector_status"] = "pending"
            result["vector_warning"] = f"Qdrant upsert failed; semantic recall may miss this source: {exc}"
        finally:
            # Even partial publication can change recall. Persist a shared cache
            # revision so a repair is safe across CLI/Web and failed Redis clears.
            self.store.advance_recall_revision()

    # ── background ingest tasks (single worker; status lives in the ledger) ──

    def submit_ingest(self, source, *, text=None, force=False):
        task_id = uuid4().hex
        self.store.task_create("ingest", {"source": source, "text": text, "force": force}, task_id)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-ingest")
        self._pool.submit(self._run_task, task_id)
        return task_id

    def _run_task(self, task_id):
        self.store.task_update(task_id, "running")
        task = self.store.task_get(task_id)
        try:
            payload = task["payload"]
            if payload.get("text") is not None:
                result = self.ingest_text(payload["source"], payload["text"], force=payload.get("force", False))
            else:
                result = self.ingest(payload["source"], force=payload.get("force", False))
            self.store.task_update(task_id, "done", result=result)
        except Exception as exc:
            self.store.task_update(task_id, "failed", error=str(exc))

    def task_get(self, task_id):
        return self.store.task_get(task_id)

    def task_list(self, limit=50):
        return self.store.task_list(limit)

    # ── retrieval ──

    def sync(self, directory, prune=False, force=False):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError("Sync source must be an existing directory")
        paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".pdf"))
        summary = {"indexed": 0, "unchanged": 0, "deleted": 0, "errors": [], "files": []}
        for path in paths:
            try:
                result = self.ingest(str(path), force=force)
                summary[result["status"]] += 1
                summary["files"].append(result)
            except Exception as exc:
                summary["errors"].append({"source": str(path), "error": str(exc)})
        if prune and not summary["errors"]:
            present = {source_key(p) for p in paths}
            for row in self.store.list_sources():
                if row["source"].startswith(("https://", "http://")):
                    continue
                old = Path(row["source"])
                if old.is_relative_to(directory) and row["source"] not in present:
                    summary["deleted"] += int(self.delete(row["source"])["deleted"])
        return summary

    def search(self, question, mode="hybrid_graph", top_k=5, threshold=.25, rerank=False, anchor=None):
        started = time.perf_counter()
        corpus_empty = self.store.stats()["chunks"] == 0
        snapshot = self._bounded_snapshot(question, mode, top_k, threshold, anchor=anchor)
        return self._search_snapshot(snapshot, question, mode, top_k, threshold, rerank, corpus_empty, started=started)

    def _search_snapshot(self, snapshot, question, mode="hybrid_graph", top_k=5, threshold=.25,
                         rerank=False, corpus_empty=False, started=None):
        started = time.perf_counter() if started is None else started
        result = DualLevelRetriever(snapshot, self.client).search(question, mode, top_k, threshold, rerank)
        result["retrieval_trace"]["snapshot_revisions"] = {s["source"]: s["revision"] for s in snapshot["sources"]}
        result.update({"status": "ok" if result["chunks_used"] else ("no_docs" if corpus_empty else "no_relevant"),
                       "sources": list(dict.fromkeys(c["metadata"]["source"] for c in result["chunks_used"])),
                       "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "offline": self.client.offline})
        return result

    def _answer(self, question, result):
        if not result["chunks_used"]:
            result["answer"] = "知识库中没有足够证据回答该问题。No supporting evidence was found."
            return result
        answer = self.client.answer(question, result["chunks_used"], result["relations"])
        if answer.strip() == "INSUFFICIENT_EVIDENCE":
            result["status"] = "insufficient_evidence"
            result["answer"] = "检索到的材料不足以支持答案，无法据此判断。Insufficient supporting evidence."
            return result
        citations = set(re.findall(r"\[(S\d+)\]", answer))
        allowed = {c["citation_id"] for c in result["chunks_used"]}
        if not citations or not citations <= allowed:
            raise ValueError("Generated answer is missing valid source citations; answer withheld")
        result["answer"] = answer
        return result

    def _corpus_signature(self):
        state = sorted((row["source"], row["content_hash"], row["revision"])
                       for row in self.store.list_sources())
        return _digest(json.dumps([self.qdrant is not None, state, self.store.recall_revision()], ensure_ascii=False))

    def query(self, question, **kwargs):
        key = None
        if self.cache is not None and self.cache.enabled:
            signature = self._corpus_signature()
            key = cache_key(self.client.fingerprint, question, kwargs.get("mode", "hybrid_graph"),
                            kwargs.get("top_k", 5), kwargs.get("threshold", .25), kwargs.get("rerank", False),
                            corpus_signature=signature)
            cached = self.cache.get(key)
            if cached is not None and signature == self._corpus_signature():
                return {**cached, "cached": True}
        result = self._answer(question, self.search(question, **kwargs))
        if (key is not None and result.get("status") in ("ok", "insufficient_evidence")
                and signature == self._corpus_signature()):
            self.cache.put(key, result)
        return result

    def agentic_query(self, question, max_iterations=3, mode="hybrid_graph", top_k=5, threshold=.25, rerank=False):
        if not 1 <= max_iterations <= 10:
            raise ValueError("max_iterations must be 1..10")
        # Validate before invoking a model planner or accumulating unbounded contexts.
        corpus_empty = self.store.stats()["chunks"] == 0
        anchor = {row["source"]: row["generation"] for row in self.store.generations()}
        initial = self.search(question, mode=mode, top_k=top_k, threshold=threshold, anchor=anchor)
        steps = [initial]
        if initial["status"] != "no_docs" and max_iterations > 1:
            seen = {canonical(question)}
            for query in self.client.plan(question):
                if len(steps) >= max_iterations:
                    break
                if not query.strip() or canonical(query) in seen:
                    continue
                seen.add(canonical(query))
                steps.append(self.search(query, mode=mode, top_k=top_k, threshold=threshold, anchor=anchor))
        chunks, relations = {}, {}
        for step in steps:
            for rank, chunk in enumerate(step["chunks_used"]):
                combined = chunks.setdefault(chunk["id"], {**chunk, "agent_score": 0.})
                combined["agent_score"] += 1 / (60 + rank + 1)
            for relation in step["relations"]:
                relations[relation["id"]] = relation
        candidates = list(islice(source_rounds(sorted(chunks.values(), key=lambda c: (-c["agent_score"], c["id"]))), min(top_k * 4, 100)))
        if rerank and candidates:
            candidates = self.client.rerank(question, candidates)
        selected = diverse_contexts(candidates, top_k)
        selected_ids = {chunk["id"] for chunk in selected}
        citations = {chunk["id"]: chunk["citation_id"] for chunk in selected}
        result = {**initial, "chunks_used": selected,
                  "relations": [{**r, "citation_id": citations[r["chunk_id"]]} for r in relations.values() if r["chunk_id"] in selected_ids][:100],
                  "sources": list(dict.fromkeys(c["metadata"]["source"] for c in selected)),
                  "iterations": len(steps), "steps": [s["retrieval_trace"] for s in steps],
                  "status": "ok" if selected else initial["status"]}
        return self._answer(question, result)

    # ── inspection ──

    def list_sources(self):
        return self.store.list_sources()

    def delete(self, source):
        source = source_key(source)
        deleted = self.store.delete(source)
        result = {"source": source, "deleted": deleted}
        if deleted and self.qdrant is not None:
            try:
                self.qdrant.delete_source(source)
            except Exception as exc:
                result["vector_warning"] = f"Qdrant cleanup deferred (harmless: invisible in PG): {exc}"
        if deleted and self.cache is not None:
            self.cache.clear()
        return result

    def graph(self):
        # Explicit whole-store export: not on the query path, no model credentials required.
        snapshot = self.store.records()
        chunks = {c["id"]: c for c in snapshot["chunks"]}
        nodes = {}
        for entity in snapshot["entities"]:
            node = nodes.setdefault(entity["key"], {"id": entity["key"], "name": entity["name"], "sources": []})
            source = chunks[entity["chunk_id"]]["metadata"]["source"]
            if source not in node["sources"]:
                node["sources"].append(source)
        relations = [{**{k: v for k, v in r.items() if k != "embedding"},
                      "source_document": chunks[r["chunk_id"]]["metadata"]["source"]} for r in snapshot["relations"]]
        return {"nodes": list(nodes.values()), "relations": relations, "stats": self.store.stats()}

    def doctor(self):
        check = self.store.check()
        stats = self.store.stats()
        fingerprint = self.store.fingerprint()
        compatible = not stats["sources"] or (self.client is not None and fingerprint == self.client.fingerprint)
        services = {"storage_mode": "services" if self.qdrant is not None else "local (SQLite adapter, exact vector scan)"}
        if self.qdrant is not None:
            try:
                services["qdrant"] = self.qdrant.health()
            except Exception as exc:
                services["qdrant"] = {"configured": True, "error": str(exc)}
        services["answer_cache"] = "enabled" if self.cache is not None and self.cache.enabled else "disabled"
        return {"storage": check, "stats": stats, "data_dir": str(self.data_dir or self.store.path),
                "model_compatible": compatible, "services": services,
                "provider": self.client.health() if self.client else {"configured": False, "network_verified": False},
                "scope": "single-node, single-tenant; PG ledger + Qdrant recall + optional Redis cache"}

    def evaluate(self, questions, modes=("hybrid", "local", "global", "hybrid_graph"), top_k=5, threshold=.25):
        if not questions:
            raise ValueError("Evaluation requires a nonempty question list")
        rows, summaries = [], {}
        for mode in modes:
            scored, reciprocal, abstain, latencies, entity_hits = [], [], [], [], []
            for item in questions:
                result = self.search(item["question"], mode=mode, top_k=top_k, threshold=threshold)
                expected = set(item.get("expected_sources", []))
                actual = [Path(c["metadata"]["source"]).name for c in result["chunks_used"]]
                answerable = item.get("answerable", True)
                recall = len(expected & set(actual)) / len(expected) if expected else None
                rr = next((1 / (rank+1) for rank, name in enumerate(actual) if name in expected), 0.) if expected else None
                if recall is not None:
                    scored.append(recall)
                    reciprocal.append(rr)
                if not answerable:
                    abstain.append(float(not actual))
                expected_entities = {canonical(n) for n in item.get("expected_entities", [])}
                actual_entities = {r[f] for r in result["relations"] for f in ("source_key", "target_key")}
                entity_recall = len(expected_entities & actual_entities) / len(expected_entities) if expected_entities else None
                if entity_recall is not None:
                    entity_hits.append(entity_recall)
                latencies.append(result["elapsed_ms"])
                rows.append({"id": item.get("id"), "question": item["question"], "mode": mode,
                             "source_recall": recall, "reciprocal_rank": rr, "expected_sources": sorted(expected),
                             "actual_sources": actual, "entity_recall": entity_recall, "status": result["status"],
                             "elapsed_ms": result["elapsed_ms"]})
            average = lambda values: round(sum(values) / len(values), 4) if values else None
            summaries[mode] = {"source_recall": average(scored), "mrr": average(reciprocal),
                               "unanswerable_abstention": average(abstain), "entity_recall": average(entity_hits),
                               "mean_latency_ms": average(latencies),
                               "p95_latency_ms": sorted(latencies)[max(0, math.ceil(len(latencies)*.95)-1)]}
        return {"summary": summaries, "individual": rows, "num_questions": len(questions),
                "offline": self.client.offline, "metrics": "retrieval only; not answer factuality or online SLA",
                "evaluated_at": datetime.now(timezone.utc).isoformat()}
