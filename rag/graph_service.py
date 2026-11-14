"""Single-node CLI application service: atomic indexing, dual-level retrieval and grounded answers."""
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

from PyPDF2 import PdfReader

from rag.text_chunks import split_chunks
from rag.graph_retrieval import DualLevelRetriever, canonical, diverse_contexts, source_rounds
from rag.knowledge_store import KnowledgeStore


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


class GraphService:
    def __init__(self, data_dir, client, chunk_size=900, overlap=120, workers=4):
        if not 128 <= chunk_size <= 6000 or not 0 <= overlap < chunk_size or not 1 <= workers <= 8:
            raise ValueError("chunk_size must be 128..6000, overlap < chunk_size, workers 1..8")
        self.data_dir = Path(data_dir).resolve()
        self.client = client
        self.chunk_size, self.overlap, self.workers = chunk_size, overlap, workers
        self.store = KnowledgeStore(self.data_dir / "knowledge.sqlite3")

    def _snapshot(self):
        snapshot = self.store.records()
        if snapshot["sources"] and snapshot["fingerprint"] != self.client.fingerprint:
            raise ValueError("Index model fingerprint differs. Select the original model/--offline setting or a fresh --data-dir.")
        return snapshot

    def ingest(self, source, force=False):
        source = source_key(source)
        return self.ingest_text(source, read_document(source), force=force)

    def ingest_text(self, source, text, force=False):
        source = source_key(source)
        if not text.strip() or len(text) > 500000:
            raise ValueError("Document text must contain 1..500000 characters")
        self._snapshot()
        content_hash = _digest(f"index-v2:{self.chunk_size}:{self.overlap}:" + text)
        source_state = self.store.source_state(source)
        old = source_state["record"]
        if old and old["content_hash"] == content_hash and not force:
            return {"source": source, "status": "unchanged", "revision": old["revision"]}
        chunks = split_chunks(text, source, self.chunk_size, self.overlap)
        if not chunks or len(chunks) > 1024:
            raise ValueError("Document must produce 1..1024 chunks")
        for chunk in chunks:
            chunk["id"] = _digest(f"{source}:{content_hash}:{chunk['chunk_index']}")
        # Nothing is published until every extraction and all three vector indexes succeed.
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
        self.store.replace_source(source, content_hash, self.client.fingerprint, chunks=stored_chunks,
                                  entities=entities, relations=relations, expected_revision=source_state["revision"])
        return {"source": source, "status": "indexed", "chunks": len(chunks), "entities": len(entities),
                "relations": len(relations), "graph_status": "ready" if relations else "no_relations",
                "revision": source_state["revision"] + 1, "offline": self.client.offline}

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
                    summary["deleted"] += int(self.store.delete(row["source"]))
        return summary

    def search(self, question, mode="hybrid_graph", top_k=5, threshold=.25, rerank=False):
        return self._search_snapshot(self._snapshot(), question, mode, top_k, threshold, rerank)

    def _search_snapshot(self, snapshot, question, mode="hybrid_graph", top_k=5, threshold=.25, rerank=False):
        started = time.perf_counter()
        result = DualLevelRetriever(snapshot, self.client).search(question, mode, top_k, threshold, rerank)
        result["retrieval_trace"]["snapshot_revisions"] = {s["source"]: s["revision"] for s in snapshot["sources"]}
        result.update({"status": "ok" if result["chunks_used"] else ("no_docs" if not snapshot["chunks"] else "no_relevant"),
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

    def query(self, question, **kwargs):
        return self._answer(question, self.search(question, **kwargs))

    def agentic_query(self, question, max_iterations=3, mode="hybrid_graph", top_k=5, threshold=.25, rerank=False):
        if not 1 <= max_iterations <= 10:
            raise ValueError("max_iterations must be 1..10")
        # Validate before invoking a model planner or accumulating unbounded contexts.
        snapshot = self._snapshot()
        initial = self._search_snapshot(snapshot, question, mode=mode, top_k=top_k, threshold=threshold)
        steps = [initial]
        if initial["status"] != "no_docs" and max_iterations > 1:
            seen = {canonical(question)}
            for query in self.client.plan(question):
                if len(steps) >= max_iterations:
                    break
                if not query.strip() or canonical(query) in seen:
                    continue
                seen.add(canonical(query))
                steps.append(self._search_snapshot(snapshot, query, mode=mode, top_k=top_k, threshold=threshold))
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

    def list_sources(self):
        return self.store.list_sources()

    def delete(self, source):
        return {"source": source_key(source), "deleted": self.store.delete(source_key(source))}

    def graph(self):
        # Exporting stored evidence does not require model credentials or model calls.
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
        snapshot = self.store.records()
        compatible = not snapshot["sources"] or (self.client is not None and snapshot["fingerprint"] == self.client.fingerprint)
        return {"storage": check, "stats": self.store.stats(), "data_dir": str(self.data_dir),
                "model_compatible": compatible, "provider": self.client.health() if self.client else {"configured": False, "network_verified": False},
                "scope": "single-node, single-tenant SQLite; exact vector search"}

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
