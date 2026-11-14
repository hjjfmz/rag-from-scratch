"""LightRAG-inspired entity (local) and relationship/theme (global) retrieval."""
import re
import unicodedata
from collections import deque
from itertools import islice

from rag.bm25 import BM25
from rag.vector_store import cosine_similarity


MODES = ("vector", "keyword", "hybrid", "local", "global", "hybrid_graph")


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def tokens(text):
    text = canonical(text)
    words = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(run if len(run) == 1 else [run[i:i+2] for i in range(len(run)-1)])
    return words


def _overlap(query, text):
    wanted = set(tokens(query)) - {"the", "a", "an", "is", "are", "how", "what", "does", "to", "of", "and", "in"}
    return len(wanted & set(tokens(text))) / max(len(wanted), 1)


def _name_match(name, text):
    name, text = canonical(name), canonical(text)
    if not name:
        return False
    if name.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", text) is not None
    return name in text


def source_rounds(candidates, source_of=lambda row: row["metadata"]["source"]):
    """Preserve within-file ranking while giving matching files ranked turns."""
    groups = {}
    for row in candidates:
        groups.setdefault(source_of(row), deque()).append(row)
    while groups:
        for source in list(groups):
            row = groups[source].popleft()
            if not groups[source]:
                del groups[source]
            yield row


def diverse_contexts(candidates, top_k, max_chars=16000):
    """First cover distinct matching files, then add more chunks from each file in ranked rounds."""
    selected, size = [], 0
    for row in source_rounds(candidates):
        if size + len(row["text"]) > max_chars:
            continue
        row = {key: value for key, value in row.items() if key != "embedding"}
        row["citation_id"] = f"S{len(selected)+1}"
        selected.append(row)
        size += len(row["text"])
        if len(selected) == top_k:
            break
    return selected


class DualLevelRetriever:
    def __init__(self, snapshot, client):
        self.snapshot, self.client = snapshot, client
        self.chunks = {c["id"]: c for c in snapshot["chunks"]}

    def _candidate_ids(self, ids, budget):
        unique = dict.fromkeys(ids)
        return list(islice(source_rounds(unique, lambda cid: self.chunks[cid]["metadata"]["source"]), budget))

    def search(self, question, mode="hybrid_graph", top_k=5, threshold=0.25, rerank=False, max_chars=16000):
        if mode not in MODES or not 1 <= top_k <= 100 or not 0 <= threshold <= 1:
            raise ValueError("Invalid search mode, top_k (1..100), or threshold (0..1)")
        if not question.strip() or len(question) > 8000:
            raise ValueError("Question must contain 1..8000 characters")
        trace = {"mode": mode, "query": question, "local": {"seed_entities": [], "chunk_ids": []},
                 "global": {"seed_relations": [], "chunk_ids": []}, "counts": {}}
        if not self.chunks:
            return {"chunks_used": [], "relations": [], "retrieval_trace": trace}
        graph_mode = mode in ("local", "global", "hybrid_graph")
        keywords = self.client.keywords(question) if graph_mode else {"low_level": [], "high_level": []}
        trace["keywords"] = keywords
        query_vector = self.client.embed([question])[0] if mode != "keyword" else None
        budget = min(top_k * 4, 100)
        ranked = {}
        if mode in ("vector", "hybrid", "hybrid_graph"):
            scored = [(cosine_similarity(query_vector, c["embedding"]), c["id"]) for c in self.chunks.values()]
            ranked["vector"] = self._candidate_ids((cid for score, cid in sorted(scored, reverse=True) if score >= threshold), budget)
        if mode in ("keyword", "hybrid", "hybrid_graph"):
            # Reuse BM25's scoring, supplying Unicode-safe tokens via its document tokenizer boundary.
            bm = BM25()
            documents = [{"text": " ".join(tokens(c["text"])), "metadata": {"chunk_id": c["id"]}}
                         for c in self.chunks.values()]
            bm.index(documents)
            ranked["keyword"] = self._candidate_ids((r["metadata"]["chunk_id"] for r in bm.search(" ".join(tokens(question)), top_k=len(documents))), budget)
        if mode in ("local", "hybrid_graph"):
            names = " ".join([question] + keywords.get("low_level", []))
            scored_entities = []
            for entity in self.snapshot["entities"]:
                exact = _name_match(entity["name"], names)
                semantic = cosine_similarity(query_vector, entity["embedding"])
                if exact or semantic >= max(threshold, .30):
                    scored_entities.append((2.0 * exact + semantic, entity))
            scored_entities.sort(key=lambda item: (-item[0], item[1]["key"], item[1]["id"]))
            balanced_entities = source_rounds((e for _, e in scored_entities), lambda e: self.chunks[e["chunk_id"]]["metadata"]["source"])
            seeds = list(dict.fromkeys(e["key"] for e in balanced_entities))[:min(budget, 12)]
            seed_set = set(seeds)
            related = [r for r in self.snapshot["relations"] if r["source_key"] in seed_set or r["target_key"] in seed_set]
            related = list(islice(source_rounds(related, lambda r: self.chunks[r["chunk_id"]]["metadata"]["source"]), budget))
            neighbors = seed_set | {r[field] for r in related for field in ("source_key", "target_key")}
            ids = [e["chunk_id"] for _, e in scored_entities if e["key"] in seed_set]
            ids += [r["chunk_id"] for r in related]
            ids += [e["chunk_id"] for e in self.snapshot["entities"] if e["key"] in neighbors]
            ranked["local"] = self._candidate_ids(ids, budget)
            trace["local"] = {"seed_entities": seeds, "chunk_ids": ranked["local"]}
        if mode in ("global", "hybrid_graph"):
            theme = " ".join(keywords.get("high_level", [])) or question
            theme_vector = self.client.embed([theme])[0] if theme != question else query_vector
            scored_relations = []
            for relation in self.snapshot["relations"]:
                text = relation["description"] + " " + " ".join(relation["keywords"]) + " " + relation["relation"]
                overlap = _overlap(theme, text)
                semantic = cosine_similarity(theme_vector, relation["embedding"])
                if overlap > 0 or semantic >= max(threshold, .30):
                    scored_relations.append((overlap + semantic, relation))
            scored_relations.sort(key=lambda item: (-item[0], item[1]["id"]))
            selected = list(islice(source_rounds((r for _, r in scored_relations), lambda r: self.chunks[r["chunk_id"]]["metadata"]["source"]), budget))
            endpoints = {r[field] for r in selected for field in ("source_key", "target_key")}
            ids = [r["chunk_id"] for r in selected]
            ids += [e["chunk_id"] for e in self.snapshot["entities"] if e["key"] in endpoints]
            ranked["global"] = self._candidate_ids(ids, budget)
            trace["global"] = {"seed_relations": [r["id"] for r in selected], "chunk_ids": ranked["global"]}
        fused = {}
        for channel, ids in ranked.items():
            trace["counts"][channel] = len(ids)
            for rank, cid in enumerate(ids):
                row = fused.setdefault(cid, {**self.chunks[cid], "score": 0.0, "channels": []})
                row["score"] += 1 / (60 + rank + 1)
                row["channels"].append(channel)
        candidates = list(islice(source_rounds(sorted(fused.values(), key=lambda row: (-row["score"], row["id"]))), budget))
        if rerank and candidates:
            candidates = self.client.rerank(question, candidates)
        selected = diverse_contexts(candidates, top_k, max_chars)
        chars = sum(len(row["text"]) for row in selected)
        ids = {row["id"] for row in selected}
        citations = {row["id"]: row["citation_id"] for row in selected}
        relations = [{**{key: value for key, value in r.items() if key != "embedding"}, "citation_id": citations[r["chunk_id"]]}
                     for r in self.snapshot["relations"] if r["chunk_id"] in ids]
        trace["context_characters"] = chars
        trace["graph_available"] = bool(self.snapshot["relations"])
        trace["graph_evidence_used"] = any(set(row["channels"]) & {"local", "global"} for row in selected)
        return {"chunks_used": selected, "relations": relations[:100], "retrieval_trace": trace}
