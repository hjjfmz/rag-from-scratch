"""JSON adjacency graph of entities extracted from chunks."""

import json
import os


def chunk_id_for(source: str, chunk_index: int) -> str:
    normalized = str(source or "").replace("\\", "/")
    return f"{normalized}::{chunk_index}"


def graph_path_for(index_path: str) -> str:
    root, _ext = os.path.splitext(index_path)
    return root + "_graph.json"


class GraphStore:
    """Entity graph stored as a JSON adjacency list. No database server."""

    def __init__(self):
        # name -> list of chunk ids where the entity was seen
        self.nodes: dict[str, list[str]] = {}
        self.edges: list[dict] = []

    def drop_source(self, source: str):
        prefix = str(source or "").replace("\\", "/") + "::"
        for name in list(self.nodes):
            kept = [cid for cid in self.nodes[name] if not cid.startswith(prefix)]
            if kept:
                self.nodes[name] = kept
            else:
                del self.nodes[name]
        self.edges = [
            edge for edge in self.edges if not str(edge.get("chunk_id", "")).startswith(prefix)
        ]

    def add_extraction(self, entities: list[str], relations: list[dict], chunk_id: str):
        for name in entities:
            cleaned = str(name).strip()
            if not cleaned:
                continue
            bucket = self.nodes.setdefault(cleaned, [])
            if chunk_id not in bucket:
                bucket.append(chunk_id)
        for relation in relations:
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            label = str(relation.get("relation", "")).strip() or "related_to"
            if not source or not target:
                continue
            for endpoint in (source, target):
                bucket = self.nodes.setdefault(endpoint, [])
                if chunk_id not in bucket:
                    bucket.append(chunk_id)
            self.edges.append({
                "source": source,
                "relation": label,
                "target": target,
                "chunk_id": chunk_id,
            })

    def match_names(self, query: str) -> list[str]:
        """Entity names that appear in the query, longest first."""
        text = query.lower()
        matched = [name for name in self.nodes if name.lower() in text]
        matched.sort(key=len, reverse=True)
        return matched

    def one_hop_chunk_ids(self, names: list[str]) -> list[str]:
        """Chunk ids for the named entities and their direct neighbors.

        Seed chunks come first, then neighbor chunks. Order is the rank
        later used by reciprocal rank fusion.
        """
        wanted = {name.lower() for name in names}
        if not wanted:
            return []

        def is_hit(name: str) -> bool:
            return name.lower() in wanted

        seed_ids: list[str] = []
        neighbor_names: set[str] = set()
        for name, chunk_ids in self.nodes.items():
            if is_hit(name):
                seed_ids.extend(chunk_ids)

        for edge in self.edges:
            source_hit = is_hit(edge["source"])
            target_hit = is_hit(edge["target"])
            if source_hit:
                neighbor_names.add(edge["target"])
            if target_hit:
                neighbor_names.add(edge["source"])

        neighbor_ids: list[str] = []
        for name in neighbor_names:
            if is_hit(name):
                continue
            neighbor_ids.extend(self.nodes.get(name, []))

        ordered: list[str] = []
        seen: set[str] = set()
        for chunk_id in seed_ids + neighbor_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)
        return ordered

    def save(self, path: str):
        payload = {"nodes": self.nodes, "edges": self.edges}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            self.nodes = {}
            self.edges = []
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.nodes = payload.get("nodes") or {}
        self.edges = payload.get("edges") or []
