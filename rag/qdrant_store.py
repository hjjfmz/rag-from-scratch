"""Qdrant server client over plain REST (no vendor SDK).

Qdrant is a recall index only: it answers "which stored vectors are closest?".
PG stays the source of truth — every candidate ID returned here must hydrate
successfully from the ledger before it can influence an answer.
"""
from uuid import NAMESPACE_URL, uuid5

import requests


def _point_id(ref):
    """Qdrant point IDs must be uint/UUID; map our stable string IDs deterministically."""
    return str(uuid5(NAMESPACE_URL, "rag-knowledge:" + ref))


class QdrantStore:
    def __init__(self, url, fingerprint, timeout=10):
        if not url:
            raise ValueError("Qdrant URL is required (RAG_QDRANT_URL)")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.fingerprint = fingerprint
        self._collections = {kind: f"rag_{kind}_{fingerprint[:12]}" for kind in ("chunks", "entities", "relations")}

    def _request(self, method, path, ok_codes=(200,), **kwargs):
        response = requests.request(method, self.url + path, timeout=self.timeout, **kwargs)
        if response.status_code not in ok_codes:
            raise RuntimeError(f"Qdrant {method} {path} -> {response.status_code}: {response.text[:200]}")
        return response.json() if response.content else {}

    def collections(self):
        return self._collections

    def ensure(self, vector_size):
        """Create the three fingerprint-scoped collections and source payload indexes (idempotent)."""
        for name in self._collections.values():
            existing = self._request("GET", f"/collections/{name}", ok_codes=(200, 404))
            if not existing.get("result"):
                self._request("PUT", f"/collections/{name}",
                              json={"vectors": {"size": vector_size, "distance": "Cosine"}})
            self._request("PUT", f"/collections/{name}/index",
                          json={"field_name": "source", "field_type": "keyword"})

    def upsert(self, kind, rows, source_of):
        """rows: dicts with id/embedding; source_of maps a row to its source for filter deletes."""
        if not rows:
            return 0
        points = [{"id": _point_id(row["id"]), "vector": row["embedding"],
                   "payload": {"ref": row["id"], "source": source_of(row)}} for row in rows]
        self._request("PUT", f"/collections/{self._collections[kind]}/points",
                      params={"wait": "true"}, json={"points": points})
        return len(points)

    def query(self, kind, vector, limit, floor=0.0):
        """Bounded recall: at most `limit` candidate refs above the similarity floor."""
        data = self._request("POST", f"/collections/{self._collections[kind]}/points/query",
                             json={"query": vector, "limit": limit, "with_payload": True})
        result = data.get("result", {})
        hits = result.get("points", []) if isinstance(result, dict) else result
        return [(hit["payload"]["ref"], hit["score"]) for hit in hits if hit["score"] >= floor]

    def delete_source(self, source):
        deleted = 0
        for name in self._collections.values():
            data = self._request("POST", f"/collections/{name}/points/delete",
                                 params={"wait": "true"},
                                 json={"filter": {"must": [{"key": "source", "match": {"value": source}}]}})
            deleted += int(data.get("result", {}).get("deleted", 0) or 0)
        return deleted

    def health(self):
        data = self._request("GET", "/collections")
        return {"configured": True, "collections": len(data.get("result", []))}
