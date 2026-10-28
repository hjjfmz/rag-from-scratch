"""Qdrant adapter with the same methods as the JSON vector store.

The JSON store shows exact cosine search. This adapter is the path that
keeps vectors in Qdrant (HNSW index, payload filters, server-side persistence).
Switch with VECTOR_BACKEND=qdrant. The client is imported lazily so the JSON
tests run without qdrant-client installed.
"""

import os
import uuid

from rag.vector_store import cosine_similarity


def _point_id(metadata: dict, text: str) -> str:
    chunk_id = (metadata or {}).get("chunk_id") or text[:80]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None, client=None):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "rag_chunks")
        self.entries: list[dict] = []
        self._client = client
        self._vector_size: int | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "VECTOR_BACKEND=qdrant requires qdrant-client. "
                "Install requirements.txt and start Qdrant, or set VECTOR_BACKEND=json."
            ) from exc
        self._client = QdrantClient(url=self.url)
        return self._client

    def _collection_exists(self, client) -> bool:
        names = [item.name for item in client.get_collections().collections]
        return self.collection in names

    def _ensure_collection(self, dimension: int):
        client = self._get_client()
        if self._collection_exists(client):
            self._vector_size = dimension
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            config = VectorParams(size=dimension, distance=Distance.COSINE)
        except ImportError:
            config = type("VectorParams", (), {"size": dimension, "distance": "Cosine"})()
        client.create_collection(
            collection_name=self.collection,
            vectors_config=config,
        )
        self._vector_size = dimension

    def add(self, text: str, embedding: list[float], metadata: dict = None):
        metadata = metadata or {}
        self.entries.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        })
        self._ensure_collection(len(embedding))
        payload = {"text": text, "metadata": metadata}
        point_id = _point_id(metadata, text)
        try:
            from qdrant_client.models import PointStruct
            point = PointStruct(id=point_id, vector=embedding, payload=payload)
        except ImportError:
            point = type("Point", (), {"id": point_id, "vector": embedding, "payload": payload})()
        self._get_client().upsert(collection_name=self.collection, points=[point])

    def add_chunks(self, chunks: list[dict]):
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k not in ("text", "embedding")}
            self.add(chunk["text"], chunk["embedding"], metadata)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
        source: str | None = None,
    ) -> list[dict]:
        """Search Qdrant. `source` is an optional payload filter on metadata.source."""
        client = self._get_client()
        query_filter = None
        if source:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=source),
                        )
                    ]
                )
            except ImportError:
                query_filter = {"metadata.source": source}

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                score_threshold=threshold if threshold > 0 else None,
                query_filter=query_filter,
            )
            points = getattr(response, "points", response)
        else:
            points = client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=threshold if threshold > 0 else None,
                query_filter=query_filter,
            )

        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata") or {},
                "score": float(point.score),
            })
        return results

    def save(self, path: str):
        """Qdrant already persists the collection. `path` is the JSON index name and is unused."""
        print(f"  💾 Qdrant collection '{self.collection}' has {len(self.entries)} entries")

    def load(self, path: str):
        """Scroll the collection into `entries` so BM25 and the graph can read the same chunks."""
        del path  # persistence lives in Qdrant, not the JSON file
        client = self._get_client()
        if not self._collection_exists(client):
            self.entries = []
            print(f"  📂 Qdrant collection '{self.collection}' is empty")
            return

        loaded = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                limit=128,
                offset=offset,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload or {}
                vector = point.vector
                if isinstance(vector, dict):
                    vector = next(iter(vector.values()))
                loaded.append({
                    "text": payload.get("text", ""),
                    "embedding": list(vector or []),
                    "metadata": payload.get("metadata") or {},
                })
            if offset is None:
                break
        self.entries = loaded
        print(f"  📂 Loaded {len(self.entries)} entries from Qdrant '{self.collection}'")

    def replace_entries(self, entries: list[dict]):
        client = self._get_client()
        if self._collection_exists(client):
            client.delete_collection(collection_name=self.collection)
        self.entries = []
        self._vector_size = None
        for entry in entries:
            self.add(entry["text"], entry["embedding"], entry.get("metadata") or {})

    def __len__(self) -> int:
        return len(self.entries)

    def list_sources(self) -> list[str]:
        sources = set()
        for entry in self.entries:
            sources.add(entry.get("metadata", {}).get("source", "unknown"))
        return sorted(sources)

    def cosine_against_entries(self, query_vector: list[float]) -> list[dict]:
        """Exact cosine over the mirrored entries. Used to check the adapter, not as the query path."""
        scored = []
        for entry in self.entries:
            scored.append({
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": cosine_similarity(query_vector, entry["embedding"]),
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored
