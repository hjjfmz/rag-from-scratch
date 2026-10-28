"""Qdrant adapter tests. A fake client stands in for the server."""

from rag.qdrant_store import QdrantVectorStore


class _Name:
    def __init__(self, name):
        self.name = name


class _Point:
    def __init__(self, payload, score=0.0, vector=None):
        self.payload = payload
        self.score = score
        self.vector = vector


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.last_filter = None

    def get_collections(self):
        listing = type("Listing", (), {})()
        listing.collections = [_Name(name) for name in self.collections]
        return listing

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {
            "size": vectors_config.size,
            "points": {},
        }

    def upsert(self, collection_name, points):
        for point in points:
            self.collections[collection_name]["points"][point.id] = point

    def query_points(self, collection_name, query, limit, score_threshold=None, query_filter=None):
        self.last_filter = query_filter
        hits = []
        for point in self.collections[collection_name]["points"].values():
            hits.append(_Point(point.payload, score=0.91, vector=point.vector))
        response = type("Response", (), {})()
        response.points = hits[:limit]
        return response

    def scroll(self, collection_name, limit, offset, with_vectors):
        points = [
            _Point(point.payload, vector=point.vector)
            for point in self.collections[collection_name]["points"].values()
        ]
        return points[:limit], None

    def delete_collection(self, collection_name):
        del self.collections[collection_name]


def test_add_search_and_payload_filter():
    client = FakeClient()
    store = QdrantVectorStore(client=client, collection="rag_chunks")
    store.add("return window is 30 days", [1.0, 0.0], {"source": "refund.md", "chunk_id": "refund.md::0"})
    store.add("office hours", [0.0, 1.0], {"source": "hr.md", "chunk_id": "hr.md::0"})

    hits = store.search([1.0, 0.0], top_k=2)
    assert hits[0]["text"] == "return window is 30 days"
    assert client.collections["rag_chunks"]["size"] == 2

    store.search([1.0, 0.0], top_k=1, source="refund.md")
    assert client.last_filter is not None


def test_load_scrolls_points_into_entries():
    client = FakeClient()
    store = QdrantVectorStore(client=client, collection="rag_chunks")
    store.add("policy text", [0.2, 0.3], {"source": "policy.md", "chunk_index": 0})

    loaded = QdrantVectorStore(client=client, collection="rag_chunks")
    loaded.load("rag_index.json")
    assert len(loaded) == 1
    assert loaded.entries[0]["text"] == "policy text"
    assert loaded.entries[0]["embedding"] == [0.2, 0.3]


def test_replace_entries_rebuilds_collection():
    client = FakeClient()
    store = QdrantVectorStore(client=client, collection="rag_chunks")
    store.add("old", [1.0, 0.0], {"source": "old.md"})
    store.replace_entries([
        {"text": "new", "embedding": [0.0, 1.0], "metadata": {"source": "new.md"}},
    ])
    assert len(store) == 1
    assert store.entries[0]["text"] == "new"
    assert len(client.collections["rag_chunks"]["points"]) == 1
