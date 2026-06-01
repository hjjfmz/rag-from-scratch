"""
Tests for rag.vector_store — verify cosine similarity and vector search.
"""

import json
import os
import tempfile
import pytest
from rag.vector_store import cosine_similarity, VectorStore


# ── Cosine similarity tests ───────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Same vector should have similarity of 1.0"""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0"""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Perpendicular vectors should have similarity of 0.0"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similar_vectors_high_score(self):
        """Similar vectors should have a high similarity score"""
        a = [1.0, 0.0, 0.0]
        b = [0.9, 0.1, 0.0]
        score = cosine_similarity(a, b)
        assert score > 0.9

    def test_different_vectors_low_score(self):
        """Different vectors should have a low similarity score"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0]
        score = cosine_similarity(a, b)
        assert score < 0.1

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 (not crash)"""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_lengths_raises_error(self):
        """Vectors of different lengths should raise ValueError"""
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_magnitude_invariant(self):
        """Cosine similarity should not change with magnitude scaling"""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        score1 = cosine_similarity(a, b)
        # Scale b by 100x — should give same similarity
        b_scaled = [x * 100 for x in b]
        score2 = cosine_similarity(a, b_scaled)
        assert score1 == pytest.approx(score2)


# ── VectorStore tests ─────────────────────────────────────────

class TestVectorStore:
    def setup_method(self):
        """Create a fresh store with test data for each test."""
        self.store = VectorStore()
        self.store.add("cat text", [1.0, 0.0, 0.0], {"source": "cats.txt"})
        self.store.add("kitten text", [0.9, 0.1, 0.0], {"source": "cats.txt"})
        self.store.add("stock text", [0.0, 0.0, 1.0], {"source": "finance.txt"})

    def test_add_increases_length(self):
        store = VectorStore()
        assert len(store) == 0
        store.add("test", [1.0], {})
        assert len(store) == 1

    def test_search_returns_correct_order(self):
        """Most similar should be first"""
        results = self.store.search([1.0, 0.0, 0.0], top_k=3)
        assert results[0]["text"] == "cat text"
        assert results[1]["text"] == "kitten text"
        assert results[2]["text"] == "stock text"

    def test_search_top_k_limits_results(self):
        results = self.store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_search_threshold_filters(self):
        """Results below threshold should be excluded"""
        results = self.store.search([1.0, 0.0, 0.0], top_k=10, threshold=0.5)
        # Only cat and kitten should pass (stock is score 0.0)
        assert len(results) == 2
        for r in results:
            assert r["score"] >= 0.5

    def test_search_empty_store(self):
        empty = VectorStore()
        results = empty.search([1.0, 0.0, 0.0])
        assert results == []

    def test_search_result_has_required_fields(self):
        results = self.store.search([1.0, 0.0, 0.0], top_k=1)
        result = results[0]
        assert "text" in result
        assert "metadata" in result
        assert "score" in result

    def test_list_sources(self):
        sources = self.store.list_sources()
        assert "cats.txt" in sources
        assert "finance.txt" in sources

    def test_add_chunks(self):
        """Test adding chunks from the embedder format"""
        store = VectorStore()
        chunks = [
            {"text": "chunk 1", "embedding": [1.0, 0.0], "source": "a.txt", "chunk_index": 0},
            {"text": "chunk 2", "embedding": [0.0, 1.0], "source": "a.txt", "chunk_index": 1},
        ]
        store.add_chunks(chunks)
        assert len(store) == 2


# ── Save / Load tests ─────────────────────────────────────────

class TestVectorStorePersistence:
    def test_save_and_load_roundtrip(self):
        """Save then load should produce identical search results"""
        store = VectorStore()
        store.add("hello", [1.0, 0.0, 0.0], {"source": "test.txt"})
        store.add("world", [0.0, 1.0, 0.0], {"source": "test.txt"})

        tmp = os.path.join(tempfile.gettempdir(), "test_vs.json")
        try:
            store.save(tmp)

            loaded = VectorStore()
            loaded.load(tmp)

            assert len(loaded) == 2
            results = loaded.search([1.0, 0.0, 0.0], top_k=1)
            assert results[0]["text"] == "hello"
        finally:
            os.remove(tmp)

    def test_saved_file_is_valid_json(self):
        store = VectorStore()
        store.add("test", [1.0, 2.0], {"key": "val"})

        tmp = os.path.join(tempfile.gettempdir(), "test_json.json")
        try:
            store.save(tmp)
            with open(tmp, "r") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
        finally:
            os.remove(tmp)
