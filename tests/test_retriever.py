"""
Tests for rag.retriever — verify retrieval logic.

📚 LEARN: We "mock" the embedding function so tests run without an API key.
A mock replaces a real function with a fake one that returns predictable values.
This lets us test the retriever's logic (searching, ranking, filtering)
without making actual API calls.
"""

import pytest
from unittest.mock import patch
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.graph_store import GraphStore


def _make_store_with_fake_vectors():
    """Create a vector store with hand-crafted vectors for testing."""
    store = VectorStore()
    # "ML topic" vectors — point roughly in the same direction
    store.add(
        text="Machine learning is a subset of AI.",
        embedding=[0.9, 0.1, 0.0],
        metadata={"source": "ml.md", "chunk_index": 0},
    )
    store.add(
        text="Overfitting happens when a model memorizes training data.",
        embedding=[0.8, 0.2, 0.0],
        metadata={"source": "ml.md", "chunk_index": 1},
    )
    # "Finance topic" vector — points in a completely different direction
    store.add(
        text="Stock prices rose 5% today.",
        embedding=[0.0, 0.0, 1.0],
        metadata={"source": "finance.md", "chunk_index": 0},
    )
    return store


class TestRetriever:
    def setup_method(self):
        self.store = _make_store_with_fake_vectors()

    @patch("rag.retriever.embed_query")
    def test_retrieve_returns_ranked_results(self, mock_embed):
        """Most relevant chunk should be first."""
        # Fake query vector similar to ML topics
        mock_embed.return_value = [0.85, 0.15, 0.0]

        retriever = Retriever(self.store, top_k=3)
        results = retriever.retrieve("What is machine learning?")

        assert len(results) > 0
        # ML chunks should score higher than finance
        assert "machine learning" in results[0]["text"].lower() or \
               "overfitting" in results[0]["text"].lower()
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    @patch("rag.retriever.embed_query")
    def test_top_k_limits_results(self, mock_embed):
        mock_embed.return_value = [0.85, 0.15, 0.0]

        retriever = Retriever(self.store, top_k=1)
        results = retriever.retrieve("test query")

        assert len(results) == 1

    @patch("rag.retriever.embed_query")
    def test_threshold_filters_low_scores(self, mock_embed):
        # Vector pointing toward ML topics
        mock_embed.return_value = [1.0, 0.0, 0.0]

        retriever = Retriever(self.store, top_k=10, threshold=0.5)
        results = retriever.retrieve("test query")

        # Only ML chunks should pass the threshold (finance is ~0.0)
        for r in results:
            assert r["score"] >= 0.5

    @patch("rag.retriever.embed_query")
    def test_results_have_required_fields(self, mock_embed):
        mock_embed.return_value = [0.5, 0.5, 0.0]

        retriever = Retriever(self.store, top_k=1)
        results = retriever.retrieve("test")

        result = results[0]
        assert "text" in result
        assert "metadata" in result
        assert "score" in result
        assert isinstance(result["score"], float)

    @patch("rag.retriever.embed_query")
    def test_empty_store_returns_empty(self, mock_embed):
        mock_embed.return_value = [1.0, 0.0, 0.0]

        empty_store = VectorStore()
        retriever = Retriever(empty_store)
        results = retriever.retrieve("test")

        assert results == []


class TestHybridGraph:
    @patch("rag.retriever.extract_entity_names", return_value=[])
    @patch("rag.retriever.embed_query")
    def test_graph_chunk_is_fused_ahead_of_unrelated_vector_hit(self, mock_embed, _mock_names):
        from rag.graph_store import GraphStore

        mock_embed.return_value = [0.0, 0.0, 1.0]
        store = VectorStore()
        store.add(
            text="Stock prices rose 5% today and the finance desk closed.",
            embedding=[0.0, 0.0, 1.0],
            metadata={"source": "finance.md", "chunk_index": 0, "chunk_id": "finance.md::0"},
        )
        store.add(
            text="GlowBeauty return window is 30 days from delivery.",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "refund.md", "chunk_index": 0, "chunk_id": "refund.md::0"},
        )
        graph = GraphStore()
        graph.add_extraction(["GlowBeauty"], [], "refund.md::0")

        retriever = Retriever(store, top_k=1, threshold=0.0, search_mode="hybrid_graph", graph_store=graph)
        results = retriever.retrieve("GlowBeauty return window")

        assert results
        assert results[0]["metadata"]["source"] == "refund.md"

    @patch("rag.retriever.embed_query")
    def test_empty_graph_falls_back_to_hybrid(self, mock_embed):
        mock_embed.return_value = [0.9, 0.1, 0.0]
        store = _make_store_with_fake_vectors()
        retriever = Retriever(store, top_k=2, threshold=0.0, search_mode="hybrid_graph", graph_store=GraphStore())
        results = retriever.retrieve("machine learning")
        assert results
        assert "score" in results[0]
