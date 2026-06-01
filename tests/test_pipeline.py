"""
Tests for rag.pipeline — focus on retrieval miss / low-confidence handling.
"""

from unittest.mock import MagicMock, patch

from rag.pipeline import RAGPipeline


def _make_pipeline_with_mock_store(search_mode: str = "vector"):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.index_path = "rag_index.json"
    pipeline.chunk_size = 500
    pipeline.chunk_overlap = 100
    pipeline.top_k = 5
    pipeline.threshold = 0.3
    pipeline.search_mode = search_mode
    pipeline.use_reranker = False
    pipeline.store = MagicMock()
    pipeline.store.__len__ = MagicMock(return_value=10)
    return pipeline


class TestPipelineFallbacks:
    @patch("rag.pipeline.generate_chat")
    def test_empty_store_returns_no_docs(self, mock_chat):
        mock_chat.return_value = "chat fallback answer"
        pipeline = _make_pipeline_with_mock_store()
        pipeline.store.__len__ = MagicMock(return_value=0)

        result = pipeline.query("test?")
        assert result["status"] == "no_docs"
        assert result["sources"] == []
        assert result["user_guidance"]
        assert result["answer"] == "chat fallback answer"

    @patch("rag.pipeline.generate_chat")
    @patch("rag.pipeline.Retriever")
    def test_no_relevant_returns_status(self, mock_retriever_cls, mock_chat):
        mock_chat.return_value = "chat fallback answer"
        pipeline = _make_pipeline_with_mock_store()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        result = pipeline.query("test?")
        assert result["status"] == "no_relevant"
        assert result["sources"] == []
        assert result["chunks_used"] == []
        assert result["user_guidance"]

    @patch("rag.pipeline.generate_chat")
    @patch("rag.pipeline.Retriever")
    def test_low_confidence_returns_status(self, mock_retriever_cls, mock_chat):
        mock_chat.return_value = "chat fallback answer"
        pipeline = _make_pipeline_with_mock_store(search_mode="vector")
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": "x", "metadata": {"source": "a.md", "chunk_index": 0}, "score": 0.31},
            {"text": "y", "metadata": {"source": "b.md", "chunk_index": 1}, "score": 0.30},
        ]
        mock_retriever_cls.return_value = mock_retriever

        result = pipeline.query("test?")
        assert result["status"] == "low_confidence"
        assert result["sources"] == []
        assert result["chunks_used"] == []
        assert "retrieval_quality" in result
        assert result["retrieval_quality"].get("search_mode") == "vector"
        assert result["answer"] == "chat fallback answer"
