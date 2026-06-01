"""
Tests for rag/reranker.py — LLM-based Reranking (TDD)

📚 LEARN: We write tests FIRST to define the reranker's behavior,
then implement it to make the tests pass.

The reranker takes (query, chunks) and asks the LLM to score each
chunk's relevance on a 0-10 scale. We mock the LLM calls in tests.
"""

import pytest
from unittest.mock import patch, MagicMock

from rag.reranker import (
    build_rerank_prompt,
    parse_rerank_score,
    rerank,
)


# ── Prompt Building Tests ─────────────────────────────────────


class TestBuildRerankPrompt:
    """Test that reranking prompts are well-formed."""

    def test_contains_query(self):
        prompt = build_rerank_prompt("What is the return policy?", "You can return items.")
        assert "return policy" in prompt

    def test_contains_passage(self):
        prompt = build_rerank_prompt("What is the return policy?", "30-day returns accepted.")
        assert "30-day returns accepted" in prompt

    def test_asks_for_score(self):
        prompt = build_rerank_prompt("test?", "passage")
        # Prompt should instruct the LLM to return a numeric score
        assert "0" in prompt and "10" in prompt


# ── Score Parsing Tests ───────────────────────────────────────


class TestParseRerankScore:
    """Test extracting a 0-10 score from LLM rerank output."""

    def test_simple_integer(self):
        assert parse_rerank_score("8") == 8

    def test_score_with_text(self):
        assert parse_rerank_score("Relevance: 7") == 7

    def test_decimal_score(self):
        assert parse_rerank_score("8.5") == 8.5

    def test_score_zero(self):
        assert parse_rerank_score("0") == 0

    def test_score_ten(self):
        assert parse_rerank_score("10") == 10

    def test_clamp_above_ten(self):
        assert parse_rerank_score("15") == 10

    def test_clamp_below_zero(self):
        assert parse_rerank_score("-3") == 0

    def test_no_score_found(self):
        assert parse_rerank_score("I cannot determine relevance") == 0

    def test_score_in_sentence(self):
        assert parse_rerank_score("This passage is highly relevant. Score: 9") == 9


# ── Rerank Integration Tests (Mocked LLM) ────────────────────


class TestRerank:
    """Test the full reranking flow with mocked LLM responses."""

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_reorders_by_score(self, mock_llm):
        """Chunks should be reordered by the LLM's relevance scores."""
        # LLM returns scores: chunk 0 → 3, chunk 1 → 9, chunk 2 → 6
        mock_llm.side_effect = ["3", "9", "6"]

        chunks = [
            {"text": "About shipping", "metadata": {"source": "a.md"}, "score": 0.9},
            {"text": "About returns", "metadata": {"source": "b.md"}, "score": 0.8},
            {"text": "About loyalty", "metadata": {"source": "c.md"}, "score": 0.7},
        ]

        result = rerank("What is the return policy?", chunks, top_k=3)

        # chunk 1 (score 9) should be first, chunk 2 (score 6) second
        assert result[0]["text"] == "About returns"
        assert result[1]["text"] == "About loyalty"
        assert result[2]["text"] == "About shipping"

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_top_k_limits(self, mock_llm):
        """Should only return top_k results after reranking."""
        mock_llm.side_effect = ["5", "8", "2", "9"]

        chunks = [
            {"text": f"chunk {i}", "metadata": {}, "score": 0.5}
            for i in range(4)
        ]

        result = rerank("query", chunks, top_k=2)
        assert len(result) == 2

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_preserves_metadata(self, mock_llm):
        """Original metadata should be preserved after reranking."""
        mock_llm.return_value = "7"

        chunks = [
            {"text": "test", "metadata": {"source": "policy.md", "chunk_index": 3}, "score": 0.8},
        ]

        result = rerank("query", chunks, top_k=1)
        assert result[0]["metadata"]["source"] == "policy.md"
        assert result[0]["metadata"]["chunk_index"] == 3

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_adds_rerank_score(self, mock_llm):
        """Reranked results should have a 'rerank_score' field."""
        mock_llm.return_value = "7"

        chunks = [
            {"text": "test", "metadata": {}, "score": 0.8},
        ]

        result = rerank("query", chunks, top_k=1)
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] == 7.0

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_empty_chunks(self, mock_llm):
        """Should handle empty chunk list gracefully."""
        result = rerank("query", [], top_k=5)
        assert result == []
        mock_llm.assert_not_called()

    @patch("rag.reranker._call_rerank_llm")
    def test_rerank_llm_failure_uses_fallback(self, mock_llm):
        """If LLM call fails, chunk should get score 0 (not crash)."""
        mock_llm.side_effect = Exception("API error")

        chunks = [
            {"text": "test", "metadata": {}, "score": 0.8},
        ]

        result = rerank("query", chunks, top_k=1)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0
