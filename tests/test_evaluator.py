"""
Tests for rag/evaluator.py — Evaluation Framework

📚 LEARN: We test the evaluator with MOCKED retriever and generator responses.
This lets us verify scoring logic without making real API calls.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from rag.evaluator import (
    compute_retrieval_precision,
    parse_llm_score,
    build_faithfulness_prompt,
    build_relevance_prompt,
    run_evaluation,
    load_test_questions,
)


# ── Retrieval Precision Tests ─────────────────────────────────


class TestRetrievalPrecision:
    """Test the retrieval precision metric (no LLM needed)."""

    def test_perfect_precision(self):
        """All retrieved chunks come from the expected source."""
        retrieved = [
            {"metadata": {"source": "policies/refund-policy.md"}, "score": 0.9},
            {"metadata": {"source": "policies/refund-policy.md"}, "score": 0.8},
        ]
        score = compute_retrieval_precision(retrieved, "policies/refund-policy.md")
        assert score == 1.0

    def test_partial_precision(self):
        """Some chunks from expected source, some from other sources."""
        retrieved = [
            {"metadata": {"source": "policies/refund-policy.md"}, "score": 0.9},
            {"metadata": {"source": "faq/customer-faq.md"}, "score": 0.8},
            {"metadata": {"source": "hr/pto-policy.md"}, "score": 0.7},
            {"metadata": {"source": "policies/refund-policy.md"}, "score": 0.6},
        ]
        score = compute_retrieval_precision(retrieved, "policies/refund-policy.md")
        assert score == 0.5  # 2 out of 4

    def test_zero_precision(self):
        """No chunks from the expected source."""
        retrieved = [
            {"metadata": {"source": "hr/pto-policy.md"}, "score": 0.9},
            {"metadata": {"source": "faq/customer-faq.md"}, "score": 0.8},
        ]
        score = compute_retrieval_precision(retrieved, "policies/refund-policy.md")
        assert score == 0.0

    def test_empty_retrieved(self):
        """No chunks retrieved at all."""
        score = compute_retrieval_precision([], "policies/refund-policy.md")
        assert score == 0.0

    def test_no_expected_source(self):
        """Question has no expected source (unanswerable) — skip precision."""
        score = compute_retrieval_precision(
            [{"metadata": {"source": "anything.md"}, "score": 0.5}],
            expected_source=None,
        )
        assert score is None  # Can't measure precision without expected source

    def test_partial_path_match(self):
        """Expected source is a partial path — matches if chunk source ends with it."""
        retrieved = [
            {"metadata": {"source": "sample_data/policies/refund-policy.md"}, "score": 0.9},
        ]
        score = compute_retrieval_precision(retrieved, "policies/refund-policy.md")
        assert score == 1.0


# ── LLM Score Parsing Tests ───────────────────────────────────


class TestParseLlmScore:
    """Test extracting a 0.0-1.0 score from LLM text output."""

    def test_simple_score(self):
        assert parse_llm_score("0.8") == 0.8

    def test_score_with_text(self):
        assert parse_llm_score("Score: 0.9") == 0.9

    def test_score_in_sentence(self):
        assert parse_llm_score("The answer is well grounded. Score: 0.85") == 0.85

    def test_score_one(self):
        assert parse_llm_score("1.0") == 1.0

    def test_score_zero(self):
        assert parse_llm_score("0.0") == 0.0

    def test_integer_score(self):
        """LLM might return '1' instead of '1.0'."""
        result = parse_llm_score("1")
        assert result == 1.0

    def test_no_score_found(self):
        """If no number found, return 0.0 as default."""
        assert parse_llm_score("I can't determine a score") == 0.0

    def test_clamp_above_one(self):
        """Scores above 1.0 get clamped."""
        assert parse_llm_score("Score: 1.5") == 1.0

    def test_negative_score(self):
        """Negative scores get clamped to 0.0."""
        assert parse_llm_score("Score: -0.3") == 0.0


# ── Prompt Building Tests ─────────────────────────────────────


class TestPromptBuilding:
    """Test that evaluation prompts are well-formed."""

    def test_faithfulness_prompt_contains_context(self):
        context = "GlowBeauty offers 30-day returns."
        answer = "You can return items within 30 days."
        prompt = build_faithfulness_prompt(context, answer)
        assert "30-day returns" in prompt
        assert "return items within 30 days" in prompt

    def test_relevance_prompt_contains_question(self):
        question = "What is the return policy?"
        answer = "You can return items within 30 days."
        prompt = build_relevance_prompt(question, answer)
        assert "return policy" in prompt
        assert "return items within 30 days" in prompt


# ── Load Test Questions ───────────────────────────────────────


class TestLoadTestQuestions:
    """Test loading the eval test set from JSON."""

    def test_load_from_file(self):
        data = [
            {"question": "test?", "expected_source": "src.md", "expected_keywords": ["test"]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            questions = load_test_questions(path)
            assert len(questions) == 1
            assert questions[0]["question"] == "test?"
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_test_questions("nonexistent.json")


# ── Integration Test (Mocked) ─────────────────────────────────


class TestRunEvaluation:
    """Test the full evaluation run with mocked retriever and generator."""

    @patch("rag.evaluator.generate", return_value="Opened products can be returned within 30 days.")
    @patch("rag.evaluator._call_llm_judge")
    @patch("rag.evaluator._retrieve_for_question")
    def test_run_evaluation_basic(self, mock_retrieve, mock_judge, _mock_generate):
        """Test that run_evaluation produces a results dict with all metrics."""
        # Mock retrieval: returns chunks from expected source
        mock_retrieve.return_value = [
            {"text": "GlowBeauty offers 30-day returns on opened products.",
             "metadata": {"source": "sample_data/policies/refund-policy.md"},
             "score": 0.85},
        ]
        # Mock LLM judge: returns high scores
        mock_judge.return_value = "0.9"

        test_questions = [
            {
                "question": "Can I return an opened product?",
                "expected_source": "policies/refund-policy.md",
                "expected_keywords": ["30 days", "opened"],
            },
        ]

        results = run_evaluation(test_questions)

        assert "summary" in results
        assert "individual" in results
        assert results["summary"]["num_questions"] == 1
        assert results["summary"]["avg_retrieval_precision"] == 1.0
        assert results["summary"]["avg_faithfulness"] >= 0.0
        assert results["summary"]["avg_relevance"] >= 0.0

    @patch("rag.evaluator._call_llm_judge")
    @patch("rag.evaluator._retrieve_for_question")
    def test_unanswerable_question(self, mock_retrieve, mock_judge):
        """Unanswerable questions (expected_source=null) skip precision."""
        mock_retrieve.return_value = []  # No results
        mock_judge.return_value = "0.0"

        test_questions = [
            {
                "question": "What is the capital of France?",
                "expected_source": None,
                "expected_keywords": [],
            },
        ]

        results = run_evaluation(test_questions)
        # Precision should be None for unanswerable, so avg excludes it
        assert results["summary"]["num_questions"] == 1
