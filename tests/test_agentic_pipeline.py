"""
Tests for rag/agentic_pipeline.py — End-to-End Agentic RAG

📚 LEARN: These tests verify the full ReAct loop — from receiving a
question, through multiple agent iterations, to producing a final answer.
We mock both the LLM (agent decisions) and the retriever (search results)
to isolate the agentic orchestration logic.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from rag.agentic_pipeline import AgenticRAGPipeline
from rag.agent import (
    ReActAgent,
    AgentStep,
    ReformulateQuery,
    RequestMoreContext,
    Answer,
)


# ── Helper: Create a Pipeline with Mocked Store ──────────────

def _make_pipeline(max_iterations=5, store_size=10):
    """Create an AgenticRAGPipeline with a mocked non-empty vector store."""
    pipeline = AgenticRAGPipeline.__new__(AgenticRAGPipeline)
    pipeline.rag_pipeline = MagicMock()
    pipeline.rag_pipeline.store = MagicMock()
    pipeline.rag_pipeline.store.__len__ = MagicMock(return_value=store_size)
    pipeline.max_iterations = max_iterations
    pipeline.default_top_k = 5
    pipeline.default_threshold = 0.3
    pipeline.default_search_mode = "vector"
    pipeline.use_reranker = False
    return pipeline


def _make_chunks(texts, source="test.md"):
    """Create mock retrieval results."""
    return [
        {
            "text": t,
            "metadata": {"source": source, "chunk_index": i},
            "score": 0.9 - i * 0.1,
        }
        for i, t in enumerate(texts)
    ]


# ── Basic Flow Tests ─────────────────────────────────────────


class TestAgenticPipelineBasicFlow:
    """Test the core ReAct loop orchestration."""

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_single_iteration_answer(self, mock_llm, mock_retriever_cls, mock_gen):
        """Agent retrieves once and answers — the simplest flow."""
        pipeline = _make_pipeline(max_iterations=5)

        # Iteration 1: Agent requests context
        # Iteration 1 (cont): Agent sees context and answers
        mock_llm.side_effect = [
            "Thought: Let me search for info.\n"
            'Action: RequestMoreContext("test query", 5, "vector")',

            "Thought: I have good context now.\n"
            'Action: Answer("The answer based on docs.")',
        ]

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_chunks(["Relevant chunk about topic"])
        mock_retriever_cls.return_value = mock_retriever

        # Mock generator
        mock_gen.return_value = {
            "answer": "Generated answer",
            "sources": ["test.md"],
            "chunks_used": _make_chunks(["Relevant chunk about topic"]),
        }

        result = pipeline.query("What is the topic?")

        assert result["answer"] == "Generated answer"
        assert result["iterations"] == 2
        assert len(result["steps"]) == 2

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_reformulate_then_answer(self, mock_llm, mock_retriever_cls, mock_gen):
        """Agent reformulates query, retrieves, then answers — 3 iterations."""
        pipeline = _make_pipeline(max_iterations=5)

        mock_llm.side_effect = [
            # Iteration 1: Initial retrieval
            "Thought: Let me search.\n"
            'Action: RequestMoreContext("initial query")',
            # Iteration 2: Reformulate
            "Thought: Results aren't great, let me try different terms.\n"
            'Action: ReformulateQuery("better search terms")',
            # Iteration 3: Answer
            "Thought: Now I have good context.\n"
            'Action: Answer("Found it.")',
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_chunks(["Some chunk"])
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "Final answer",
            "sources": ["test.md"],
            "chunks_used": [],
        }

        result = pipeline.query("Complex question?")

        assert result["iterations"] == 3
        # Retriever should have been called twice (initial + reformulated)
        assert mock_retriever.retrieve.call_count == 2

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_max_iterations_forces_answer(self, mock_llm, mock_retriever_cls, mock_gen):
        """When max iterations hit, pipeline should still produce an answer."""
        pipeline = _make_pipeline(max_iterations=2)

        # Agent never chooses Answer
        mock_llm.side_effect = [
            "Thought: Need more.\n"
            'Action: RequestMoreContext("query 1")',
            "Thought: Still need more.\n"
            'Action: RequestMoreContext("query 2")',
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_chunks(["Chunk"])
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "Forced answer",
            "sources": ["test.md"],
            "chunks_used": [],
        }

        result = pipeline.query("Hard question?")

        assert result["answer"] == "Forced answer"
        assert result["iterations"] == 2


# ── Edge Case Tests ──────────────────────────────────────────


class TestAgenticPipelineEdgeCases:
    """Test error handling and edge cases."""

    def test_empty_store_returns_early(self):
        """Pipeline should return immediately if no documents indexed."""
        pipeline = _make_pipeline(store_size=0)
        result = pipeline.query("Any question?")

        assert "No documents" in result["answer"]
        assert result["iterations"] == 0

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_parse_error_falls_back(self, mock_llm, mock_retriever_cls, mock_gen):
        """If agent output can't be parsed, pipeline should still answer."""
        pipeline = _make_pipeline(max_iterations=3)

        # LLM returns unparseable garbage
        mock_llm.side_effect = ValueError("Can't parse")

        result = pipeline.query("Any question?")

        # Should still return something (no crash)
        assert "answer" in result

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_no_chunks_found(self, mock_llm, mock_retriever_cls, mock_gen):
        """If retriever returns no chunks, agent should still work."""
        pipeline = _make_pipeline(max_iterations=3)

        mock_llm.side_effect = [
            "Thought: Let me search.\n"
            'Action: RequestMoreContext("obscure topic")',
            "Thought: Nothing found, let me answer anyway.\n"
            'Action: Answer("I could not find relevant information.")',
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []  # No results
        mock_retriever_cls.return_value = mock_retriever

        result = pipeline.query("Obscure question?")

        # Should handle gracefully
        assert "answer" in result


# ── Context Accumulation Tests ────────────────────────────────


class TestContextAccumulation:
    """Test that chunks accumulate and deduplicate across iterations."""

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_chunks_accumulate_across_iterations(
        self, mock_llm, mock_retriever_cls, mock_gen
    ):
        """Chunks from multiple retrievals should be merged."""
        pipeline = _make_pipeline(max_iterations=5)

        mock_llm.side_effect = [
            "Thought: Search first.\n"
            'Action: RequestMoreContext("query 1")',
            "Thought: Need more from a different angle.\n"
            'Action: RequestMoreContext("query 2")',
            "Thought: Good enough now.\n"
            'Action: Answer("Done.")',
        ]

        # Different chunks for each retrieval
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = [
            _make_chunks(["Chunk A", "Chunk B"]),
            _make_chunks(["Chunk C", "Chunk D"]),
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "Combined answer",
            "sources": ["test.md"],
            "chunks_used": [],
        }

        result = pipeline.query("Multi-step question?")

        # generate_with_sources should receive accumulated chunks
        call_args = mock_gen.call_args
        chunks_passed = call_args[0][1]  # second positional arg
        assert len(chunks_passed) == 4  # All 4 unique chunks

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_duplicate_chunks_deduplicated(
        self, mock_llm, mock_retriever_cls, mock_gen
    ):
        """Same chunk from different queries should not be duplicated."""
        pipeline = _make_pipeline(max_iterations=5)

        mock_llm.side_effect = [
            "Thought: Search.\n"
            'Action: RequestMoreContext("query 1")',
            "Thought: Try again.\n"
            'Action: RequestMoreContext("query 2")',
            "Thought: Done.\n"
            'Action: Answer("Answer.")',
        ]

        # Both retrievals return the same chunk
        same_chunk = _make_chunks(["Same chunk text"])
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = [same_chunk, same_chunk]
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "Answer",
            "sources": [],
            "chunks_used": [],
        }

        result = pipeline.query("Question?")

        call_args = mock_gen.call_args
        chunks_passed = call_args[0][1]
        assert len(chunks_passed) == 1  # Deduplicated


# ── Verbose Output Tests ─────────────────────────────────────


class TestVerboseOutput:
    """Test that verbose mode prints the reasoning trace."""

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_verbose_prints_trace(self, mock_llm, mock_retriever_cls, mock_gen, capsys):
        """Verbose mode should print Thought, Action, and Observation."""
        pipeline = _make_pipeline(max_iterations=5)

        mock_llm.side_effect = [
            "Thought: I need to search for info.\n"
            'Action: RequestMoreContext("test query")',
            "Thought: Found what I need.\n"
            'Action: Answer("The answer.")',
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_chunks(["A chunk"])
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "Answer",
            "sources": [],
            "chunks_used": [],
        }

        pipeline.query("Question?", verbose=True)

        captured = capsys.readouterr()
        assert "Thought:" in captured.out or "💭" in captured.out
        assert "Action:" in captured.out or "🎯" in captured.out
        assert "Iteration" in captured.out


# ── Return Value Tests ────────────────────────────────────────


class TestReturnValue:
    """Test the structure of the returned result dict."""

    @patch("rag.agentic_pipeline.generate_with_sources")
    @patch("rag.agentic_pipeline.Retriever")
    @patch("rag.agent._call_agent_llm")
    def test_result_has_expected_keys(self, mock_llm, mock_retriever_cls, mock_gen):
        """Result should include answer, sources, iterations, and steps."""
        pipeline = _make_pipeline(max_iterations=5)

        mock_llm.side_effect = [
            "Thought: Search.\n"
            'Action: RequestMoreContext("query")',
            "Thought: Done.\n"
            'Action: Answer("Answer.")',
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = _make_chunks(["Chunk"])
        mock_retriever_cls.return_value = mock_retriever

        mock_gen.return_value = {
            "answer": "The answer",
            "sources": ["test.md"],
            "chunks_used": [],
        }

        result = pipeline.query("Question?")

        assert "answer" in result
        assert "sources" in result
        assert "iterations" in result
        assert "steps" in result
        assert isinstance(result["iterations"], int)
        assert isinstance(result["steps"], list)
