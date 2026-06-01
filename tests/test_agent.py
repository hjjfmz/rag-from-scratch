"""
Tests for rag/agent.py — ReAct Agent Core (Action Parsing & State)

📚 LEARN: These tests verify the agent's "brain" — how it parses LLM
output into structured actions, manages state across iterations, and
handles edge cases gracefully. We mock the LLM so tests run without
API calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from rag.agent import (
    parse_agent_response,
    _parse_action_string,
    ReActAgent,
    AgentStep,
    ReformulateQuery,
    RequestMoreContext,
    Answer,
    AGENT_SYSTEM_PROMPT,
)


# ── Action Parsing Tests ─────────────────────────────────────


class TestParseActionString:
    """Test parsing individual action strings into dataclasses."""

    # -- Answer action --

    def test_answer_double_quotes(self):
        action = _parse_action_string('Answer("The answer is 42.")')
        assert isinstance(action, Answer)
        assert action.response == "The answer is 42."

    def test_answer_single_quotes(self):
        action = _parse_action_string("Answer('Based on the docs, yes.')")
        assert isinstance(action, Answer)
        assert action.response == "Based on the docs, yes."

    def test_answer_multiline(self):
        action = _parse_action_string('Answer("Line one.\nLine two.")')
        assert isinstance(action, Answer)
        assert "Line one" in action.response
        assert "Line two" in action.response

    def test_answer_with_special_chars(self):
        action = _parse_action_string('Answer("It costs $5.99 (30% off).")')
        assert isinstance(action, Answer)
        assert "$5.99" in action.response

    # -- ReformulateQuery action --

    def test_reformulate_double_quotes(self):
        action = _parse_action_string('ReformulateQuery("better search terms")')
        assert isinstance(action, ReformulateQuery)
        assert action.new_query == "better search terms"

    def test_reformulate_single_quotes(self):
        action = _parse_action_string("ReformulateQuery('overfitting prevention')")
        assert isinstance(action, ReformulateQuery)
        assert action.new_query == "overfitting prevention"

    # -- RequestMoreContext action --

    def test_request_all_params(self):
        action = _parse_action_string('RequestMoreContext("my query", 10, "hybrid")')
        assert isinstance(action, RequestMoreContext)
        assert action.query == "my query"
        assert action.top_k == 10
        assert action.search_mode == "hybrid"

    def test_request_query_only(self):
        action = _parse_action_string('RequestMoreContext("my query")')
        assert isinstance(action, RequestMoreContext)
        assert action.query == "my query"
        assert action.top_k == 5  # default
        assert action.search_mode == "vector"  # default

    def test_request_query_and_top_k(self):
        action = _parse_action_string('RequestMoreContext("my query", 3)')
        assert isinstance(action, RequestMoreContext)
        assert action.query == "my query"
        assert action.top_k == 3
        assert action.search_mode == "vector"  # default

    def test_request_keyword_mode(self):
        action = _parse_action_string('RequestMoreContext("exact terms", 5, "keyword")')
        assert isinstance(action, RequestMoreContext)
        assert action.search_mode == "keyword"

    # -- Error cases --

    def test_unknown_action_raises(self):
        with pytest.raises(ValueError, match="Unknown action"):
            _parse_action_string("DoSomethingElse(123)")

    def test_garbage_input_raises(self):
        with pytest.raises(ValueError, match="Unknown action"):
            _parse_action_string("not an action at all")


# ── Full Response Parsing Tests ───────────────────────────────


class TestParseAgentResponse:
    """Test parsing complete Thought + Action responses."""

    def test_standard_format(self):
        text = (
            "Thought: I need to find information about return policies.\n"
            'Action: RequestMoreContext("return policy", 5, "vector")'
        )
        thought, action = parse_agent_response(text)
        assert "return policies" in thought
        assert isinstance(action, RequestMoreContext)
        assert action.query == "return policy"

    def test_answer_format(self):
        text = (
            "Thought: I have enough context to answer the question.\n"
            'Action: Answer("The return policy allows 30-day returns.")'
        )
        thought, action = parse_agent_response(text)
        assert "enough context" in thought
        assert isinstance(action, Answer)
        assert "30-day" in action.response

    def test_reformulate_format(self):
        text = (
            "Thought: The chunks don't mention the specific product.\n"
            'Action: ReformulateQuery("specific product warranty terms")'
        )
        thought, action = parse_agent_response(text)
        assert isinstance(action, ReformulateQuery)
        assert "warranty" in action.new_query

    def test_multiline_thought(self):
        text = (
            "Thought: I found some info about returns but nothing about\n"
            "exchanges specifically. Let me search differently.\n"
            'Action: ReformulateQuery("exchange policy")'
        )
        thought, action = parse_agent_response(text)
        assert "exchanges" in thought
        assert isinstance(action, ReformulateQuery)

    def test_no_action_raises(self):
        text = "Thought: I'm thinking but never acting."
        with pytest.raises(ValueError, match="No Action found"):
            parse_agent_response(text)

    def test_extra_whitespace(self):
        text = (
            "Thought:   Lots of spaces here.  \n"
            'Action:   RequestMoreContext("query")  '
        )
        thought, action = parse_agent_response(text)
        assert "Lots of spaces" in thought
        assert isinstance(action, RequestMoreContext)


# ── ReActAgent State Management Tests ─────────────────────────


class TestReActAgentState:
    """Test the agent's state management (chunks, queries, history)."""

    def test_initial_state(self):
        agent = ReActAgent(max_iterations=5)
        assert agent.max_iterations == 5
        assert agent.steps == []
        assert agent.accumulated_chunks == []
        assert agent.tried_queries == set()

    def test_add_chunks_deduplicates(self):
        """Same chunk text should not be added twice."""
        agent = ReActAgent()
        chunk_a = {"text": "Hello world", "metadata": {"source": "a.md"}, "score": 0.9}
        chunk_b = {"text": "Hello world", "metadata": {"source": "b.md"}, "score": 0.8}
        chunk_c = {"text": "Different text", "metadata": {"source": "c.md"}, "score": 0.7}

        agent.add_chunks([chunk_a, chunk_b, chunk_c])
        assert len(agent.accumulated_chunks) == 2  # deduplicated

    def test_add_chunks_preserves_first_occurrence(self):
        """When deduplicating, the first occurrence should be kept."""
        agent = ReActAgent()
        chunk1 = {"text": "Same text", "metadata": {"source": "first.md"}, "score": 0.9}
        chunk2 = {"text": "Same text", "metadata": {"source": "second.md"}, "score": 0.5}

        agent.add_chunks([chunk1])
        agent.add_chunks([chunk2])
        assert agent.accumulated_chunks[0]["metadata"]["source"] == "first.md"

    def test_add_chunks_across_iterations(self):
        """Chunks from different iterations should accumulate."""
        agent = ReActAgent()
        agent.add_chunks([{"text": "chunk 1", "metadata": {}, "score": 0.9}])
        agent.add_chunks([{"text": "chunk 2", "metadata": {}, "score": 0.8}])
        assert len(agent.accumulated_chunks) == 2

    def test_tried_queries_tracked(self):
        """The agent should track which queries it has tried."""
        agent = ReActAgent()
        agent.tried_queries.add("first query")
        agent.tried_queries.add("second query")
        assert "first query" in agent.tried_queries
        assert "third query" not in agent.tried_queries


# ── Message Building Tests ────────────────────────────────────


class TestBuildMessages:
    """Test that conversation history is correctly built."""

    def test_initial_messages(self):
        """First call should have system prompt + user question."""
        agent = ReActAgent()
        messages = agent.build_messages("What is ML?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is ML?" in messages[1]["content"]

    def test_messages_include_history(self):
        """After one step, messages should include the previous cycle."""
        agent = ReActAgent()
        step = AgentStep(
            thought="I need to search",
            action=RequestMoreContext(query="machine learning"),
            observation="Retrieved 3 chunks...",
        )
        agent.steps.append(step)

        messages = agent.build_messages("What is ML?")
        assert len(messages) == 4  # system + user + assistant + observation
        assert messages[2]["role"] == "assistant"
        assert "I need to search" in messages[2]["content"]
        assert messages[3]["role"] == "user"
        assert "Retrieved 3 chunks" in messages[3]["content"]

    def test_messages_grow_with_steps(self):
        """Each step adds 2 messages (assistant + observation)."""
        agent = ReActAgent()
        for i in range(3):
            agent.steps.append(AgentStep(
                thought=f"Thought {i}",
                action=RequestMoreContext(query=f"query {i}"),
                observation=f"Observation {i}",
            ))

        messages = agent.build_messages("Question?")
        # 2 base + (3 steps × 2 messages each) = 8
        assert len(messages) == 8


# ── Action Formatting Tests ───────────────────────────────────


class TestFormatAction:
    """Test formatting actions back to string representation."""

    def test_format_answer(self):
        formatted = ReActAgent._format_action(Answer(response="The answer."))
        assert "Answer" in formatted
        assert "The answer." in formatted

    def test_format_reformulate(self):
        formatted = ReActAgent._format_action(ReformulateQuery(new_query="better query"))
        assert "ReformulateQuery" in formatted
        assert "better query" in formatted

    def test_format_request_more_context(self):
        formatted = ReActAgent._format_action(
            RequestMoreContext(query="my query", top_k=10, search_mode="hybrid")
        )
        assert "RequestMoreContext" in formatted
        assert "my query" in formatted
        assert "10" in formatted
        assert "hybrid" in formatted


# ── System Prompt Tests ───────────────────────────────────────


class TestAgentSystemPrompt:
    """Verify the system prompt contains essential instructions."""

    def test_explains_react_format(self):
        assert "Thought" in AGENT_SYSTEM_PROMPT
        assert "Action" in AGENT_SYSTEM_PROMPT
        assert "Observation" in AGENT_SYSTEM_PROMPT

    def test_lists_all_actions(self):
        assert "RequestMoreContext" in AGENT_SYSTEM_PROMPT
        assert "ReformulateQuery" in AGENT_SYSTEM_PROMPT
        assert "Answer" in AGENT_SYSTEM_PROMPT

    def test_mentions_search_modes(self):
        assert "vector" in AGENT_SYSTEM_PROMPT
        assert "keyword" in AGENT_SYSTEM_PROMPT
        assert "hybrid" in AGENT_SYSTEM_PROMPT

    def test_includes_examples(self):
        assert "Example:" in AGENT_SYSTEM_PROMPT or "example" in AGENT_SYSTEM_PROMPT.lower()

    def test_mentions_no_repeat_rule(self):
        assert "repeat" in AGENT_SYSTEM_PROMPT.lower()


# ── Agent LLM Call Tests (Mocked) ─────────────────────────────


class TestGetNextAction:
    """Test the agent's decision-making with mocked LLM."""

    @patch("rag.agent._call_agent_llm")
    def test_first_action_is_retrieval(self, mock_llm):
        """On the first call, the agent should retrieve context."""
        mock_llm.return_value = (
            "Thought: I need to find information about ML.\n"
            'Action: RequestMoreContext("machine learning basics", 5, "vector")'
        )
        agent = ReActAgent()
        thought, action = agent.get_next_action("What is ML?")

        assert isinstance(action, RequestMoreContext)
        assert "machine learning" in action.query

    @patch("rag.agent._call_agent_llm")
    def test_agent_can_answer(self, mock_llm):
        """Agent should be able to choose Answer."""
        mock_llm.return_value = (
            "Thought: I have enough context now.\n"
            'Action: Answer("ML is a subset of AI.")'
        )
        agent = ReActAgent()
        thought, action = agent.get_next_action("What is ML?")

        assert isinstance(action, Answer)
        assert "ML" in action.response

    @patch("rag.agent._call_agent_llm")
    def test_agent_can_reformulate(self, mock_llm):
        """Agent should be able to reformulate queries."""
        mock_llm.return_value = (
            "Thought: Results were not relevant, let me try different terms.\n"
            'Action: ReformulateQuery("artificial intelligence machine learning definition")'
        )
        agent = ReActAgent()
        thought, action = agent.get_next_action("What is ML?")

        assert isinstance(action, ReformulateQuery)
        assert "artificial intelligence" in action.new_query

    @patch("rag.agent._call_agent_llm")
    def test_invalid_response_raises(self, mock_llm):
        """If LLM doesn't follow format, should raise ValueError."""
        mock_llm.return_value = "I don't know what to do."

        agent = ReActAgent()
        with pytest.raises(ValueError):
            agent.get_next_action("What is ML?")
