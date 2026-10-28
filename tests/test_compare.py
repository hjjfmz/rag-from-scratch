"""Retrieval comparison does not invent scores when the index is empty."""

from rag.evaluator import summarize_retrieval_comparison, compare_hybrid_and_graph


def test_summarize_averages_only_scored_questions():
    summary = summarize_retrieval_comparison([
        {"hybrid_precision": 1.0, "hybrid_graph_precision": 0.5},
        {"hybrid_precision": 0.0, "hybrid_graph_precision": 1.0},
        {"hybrid_precision": None, "hybrid_graph_precision": None},
    ])
    assert summary["num_questions"] == 3
    assert summary["hybrid_precision"] == 0.5
    assert summary["hybrid_graph_precision"] == 0.75


def test_compare_skips_when_index_is_empty(monkeypatch):
    monkeypatch.setattr("rag.evaluator.index_has_chunks", lambda: False)
    result = compare_hybrid_and_graph([
        {"question": "退货政策是什么？", "expected_source": "policies/refund-policy.md"},
    ])
    assert result["skipped"] is True
    assert result["hybrid_precision"] is None
    assert result["hybrid_graph_precision"] is None
    assert result["individual"] == []
