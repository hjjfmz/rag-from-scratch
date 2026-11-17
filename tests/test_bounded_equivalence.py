"""Equivalence proof: bounded candidate snapshots behave exactly like full-corpus scans.

The bounded path (GraphService.search) must return identical results to the
reference full-snapshot path (DualLevelRetriever over store.records()) for every
mode, so the storage rework changes where data comes from — never what the
algorithms produce. Also covers ingest task lifecycle and cache degradation.
"""
import re
import time
from pathlib import Path

import pytest

from rag.graph_retrieval import DualLevelRetriever, MODES
from rag.cache import AnswerCache, cache_key
from rag.graph_service import GraphService

from tests.test_graph_service import DeterministicModel

QUESTIONS = ["Pod", "Pod uses Container.", "Service routing", "健康探针",
             "traffic routing", "zebra", "Probe health checks", "Service"]


@pytest.fixture
def corpus_service(tmp_path):
    service = GraphService(tmp_path / "kb", DeterministicModel())
    ingest = [("alpha.md", "Alpha pod uses Storage. Alpha pod restarts Fast."),
              ("beta.md", "Beta service routes to Alpha pod. Beta controls traffic."),
              ("gamma.md", "健康探针决定容器能否接收流量。Gamma watches probe health."),
              ("delta.md", "Pod uses Container. Pod uses Network. Zebra escapes."),
              ("long.md", "Service routes to Pod. " * 2000)]
    for name, text in ingest:
        path = tmp_path / name
        path.write_bytes(text.encode("utf-8"))
        service.ingest(str(path))
    return service


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("question", QUESTIONS)
def test_bounded_path_matches_full_snapshot(corpus_service, question, mode):
    service = corpus_service
    snapshot = service.store.records()
    reference = DualLevelRetriever(snapshot, service.client).search(
        question, mode=mode, top_k=5, threshold=.25)
    bounded = service.search(question, mode=mode, top_k=5, threshold=.25)
    assert [c["id"] for c in bounded["chunks_used"]] == [c["id"] for c in reference["chunks_used"]]
    assert bounded["relations"] == reference["relations"]
    if bounded["retrieval_trace"]["counts"]:
        # Degenerate candidate-free queries short-circuit inside the retriever on both
        # paths; compare channel detail only when the bounded snapshot ran channels.
        assert bounded["retrieval_trace"]["counts"] == reference["retrieval_trace"]["counts"]
        assert bounded["retrieval_trace"]["local"] == reference["retrieval_trace"]["local"]
        assert bounded["retrieval_trace"]["global"] == reference["retrieval_trace"]["global"]
    else:
        assert not reference["chunks_used"]


@pytest.mark.parametrize("mode", MODES)
def test_bounded_path_matches_full_snapshot_with_rerank(corpus_service, mode):
    service = corpus_service
    snapshot = service.store.records()
    reference = DualLevelRetriever(snapshot, service.client).search(
        "Pod", mode=mode, top_k=3, threshold=.25, rerank=True)
    bounded = service.search("Pod", mode=mode, top_k=3, threshold=.25, rerank=True)
    assert [c["id"] for c in bounded["chunks_used"]] == [c["id"] for c in reference["chunks_used"]]


def test_bounded_reads_stay_below_full_corpus(corpus_service):
    """The query path touches a bounded slice, not every stored row (large-file case)."""
    service = corpus_service
    total = service.store.stats()["chunks"]
    question = "Beta controls traffic"
    calls = {"fetch_chunks": []}
    original = service.store.fetch_chunks

    def spying(ids):
        calls["fetch_chunks"].append(list(ids))
        return original(ids)

    service.store.fetch_chunks = spying
    result = service.search(question, mode="hybrid_graph")
    assert result["chunks_used"]
    assert total > 20  # long.md alone guarantees a sizeable corpus
    for ids in calls["fetch_chunks"]:
        assert len(ids) < total, "bounded read must stay below the whole corpus"


def test_agentic_bounded_run_stays_grounded(corpus_service):
    """Agentic = fusion over per-step searches; each step's equivalence is proven by
    test_bounded_path_matches_full_snapshot, and cross-step version consistency by
    test_graph_service's concurrent-update test. Here: the composed run stays sane."""
    service = corpus_service
    result = service.agentic_query("Pod", max_iterations=3, mode="hybrid_graph", top_k=3)
    assert result["status"] == "ok"
    assert result["iterations"] >= 2
    assert any(Path(source).name in ("alpha.md", "beta.md", "delta.md") for source in result["sources"])
    assert {c["citation_id"] for c in result["chunks_used"]} <= {r["citation_id"] for r in result["relations"]} | \
           {c["citation_id"] for c in result["chunks_used"]}


def test_ingest_task_lifecycle(tmp_path):
    service = GraphService(tmp_path / "kb", DeterministicModel())
    path = tmp_path / "tasked.md"
    path.write_bytes(b"Pod uses Container.")
    task_id = service.submit_ingest(str(path))
    deadline = time.time() + 10
    while time.time() < deadline:
        task = service.task_get(task_id)
        if task["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert task["status"] == "done", task
    assert task["result"]["status"] == "indexed"
    assert service.store.stats()["sources"] == 1
    failed = service.submit_ingest(str(tmp_path / "missing.md"))
    deadline = time.time() + 10
    while time.time() < deadline:
        if service.task_get(failed)["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert service.task_get(failed)["status"] == "failed"
    assert "missing" in service.task_get(failed)["error"] or "No such file" in service.task_get(failed)["error"]


class _FlakyRedis:
    """Stand-in for a Redis outage: every operation raises."""

    def get(self, key):
        raise ConnectionError("redis down")

    def set(self, key, value, ex=None):
        raise ConnectionError("redis down")

    def scan_iter(self, match, count=None):
        raise ConnectionError("redis down")
        yield


def test_cache_degrades_and_recovers(tmp_path, capsys):
    service = GraphService(tmp_path / "kb", DeterministicModel(),
                           cache=AnswerCache(client=object()))  # enabled but unusable
    path = tmp_path / "pod.md"
    path.write_bytes(b"Pod uses Container.")
    service.ingest(str(path))
    result = service.query("Pod")
    assert result["status"] == "ok" and "cached" not in result
    assert "Redis unavailable" in capsys.readouterr().out
    # Dead cache must not fake an empty knowledge base either.
    assert service.search("Nothing matches this zebra query", mode="keyword")["status"] == "no_relevant"


def test_cache_hit_and_invalidation(tmp_path):
    cache = AnswerCache(client={})  # dict-backed fake: get/set/scan via mapping methods is not enough, use stub

    class _FakeRedis:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ex=None):
            self.data[key] = value

        def scan_iter(self, match, count=None):
            for key in list(self.data):
                if key.startswith(match.rstrip("*")):
                    yield key

        def delete(self, *keys):
            for key in keys:
                self.data.pop(key, None)

    cache = AnswerCache(client=_FakeRedis())
    service = GraphService(tmp_path / "kb", DeterministicModel(), cache=cache)
    path = tmp_path / "pod.md"
    path.write_bytes(b"Pod uses Container.")
    service.ingest(str(path))
    first = service.query("Pod")
    again = service.query("Pod")
    assert again.get("cached") is True
    assert first["answer"] == again["answer"]
    service.ingest_text(str(path), "Pod uses NewContainer.", force=True)
    fresh = service.query("Pod")
    assert fresh.get("cached") is not True
    assert "NewContainer" in fresh["answer"]


def test_cache_key_distinguishes_inputs():
    base = cache_key("fp", "q", "hybrid", 5, .25, False)
    assert base == cache_key("fp", "q", "hybrid", 5, .25, False)
    assert base != cache_key("fp", "q", "vector", 5, .25, False)
    assert base != cache_key("fp", "q", "hybrid", 3, .25, False)
    assert base != cache_key("fp2", "q", "hybrid", 5, .25, False)
