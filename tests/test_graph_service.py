"""Behavioral checks for ingestion consistency and genuinely separate graph retrieval levels."""
import re
from pathlib import Path
import pytest

from rag.graph_service import GraphService


class DeterministicModel:
    offline = True
    fingerprint = "test-model-v1"
    metrics = {}
    broken = False

    def embed(self, texts):
        def vector(text):
            text = text.lower()
            return [float(any(w in text for w in words)) for words in (
                ("pod", "container"), ("traffic", "routing", "service"), ("probe", "health"), ("zebra",))]
        return [vector(t) for t in texts]

    def extract(self, text):
        if self.broken:
            raise RuntimeError("extractor unavailable")
        entities, relations = {}, []
        for match in re.finditer(r"(\w+) (uses|restarts|routes to|controls) (\w+)\.", text):
            head, label, tail = match.groups()
            for name in (head, tail):
                entities[name] = {"name": name, "type": "concept", "description": name}
            relations.append({"source": head, "target": tail, "relation": label,
                              "description": match.group(), "keywords": ["traffic routing"] if "Service" in match.group() else ["health"],
                              "evidence": match.group()})
        return {"entities": list(entities.values()), "relations": relations}

    def keywords(self, question):
        return {"low_level": re.findall(r"[A-Z][A-Za-z]+", question), "high_level": [question]}

    def answer(self, question, contexts, relations):
        return "\n".join(f"[{c['citation_id']}] {c['text']}" for c in contexts)

    def plan(self, question):
        return [question, "Service traffic routing"]

    def rerank(self, question, contexts):
        return list(reversed(contexts))

    def health(self):
        return {"offline": True}


@pytest.fixture
def service(tmp_path):
    return GraphService(tmp_path / "kb", DeterministicModel())


def ingest(service, path, text):
    path.write_bytes(text.encode("utf-8"))
    return service.ingest(str(path))


def test_same_filename_content_changes_replace_text_and_graph(service, tmp_path):
    path = tmp_path / "guide.md"
    ingest(service, path, "Pod uses OldContainer.")
    updated = ingest(service, path, "Pod uses NewContainer.")
    assert updated["status"] == "indexed"
    records = service.store.records()
    assert len(records["chunks"]) == 1
    assert [r["target_key"] for r in records["relations"]] == ["newcontainer"]
    assert "OldContainer" not in records["chunks"][0]["text"]
    assert service.ingest(str(path))["status"] == "unchanged"


def test_failed_update_keeps_entire_previous_graph(service, tmp_path):
    path = tmp_path / "guide.md"
    ingest(service, path, "Pod uses Container.")
    service.client.broken = True
    path.write_text("Pod uses BrokenContainer.", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unavailable"):
        service.ingest(str(path))
    assert service.store.records()["relations"][0]["target_key"] == "container"
    assert service.store.check()["ok"] is True


def test_fine_entities_and_coarse_relations_have_independent_evidence(service, tmp_path):
    ingest(service, tmp_path / "pod.md", "Pod uses Container.")
    ingest(service, tmp_path / "routing.md", "Service routes to Pod.")
    fine = service.search("Pod", mode="local", top_k=5)
    coarse = service.search("traffic routing", mode="global", top_k=5)
    assert fine["retrieval_trace"]["local"]["seed_entities"]
    assert fine["retrieval_trace"]["global"]["seed_relations"] == []
    assert coarse["retrieval_trace"]["global"]["seed_relations"]
    assert coarse["retrieval_trace"]["local"]["seed_entities"] == []
    assert any(r["source_key"] == "service" for r in coarse["relations"])
    assert all(r["evidence"] in next(c["text"] for c in service.store.records()["chunks"] if c["id"] == r["chunk_id"]) for r in coarse["relations"])


def test_blank_index_and_irrelevant_query_do_not_invent_answers(service, tmp_path):
    assert service.query("Where?")["status"] == "no_docs"
    ingest(service, tmp_path / "pod.md", "Pod uses Container.")
    assert service.query("zebra vaccine", mode="keyword")["status"] == "no_relevant"


def test_sync_detects_content_and_prunes_only_opt_in(service, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "guide.md"
    path.write_text("Pod uses OldContainer.", encoding="utf-8")
    service.sync(folder)
    path.write_text("Pod uses NewContainer.", encoding="utf-8")
    summary = service.sync(folder)
    assert summary["indexed"] == 1
    path.unlink()
    service.sync(folder)
    assert service.store.stats()["sources"] == 1
    service.sync(folder, prune=True)
    assert service.store.stats()["sources"] == 0


def test_agentic_accumulates_graph_sources_with_bounded_queries(service, tmp_path):
    ingest(service, tmp_path / "pod.md", "Pod uses Container.")
    ingest(service, tmp_path / "service.md", "Service routes to Pod.")
    result = service.agentic_query("Pod", max_iterations=2)
    assert 1 <= result["iterations"] <= 2
    assert result["relations"]
    assert result["sources"]
    assert "[S1]" in result["answer"]


def test_model_change_refuses_to_search_mixed_index(service, tmp_path):
    ingest(service, tmp_path / "pod.md", "Pod uses Container.")
    other = DeterministicModel()
    other.fingerprint = "another-model"
    reopened = GraphService(service.data_dir, other)
    with pytest.raises(ValueError, match="fingerprint|model"):
        reopened.search("Pod")


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": 101}, {"mode": "fake"}, {"threshold": -1}])
def test_retrieval_rejects_invalid_limits(service, kwargs):
    with pytest.raises(ValueError):
        service.search("Pod", **kwargs)


def test_chinese_long_documents_have_bounded_exact_source_chunks(service, tmp_path):
    text = "就绪探针决定容器能否接收流量。" * 1400
    path = tmp_path / "long.md"
    ingest(service, path, text)
    chunks = service.store.records()["chunks"]
    assert len(chunks) > 1
    assert max(len(c["text"]) for c in chunks) <= 900
    assert all(text[c["metadata"]["char_start"]:c["metadata"]["char_end"]] == c["text"] for c in chunks)
    assert service.search("就绪探针", mode="keyword")["chunks_used"]


def test_chunk_positions_preserve_original_line_breaks(service, tmp_path):
    text = "Pod uses Container.\n\nService routes to Pod."
    ingest(service, tmp_path / "spacing.md", text)
    for chunk in service.store.records()["chunks"]:
        assert text[chunk["metadata"]["char_start"]:chunk["metadata"]["char_end"]] == chunk["text"]


def test_agentic_rank_fusion_can_select_newly_discovered_evidence(service, tmp_path):
    ingest(service, tmp_path / "alpha.md", "Alpha uses One.")
    ingest(service, tmp_path / "beta.md", "Beta uses Two.")
    service.client.plan = lambda question: ["Beta", "Beta Two"]
    result = service.agentic_query("Alpha", mode="keyword", top_k=1, max_iterations=3)
    assert result["sources"][0].endswith("beta.md")


def test_agentic_uses_one_snapshot_despite_a_concurrent_source_update(service, tmp_path):
    path = tmp_path / "guide.md"
    ingest(service, path, "Pod uses OldContainer.")
    def planner(question):
        ingest(service, path, "Pod uses NewContainer.")
        return ["NewContainer", "Pod"]
    service.client.plan = planner
    result = service.agentic_query("OldContainer", mode="keyword", top_k=3, max_iterations=3)
    assert "NewContainer" not in result["answer"]
    assert "OldContainer" in result["answer"]


def test_model_insufficient_evidence_is_a_normal_refusal(service, tmp_path):
    ingest(service, tmp_path / "pod.md", "Pod uses Container.")
    service.client.answer = lambda *args: "INSUFFICIENT_EVIDENCE"
    result = service.query("Pod")
    assert result["status"] == "insufficient_evidence"


def test_large_single_file_does_not_crowd_out_other_matching_files(service, tmp_path):
    ingest(service, tmp_path / "long.md", "Pod uses Container. " * 140)
    ingest(service, tmp_path / "short.md", "Pod uses Network.")
    result = service.search("Pod", mode="keyword", top_k=2)
    assert {Path(c["metadata"]["source"]).name for c in result["chunks_used"]} == {"long.md", "short.md"}
