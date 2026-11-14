"""Cross-file evidence must survive candidate limits before final context selection."""
import pytest

from rag.graph_retrieval import DualLevelRetriever


class QueryModel:
    def keywords(self, question):
        return {"low_level": ["Pod"], "high_level": ["uses"]}

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid", "local", "global", "hybrid_graph"])
@pytest.mark.parametrize("distinct_entities", [False, True])
def test_cross_file_evidence_survives_early_candidate_truncation(mode, distinct_entities):
    snapshot = {"chunks": [], "entities": [], "relations": []}
    for index in range(41):
        short = index == 40
        cid = f"chunk-{index:03}"
        name = f"Node{index}" if distinct_entities else "Pod"
        vector = [.9, .1] if short else [1., 0.]
        text = "Pod uses Network. " + "background " * 70 if short else "Pod uses Container. " * 30
        snapshot["chunks"].append({"id": cid, "text": text, "embedding": vector,
                                    "metadata": {"source": "short.md" if short else "long.md"}})
        snapshot["entities"].append({"id": f"entity-{index:03}", "key": name.lower(), "name": name,
                                      "chunk_id": cid, "embedding": vector})
        snapshot["relations"].append({"id": f"relation-{index:03}", "chunk_id": cid,
                                       "source_key": name.lower(), "target_key": "network" if short else "container",
                                       "description": "uses", "keywords": ["uses"], "relation": "uses",
                                       "embedding": vector})
    result = DualLevelRetriever(snapshot, QueryModel()).search("Pod uses", mode=mode, top_k=2)
    assert {c["metadata"]["source"] for c in result["chunks_used"]} == {"long.md", "short.md"}
