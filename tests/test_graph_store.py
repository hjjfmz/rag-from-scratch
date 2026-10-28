"""Tests for the JSON entity graph."""

import os

from rag.graph_store import GraphStore, chunk_id_for


def test_chunk_id_normalizes_slashes():
    assert chunk_id_for(r"sample_data\policy.md", 2) == "sample_data/policy.md::2"


def test_one_hop_puts_seed_chunks_before_neighbors():
    graph = GraphStore()
    graph.add_extraction(
        ["GlowBeauty"],
        [{"source": "GlowBeauty", "relation": "has_policy", "target": "Return Policy"}],
        "policies/refund.md::0",
    )
    graph.add_extraction(
        ["Return Policy"],
        [],
        "policies/refund.md::1",
    )

    ids = graph.one_hop_chunk_ids(graph.match_names("What is the GlowBeauty return window?"))
    assert ids[0] == "policies/refund.md::0"
    assert "policies/refund.md::1" in ids


def test_drop_source_removes_only_that_document():
    graph = GraphStore()
    graph.add_extraction(["A"], [], "a.md::0")
    graph.add_extraction(["B"], [{"source": "B", "relation": "sees", "target": "C"}], "b.md::0")
    graph.drop_source("a.md")
    assert "A" not in graph.nodes
    assert "B" in graph.nodes
    assert graph.edges[0]["chunk_id"] == "b.md::0"


def test_save_and_load_round_trip(tmp_path):
    graph = GraphStore()
    graph.add_extraction(["Product"], [], "catalog.md::0")
    path = os.path.join(tmp_path, "index_graph.json")
    graph.save(path)

    loaded = GraphStore()
    loaded.load(path)
    assert loaded.nodes["Product"] == ["catalog.md::0"]


def test_load_missing_file_clears_graph(tmp_path):
    graph = GraphStore()
    graph.add_extraction(["Keep"], [], "x.md::0")
    graph.load(os.path.join(tmp_path, "missing_graph.json"))
    assert graph.nodes == {}
    assert graph.edges == []
