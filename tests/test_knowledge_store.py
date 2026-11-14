"""Storage regressions: published readers must never observe mixed source versions."""
import pytest

from rag.knowledge_store import KnowledgeStore, RevisionConflict


def bundle(text="old", source="guide.md"):
    cid = source + "::" + text
    return {
        "chunks": [{"id": cid, "text": text, "embedding": [1., 0.], "metadata": {"source": source}}],
        "entities": [{"id": cid + ":e", "chunk_id": cid, "key": "pod", "name": "Pod", "description": text, "embedding": [1., 0.]}],
        "relations": [{"id": cid + ":r", "chunk_id": cid, "source_key": "pod", "target_key": "pod", "relation": "describes", "description": text, "keywords": ["health"], "evidence": text, "embedding": [1., 0.]}],
    }


def test_replace_cascades_old_evidence_and_survives_reopen(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    store.replace_source("guide.md", "old", "model-a", **bundle(), expected_revision=0)
    store.replace_source("guide.md", "new", "model-a", **bundle("new"), expected_revision=1)
    reopened = KnowledgeStore(tmp_path / "kb.db")
    snapshot = reopened.records()
    assert [c["text"] for c in snapshot["chunks"]] == ["new"]
    assert [r["evidence"] for r in snapshot["relations"]] == ["new"]
    assert reopened.source("guide.md")["revision"] == 2
    assert reopened.check()["ok"] is True


def test_bad_graph_rolls_back_old_source(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    store.replace_source("guide.md", "old", "model-a", **bundle(), expected_revision=0)
    broken = bundle("new")
    broken["relations"][0]["chunk_id"] = "missing-chunk"
    with pytest.raises(Exception):
        store.replace_source("guide.md", "new", "model-a", **broken, expected_revision=1)
    assert store.records()["chunks"][0]["text"] == "old"
    assert store.source("guide.md")["revision"] == 1


def test_optimistic_writes_reject_stale_snapshot(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    store.replace_source("guide.md", "one", "model-a", **bundle("one"), expected_revision=0)
    with pytest.raises(RevisionConflict):
        store.replace_source("guide.md", "two", "model-a", **bundle("two"), expected_revision=0)
    assert store.records()["chunks"][0]["text"] == "one"


def test_delete_preserves_other_sources_and_prevents_model_mixing(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    store.replace_source("guide.md", "old", "model-a", **bundle(), expected_revision=0)
    store.replace_source("other.md", "other", "model-a", **bundle("other", "other.md"), expected_revision=0)
    with pytest.raises(ValueError, match="model|fingerprint"):
        store.replace_source("third.md", "new", "model-b", **bundle("new", "third.md"), expected_revision=0)
    assert store.delete("guide.md") is True
    assert [s["source"] for s in store.list_sources()] == ["other.md"]
    assert len(store.records()["relations"]) == 1


def test_concurrent_writers_do_not_lose_other_sources(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "kb.db"
    KnowledgeStore(path)
    def write(n):
        name = f"{n}.md"
        KnowledgeStore(path).replace_source(name, str(n), "model-a", **bundle(str(n), name), expected_revision=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    store = KnowledgeStore(path)
    assert store.stats()["sources"] == 12
    assert store.check()["ok"] is True


def test_delete_reinsert_cannot_let_an_old_writer_overwrite_new_version(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    store.replace_source("guide.md", "one", "model-a", **bundle("one"), expected_revision=0)
    store.delete("guide.md")
    store.replace_source("guide.md", "new", "model-a", **bundle("new"))
    with pytest.raises(RevisionConflict):
        store.replace_source("guide.md", "stale", "model-a", **bundle("stale"), expected_revision=1)
    assert store.records()["chunks"][0]["text"] == "new"


def test_new_process_same_fingerprint_cannot_publish_different_vector_dimension(tmp_path):
    path = tmp_path / "kb.db"
    store = KnowledgeStore(path)
    store.replace_source("guide.md", "old", "model-a", **bundle(), expected_revision=0)
    other = bundle("new", "other.md")
    for rows in other.values():
        for row in rows:
            row["embedding"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="dimension"):
        KnowledgeStore(path).replace_source("other.md", "new", "model-a", **other, expected_revision=0)
    assert store.check()["ok"] is True


def test_older_database_without_dimension_metadata_still_rejects_mixed_vectors(tmp_path):
    import sqlite3
    path = tmp_path / "kb.db"
    store = KnowledgeStore(path)
    store.replace_source("guide.md", "old", "model-a", **bundle(), expected_revision=0)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM meta WHERE key='dimension'")
    other = bundle("new", "other.md")
    for rows in other.values():
        for row in rows:
            row["embedding"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="dimension"):
        KnowledgeStore(path).replace_source("other.md", "new", "model-a", **other, expected_revision=0)
    assert store.check()["ok"] is True
