"""Storage regressions without servers or model APIs.

The PG SQL harness runs statements against a disposable SQLite database. It
checks column names, cascades and term maintenance, not PostgreSQL integration.
"""
import json
import re
import sqlite3
from contextlib import contextmanager

import pytest

from rag.cache import AnswerCache
from rag.graph_service import GraphService
from rag.knowledge_store import KnowledgeStore
from rag.pg_store import PgStore, SCHEMA_PATH
from rag.qdrant_store import QdrantStore
from tests.test_graph_service import DeterministicModel
from tests.test_knowledge_store import bundle


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This regression suite must not contact external services")
    monkeypatch.setattr("requests.request", forbidden)
    monkeypatch.setattr("psycopg.connect", forbidden)


class _PgSqliteConnection:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, params=()):
        values = []
        for value in params:
            if isinstance(value, list):
                sql = re.sub(r"= ANY\(%s\)", "IN (" + ",".join("?" for _ in value) + ")", sql, count=1)
                values.extend(value)
            else:
                sql = sql.replace("%s", "?", 1)
                values.append(value)
        return self.raw.execute(sql, values)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()


@pytest.fixture(params=["sqlite", "pg-sql-harness"])
def store(request, tmp_path):
    if request.param == "sqlite":
        return KnowledgeStore(tmp_path / "knowledge.db")

    path = tmp_path / "pg-sql.db"
    schema = SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "BIGINT GENERATED ALWAYS AS IDENTITY", "INTEGER")
    with sqlite3.connect(path) as conn:
        conn.executescript(schema)

    def pg_row(cursor, values):
        row = dict(zip((column[0] for column in cursor.description), values))
        for field in ("embedding", "metadata", "keywords", "payload", "result"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
        return row

    @contextmanager
    def connect():
        raw = sqlite3.connect(path)
        raw.row_factory = pg_row
        raw.execute("PRAGMA foreign_keys=ON")
        try:
            with raw:
                yield _PgSqliteConnection(raw)
        finally:
            raw.close()

    pg = object.__new__(PgStore)
    pg._connect = connect
    return pg


def _put(store, source, text, words):
    records = bundle(text, source)
    terms = [{"chunk_id": records["chunks"][0]["id"], "term": word, "tf": 1} for word in words]
    store.replace_source(source, text, "model-a", **records, terms=terms)


def test_delete_refreshes_frequencies_of_cascaded_terms(store):
    _put(store, "a.md", "shared vanished", ["shared", "vanished"])
    _put(store, "b.md", "shared retained", ["shared", "retained"])
    assert store.bm25_df("shared") == 2
    store.delete("a.md")
    assert store.bm25_df("shared") == 1
    assert store.bm25_df("vanished") == 0
    assert store.bm25_corpus() == (1, 2.0)


def test_replace_refreshes_terms_absent_from_new_document(store):
    _put(store, "a.md", "shared vanished", ["shared", "vanished"])
    _put(store, "b.md", "shared retained", ["shared", "retained"])
    _put(store, "a.md", "new text", ["new", "text"])
    assert store.bm25_df("shared") == 1
    assert store.bm25_df("vanished") == 0
    assert store.bm25_df("new") == 1
    assert store.bm25_corpus() == (2, 2.0)


def test_export_uses_columns_that_exist_in_each_table(store):
    _put(store, "a.md", "Pod uses Container.", ["pod", "container"])
    result = store.records()
    assert result["sources"][0]["source"] == "a.md"
    assert result["chunks"][0]["text"] == "Pod uses Container."
    assert result["relations"][0]["evidence"] == "Pod uses Container."


@pytest.mark.parametrize("points", [[], [{"id": "uuid", "payload": {"ref": "chunk-1"}, "score": .9}]])
def test_qdrant_query_accepts_empty_and_nonempty_point_lists(monkeypatch, points):
    index = QdrantStore("http://unused.invalid", "test-model")
    monkeypatch.setattr(index, "_request", lambda *a, **kw: {"result": {"points": points}, "status": "ok", "time": .001})
    assert index.query("chunks", [1., 0.], 5) == ([] if not points else [("chunk-1", .9)])


@pytest.mark.parametrize("operation", ["upsert", "delete"])
def test_qdrant_mutations_request_completion_via_query_parameter(monkeypatch, operation):
    index = QdrantStore("http://unused.invalid", "test-model")
    calls = []

    def request(method, path, **kwargs):
        calls.append(kwargs)
        return {"result": {"operation_id": 1, "status": "completed"}, "status": "ok", "time": .001}

    monkeypatch.setattr(index, "_request", request)
    if operation == "upsert":
        index.upsert("chunks", [{"id": "chunk-1", "embedding": [1., 0.]}], lambda row: "pod.md")
    else:
        index.delete_source("pod.md")
    # Qdrant's wait flag belongs to the URL query, not the JSON point/filter body.
    assert calls
    for kwargs in calls:
        assert kwargs.get("params") == {"wait": "true"}
        assert "wait" not in kwargs["json"]


class _Redis:
    def __init__(self):
        self.data = {}
        self.fail_clear = False

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def scan_iter(self, match, count=None):
        if self.fail_clear:
            raise ConnectionError("simulated cache invalidation failure")
        return iter(list(self.data))

    def delete(self, *keys):
        for key in keys:
            self.data.pop(key, None)


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_cache_cannot_serve_old_corpus_after_failed_clear(tmp_path, operation):
    redis = _Redis()
    service = GraphService(tmp_path / "kb", DeterministicModel(), cache=AnswerCache(client=redis))
    source = str(tmp_path / "pod.md")
    service.ingest_text(source, "Pod uses OldContainer.")
    service.query("Pod")
    assert service.query("Pod")["cached"] is True
    redis.fail_clear = True
    if operation == "update":
        service.ingest_text(source, "Pod uses NewContainer.")
    else:
        service.delete(source)
    redis.fail_clear = False
    fresh = service.query("Pod")
    assert fresh.get("cached") is not True
    assert "OldContainer" not in fresh["answer"]
    if operation == "update":
        assert "NewContainer" in fresh["answer"]
    else:
        assert fresh["status"] == "no_docs"


def test_shared_cache_distinguishes_different_corpus_contents(tmp_path):
    cache = AnswerCache(client=_Redis())
    source = str(tmp_path / "same-name.md")
    first = GraphService(tmp_path / "first", DeterministicModel(), cache=cache)
    second = GraphService(tmp_path / "second", DeterministicModel(), cache=cache)
    first.ingest_text(source, "Pod uses OldContainer.")
    second.ingest_text(source, "Pod uses NewContainer.")
    first.query("Pod")
    result = second.query("Pod")
    assert "NewContainer" in result["answer"]
    assert "OldContainer" not in result["answer"]


def test_inflight_answer_cannot_repopulate_cache_for_updated_corpus(tmp_path, monkeypatch):
    model = DeterministicModel()
    service = GraphService(tmp_path / "kb", model, cache=AnswerCache(client=_Redis()))
    source = str(tmp_path / "pod.md")
    service.ingest_text(source, "Pod uses OldContainer.")
    original = model.answer

    def update_during_answer(question, contexts, relations):
        answer = original(question, contexts, relations)
        service.ingest_text(source, "Pod uses NewContainer.")
        return answer

    monkeypatch.setattr(model, "answer", update_during_answer)
    service.query("Pod")
    monkeypatch.setattr(model, "answer", original)
    result = service.query("Pod")
    assert "NewContainer" in result["answer"]
    assert result.get("cached") is not True


class _VectorIndex:
    def __init__(self):
        self.fail_relations = True
        self.unavailable = False
        self.rows = {kind: {} for kind in ("chunks", "entities", "relations")}

    def ensure(self, size):
        if self.unavailable:
            raise ConnectionError("simulated vector service outage")

    def upsert(self, kind, rows, source_of):
        if kind == "relations" and self.fail_relations:
            raise ConnectionError("simulated partial vector publication")
        for row in rows:
            self.rows[kind][row["id"]] = {**row, "source": source_of(row)}

    def query(self, kind, vector, limit, floor=0.):
        # Both test documents contain Pod/Container and have the same test vector.
        return [(ref, 1.) for ref in list(self.rows[kind])[:limit]]


@pytest.mark.parametrize("during_answer", [False, True])
def test_vector_repair_invalidates_partial_answers_despite_failed_clear(store, tmp_path, monkeypatch, during_answer):
    index, redis, model = _VectorIndex(), _Redis(), DeterministicModel()
    index.fail_relations = False
    service = GraphService(tmp_path / "kb", model, store=store, qdrant=index, cache=AnswerCache(client=redis))
    first, second = str(tmp_path / "first.md"), str(tmp_path / "second.md")
    service.ingest_text(first, "Pod uses OldContainer.")
    index.unavailable = True
    assert "vector_warning" in service.ingest_text(second, "Pod uses NewContainer.")
    index.unavailable = False
    original = model.answer

    def repair():
        redis.fail_clear = True
        result = service.ingest_text(second, "Pod uses NewContainer.")
        redis.fail_clear = False
        assert result["vector_status"] == "ready"

    if during_answer:
        def repair_during_answer(question, contexts, relations):
            answer = original(question, contexts, relations)
            repair()
            return answer
        monkeypatch.setattr(model, "answer", repair_during_answer)

    partial = service.query("Pod", mode="vector", top_k=2)
    assert len(partial["sources"]) == 1
    monkeypatch.setattr(model, "answer", original)
    if not during_answer:
        repair()
    fresh = service.query("Pod", mode="vector", top_k=2)
    assert len(fresh["sources"]) == 2
    assert "NewContainer" in fresh["answer"]
    assert fresh.get("cached") is not True


def test_repeated_ingest_repairs_partial_vectors_from_ledger_without_models(store, tmp_path):
    index = _VectorIndex()
    service = GraphService(tmp_path / "kb", DeterministicModel(), store=store, qdrant=index)
    source = str(tmp_path / "pod.md")
    failed = service.ingest_text(source, "Pod uses Container.")
    assert "vector_warning" in failed
    index.fail_relations = False

    class NoModelCalls(DeterministicModel):
        def extract(self, text):
            raise AssertionError("Repair must reuse persisted extraction")

        def embed(self, texts):
            raise AssertionError("Repair must reuse persisted vectors")

    # A fresh service has no in-memory record of the earlier publication failure.
    reopened = GraphService(tmp_path / "kb", NoModelCalls(), store=store, qdrant=index)
    repaired = reopened.ingest_text(source, "Pod uses Container.")
    assert repaired["status"] == "unchanged"
    assert "vector_warning" not in repaired
    assert len(index.rows["relations"]) == 1
    assert len(index.rows["entities"]) == 2
    assert len(index.rows["chunks"]) == 1
    assert next(iter(index.rows["relations"].values()))["evidence"] == "Pod uses Container."
    assert store.list_sources()[0]["revision"] == 1


def test_retrieval_latency_includes_candidate_recall(tmp_path, monkeypatch):
    service = GraphService(tmp_path / "kb", DeterministicModel())
    service.ingest_text(str(tmp_path / "pod.md"), "Pod uses Container.")
    clock = [0.]
    monkeypatch.setattr("rag.graph_service.time.perf_counter", lambda: clock[0])
    original = service._bounded_snapshot

    def recall_with_known_cost(*args, **kwargs):
        clock[0] += .125
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_bounded_snapshot", recall_with_known_cost)
    assert service.search("Pod", mode="keyword")["elapsed_ms"] == 125.
