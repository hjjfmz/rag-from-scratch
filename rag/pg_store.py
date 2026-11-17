"""PostgreSQL ledger: text, graph evidence, BM25 statistics and tasks.

Same interface as the SQLite adapter (rag.knowledge_store.KnowledgeStore). PG is
the source of truth; Qdrant holds recallable vector copies and is owned by
rag.qdrant_store, so the *_vector_candidates methods intentionally fail here —
the services-mode query path must recall through Qdrant and hydrate by ID below.
"""
import json
import math
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from rag.vector_store import cosine_similarity

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


class RevisionConflict(RuntimeError):
    """A source changed while its replacement was being prepared."""


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class PgStore:
    def __init__(self, dsn):
        if not dsn:
            raise ValueError("PostgreSQL DSN is required (RAG_PG_DSN)")
        self.dsn = dsn
        self.initialize()

    @property
    def path(self):
        return f"postgresql:{self.dsn.split('@')[-1]}"

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)

    def initialize(self):
        with self._connect() as conn:
            conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()

    def source(self, source):
        return self.source_state(source)["record"]

    def source_state(self, source):
        with self._connect() as conn:
            record = conn.execute("SELECT * FROM sources WHERE source=%s", (source,)).fetchone()
            generation = conn.execute("SELECT generation FROM source_generations WHERE source=%s",
                                      (source,)).fetchone()
            return {"record": record, "revision": generation["generation"] if generation else 0}

    def fingerprint(self):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            return row["value"] if row else None

    def recall_revision(self):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='recall_revision'").fetchone()
            return int(row["value"]) if row else 0

    def advance_recall_revision(self):
        """Atomic increment shared by CLI/Web; uses the existing metadata table."""
        with self._connect() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('recall_revision', '1') "
                         "ON CONFLICT (key) DO UPDATE SET value=CAST(CAST(meta.value AS BIGINT)+1 AS TEXT)")
            conn.commit()

    def generations(self):
        """Per-source version numbers; the query path anchors consistency on these."""
        with self._connect() as conn:
            rows = conn.execute("SELECT source, generation FROM source_generations ORDER BY source").fetchall()
            return [{"source": row["source"], "generation": row["generation"]} for row in rows]

    def list_sources(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT s.*, COUNT(c.id) AS chunks FROM sources s LEFT JOIN chunks c ON c.source=s.source "
                "GROUP BY s.source ORDER BY s.source").fetchall()

    # ── ingest: one transaction publishes text, graph, terms and corpus stats ──

    def _maintain_terms(self, conn, old_terms, terms):
        # The old postings have already been removed by ON DELETE CASCADE.
        affected = set(old_terms)
        for entry in terms:
            conn.execute("INSERT INTO chunk_terms(chunk_id, term, tf) VALUES(%s,%s,%s) "
                         "ON CONFLICT (chunk_id, term) DO UPDATE SET tf=EXCLUDED.tf",
                         (entry["chunk_id"], entry["term"], entry["tf"]))
            affected.add(entry["term"])
        if affected:
            values = list(affected)
            conn.execute("DELETE FROM term_stats WHERE term = ANY(%s)", (values,))
            conn.execute("INSERT INTO term_stats(term, df) SELECT term, COUNT(DISTINCT chunk_id) "
                         "FROM chunk_terms WHERE term = ANY(%s) GROUP BY term", (values,))
        n = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        total = conn.execute("SELECT COALESCE(SUM(tf), 0) AS total FROM chunk_terms").fetchone()["total"]
        avgdl = total / n if n else 0.0
        conn.execute("INSERT INTO corpus_stats(key, value) VALUES('n', %s) "
                     "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (str(n),))
        conn.execute("INSERT INTO corpus_stats(key, value) VALUES('avgdl', %s) "
                     "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (repr(avgdl),))

    def replace_source(self, source, content_hash, fingerprint, *, chunks, entities, relations,
                       expected_revision=None, terms=None):
        terms = terms or []
        with self._connect() as conn:
            try:
                row = conn.execute("SELECT generation FROM source_generations WHERE source=%s",
                                   (source,)).fetchone()
                revision = row["generation"] if row else 0
                if expected_revision is not None and revision != expected_revision:
                    raise RevisionConflict(f"Source {source} changed concurrently; retry ingestion")
                model = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
                populated = conn.execute("SELECT 1 FROM sources LIMIT 1").fetchone()
                if model and model["value"] != fingerprint and populated:
                    raise ValueError("Cannot mix model fingerprints. Use a separate --data-dir or rebuild a fresh index.")
                vectors = [row["embedding"] for group in (chunks, entities, relations) for row in group]
                dimensions = {len(vector) for vector in vectors}
                if len(dimensions) != 1 or 0 in dimensions or not all(
                        type(value) in (float, int) and math.isfinite(value) for vector in vectors for value in vector):
                    raise ValueError("Invalid embedding dimensions or nonfinite vector")
                dimension = next(iter(dimensions))
                stored_dimension = conn.execute("SELECT value FROM meta WHERE key='dimension'").fetchone()
                if stored_dimension and int(stored_dimension["value"]) != dimension and populated:
                    raise ValueError("Embedding dimension differs from the persisted index; source was not published")
                old_terms = [row["term"] for row in conn.execute(
                    "SELECT DISTINCT t.term FROM chunk_terms t JOIN chunks c ON c.id=t.chunk_id WHERE c.source=%s",
                    (source,))]
                conn.execute("INSERT INTO meta(key, value) VALUES('fingerprint', %s) "
                             "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (fingerprint,))
                conn.execute("INSERT INTO meta(key, value) VALUES('dimension', %s) "
                             "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (str(dimension),))
                conn.execute("INSERT INTO source_generations(source, generation) VALUES(%s,%s) "
                             "ON CONFLICT (source) DO UPDATE SET generation=EXCLUDED.generation",
                             (source, revision + 1))
                conn.execute("DELETE FROM sources WHERE source=%s", (source,))
                conn.execute("INSERT INTO sources VALUES(%s,%s,%s,%s,%s)",
                             (source, content_hash, fingerprint, revision + 1, _now()))
                for chunk in chunks:
                    conn.execute("INSERT INTO chunks(id, source, text, embedding, metadata) VALUES(%s,%s,%s,%s,%s)",
                                 (chunk["id"], source, chunk["text"], json.dumps(chunk["embedding"]),
                                  json.dumps(chunk.get("metadata", {}), ensure_ascii=False)))
                for entity in entities:
                    conn.execute("INSERT INTO entities(id, chunk_id, key, name, description, embedding) "
                                 "VALUES(%s,%s,%s,%s,%s,%s)",
                                 (entity["id"], entity["chunk_id"], entity["key"], entity["name"],
                                  entity["description"], json.dumps(entity["embedding"])))
                for relation in relations:
                    conn.execute("INSERT INTO relations(id, chunk_id, source_key, target_key, relation, "
                                 "description, keywords, evidence, embedding) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                                 (relation["id"], relation["chunk_id"], relation["source_key"], relation["target_key"],
                                  relation["relation"], relation["description"],
                                  json.dumps(relation["keywords"], ensure_ascii=False), relation["evidence"],
                                  json.dumps(relation["embedding"])))
                self._maintain_terms(conn, old_terms, terms)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def delete(self, source):
        with self._connect() as conn:
            try:
                old_terms = [row["term"] for row in conn.execute(
                    "SELECT DISTINCT t.term FROM chunk_terms t JOIN chunks c ON c.id=t.chunk_id WHERE c.source=%s",
                    (source,))]
                deleted = conn.execute("DELETE FROM sources WHERE source=%s", (source,)).rowcount > 0
                if deleted:
                    conn.execute("UPDATE source_generations SET generation=generation+1 WHERE source=%s", (source,))
                    self._maintain_terms(conn, old_terms, [])
                conn.commit()
                return deleted
            except Exception:
                conn.rollback()
                raise

    # ── explicit export only: never on the query path ──

    def source_records(self, source):
        """Read persisted vectors belonging to one source, without model calls."""
        with self._connect() as conn:
            result = {}
            for table in ("chunks", "entities", "relations"):
                where = "source=%s" if table == "chunks" else "chunk_id IN (SELECT id FROM chunks WHERE source=%s)"
                result[table] = conn.execute(
                    f"SELECT * FROM {table} WHERE {where} ORDER BY seq", (source,)).fetchall()
            return result

    def records(self):
        with self._connect() as conn:
            result = {table: conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
                      for table, order in (("sources", "source"), ("chunks", "seq"),
                                           ("entities", "seq"), ("relations", "seq"))}
            model = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            result["fingerprint"] = model["value"] if model else None
            return result

    def stats(self):
        with self._connect() as conn:
            counts = {table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                      for table in ("sources", "chunks", "entities", "relations")}
            counts["terms"] = conn.execute("SELECT COUNT(*) AS n FROM term_stats").fetchone()["n"]
            return counts

    def check(self):
        errors = []
        with self._connect() as conn:
            broken = conn.execute(
                "SELECT r.id FROM relations r JOIN chunks c ON c.id=r.chunk_id "
                "WHERE position(r.evidence IN c.text) = 0 LIMIT 10").fetchall()
            errors.extend(f"Relation {row['id']} has no matching source quote" for row in broken)
            stored = conn.execute("SELECT value FROM meta WHERE key='dimension'").fetchone()
            if stored:
                for table in ("chunks", "entities", "relations"):
                    mismatch = conn.execute(
                        f"SELECT id FROM {table} WHERE jsonb_array_length(embedding) <> %s LIMIT 1",
                        (int(stored["value"]),)).fetchone()
                    if mismatch:
                        errors.append(f"{table} {mismatch['id']} has an unexpected embedding dimension")
        return {"ok": not errors, "errors": errors}

    # ── bounded reads: the only storage calls the query path may make ──

    def fetch_chunks(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM chunks WHERE id = ANY(%s) ORDER BY seq", (list(ids),)).fetchall()

    def fetch_chunk_texts(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT id, source, text, metadata FROM chunks WHERE id = ANY(%s) ORDER BY seq",
                                (list(ids),)).fetchall()

    def fetch_entities(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM entities WHERE id = ANY(%s) ORDER BY seq", (list(ids),)).fetchall()

    def fetch_entities_by_keys(self, keys):
        if not keys:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM entities WHERE key = ANY(%s) ORDER BY seq", (list(keys),)).fetchall()

    def entity_catalog(self):
        with self._connect() as conn:
            return conn.execute("SELECT id, chunk_id, key, name FROM entities ORDER BY seq").fetchall()

    def fetch_relations(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM relations WHERE id = ANY(%s) ORDER BY seq", (list(ids),)).fetchall()

    def fetch_relations_by_keys(self, keys):
        """One-hop adjacency: relations whose either endpoint matches a seed key."""
        if not keys:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM relations WHERE source_key = ANY(%s) OR target_key = ANY(%s) "
                                "ORDER BY seq", (list(keys), list(keys))).fetchall()

    def fetch_relations_by_chunk_ids(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            return conn.execute("SELECT * FROM relations WHERE chunk_id = ANY(%s) ORDER BY seq",
                                (list(ids),)).fetchall()

    def relation_catalog(self):
        with self._connect() as conn:
            return conn.execute("SELECT id, chunk_id, source_key, target_key, relation, description, keywords "
                                "FROM relations ORDER BY seq").fetchall()

    def chunk_ids_with_terms(self, terms, limit=500):
        if not terms:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_terms WHERE term = ANY(%s) "
                "GROUP BY chunk_id ORDER BY SUM(tf) DESC, chunk_id LIMIT %s", (list(terms), limit)).fetchall()
            return [row["chunk_id"] for row in rows]

    def all_chunk_ids(self):
        with self._connect() as conn:
            return [row["id"] for row in conn.execute("SELECT id FROM chunks ORDER BY seq").fetchall()]

    def bm25_df(self, term):
        with self._connect() as conn:
            row = conn.execute("SELECT df FROM term_stats WHERE term=%s", (term,)).fetchone()
            return row["df"] if row else 0

    def bm25_corpus(self):
        with self._connect() as conn:
            rows = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM corpus_stats").fetchall()}
            return int(rows.get("n", 0)), float(rows.get("avgdl", 0.0))

    def vector_candidates(self, vector, floor=0.0, limit=1000):
        raise NotImplementedError("services mode recalls vectors through Qdrant, not PostgreSQL")

    entity_vector_candidates = vector_candidates
    relation_vector_candidates = vector_candidates

    # ── ingest task ledger ──

    def task_create(self, type, payload, task_id=None):
        task_id = task_id or str(uuid4())
        with self._connect() as conn:
            conn.execute("INSERT INTO tasks(id, type, payload, status, created_at, updated_at) "
                         "VALUES(%s,%s,%s,'pending',%s,%s)",
                         (task_id, type, json.dumps(payload, ensure_ascii=False), _now(), _now()))
            conn.commit()
        return task_id

    def task_update(self, task_id, status, error=None, result=None):
        with self._connect() as conn:
            conn.execute("UPDATE tasks SET status=%s, error=%s, result=%s, updated_at=%s WHERE id=%s",
                         (status, error, json.dumps(result, ensure_ascii=False) if result is not None else None,
                          _now(), task_id))
            conn.commit()

    def _task(self, row):
        if not row:
            return None
        row["payload"] = row["payload"] if isinstance(row["payload"], dict) else json.loads(row["payload"])
        if isinstance(row["result"], str):
            row["result"] = json.loads(row["result"])
        return row

    def task_get(self, task_id):
        with self._connect() as conn:
            return self._task(conn.execute("SELECT * FROM tasks WHERE id=%s", (task_id,)).fetchone())

    def task_list(self, limit=50):
        with self._connect() as conn:
            return [self._task(row) for row in
                    conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()]
