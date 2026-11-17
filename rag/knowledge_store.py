"""Transactional, single-node knowledge storage. All graph evidence belongs to a chunk.

This SQLite store doubles as the dependency-free test adapter for the three-service
stack: PgStore implements the same interface against PostgreSQL, while the query
path only ever calls the bounded reads below (never records(), which exists for
explicit graph export and exact-retrieval benchmarks).
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3

from rag.vector_store import cosine_similarity


class RevisionConflict(RuntimeError):
    """A source changed while its replacement was being prepared."""


def _marks(values):
    return ",".join("?" * len(values))


def _now():
    return datetime.now(timezone.utc).isoformat()


class KnowledgeStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources(
                    source TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, revision INTEGER NOT NULL, indexed_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS source_generations(source TEXT PRIMARY KEY, generation INTEGER NOT NULL);
                INSERT OR IGNORE INTO source_generations SELECT source,revision FROM sources;
                CREATE TABLE IF NOT EXISTS chunks(
                    id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES sources(source) ON DELETE CASCADE,
                    text TEXT NOT NULL, embedding TEXT NOT NULL, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS entities(
                    id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL, embedding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS relations(
                    id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    source_key TEXT NOT NULL, target_key TEXT NOT NULL, relation TEXT NOT NULL,
                    description TEXT NOT NULL, keywords TEXT NOT NULL, evidence TEXT NOT NULL,
                    embedding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunk_terms(
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    term TEXT NOT NULL, tf INTEGER NOT NULL, PRIMARY KEY(chunk_id, term));
                CREATE INDEX IF NOT EXISTS idx_chunk_terms_term ON chunk_terms(term, tf);
                CREATE TABLE IF NOT EXISTS term_stats(term TEXT PRIMARY KEY, df INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS corpus_stats(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks(
                    id TEXT PRIMARY KEY, type TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                    error TEXT, result TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
                CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(key);
                CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_key);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_key);
                CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def source(self, source):
        return self.source_state(source)["record"]

    def source_state(self, source):
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = conn.execute("SELECT * FROM sources WHERE source=?", (source,)).fetchone()
            generation = conn.execute("SELECT generation FROM source_generations WHERE source=?", (source,)).fetchone()
            return {"record": dict(row) if row else None, "revision": generation[0] if generation else 0}

    def fingerprint(self):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            return row[0] if row else None

    def recall_revision(self):
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='recall_revision'").fetchone()
            return int(row[0]) if row else 0

    def advance_recall_revision(self):
        """Invalidate answer keys after a vector publication attempt changes recall."""
        with self._connect() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('recall_revision', '1') "
                         "ON CONFLICT (key) DO UPDATE SET value=CAST(CAST(value AS BIGINT)+1 AS TEXT)")

    def generations(self):
        """Per-source version numbers; the query path anchors consistency on these."""
        with self._connect() as conn:
            return [{"source": row[0], "generation": row[1]}
                    for row in conn.execute("SELECT source, generation FROM source_generations ORDER BY source")]

    def list_sources(self):
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT s.*, COUNT(c.id) AS chunks FROM sources s LEFT JOIN chunks c ON c.source=s.source "
                "GROUP BY s.source ORDER BY s.source")]

    # ── ingest: one transaction publishes text, graph, terms and corpus stats ──

    def _maintain_terms(self, conn, old_terms, terms):
        """Rewrite persistent BM25 statistics for the affected terms inside the caller's transaction."""
        # Capture old terms BEFORE deleting the source cascades its postings away.
        affected = set(old_terms)
        for entry in terms:
            conn.execute("INSERT OR REPLACE INTO chunk_terms(chunk_id, term, tf) VALUES(?,?,?)",
                         (entry["chunk_id"], entry["term"], entry["tf"]))
            affected.add(entry["term"])
        if affected:
            marks = _marks(list(affected))
            conn.execute(f"DELETE FROM term_stats WHERE term IN ({marks})", list(affected))
            conn.execute(f"INSERT INTO term_stats(term, df) SELECT term, COUNT(DISTINCT chunk_id) "
                         f"FROM chunk_terms WHERE term IN ({marks}) GROUP BY term", list(affected))
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        total = conn.execute("SELECT COALESCE(SUM(tf), 0) FROM chunk_terms").fetchone()[0]
        n = row[0]
        avgdl = total / n if n else 0.0
        conn.execute("INSERT OR REPLACE INTO corpus_stats(key, value) VALUES('n', ?)", (str(n),))
        conn.execute("INSERT OR REPLACE INTO corpus_stats(key, value) VALUES('avgdl', ?)", (repr(avgdl),))

    def replace_source(self, source, content_hash, fingerprint, *, chunks, entities, relations,
                       expected_revision=None, terms=None):
        # Model calls finish before this short transaction. Failed writes keep the previous graph AND text.
        terms = terms or []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT generation FROM source_generations WHERE source=?", (source,)).fetchone()
            revision = row[0] if row else 0
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
            if stored_dimension is None and populated:
                # Upgrade early local indexes without silently letting a different dimension enter.
                old_dimensions = {len(json.loads(row[0])) for table in ("chunks", "entities", "relations")
                                  for row in conn.execute(f"SELECT embedding FROM {table}")}
                if len(old_dimensions) != 1 or dimension not in old_dimensions:
                    raise ValueError("Existing index has incompatible embedding dimension; migration refused")
            if stored_dimension and int(stored_dimension[0]) != dimension and populated:
                raise ValueError("Embedding dimension differs from the persisted index; source was not published")
            old_terms = [row[0] for row in conn.execute(
                "SELECT DISTINCT t.term FROM chunk_terms t JOIN chunks c ON c.id=t.chunk_id WHERE c.source=?",
                (source,))]
            conn.execute("INSERT OR REPLACE INTO meta VALUES('fingerprint', ?)", (fingerprint,))
            conn.execute("INSERT OR REPLACE INTO meta VALUES('dimension', ?)", (str(dimension),))
            conn.execute("INSERT OR REPLACE INTO source_generations VALUES(?,?)", (source, revision + 1))
            conn.execute("DELETE FROM sources WHERE source=?", (source,))
            conn.execute("INSERT INTO sources VALUES(?,?,?,?,?)", (
                source, content_hash, fingerprint, revision + 1, _now()))
            for chunk in chunks:
                conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?)", (
                    chunk["id"], source, chunk["text"], json.dumps(chunk["embedding"]),
                    json.dumps(chunk.get("metadata", {}), ensure_ascii=False)))
            for entity in entities:
                conn.execute("INSERT INTO entities VALUES(?,?,?,?,?,?)", (
                    entity["id"], entity["chunk_id"], entity["key"], entity["name"],
                    entity["description"], json.dumps(entity["embedding"])))
            for relation in relations:
                conn.execute("INSERT INTO relations VALUES(?,?,?,?,?,?,?,?,?)", (
                    relation["id"], relation["chunk_id"], relation["source_key"], relation["target_key"],
                    relation["relation"], relation["description"], json.dumps(relation["keywords"], ensure_ascii=False),
                    relation["evidence"], json.dumps(relation["embedding"])))
            self._maintain_terms(conn, old_terms, terms)

    def delete(self, source):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old_terms = [row[0] for row in conn.execute(
                "SELECT DISTINCT t.term FROM chunk_terms t JOIN chunks c ON c.id=t.chunk_id WHERE c.source=?",
                (source,))]
            deleted = conn.execute("DELETE FROM sources WHERE source=?", (source,)).rowcount > 0
            if deleted:
                conn.execute("UPDATE source_generations SET generation=generation+1 WHERE source=?", (source,))
                self._maintain_terms(conn, old_terms, [])
            return deleted

    def source_records(self, source):
        """Read one source's persisted vectors for explicit Qdrant publication/repair."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            result = {}
            for table in ("chunks", "entities", "relations"):
                where = "source=?" if table == "chunks" else "chunk_id IN (SELECT id FROM chunks WHERE source=?)"
                rows = [dict(row) for row in conn.execute(
                    f"SELECT * FROM {table} WHERE {where} ORDER BY rowid", (source,))]
                for row in rows:
                    for field in ("embedding", "metadata", "keywords"):
                        if field in row:
                            row[field] = json.loads(row[field])
                result[table] = rows
            return result

    def records(self):
        with self._connect() as conn:
            conn.execute("BEGIN")
            result = {}
            for table in ("sources", "chunks", "entities", "relations"):
                rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                for row in rows:
                    for field in ("embedding", "metadata", "keywords"):
                        if field in row:
                            row[field] = json.loads(row[field])
                result[table] = rows
            model = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            result["fingerprint"] = model["value"] if model else None
            return result

    def stats(self):
        with self._connect() as conn:
            counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for table in ("sources", "chunks", "entities", "relations")}
            counts["terms"] = conn.execute("SELECT COUNT(*) FROM term_stats").fetchone()[0]
            return counts

    def check(self):
        with self._connect() as conn:
            errors = [str(tuple(row)) for row in conn.execute("PRAGMA foreign_key_check")]
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                errors.append(integrity)
        snapshot = self.records()
        dimensions = {len(item["embedding"]) for table in ("chunks", "entities", "relations") for item in snapshot[table]}
        if len(dimensions) > 1 or 0 in dimensions:
            errors.append("Inconsistent or empty embedding dimensions")
        by_id = {chunk["id"]: chunk for chunk in snapshot["chunks"]}
        for relation in snapshot["relations"]:
            if relation["evidence"] not in by_id[relation["chunk_id"]]["text"]:
                errors.append(f"Relation {relation['id']} has no matching source quote")
        return {"ok": not errors, "errors": errors}

    # ── bounded reads: the only storage calls the query path may make ──

    def _rows(self, conn, table, ids):
        if not ids:
            return []
        rows = [dict(row) for row in conn.execute(
            f"SELECT * FROM {table} WHERE id IN ({_marks(ids)}) ORDER BY rowid", list(ids))]
        for row in rows:
            for field in ("embedding", "metadata", "keywords"):
                if field in row:
                    row[field] = json.loads(row[field])
        return rows

    def fetch_chunks(self, ids):
        with self._connect() as conn:
            return self._rows(conn, "chunks", ids)

    def fetch_chunk_texts(self, ids):
        """Candidate texts without embeddings, in stable rowid order."""
        if not ids:
            return []
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                f"SELECT id, source, text, metadata FROM chunks WHERE id IN ({_marks(ids)}) ORDER BY rowid",
                list(ids))]

    def fetch_entities(self, ids):
        with self._connect() as conn:
            return self._rows(conn, "entities", ids)

    def fetch_entities_by_keys(self, keys):
        if not keys:
            return []
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT * FROM entities WHERE key IN ({_marks(keys)}) ORDER BY rowid", list(keys))]
            for row in rows:
                row["embedding"] = json.loads(row["embedding"])
            return rows

    def entity_catalog(self):
        """Lightweight name catalog for seed matching: no text bodies, no vectors."""
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT id, chunk_id, key, name FROM entities ORDER BY rowid")]

    def fetch_relations(self, ids):
        with self._connect() as conn:
            return self._rows(conn, "relations", ids)

    def fetch_relations_by_keys(self, keys):
        """One-hop adjacency: relations whose either endpoint matches a seed key."""
        if not keys:
            return []
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT * FROM relations WHERE source_key IN ({_marks(keys)}) "
                f"OR target_key IN ({_marks(keys)}) ORDER BY rowid", list(keys) + list(keys))]
            for row in rows:
                row["embedding"] = json.loads(row["embedding"])
                row["keywords"] = json.loads(row["keywords"])
            return rows

    def fetch_relations_by_chunk_ids(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(
                f"SELECT * FROM relations WHERE chunk_id IN ({_marks(ids)}) ORDER BY rowid", list(ids))]
            for row in rows:
                row["embedding"] = json.loads(row["embedding"])
                row["keywords"] = json.loads(row["keywords"])
            return rows

    def relation_catalog(self):
        """Lightweight relation catalog for theme-overlap seed matching: no vectors."""
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT id, chunk_id, source_key, target_key, relation, description, keywords "
                "FROM relations ORDER BY rowid")]
            for row in rows:
                row["keywords"] = json.loads(row["keywords"])
            return rows

    def chunk_ids_with_terms(self, terms, limit=500):
        """Keyword recall window: chunks sharing any query term, widest tf first."""
        if not terms:
            return []
        with self._connect() as conn:
            return [row[0] for row in conn.execute(
                f"SELECT chunk_id FROM chunk_terms WHERE term IN ({_marks(terms)}) "
                f"GROUP BY chunk_id ORDER BY SUM(tf) DESC, chunk_id LIMIT ?", list(terms) + [limit])]

    def all_chunk_ids(self):
        with self._connect() as conn:
            return [row[0] for row in conn.execute("SELECT id FROM chunks ORDER BY rowid")]

    def bm25_df(self, term):
        with self._connect() as conn:
            row = conn.execute("SELECT df FROM term_stats WHERE term=?", (term,)).fetchone()
            return row[0] if row else 0

    def bm25_corpus(self):
        with self._connect() as conn:
            rows = dict(conn.execute("SELECT key, value FROM corpus_stats").fetchall())
            return int(rows.get("n", 0)), float(rows.get("avgdl", 0.0))

    def _exact_candidates(self, table, vector, floor, limit):
        """Exact cosine scan — the reference/benchmark recall used by the SQLite adapter."""
        with self._connect() as conn:
            scored = [(cosine_similarity(vector, json.loads(row[0])), row[1])
                      for row in conn.execute(f"SELECT embedding, id FROM {table}")]
        scored = [item for item in scored if item[0] >= floor]
        scored.sort(reverse=True)
        return [(cid, score) for score, cid in scored[:limit]]

    def vector_candidates(self, vector, floor=0.0, limit=1000):
        return self._exact_candidates("chunks", vector, floor, limit)

    def entity_vector_candidates(self, vector, floor=0.0, limit=1000):
        return self._exact_candidates("entities", vector, floor, limit)

    def relation_vector_candidates(self, vector, floor=0.0, limit=1000):
        return self._exact_candidates("relations", vector, floor, limit)

    # ── ingest task ledger (single in-process worker; status survives restarts) ──

    def task_create(self, type, payload, task_id):
        with self._connect() as conn:
            conn.execute("INSERT INTO tasks(id, type, payload, status, created_at, updated_at) VALUES(?,?,?,?,?,?)",
                         (task_id, type, json.dumps(payload, ensure_ascii=False), "pending", _now(), _now()))
        return task_id

    def task_update(self, task_id, status, error=None, result=None):
        with self._connect() as conn:
            conn.execute("UPDATE tasks SET status=?, error=?, result=?, updated_at=? WHERE id=?",
                         (status, error, json.dumps(result, ensure_ascii=False) if result is not None else None,
                          _now(), task_id))

    def task_get(self, task_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row:
                return None
            task = dict(row)
            task["payload"] = json.loads(task["payload"])
            task["result"] = json.loads(task["result"]) if task["result"] else None
            return task

    def task_list(self, limit=50):
        with self._connect() as conn:
            tasks = [dict(row) for row in conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))]
            for task in tasks:
                task["payload"] = json.loads(task["payload"])
                task["result"] = json.loads(task["result"]) if task["result"] else None
            return tasks
