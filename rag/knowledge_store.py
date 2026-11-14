"""Transactional, single-node knowledge storage. All graph evidence belongs to a chunk."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3


class RevisionConflict(RuntimeError):
    """A source changed while its replacement was being prepared."""


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

    def list_sources(self):
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT s.*, COUNT(c.id) AS chunks FROM sources s LEFT JOIN chunks c ON c.source=s.source "
                "GROUP BY s.source ORDER BY s.source")]

    def replace_source(self, source, content_hash, fingerprint, *, chunks, entities, relations, expected_revision=None):
        # Model calls finish before this short transaction. Failed writes keep the previous graph AND text.
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
            conn.execute("INSERT OR REPLACE INTO meta VALUES('fingerprint', ?)", (fingerprint,))
            conn.execute("INSERT OR REPLACE INTO meta VALUES('dimension', ?)", (str(dimension),))
            conn.execute("INSERT OR REPLACE INTO source_generations VALUES(?,?)", (source, revision + 1))
            conn.execute("DELETE FROM sources WHERE source=?", (source,))
            conn.execute("INSERT INTO sources VALUES(?,?,?,?,?)", (
                source, content_hash, fingerprint, revision + 1, datetime.now(timezone.utc).isoformat()))
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

    def delete(self, source):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            deleted = conn.execute("DELETE FROM sources WHERE source=?", (source,)).rowcount > 0
            if deleted:
                conn.execute("UPDATE source_generations SET generation=generation+1 WHERE source=?", (source,))
            return deleted

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
            return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("sources", "chunks", "entities", "relations")}

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
