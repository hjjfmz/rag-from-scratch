-- PostgreSQL schema for the three-service stack. Idempotent; applied by rag.pg_store.PgStore.initialize().
-- PG is the ledger and source of truth (text, graph evidence, BM25 term statistics, tasks, sessions).
-- Qdrant only holds recallable copies of the vectors; Redis only holds expiring answer caches.

CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sources(
    source TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    revision INTEGER NOT NULL,
    indexed_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS source_generations(
    source TEXT PRIMARY KEY,
    generation INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS chunks(
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL REFERENCES sources(source) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding JSONB NOT NULL,
    metadata JSONB NOT NULL);

CREATE TABLE IF NOT EXISTS entities(
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding JSONB NOT NULL);

CREATE TABLE IF NOT EXISTS relations(
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords JSONB NOT NULL,
    evidence TEXT NOT NULL,
    embedding JSONB NOT NULL);

-- Persistent BM25 statistics: term postings, document frequencies and corpus averages.
CREATE TABLE IF NOT EXISTS chunk_terms(
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY(chunk_id, term));
CREATE INDEX IF NOT EXISTS idx_chunk_terms_term ON chunk_terms(term, tf);

CREATE TABLE IF NOT EXISTS term_stats(
    term TEXT PRIMARY KEY,
    df INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS corpus_stats(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL);

-- Ingest task ledger; a single in-process worker executes, status survives restarts.
CREATE TABLE IF NOT EXISTS tasks(
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    result JSONB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(key);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_key);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_key);
CREATE INDEX IF NOT EXISTS idx_relations_chunk ON relations(chunk_id);
