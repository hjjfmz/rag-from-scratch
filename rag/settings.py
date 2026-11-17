"""Central configuration: storage mode, service addresses and ingest budget.

RAG_STORAGE selects the storage stack:
  local    — SQLite adapter with exact in-process vector scan (default; no services needed)
  services — PostgreSQL + Qdrant server + optional Redis; addresses are required explicitly
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    storage: str = "local"
    data_dir: Path = ROOT / ".runtime" / "knowledge"
    pg_dsn: str = ""
    qdrant_url: str = ""
    redis_url: str = ""
    chunk_size: int = 900
    overlap: int = 120
    workers: int = 4


def load_env():
    load_dotenv(Path(os.getenv("RAG_ENV_FILE", str(ROOT / ".env"))), override=False)


def load_settings(env=None):
    env = os.environ if env is None else env
    storage = (env.get("RAG_STORAGE") or "local").strip().lower()
    if storage not in ("local", "services"):
        raise ValueError("RAG_STORAGE must be 'local' or 'services'")
    settings = Settings(
        storage=storage,
        data_dir=Path(env.get("RAG_DATA_DIR") or ROOT / ".runtime" / "knowledge").resolve(),
        pg_dsn=(env.get("RAG_PG_DSN") or "").strip(),
        qdrant_url=(env.get("RAG_QDRANT_URL") or "").strip(),
        redis_url=(env.get("RAG_REDIS_URL") or "").strip(),
        chunk_size=int(env.get("RAG_CHUNK_SIZE") or 900),
        overlap=int(env.get("RAG_OVERLAP") or 120),
        workers=int(env.get("RAG_WORKERS") or 4),
    )
    if storage == "services":
        missing = [name for name, value in (("RAG_PG_DSN", settings.pg_dsn), ("RAG_QDRANT_URL", settings.qdrant_url))
                   if not value]
        if missing:
            raise ValueError(
                "RAG_STORAGE=services requires " + ", ".join(missing)
                + "; set them in .env or switch to RAG_STORAGE=local for the SQLite adapter")
    return settings


def build_store(settings):
    """Storage stack factory. Missing production configuration fails loudly, never silently falls back."""
    if settings.storage == "services":
        from rag.pg_store import PgStore
        return PgStore(settings.pg_dsn)
    from rag.knowledge_store import KnowledgeStore
    return KnowledgeStore(settings.data_dir / "knowledge.sqlite3")


def build_runtime(settings=None, offline=None):
    """One factory for CLI and Web: model client + storage + recall + cache."""
    settings = settings or load_settings()
    if offline is None:
        offline = (os.getenv("RAG_OFFLINE", "").strip().lower() in ("1", "true", "yes"))
    from rag.model_client import ModelClient
    client = ModelClient(offline=offline, cache_path=settings.data_dir / "model-cache.sqlite3")
    store = build_store(settings)
    qdrant = None
    if settings.storage == "services":
        from rag.qdrant_store import QdrantStore
        qdrant = QdrantStore(settings.qdrant_url, client.fingerprint)
    cache = None
    if settings.redis_url:
        from rag.cache import AnswerCache
        cache = AnswerCache(settings.redis_url)
    from rag.graph_service import GraphService
    service = GraphService(settings.data_dir, client, store=store, qdrant=qdrant, cache=cache,
                           chunk_size=settings.chunk_size, overlap=settings.overlap, workers=settings.workers)
    return service
