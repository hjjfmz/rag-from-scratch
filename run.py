"""One command to start the web workbench: python run.py

Storage mode comes from .env (RAG_STORAGE=local by default; RAG_STORAGE=services
uses PostgreSQL + Qdrant + optional Redis after `docker compose up -d`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

from rag.settings import load_env

load_env()

if __name__ == "__main__":
    print("RAG 知识库工作台: http://127.0.0.1:8000")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
