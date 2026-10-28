"""OpenAI-compatible chat and embedding settings.

The pipeline talks to whatever base URL you set. OpenRouter, SiliconFlow,
DashScope, DeepSeek, and a local Ollama server all work if they speak the
same chat and embeddings HTTP API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

CHAT_API_URL = os.getenv(
    "CHAT_API_URL", "https://openrouter.ai/api/v1/chat/completions"
)
EMBEDDING_API_URL = os.getenv(
    "EMBEDDING_API_URL", "https://openrouter.ai/api/v1/embeddings"
)
CHAT_MODEL = os.getenv("MODEL_CHAT", "qwen3-8b")
JUDGE_MODEL = os.getenv("MODEL_JUDGE", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("MODEL_EMBEDDING", "text-embedding-v3")


def get_api_key() -> str:
    """Return the first configured key.

    LLM_API_KEY is the generic name. OPENAI_API_KEY and OPENROUTER_API_KEY
    remain accepted so older .env files keep working.
    """
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise ValueError(
        "No API key found. Set LLM_API_KEY to a key for your OpenAI-compatible endpoint."
    )


def build_headers(api_key: str | None = None) -> dict:
    token = api_key or get_api_key()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if app_name:
        headers["X-Title"] = app_name
    return headers
