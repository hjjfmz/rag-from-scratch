"""
📚 LEARN: Generator — The "AG" in RAG (Augmented Generation)
==============================================================
This is the final step: we take the retrieved context chunks and the user's
question, build a prompt, and send it to the LLM to generate an answer.

The key insight of RAG is prompt engineering:
  - WITHOUT RAG: "What is overfitting?" → LLM answers from training data (may hallucinate)
  - WITH RAG: "Given these excerpts from the user's docs: [chunks], answer: What is overfitting?"
    → LLM answers grounded in the actual documents

📚 HOW WE CALL THE CHAT API:
We use the raw Azure AI Foundry REST API:
  POST https://<resource>.cognitiveservices.azure.com/openai/deployments/<model>/chat/completions?api-version=...

The request body looks like:
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "Context: [chunks]\\nQuestion: [query]"}
    ],
    "stream": true
  }

With "stream": true, the API sends tokens one at a time, so the user sees
the answer appear progressively instead of waiting for the full response.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────

# OpenRouter API configuration
CHAT_API_URL = os.getenv("CHAT_API_URL", "https://openrouter.ai/api/v1/chat/completions")
CHAT_MODEL = os.getenv("MODEL_CHAT", "qwen3-8b")
JUDGE_MODEL = os.getenv("MODEL_JUDGE", CHAT_MODEL)

# 📚 LEARN: The system message sets the LLM's behavior. It tells the model
# to answer ONLY from the provided context and to cite sources. This is
# crucial for RAG — without this instruction, the LLM might ignore the
# context and make up answers from its training data.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Answer ONLY based on the context provided below. Do not use your training data.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Keep answers concise and accurate.
4. At the end of your answer, cite which source(s) you used."""

FALLBACK_CHAT_PROMPT = """You are a helpful assistant. The user's knowledge base does not contain relevant information for this question, so answer based on your general knowledge.

Rules:
1. Be honest: if you are unsure, say so.
2. Do NOT fabricate references to the user's documents.
3. Keep answers concise and helpful."""


def generate_chat(query: str, stream: bool = True, yield_chunks: bool = False):
    """
    Generate an answer using the LLM without any RAG context (pure chat fallback).

    Used when retrieval quality is too low or no relevant documents exist,
    so the user still gets a helpful response instead of silence.
    """
    api_key = _get_api_key()
    messages = [
        {"role": "system", "content": FALLBACK_CHAT_PROMPT},
        {"role": "user", "content": query},
    ]
    headers = _build_headers(api_key)
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": stream,
    }
    if stream:
        return _generate_streaming(headers, payload, yield_chunks=yield_chunks)
    else:
        return _generate_non_streaming(headers, payload)


def build_no_answer_response(
    status: str = "no_relevant",
    user_guidance: str = "",
    retrieval_quality: dict | None = None,
) -> dict:
    """
    Build a standardized no-answer payload for retrieval miss scenarios.
    """
    if status == "no_docs":
        answer = "No documents indexed."
    else:
        answer = "No relevant documents found for your question."

    return {
        "answer": answer,
        "sources": [],
        "chunks_used": [],
        "status": status,
        "user_guidance": user_guidance,
        "retrieval_quality": retrieval_quality or {},
    }


def _get_api_key() -> str:
    """Get API key for OpenRouter endpoint."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY"
    )


def _build_headers(api_key: str | None = None) -> dict:
    """Build request headers for OpenAI-compatible providers (Qwen/OpenRouter/etc.)."""
    token = api_key or _get_api_key()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # OpenRouter recommends these headers; they are optional.
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_name:
        headers["X-Title"] = app_name

    return headers


def build_prompt(query: str, context_chunks: list[dict]) -> list[dict]:
    """
    Build the chat messages array from the query and retrieved context.

    📚 LEARN: This is where "Augmentation" happens — we inject the retrieved
    document chunks directly into the prompt. The LLM sees them as context
    and uses them to generate a grounded answer.

    The prompt structure is:
      System: "You are a helpful assistant that answers from context..."
      User: "Context:\n[chunk 1]\n[chunk 2]\n\nQuestion: [query]"

    Args:
        query: The user's question
        context_chunks: List of retrieval results (each has text, metadata, score)

    Returns:
        A list of message dicts ready for the OpenAI Chat API
    """
    # 📚 LEARN: We format each chunk with its source and relevance score
    # so the LLM knows where each piece of information came from.
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "unknown")
        score = chunk.get("score", 0.0)
        context_parts.append(
            f"[Source: {source} | Relevance: {score:.2f}]\n{chunk['text']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # 📚 LEARN: The messages format is what OpenAI's Chat API expects.
    # "system" sets behavior, "user" is the actual prompt.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]

    return messages


def generate(query: str, context_chunks: list[dict], stream: bool = True, yield_chunks: bool = False):
    """
    Generate an answer using the Azure AI Foundry Chat API.

    📚 LEARN: We call the API with stream=True, which uses Server-Sent Events
    (SSE). Instead of waiting for the full answer, the API sends one token at
    a time. Each line looks like:
      data: {"choices": [{"delta": {"content": "The"}}]}
      data: {"choices": [{"delta": {"content": " answer"}}]}
      data: {"choices": [{"delta": {"content": " is"}}]}
      data: [DONE]

    We print each token immediately, so the user sees the answer build up
    in real time — just like ChatGPT's typing effect.

    Args:
        query: The user's question
        context_chunks: Retrieved context from the retriever
        stream: Whether to stream tokens (default: True)

    Returns:
        The complete generated answer as a string
    """
    api_key = _get_api_key()
    messages = build_prompt(query, context_chunks)

    headers = _build_headers(api_key)

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": stream,
    }

    # 📚 LEARN: temperature controls randomness in the LLM's output:
    #   0.0 = always pick the most likely next token (deterministic)
    #   1.0 = sample from the full distribution (more creative/random)
    # For RAG, we want factual answers, so we use a low temperature (0.3).

    if stream:
        return _generate_streaming(headers, payload, yield_chunks=yield_chunks)
    else:
        return _generate_non_streaming(headers, payload)


def _generate_streaming(headers: dict, payload: dict, yield_chunks: bool = False):
    """
    Call the API with streaming.
    If yield_chunks is False, print tokens as they arrive and return the full string.
    If yield_chunks is True, act as a generator yielding chunks.
    """
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    )

    def stream_generator():
        if response.status_code != 200:
            yield f"Error: {response.status_code} - {response.text[:500]}"
            return

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    if yield_chunks:
        return stream_generator()
    else:
        if response.status_code != 200:
            print(f"  ❌ API Error {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        full_response = []
        for content in stream_generator():
            print(content, end="", flush=True)
            full_response.append(content)
        print()
        return "".join(full_response)


def _generate_non_streaming(headers: dict, payload: dict) -> str:
    """
    Call the API without streaming — wait for the full response.
    """
    response = requests.post(
        CHAT_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    return answer


def generate_with_sources(
    query: str, context_chunks: list[dict], stream: bool = True
) -> dict:
    """
    Generate an answer and return it with source attribution.

    📚 LEARN: Source attribution is what makes RAG trustworthy. Instead of
    just giving an answer, we also show WHICH documents were used. The user
    can then verify the answer against the original sources.

    Args:
        query: The user's question
        context_chunks: Retrieved context chunks

    Returns:
        A dict with:
          - "answer": the generated text
          - "sources": list of source files used
          - "chunks_used": the context chunks with scores
    """
    answer = generate(query, context_chunks, stream=stream)

    # Extract unique sources from the chunks
    sources = list(set(
        chunk.get("metadata", {}).get("source", "unknown")
        for chunk in context_chunks
    ))

    return {
        "answer": answer,
        "sources": sorted(sources),
        "chunks_used": context_chunks,
    }
