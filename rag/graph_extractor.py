"""Ask the chat model for entities and relations in a chunk or question."""

import json
import re

import requests

from rag.llm_config import CHAT_API_URL, CHAT_MODEL, build_headers, get_api_key


_EXTRACT_INSTRUCTIONS = """Extract a compact knowledge graph from the text.
Return JSON only, with this shape:
{"entities": ["Name"], "relations": [{"source": "Name", "relation": "verb", "target": "Name"}]}
Use short canonical names. Do not invent facts that are not in the text.
If nothing useful is present, return {"entities": [], "relations": []}.
"""


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("extractor response was not a JSON object")
    return payload


def extract_graph(text: str) -> dict:
    """Return {"entities": [...], "relations": [...]} for one piece of text."""
    api_key = get_api_key()
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": _EXTRACT_INSTRUCTIONS},
            {"role": "user", "content": text[:6000]},
        ],
        "stream": False,
        "temperature": 0,
    }
    response = requests.post(
        CHAT_API_URL,
        headers=build_headers(api_key),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    parsed = _parse_json_object(content)
    entities = [str(item).strip() for item in parsed.get("entities") or [] if str(item).strip()]
    relations = []
    for item in parsed.get("relations") or []:
        if not isinstance(item, dict):
            continue
        relations.append({
            "source": str(item.get("source", "")).strip(),
            "relation": str(item.get("relation", "")).strip(),
            "target": str(item.get("target", "")).strip(),
        })
    return {"entities": entities[:12], "relations": relations[:16]}


def extract_entity_names(text: str) -> list[str]:
    """Entity names mentioned in a question. Empty list if the call fails."""
    try:
        return extract_graph(text)["entities"]
    except Exception as exc:
        print(f"  ⚠️  Entity extraction skipped: {exc}")
        return []
