"""Bounded Unicode chunks whose offsets always select the original verbatim text."""
import re


def split_chunks(text, source, chunk_size=900, overlap=120):
    if not 128 <= chunk_size <= 6000 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid bounded chunk parameters")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer paragraph/sentence boundaries near the end. Unpunctuated text has a hard cap.
            lower = start + max(chunk_size // 2, overlap + 1)
            endings = [match.end() + lower for match in re.finditer(r"\n\n|[。！？]|[.!?](?=\s)", text[lower:end])]
            if endings:
                end = endings[-1]
        piece = text[start:end]
        if piece.strip():
            chunks.append({"text": piece, "source": source, "chunk_index": len(chunks),
                           "char_start": start, "char_end": end})
        if end == len(text):
            break
        start = end - overlap
    return chunks
