"""
Tests for rag.chunker — verify all three chunking strategies work correctly.
"""

import pytest
from rag.chunker import (
    chunk_fixed_size,
    chunk_with_overlap,
    chunk_sentences,
    create_chunks,
)


# ── Fixed-size chunking tests ──────────────────────────────────

class TestFixedSizeChunker:
    def test_empty_text(self):
        assert chunk_fixed_size("") == []

    def test_text_shorter_than_chunk_size(self):
        result = chunk_fixed_size("Hello world", chunk_size=100)
        assert result == ["Hello world"]

    def test_text_exactly_chunk_size(self):
        text = "A" * 500
        result = chunk_fixed_size(text, chunk_size=500)
        assert len(result) == 1
        assert len(result[0]) == 500

    def test_splits_into_multiple_chunks(self):
        text = "A" * 1000
        result = chunk_fixed_size(text, chunk_size=500)
        assert len(result) == 2

    def test_last_chunk_can_be_smaller(self):
        text = "A" * 750
        result = chunk_fixed_size(text, chunk_size=500)
        assert len(result) == 2
        assert len(result[0]) == 500
        assert len(result[1]) == 250


# ── Overlapping chunking tests ─────────────────────────────────

class TestOverlapChunker:
    def test_empty_text(self):
        assert chunk_with_overlap("") == []

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_with_overlap("Hello", chunk_size=100, overlap=100)

    def test_overlap_creates_more_chunks(self):
        text = "A" * 1000
        no_overlap = chunk_fixed_size(text, chunk_size=500)
        with_overlap = chunk_with_overlap(text, chunk_size=500, overlap=100)
        # Overlap should create more chunks since we advance less each step
        assert len(with_overlap) >= len(no_overlap)

    def test_chunks_share_content(self):
        text = "ABCDEFGHIJ"
        result = chunk_with_overlap(text, chunk_size=6, overlap=2)
        # Chunk 1: ABCDEF, Chunk 2: EFGHIJ — "EF" shared
        assert len(result) == 2
        # The end of chunk 1 should overlap with the start of chunk 2
        overlap_text = result[0][-2:]  # Last 2 chars of chunk 1
        assert overlap_text in result[1]  # Should appear in chunk 2


# ── Sentence-aware chunking tests ──────────────────────────────

class TestSentenceChunker:
    def test_empty_text(self):
        assert chunk_sentences("") == []

    def test_single_sentence(self):
        result = chunk_sentences("Hello world.", chunk_size=500)
        assert result == ["Hello world."]

    def test_keeps_sentences_intact(self):
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_sentences(text, chunk_size=50, overlap=0)
        # Each chunk should end with a period (complete sentence)
        for chunk in result:
            assert chunk.rstrip().endswith(".")

    def test_groups_sentences_within_size_limit(self):
        text = "Short. Also short. Tiny."
        result = chunk_sentences(text, chunk_size=500, overlap=0)
        # All sentences fit in one chunk
        assert len(result) == 1

    def test_respects_chunk_size(self):
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth sentence."
        )
        result = chunk_sentences(text, chunk_size=60, overlap=0)
        assert len(result) > 1


# ── Metadata enrichment tests ──────────────────────────────────

class TestCreateChunks:
    def test_returns_dicts_with_required_keys(self):
        chunks = create_chunks("Hello world.", source="test.md")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "text" in chunk
        assert "source" in chunk
        assert "chunk_index" in chunk
        assert "char_start" in chunk
        assert "char_end" in chunk

    def test_source_is_preserved(self):
        chunks = create_chunks("Hello.", source="my_doc.pdf")
        assert chunks[0]["source"] == "my_doc.pdf"

    def test_chunk_indices_are_sequential(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = create_chunks(text, source="test.md", chunk_size=40, overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_invalid_strategy_raises_error(self):
        with pytest.raises(ValueError):
            create_chunks("Hello.", source="test.md", strategy="magic")

    def test_all_strategies_accepted(self):
        text = "Hello world. Goodbye world."
        for strategy in ["fixed", "overlap", "sentence"]:
            chunks = create_chunks(text, source="test.md", strategy=strategy)
            assert len(chunks) >= 1
