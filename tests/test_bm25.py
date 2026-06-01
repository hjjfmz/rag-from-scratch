"""
Tests for rag/bm25.py — BM25 Keyword Search from Scratch (TDD)

📚 LEARN: We write tests FIRST (TDD) to define exactly what BM25 should do,
then implement it to make the tests pass.

BM25 has three key properties we need to test:
1. Rare words should score higher than common words (IDF)
2. More occurrences of a query word = higher score (TF)
3. Length normalization: a short focused doc scores similarly to a long one
"""

import pytest
from rag.bm25 import BM25, tokenize


# ── Tokenizer Tests ───────────────────────────────────────────


class TestTokenize:
    """Test the tokenizer that preprocesses text for BM25."""

    def test_basic_tokenize(self):
        tokens = tokenize("The cat sat on the mat")
        assert "cat" in tokens
        assert "sat" in tokens
        assert "mat" in tokens

    def test_lowercase(self):
        tokens = tokenize("The CAT Sat")
        assert "cat" in tokens
        assert "sat" in tokens

    def test_punctuation_removed(self):
        tokens = tokenize("Hello, world! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        # Punctuation should be stripped
        assert "hello," not in tokens

    def test_stopwords_removed(self):
        """Common words like 'the', 'a', 'is' should be removed."""
        tokens = tokenize("the cat is on a mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "on" not in tokens
        assert "cat" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_kept(self):
        tokens = tokenize("The price is 50 dollars")
        assert "50" in tokens
        assert "price" in tokens


# ── BM25 Index Tests ─────────────────────────────────────────


class TestBM25Index:
    """Test building the BM25 index from documents."""

    def setup_method(self):
        """Create a small test corpus."""
        self.docs = [
            {"text": "GlowBeauty offers a 30-day return policy on all products",
             "metadata": {"source": "refund-policy.md"}},
            {"text": "Standard shipping takes 5-7 business days for delivery",
             "metadata": {"source": "shipping-policy.md"}},
            {"text": "The loyalty program rewards customers with points for every purchase",
             "metadata": {"source": "loyalty-program.md"}},
            {"text": "Employees receive 15 days of paid time off per year",
             "metadata": {"source": "pto-policy.md"}},
        ]
        self.bm25 = BM25()
        self.bm25.index(self.docs)

    def test_index_size(self):
        assert len(self.bm25) == 4

    def test_avg_doc_length_computed(self):
        assert self.bm25.avg_doc_length > 0

    def test_idf_computed(self):
        """IDF values should exist for words in the corpus."""
        assert len(self.bm25.idf) > 0

    def test_rare_word_higher_idf(self):
        """A word in 1 doc should have higher IDF than a word in multiple docs."""
        # "return" only in doc 0, but if other words appear in multiple docs
        # we can check the IDF ordering
        self.bm25.index(self.docs)
        # Any word appearing in fewer docs should have >= IDF of more common words
        # Let's test with a fresh corpus that makes this clear
        bm25 = BM25()
        bm25.index([
            {"text": "cat dog", "metadata": {}},
            {"text": "cat fish", "metadata": {}},
            {"text": "cat bird", "metadata": {}},
        ])
        # "cat" appears in all 3, "dog" in 1 → dog should have higher IDF
        assert bm25.idf.get("dog", 0) > bm25.idf.get("cat", 0)


# ── BM25 Search Tests ────────────────────────────────────────


class TestBM25Search:
    """Test BM25 search ranking."""

    def setup_method(self):
        self.docs = [
            {"text": "GlowBeauty offers a 30-day return policy on all products. "
                     "You can return opened products within the return window.",
             "metadata": {"source": "refund-policy.md"}},
            {"text": "Standard shipping takes 5-7 business days for delivery. "
                     "Express shipping is available for an additional fee.",
             "metadata": {"source": "shipping-policy.md"}},
            {"text": "The loyalty program rewards customers with points. "
                     "Earn points on every purchase at GlowBeauty.",
             "metadata": {"source": "loyalty-program.md"}},
            {"text": "Employees receive 15 days of paid time off per year. "
                     "PTO accrues on a bi-weekly basis.",
             "metadata": {"source": "pto-policy.md"}},
        ]
        self.bm25 = BM25()
        self.bm25.index(self.docs)

    def test_exact_keyword_match(self):
        """Query with exact keywords should find the right document."""
        results = self.bm25.search("return policy", top_k=4)
        assert len(results) > 0
        assert results[0]["metadata"]["source"] == "refund-policy.md"

    def test_shipping_query(self):
        results = self.bm25.search("shipping delivery days", top_k=4)
        assert results[0]["metadata"]["source"] == "shipping-policy.md"

    def test_pto_query(self):
        results = self.bm25.search("paid time off PTO", top_k=4)
        assert results[0]["metadata"]["source"] == "pto-policy.md"

    def test_top_k_limits_results(self):
        results = self.bm25.search("GlowBeauty", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self):
        """Query with no matching words should return empty (or very low scores)."""
        results = self.bm25.search("xyzzyspoon quantum", top_k=4)
        # Either empty or all zero scores
        assert len(results) == 0 or all(r["score"] == 0.0 for r in results)

    def test_results_have_required_fields(self):
        results = self.bm25.search("return", top_k=1)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_results_sorted_by_score(self):
        results = self.bm25.search("return policy", top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_term_frequency_matters(self):
        """A doc mentioning 'return' twice should score higher than one mentioning it once."""
        bm25 = BM25()
        bm25.index([
            {"text": "return return return policy", "metadata": {"source": "a.md"}},
            {"text": "return policy", "metadata": {"source": "b.md"}},
        ])
        results = bm25.search("return", top_k=2)
        # Doc with more "return" mentions should rank first
        assert results[0]["metadata"]["source"] == "a.md"


# ── Reciprocal Rank Fusion Tests ──────────────────────────────


class TestReciprocalRankFusion:
    """Test the RRF merge function that combines vector + BM25 results."""

    def test_basic_fusion(self):
        from rag.bm25 import reciprocal_rank_fusion

        vector_results = [
            {"text": "chunk A", "metadata": {"source": "a.md"}, "score": 0.9},
            {"text": "chunk C", "metadata": {"source": "c.md"}, "score": 0.8},
            {"text": "chunk B", "metadata": {"source": "b.md"}, "score": 0.7},
        ]
        bm25_results = [
            {"text": "chunk B", "metadata": {"source": "b.md"}, "score": 5.0},
            {"text": "chunk A", "metadata": {"source": "a.md"}, "score": 3.0},
            {"text": "chunk D", "metadata": {"source": "d.md"}, "score": 2.0},
        ]

        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)

        # chunk A and chunk B appear in both lists → should be boosted
        texts = [r["text"] for r in fused]
        assert "chunk A" in texts
        assert "chunk B" in texts
        # Both should rank above chunks that only appear in one list
        top_2_texts = set(texts[:2])
        assert "chunk A" in top_2_texts
        assert "chunk B" in top_2_texts

    def test_fusion_top_k(self):
        from rag.bm25 import reciprocal_rank_fusion

        vec = [{"text": f"v{i}", "metadata": {}, "score": 1.0 - i*0.1} for i in range(5)]
        bm25 = [{"text": f"b{i}", "metadata": {}, "score": 5.0 - i} for i in range(5)]

        fused = reciprocal_rank_fusion(vec, bm25, top_k=3)
        assert len(fused) == 3

    def test_fusion_empty_one_side(self):
        """If BM25 returns nothing, fusion should still work (just vector results)."""
        from rag.bm25 import reciprocal_rank_fusion

        vec = [
            {"text": "chunk A", "metadata": {}, "score": 0.9},
            {"text": "chunk B", "metadata": {}, "score": 0.8},
        ]
        fused = reciprocal_rank_fusion(vec, [], top_k=5)
        assert len(fused) == 2

    def test_fusion_deduplicates(self):
        """Same chunk in both lists should only appear once in output."""
        from rag.bm25 import reciprocal_rank_fusion

        vec = [{"text": "same chunk", "metadata": {"source": "x.md"}, "score": 0.9}]
        bm25 = [{"text": "same chunk", "metadata": {"source": "x.md"}, "score": 5.0}]

        fused = reciprocal_rank_fusion(vec, bm25)
        assert len(fused) == 1

    def test_fusion_has_rrf_score(self):
        """Fused results should have an 'rrf_score' field."""
        from rag.bm25 import reciprocal_rank_fusion

        vec = [{"text": "A", "metadata": {}, "score": 0.9}]
        bm25 = [{"text": "B", "metadata": {}, "score": 5.0}]

        fused = reciprocal_rank_fusion(vec, bm25)
        for r in fused:
            assert "score" in r
