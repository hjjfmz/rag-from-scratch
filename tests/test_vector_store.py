"""
Tests for rag.vector_store — verify cosine similarity and vector search.
"""

import pytest
from rag.vector_store import cosine_similarity


# ── Cosine similarity tests ───────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Same vector should have similarity of 1.0"""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0"""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        """Perpendicular vectors should have similarity of 0.0"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similar_vectors_high_score(self):
        """Similar vectors should have a high similarity score"""
        a = [1.0, 0.0, 0.0]
        b = [0.9, 0.1, 0.0]
        score = cosine_similarity(a, b)
        assert score > 0.9

    def test_different_vectors_low_score(self):
        """Different vectors should have a low similarity score"""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0]
        score = cosine_similarity(a, b)
        assert score < 0.1

    def test_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 (not crash)"""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_lengths_raises_error(self):
        """Vectors of different lengths should raise ValueError"""
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_magnitude_invariant(self):
        """Cosine similarity should not change with magnitude scaling"""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        score1 = cosine_similarity(a, b)
        # Scale b by 100x — should give same similarity
        b_scaled = [x * 100 for x in b]
        score2 = cosine_similarity(a, b_scaled)
        assert score1 == pytest.approx(score2)


