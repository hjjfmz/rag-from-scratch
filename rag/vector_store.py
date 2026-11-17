"""
📚 LEARN: Hand-written cosine similarity — the math behind semantic search
=======================================================================

The precise-similarity reference used by the SQLite adapter (exact vector scan)
and by benchmark comparisons against Qdrant's index recall. Implemented from
scratch — no numpy.

📚 COSINE SIMILARITY
=======================

Cosine similarity measures the angle between two vectors:

  cosine_sim(A, B) = (A · B) / (|A| × |B|)

Where:
  A · B  = dot product = sum of (a_i × b_i) for each dimension
  |A|    = magnitude = sqrt(sum of a_i²)

The result is a number between -1 and 1:

  1.0  = identical direction (same meaning)
  0.0  = perpendicular (unrelated)
  -1.0 = opposite direction (opposite meaning)

Why cosine and not Euclidean distance? Because cosine similarity cares about
DIRECTION, not magnitude. Two paragraphs about cats will point in a similar
direction regardless of whether one is 10 words or 100 words long.

(The in-memory VectorStore that used to live in this file retired with the
legacy web pipeline; vector recall now happens in Qdrant or the SQLite adapter.)
"""

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors manually.

    📚 LEARN: We implement this from scratch — no numpy! Here's the math:

    Step 1: Dot product — multiply corresponding elements and sum them.
      A = [1, 2, 3], B = [4, 5, 6]
      A · B = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32

    Step 2: Magnitudes — the "length" of each vector.
      |A| = sqrt(1² + 2² + 3²) = sqrt(14) ≈ 3.74
      |B| = sqrt(4² + 5² + 6²) = sqrt(77) ≈ 8.77

    Step 3: Divide.
      cosine_sim = 32 / (3.74 × 8.77) ≈ 0.974

    Args:
        vec_a: First vector (list of floats)
        vec_b: Second vector (list of floats)

    Returns:
        Cosine similarity score between -1.0 and 1.0
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have same length: {len(vec_a)} vs {len(vec_b)}"
        )

    # Step 1: Dot product
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Step 2: Magnitudes
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    # Step 3: Divide (guard against zero-length vectors)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
