"""
📚 LEARN: Retriever — Finding the Right Context for the LLM
=============================================================
The retriever is the "R" in RAG — Retrieval-Augmented Generation.

When a user asks a question, the retriever:
  1. Embeds the question into a vector (using the same model as the documents)
  2. Searches the vector store for the most similar chunks
  3. Returns the top-k chunks ranked by relevance

These chunks become the "context" that we inject into the LLM's prompt,
so it can answer based on YOUR documents instead of guessing.

📚 SEARCH MODES:
  - "vector"  — Semantic search only (cosine similarity on embeddings)
  - "keyword" — BM25 keyword search only (term frequency + IDF)
  - "hybrid"  — Both vector + BM25, merged with Reciprocal Rank Fusion (RRF)

📚 WHY HYBRID?
Vector search finds semantically similar text but can miss exact keywords.
BM25 finds exact keyword matches but misses synonyms.
Hybrid combines both for the best retrieval quality. This is what
Azure AI Search, Elasticsearch, and most production systems use.

📚 WHY SAME EMBEDDING MODEL?
The query MUST be embedded with the same model used for the documents.
Both vectors must live in the same "vector space" for cosine similarity
to be meaningful. Using different models would be like measuring one thing
in miles and another in kilograms — the comparison makes no sense.
"""

from rag.embedder import embed_query
from rag.vector_store import VectorStore
from rag.bm25 import BM25, reciprocal_rank_fusion
from rag.reranker import rerank
from rag.graph_extractor import extract_entity_names
from rag.graph_store import GraphStore, chunk_id_for


class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.

    📚 LEARN: The retriever is a thin orchestration layer. It doesn't do
    much on its own — it delegates:
      - Embedding → embedder.py
      - Searching → vector_store.py

    But it's valuable because it:
      1. Provides a clean interface: query in → ranked results out
      2. Handles configuration (top_k, threshold) in one place
      3. Keeps the pipeline modular — you can swap the vector store
         without changing the rest of the code
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        threshold: float = 0.0,
        search_mode: str = "vector",
        use_reranker: bool = False,
        graph_store: GraphStore | None = None,
    ):
        """
        Args:
            vector_store: A populated VectorStore instance
            top_k: Maximum number of chunks to retrieve (default: 5)
            threshold: Minimum similarity score to include (default: 0.0)
            search_mode: "vector", "keyword", "hybrid", or "hybrid_graph"
            use_reranker: If True, rerank results with LLM (default: False)
            graph_store: Entity graph used by hybrid_graph. Ignored otherwise.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold
        self.search_mode = search_mode
        self.use_reranker = use_reranker
        self.graph_store = graph_store

        # 📚 LEARN: Build a BM25 index from the same data in the vector store.
        # This lets us do keyword search alongside vector search without
        # storing the data twice — BM25 just needs the text and metadata.
        self.bm25 = BM25()
        if search_mode in ("keyword", "hybrid", "hybrid_graph") and vector_store.entries:
            bm25_docs = [
                {"text": e["text"], "metadata": e["metadata"]}
                for e in vector_store.entries
            ]
            self.bm25.index(bm25_docs)

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for a natural language query.

        📚 LEARN: This method supports three search modes:

        "vector" (default):
          query → embed → cosine similarity search
          Good at finding semantically similar text (synonyms, paraphrases).

        "keyword" (BM25 only):
          query → tokenize → BM25 score
          Good at finding exact keyword matches (product names, codes).

        "hybrid" (both + RRF merge):
          query → [vector search] + [BM25 search] → Reciprocal Rank Fusion
          Best of both worlds. Chunks ranking well in BOTH methods get boosted.

        Args:
            query: The user's question in natural language

        Returns:
            List of dicts, each with:
              - "text": the chunk content
              - "metadata": source, chunk_index, etc.
              - "score": similarity score (cosine for vector, BM25 for keyword, RRF for hybrid)
            Sorted by score descending (most relevant first)
        """
        print(f"  🔍 Embedding query: \"{query[:60]}...\"" if len(query) > 60
              else f"  🔍 Embedding query: \"{query}\"")

        if self.search_mode == "keyword":
            # BM25 only — no embedding needed
            # 📚 LEARN: When reranking with BM25, we retrieve more candidates
            # initially (top_k * 4) so the reranker has a wider pool to pick from.
            retrieve_k = self.top_k * 4 if self.use_reranker else self.top_k
            results = self.bm25.search(query, top_k=retrieve_k)
            print(f"  📄 Found {len(results)} relevant chunks via BM25 "
                  f"(top score: {results[0]['score']:.4f})" if results
                  else "  ⚠️  No relevant chunks found!")
            return self._maybe_rerank(query, results)

        # Vector search (needed for both "vector" and "hybrid" modes)
        query_vector = embed_query(query)

        # Guard against stale index when embedding model changes.
        if self.vector_store.entries:
          first_vec = self.vector_store.entries[0].get("embedding", [])
          if isinstance(first_vec, list) and first_vec and len(first_vec) != len(query_vector):
            raise ValueError(
              "Embedding dimension mismatch between query and stored index. "
              f"query_dim={len(query_vector)}, index_dim={len(first_vec)}. "
              "Please delete rag_index.json and re-ingest documents."
            )

        wide = self.search_mode in ("hybrid", "hybrid_graph") or self.use_reranker
        retrieve_k = self.top_k * 4 if wide else self.top_k
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=retrieve_k,
            threshold=self.threshold,
        )

        if self.search_mode == "vector":
            print(f"  📄 Found {len(vector_results)} relevant chunks "
                  f"(top score: {vector_results[0]['score']:.4f})" if vector_results
                  else "  ⚠️  No relevant chunks found!")
            return self._maybe_rerank(query, vector_results)

        # Hybrid mode: also run BM25 and merge with RRF.
        # hybrid_graph adds a third ranked list: 1-hop chunks from the entity graph.
        bm25_results = self.bm25.search(query, top_k=self.top_k * 4)
        graph_results = self._graph_results(query) if self.search_mode == "hybrid_graph" else []
        fuse_k = self.top_k * 4 if self.use_reranker else self.top_k
        if graph_results:
            fused = reciprocal_rank_fusion(
                vector_results, bm25_results, graph_results, top_k=fuse_k
            )
        else:
            fused = reciprocal_rank_fusion(
                vector_results, bm25_results, top_k=fuse_k
            )

        label = "hybrid+graph" if self.search_mode == "hybrid_graph" else "hybrid"
        print(f"  📄 Found {len(fused)} relevant chunks via {label} search "
              f"(vector: {len(vector_results)}, BM25: {len(bm25_results)}, "
              f"graph: {len(graph_results)}, "
              f"top RRF: {fused[0]['score']:.4f})" if fused
              else "  ⚠️  No relevant chunks found!")
        return self._maybe_rerank(query, fused)

    def _graph_results(self, query: str) -> list[dict]:
        """Ranked chunks for entities mentioned in the query and their neighbors.

        An empty graph falls back to plain hybrid: this returns [] and the
        caller fuses only the vector and BM25 lists.
        """
        graph = self.graph_store
        if graph is None or not graph.nodes:
            return []

        names: list[str] = []
        seen: set[str] = set()
        for name in list(graph.match_names(query)) + extract_entity_names(query):
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)

        by_id: dict[str, dict] = {}
        for entry in self.vector_store.entries:
            metadata = entry.get("metadata") or {}
            chunk_id = metadata.get("chunk_id") or chunk_id_for(
                metadata.get("source", ""), metadata.get("chunk_index", -1)
            )
            by_id.setdefault(chunk_id, entry)

        ranked = []
        for chunk_id in graph.one_hop_chunk_ids(names):
            entry = by_id.get(chunk_id)
            if entry is None:
                continue
            ranked.append({
                "text": entry["text"],
                "metadata": entry.get("metadata") or {},
                "score": 0.0,
            })
        return ranked

    def _maybe_rerank(self, query: str, results: list[dict]) -> list[dict]:
        """
        Apply LLM reranking if enabled, otherwise return as-is.

        📚 LEARN: This is the two-stage pattern:
          Stage 1 already retrieved top-N candidates (fast, cheap)
          Stage 2 (this step): rerank N → keep top_k (slow, precise)
        """
        if not self.use_reranker or not results:
            return results

        return rerank(query, results, top_k=self.top_k)


# ── Manual testing ─────────────────────────────────────────────
if __name__ == "__main__":
    # 📚 LEARN: To test the retriever, we need:
    #   1. A populated vector store (requires API key for real embeddings)
    #   2. A query
    # If you have your .env set up, try running this!

    import sys
    from rag.loader import load_document
    from rag.chunker import create_chunks
    from rag.embedder import embed_chunks

    print("=" * 60)
    print("RETRIEVER TEST — End-to-End Search")
    print("=" * 60)

    # Load and chunk a sample document
    source = "sample_data/sample.md"
    print(f"\n1. Loading: {source}")
    doc = load_document(source)

    print(f"2. Chunking ({len(doc['text'])} chars)...")
    chunks = create_chunks(doc["text"], source=doc["source"], chunk_size=300, overlap=50)
    print(f"   → {len(chunks)} chunks")

    print("3. Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    print("4. Building vector store...")
    store = VectorStore()
    store.add_chunks(embedded_chunks)

    # Create retriever and search
    retriever = Retriever(store, top_k=3, threshold=0.3)

    query = "What is overfitting?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    print(f"\n5. Searching: \"{query}\"")
    results = retriever.retrieve(query)

    print(f"\n{'='*60}")
    print(f"TOP {len(results)} RESULTS:")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['metadata'].get('source', '?')}, "
              f"Chunk: {r['metadata'].get('chunk_index', '?')}")
        print(r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"])
