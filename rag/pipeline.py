"""
📚 LEARN: Pipeline — Wiring It All Together
=============================================
This module connects all the pieces into a complete RAG system:

  INGEST FLOW:
    Source file/URL → Loader → Chunker → Embedder → Vector Store (save to disk)

  QUERY FLOW:
    User question → Embedder → Retriever (search store) → Generator → Answer

Think of the pipeline as the "conductor" — each module is an instrument,
and the pipeline makes sure they play in the right order.
"""

import os
from datetime import datetime
from rag.loader import load_document
from rag.chunker import create_chunks
from rag.embedder import embed_chunks
from rag.vector_store import create_vector_store
from rag.retriever import Retriever
from rag.generator import generate_with_sources, build_no_answer_response, generate_chat
from rag.graph_store import GraphStore, chunk_id_for, graph_path_for
from rag.graph_extractor import extract_graph

# Default paths for the vector store index
DEFAULT_INDEX_PATH = "rag_index.json"


class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest documents and answer questions.

    📚 LEARN: This class is the "glue" that connects all our modules.
    It manages the vector store lifecycle (create, populate, save, load)
    and provides simple ingest() and query() methods.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        threshold: float = 0.3,
        search_mode: str = "vector",
        use_reranker: bool = False,
    ):
        """
        Args:
            index_path: Where to save/load the vector store
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            top_k: Number of chunks to retrieve per query
            threshold: Minimum similarity score for retrieved chunks
            search_mode: "vector", "keyword", "hybrid", or "hybrid_graph"
            use_reranker: If True, rerank results with LLM (default: False)
        """
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.threshold = threshold
        self.search_mode = search_mode
        self.use_reranker = use_reranker

        # Load existing index if it exists, otherwise create empty store
        self.store = create_vector_store()
        self.graph = GraphStore()
        self.graph_path = graph_path_for(index_path)
        if os.path.exists(index_path):
            self.store.load(index_path)
        elif os.getenv("VECTOR_BACKEND", "json").strip().lower() == "qdrant":
            self.store.load(index_path)
        if os.path.exists(self.graph_path):
            self.graph.load(self.graph_path)

    def ingest(self, source: str, save: bool = True) -> int:
        """
        Ingest a document: load → chunk → embed → store.

        📚 LEARN: This is the "indexing" phase. You run this once for each
        document. After ingestion, the chunks and their vectors are saved
        to disk so you don't need to re-embed them every time.

        Args:
            source: File path or URL to ingest

        Returns:
            Number of chunks created
        """
        print(f"\n📥 Ingesting: {source}")

        # Step 1: Load the document
        print("  1️⃣  Loading document...")
        doc = load_document(source)
        print(f"     → {doc['type']} document, {len(doc['text'])} characters")

        # Step 2: Chunk the text
        print("  2️⃣  Chunking text...")
        chunks = create_chunks(
            text=doc["text"],
            source=doc["source"],
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            strategy="sentence",
        )
        print(f"     → {len(chunks)} chunks (avg {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars each)")

        for chunk in chunks:
            chunk["chunk_id"] = chunk_id_for(chunk.get("source", doc["source"]), chunk.get("chunk_index", 0))

        print("  2️⃣b Extracting entities...")
        self.graph.drop_source(doc["source"])
        for chunk in chunks:
            try:
                extracted = extract_graph(chunk["text"])
            except Exception as exc:
                print(f"     ⚠️  Graph extraction skipped for chunk {chunk['chunk_id']}: {exc}")
                extracted = {"entities": [], "relations": []}
            self.graph.add_extraction(
                extracted["entities"], extracted["relations"], chunk["chunk_id"]
            )
        print(f"     → graph has {len(self.graph.nodes)} entities, {len(self.graph.edges)} relations")

        # Step 3: Embed the chunks
        print("  3️⃣  Embedding chunks...")
        embedded_chunks = embed_chunks(chunks)

        # Step 4: Add to vector store and save
        print("  4️⃣  Adding to vector store...")
        self.store.add_chunks(embedded_chunks)
        if save:
            self.store.save(self.index_path)
            self.graph.save(self.graph_path)

        print(f"  ✅ Done! Store now has {len(self.store)} total chunks.")
        return len(chunks)

    def query(
        self,
        question: str,
        verbose: bool = False,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
        search_mode: str | None = None,
        use_reranker: bool | None = None,
    ) -> dict:
        """
        Ask a question and get an answer grounded in your documents.

        📚 LEARN: This is the "query" phase — the whole point of RAG:
          1. Embed the question
          2. Find similar chunks (retrieval)
          3. Send chunks + question to LLM (generation)
          4. Return answer with sources

        Args:
            question: Natural language question
            verbose: If True, show retrieved chunks before the answer

        Returns:
            Dict with "answer", "sources", and "chunks_used"
        """
        if len(self.store) == 0:
            print("⚠️  No documents ingested yet! Answering from general knowledge.\n")
            print("💬 Answer:")
            answer = generate_chat(question, stream=True)
            return {
                "answer": answer,
                "sources": [],
                "chunks_used": [],
                "status": "no_docs",
                "user_guidance": "No documents indexed — the answer above is from general knowledge, not your documents. Use 'ingest' to add documents for RAG-grounded answers.",
                "retrieval_quality": {},
            }

        effective_top_k = self.top_k if top_k is None else top_k
        effective_threshold = self.threshold if threshold is None else threshold
        effective_search_mode = self.search_mode if search_mode is None else search_mode
        effective_use_reranker = self.use_reranker if use_reranker is None else use_reranker

        # Step 1 & 2: Retrieve relevant chunks
        retriever = Retriever(
            self.store,
            top_k=effective_top_k,
            threshold=effective_threshold,
            search_mode=effective_search_mode,
            use_reranker=effective_use_reranker,
            graph_store=getattr(self, "graph", None),
        )
        results = retriever.retrieve(question)

        if not results:
            print("⚠️  No relevant documents found. Answering from general knowledge.\n")
            print("💬 Answer:")
            answer = generate_chat(question, stream=True)
            return {
                "answer": answer,
                "sources": [],
                "chunks_used": [],
                "status": "no_relevant",
                "user_guidance": "No relevant document chunks found — the answer above is from general knowledge. Try --search-mode hybrid or lower --threshold for better document matching.",
                "retrieval_quality": {},
            }

        quality = self._analyze_retrieval_quality(
            results,
            search_mode=effective_search_mode,
            threshold=effective_threshold,
            top_k=effective_top_k,
        )
        if quality["status"] == "low_confidence":
            print("⚠️  Low retrieval confidence — answering from general knowledge (not your documents).\n")
            print("💬 Answer:")
            answer = generate_chat(question, stream=True)
            return {
                "answer": answer,
                "sources": [],
                "chunks_used": [],
                "status": "low_confidence",
                "user_guidance": "Retrieved context looks weak — the answer above is from general knowledge. Try --search-mode hybrid, increase --top-k, or rephrase with specific keywords for document-grounded results.",
                "retrieval_quality": quality,
            }

        # Show retrieved chunks if verbose
        if verbose:
            print(f"\n📄 Retrieved {len(results)} chunks:")
            for i, r in enumerate(results):
                source = r["metadata"].get("source", "?")
                print(f"  [{i+1}] (score: {r['score']:.4f}) {source}")
                print(f"      {r['text'][:100]}...")
            print()

        # Step 3: Generate answer
        print("\n💬 Answer:")
        result = generate_with_sources(question, results, stream=True)
        result["status"] = "ok"
        result["user_guidance"] = ""
        result["retrieval_quality"] = quality

        return result

    def _analyze_retrieval_quality(
        self,
        results: list[dict],
        *,
        search_mode: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Lightweight retrieval quality check.

        We intentionally keep this heuristic simple and conservative for now.
        """
        if not results:
            return {
                "status": "no_relevant",
                "top_score": 0.0,
                "avg_top3_score": 0.0,
            }

        top_score = float(results[0].get("score", 0.0))
        top_scores = [float(r.get("score", 0.0)) for r in results[:3]]
        avg_top3 = sum(top_scores) / len(top_scores) if top_scores else 0.0

        effective_search_mode = self.search_mode if search_mode is None else search_mode
        effective_threshold = self.threshold if threshold is None else threshold
        effective_top_k = self.top_k if top_k is None else top_k

        if effective_search_mode == "vector":
            low_bar = max(effective_threshold + 0.15, 0.45)
            is_low_confidence = top_score < low_bar or avg_top3 < max(effective_threshold + 0.1, 0.4)
        else:
            min_expected = min(2, effective_top_k)
            is_low_confidence = len(results) < min_expected

        return {
            "status": "low_confidence" if is_low_confidence else "ok",
            "top_score": top_score,
            "avg_top3_score": avg_top3,
            "num_results": len(results),
            "search_mode": effective_search_mode,
        }

    def list_documents(self) -> list[str]:
        """List all documents that have been ingested."""
        return self.store.list_sources()

    def auto_ingest(self, sample_dir: str = "sample_data"):
        """
        Auto-ingest all supported files from a directory on first run.

        📚 LEARN: This is a convenience method for new users. On first run,
        the vector store is empty, so we automatically ingest all sample
        documents. This gives users something to query immediately without
        manual setup.

        Args:
            sample_dir: Directory to scan for documents (default: sample_data/)
        """
        if len(self.store) > 0:
            return  # Already have data — skip

        if not os.path.isdir(sample_dir):
            return  # No sample_data folder — skip

        # Find all supported files (including subdirectories)
        supported_extensions = (".txt", ".md", ".pdf")
        files = []
        for root, dirs, filenames in os.walk(sample_dir):
            for f in sorted(filenames):
                if f.lower().endswith(supported_extensions):
                    files.append(os.path.join(root, f))

        if not files:
            return

        print("=" * 60)
        print("🚀 First run detected! Auto-ingesting sample documents...")
        print("=" * 60)

        for file_path in files:
            self.ingest(file_path)

        print(f"\n✅ Auto-ingested {len(files)} documents. Ready to query!")

    def _normalize_source(self, source: str) -> str:
        return str(source or "").replace("\\", "/")

    def _scan_sample_files(self, sample_dir: str) -> list[str]:
        supported_extensions = (".txt", ".md", ".pdf")
        files: list[str] = []
        for root, dirs, filenames in os.walk(sample_dir):
            for f in sorted(filenames):
                if f.lower().endswith(supported_extensions):
                    files.append(os.path.join(root, f))
        return files

    def _is_sample_source(self, source: str, sample_dir: str) -> bool:
        src = self._normalize_source(source)
        prefix = self._normalize_source(sample_dir).rstrip("/") + "/"
        return src.startswith(prefix)

    def sync_sample_data(self, sample_dir: str = "sample_data") -> dict:
        started_at = datetime.now().isoformat()

        if not os.path.isdir(sample_dir):
            return {
                "changed": False,
                "status": "no_sample_dir",
                "sample_dir": sample_dir,
                "files_total": 0,
                "files_reindexed": 0,
                "files": [],
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }

        files = self._scan_sample_files(sample_dir)
        normalized_files = sorted({self._normalize_source(p) for p in files})

        sample_sources_in_index = sorted({
            self._normalize_source(s)
            for s in self.store.list_sources()
            if self._is_sample_source(s, sample_dir)
        })

        changed = set(normalized_files) != set(sample_sources_in_index)
        if not changed:
            return {
                "changed": False,
                "status": "up_to_date",
                "sample_dir": sample_dir,
                "files_total": len(normalized_files),
                "files_reindexed": 0,
                "files": [],
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }

        original_entries = list(self.store.entries)
        graph_nodes = {name: list(ids) for name, ids in self.graph.nodes.items()}
        graph_edges = list(self.graph.edges)
        try:
            kept = [
                e for e in self.store.entries
                if not self._is_sample_source(e.get("metadata", {}).get("source", ""), sample_dir)
            ]
            self.store.replace_entries(kept)
            for source in sample_sources_in_index:
                self.graph.drop_source(source)

            for file_path in files:
                self.ingest(file_path, save=False)

            self.store.save(self.index_path)
            self.graph.save(self.graph_path)
            return {
                "changed": True,
                "status": "reindexed",
                "sample_dir": sample_dir,
                "files_total": len(normalized_files),
                "files_reindexed": len(normalized_files),
                "files": normalized_files,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.store.replace_entries(original_entries)
            self.graph.nodes = graph_nodes
            self.graph.edges = graph_edges
            return {
                "changed": True,
                "status": "error",
                "sample_dir": sample_dir,
                "files_total": len(normalized_files),
                "files_reindexed": 0,
                "files": [],
                "error": str(e),
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
            }
