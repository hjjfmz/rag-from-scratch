# How It Works — A Guided Codebase Walkthrough

This document walks you through the entire codebase, file by file, explaining how each piece connects to the next.

---

## Project Structure

```
rag-from-scratch/
├── cli.py                  ← Entry point (what the user runs)
├── rag/
│   ├── pipeline.py         ← Orchestrator (wires everything together)
│   ├── loader.py           ← Step 1: Read documents
│   ├── chunker.py          ← Step 2: Split into pieces
│   ├── embedder.py         ← Step 3: Convert to vectors
│   ├── vector_store.py     ← Step 4: Store and search vectors
│   ├── bm25.py             ← BM25 keyword search (from scratch)
│   ├── retriever.py        ← Multi-mode retrieval (vector/keyword/hybrid)
│   ├── reranker.py         ← LLM-based reranking (two-stage)
│   ├── agent.py            ← ReAct agent for iterative retrieval
│   ├── agentic_pipeline.py ← Agentic RAG pipeline orchestration
│   ├── generator.py        ← Generate answers + miss-response builder
│   └── evaluator.py        ← 3-metric evaluation framework
├── tests/                  ← Unit & integration tests (156+ tests)
├── sample_data/            ← Example documents to try
└── docs/                   ← You are here!
```

---

## 1. `cli.py` — The Front Door

When you type `python cli.py query "What is overfitting?"`, this is the file that runs.

It uses Python's `argparse` to parse commands:
- **`ingest`** — calls `pipeline.ingest()` to load and index a document
- **`query`** — calls `pipeline.query()` to search and answer
- **`list`** — calls `pipeline.list_documents()` to show indexed sources
- **`agentic-query`** — calls `AgenticRAGPipeline.query()` for ReAct-based retrieval
- **`eval`** — runs the evaluation framework against test questions

The CLI itself does no heavy lifting — it just parses arguments and delegates to the pipeline.

### Status Hints for Retrieval Misses

When the system can't find good matches, the CLI prints actionable hints:

```
ℹ️  Retrieval status: no_relevant
💡 Suggestion: Try rephrasing your question, switch to --search-mode hybrid,
   or lower --threshold.
📊 Retrieval quality: top_score=0.431, avg_top3=0.389
```

This is handled by the `_print_status_hint()` helper, which reads the `status`, `user_guidance`, and `retrieval_quality` fields from the pipeline's result dict.

---

## 2. `rag/pipeline.py` — The Conductor

The pipeline is the "glue" that calls each module in the right order.

### Ingest Flow
```python
def ingest(self, source):
    doc = load_document(source)        # loader.py
    chunks = create_chunks(doc)        # chunker.py
    embedded = embed_chunks(chunks)    # embedder.py
    self.store.add_chunks(embedded)    # vector_store.py
    self.store.save(self.index_path)   # persist to disk
```

### Query Flow (with Retrieval Miss Governance)
```python
def query(self, question, verbose=False):
    # Guard 1: Is the knowledge base empty?
    if len(self.store) == 0:
        return build_no_answer_response(status="no_docs", user_guidance="...")

    # Step 1 & 2: Retrieve relevant chunks
    results = retriever.retrieve(question)

    # Guard 2: Did we find anything above the threshold?
    if not results:
        return build_no_answer_response(status="no_relevant", user_guidance="...")

    # Guard 3: Are the results high-quality enough?
    quality = self._analyze_retrieval_quality(results)
    if quality["status"] == "low_confidence":
        return build_no_answer_response(status="low_confidence",
                                         user_guidance="...",
                                         retrieval_quality=quality)

    # All checks passed — generate the answer
    result = generate_with_sources(question, results, stream=True)
    result["status"] = "ok"
    result["user_guidance"] = ""
    result["retrieval_quality"] = quality
    return result
```

### `_analyze_retrieval_quality()` — The Confidence Gate

This is the heuristic that decides whether retrieved chunks are "good enough" to use:

```python
def _analyze_retrieval_quality(self, results):
    top_score = float(results[0].get("score", 0.0))
    top_scores = [float(r.get("score", 0.0)) for r in results[:3]]
    avg_top3 = sum(top_scores) / len(top_scores)

    if self.search_mode == "vector":
        # Cosine similarity scores on 0–1 scale.
        # Require top_score ≥ max(threshold+0.15, 0.45) AND
        #         avg_top3  ≥ max(threshold+0.1,  0.40)
        low_bar = max(self.threshold + 0.15, 0.45)
        is_low_confidence = top_score < low_bar or avg_top3 < max(self.threshold + 0.1, 0.4)
    else:
        # BM25/hybrid: scores are RRF-fused ranks, not directly comparable.
        # Use a simpler check: do we have enough results?
        min_expected = min(2, self.top_k)
        is_low_confidence = len(results) < min_expected

    return {
        "status": "low_confidence" if is_low_confidence else "ok",
        "top_score": top_score,
        "avg_top3_score": avg_top3,
        "num_results": len(results),
        "search_mode": self.search_mode,
    }
```

**Key insight**: The two strategies exist because BM25 keyword scores and embedding cosine-similarity scores are fundamentally different scales. You can't apply the same numeric thresholds to both.

---

## 3. `rag/loader.py` — Reading Documents

Three loader functions, one smart dispatcher:

| Function | Input | Output |
|----------|-------|--------|
| `load_text_file(path)` | `.txt` or `.md` file | Plain text string |
| `load_pdf(path)` | `.pdf` file | Extracted text (all pages joined) |
| `load_web_page(url)` | `http://...` URL | HTML stripped to plain text |
| `load_document(source)` | Any of the above | `{"text": ..., "source": ..., "type": ...}` |

`load_document()` auto-detects the type by checking if the source starts with `http` or ends with `.pdf`.

**Key learning:** All downstream code receives the same format (plain text string) regardless of the original document type. This is the "normalize early" pattern.

---

## 4. `rag/chunker.py` — Splitting Text

Three strategies, from simple to smart:

### `chunk_fixed_size(text, chunk_size=500)`
Slices every N characters. Simple but can cut words in half.

### `chunk_with_overlap(text, chunk_size=500, overlap=100)`
Sliding window with shared text between chunks. The step size is `chunk_size - overlap`.

### `chunk_sentences(text, chunk_size=500, overlap=100)`
Splits on sentence boundaries first (`re.split(r"(?<=[.!?])\s+")`), then groups sentences into chunks. Preserves complete thoughts.

### `create_chunks(text, source, ...)`
The main entry point. Calls the chosen strategy and adds metadata to each chunk:
```python
{
    "text": "The actual chunk content...",
    "source": "sample.md",
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 487
}
```

---

## 5. `rag/embedder.py` — Text → Vectors

Makes raw HTTP requests to the OpenRouter embeddings endpoint.

### Key design decisions:
- **No `openai` SDK** — we use `requests` directly so you see the raw HTTP request
- **Batching** — sends up to 100 chunks per API call (faster than one-by-one)
- **Retry with exponential backoff** — if rate-limited (HTTP 429), waits 5s → 10s → 20s

### `embed_chunks(chunks)` flow:
```
Input:  [{"text": "...", "source": "...", ...}, ...]
        ↓
API:    POST /v1/embeddings {"model": "text-embedding-v3", "input": [...]}
        ↓
Output: [{"text": "...", "source": "...", "embedding": [0.02, -0.15, ...], ...}, ...]
```

Each chunk now has an `"embedding"` field — a list of floats (default: 3072 dimensions).

---

## 6. `rag/vector_store.py` — Store & Search

### `cosine_similarity(vec_a, vec_b)`
Hand-written math — no numpy:
```python
dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
magnitude_a = math.sqrt(sum(a * a for a in vec_a))
magnitude_b = math.sqrt(sum(b * b for b in vec_b))
return dot_product / (magnitude_a * magnitude_b)
```

### `VectorStore` class
- **`add(text, embedding, metadata)`** — append to an in-memory list
- **`search(query_vector, top_k, threshold)`** — brute-force: compute cosine similarity against every entry, sort, return top-k
- **`save(path)` / `load(path)`** — persist as JSON (human-readable!)

### Performance note
Brute-force search is O(n × d) per query. Fine for thousands of chunks. For millions, you'd need an index structure (FAISS, HNSW). But for learning, brute-force makes the algorithm crystal clear.

---

## 7. `rag/bm25.py` — Keyword Search from Scratch

Implements the BM25 ranking algorithm (same family as Elasticsearch/Lucene):

- **`tokenize(text)`** — lowercase + split + stopword removal
- **`BM25.index(documents)`** — builds term frequency and IDF tables
- **`BM25.search(query, top_k)`** — scores every document against the query
- **`reciprocal_rank_fusion(vector, bm25, k=60)`** — merges rankings from two search methods

---

## 8. `rag/retriever.py` — Finding Context

A multi-mode retrieval orchestrator:

```python
def retrieve(self, query):
    if self.search_mode == "vector":
        query_vector = embed_query(query)
        results = self.vector_store.search(query_vector, top_k, threshold)
    elif self.search_mode == "keyword":
        results = self.bm25.search(query, top_k)
    elif self.search_mode == "hybrid":
        vec_results = vector search
        kw_results = bm25 search
        results = reciprocal_rank_fusion(vec_results, kw_results)

    if self.use_reranker:
        results = self._maybe_rerank(query, results)  # optional 2nd stage

    return results
```

**Critical invariant:** The query MUST be embedded with the same model used for documents. Both vectors must be in the same "space" for cosine similarity to be meaningful.

When `use_reranker=True`, the retriever fetches `top_k * 4` candidates, then the reranker scores each one with an LLM and keeps the top `top_k`.

---

## 9. `rag/reranker.py` — The Second, Smarter Pass

```
Stage 1 (fast): Retrieve top-20 candidates
Stage 2 (precise): LLM scores each candidate 0-10 → keep top-5
```

Each chunk is scored independently by asking the LLM: "On a scale of 0 to 10, how relevant is this passage to the question?" The final score is clamped to [0, 10] and stored as `rerank_score` on the result.

---

## 10. `rag/generator.py` — Getting Answers

### Prompt Construction
```python
messages = [
    {"role": "system", "content": "Answer ONLY from the provided context..."},
    {"role": "user", "content": f"Context:\n{chunks}\n\nQuestion: {query}"}
]
```

The system prompt tells the LLM to:
1. Only use the provided context (don't hallucinate)
2. Cite sources
3. Say "I don't know" if the context doesn't help

### Streaming
With `"stream": true`, the API sends tokens one at a time via Server-Sent Events. We print each token as it arrives for a ChatGPT-like typing effect.

### Temperature
Set to 0.3 (low) for factual, deterministic answers. Higher values (0.7-1.0) would make the output more creative but less reliable for RAG.

### `build_no_answer_response()` — Standardized Miss Handling

This function is the single source of truth for all retrieval-miss responses:

```python
def build_no_answer_response(status, user_guidance, retrieval_quality):
    if status == "no_docs":
        answer = "No documents indexed."
    elif status == "low_confidence":
        answer = "I couldn't find high-confidence evidence..."
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
```

Every part of the system (standard pipeline, agentic pipeline, CLI) uses this same function, ensuring consistent miss-response formatting.

### `generate_with_sources()` — Answer + Attribution

```python
def generate_with_sources(query, context_chunks, stream=True):
    answer = generate(query, context_chunks, stream=stream)
    sources = list(set(chunk["metadata"]["source"] for chunk in context_chunks))
    return {"answer": answer, "sources": sorted(sources), "chunks_used": context_chunks}
```

---

## 11. `rag/agent.py` — The ReAct Agent

Implements the Reason+Act pattern with three action types:

| Action | Purpose | Ends Loop? |
|--------|---------|------------|
| `RequestMoreContext(query, top_k, search_mode)` | Retrieve more chunks | No |
| `ReformulateQuery(new_query)` | Rephrase the search query | No |
| `Answer(response)` | Provide the final answer | **Yes** |

The agent loop:
1. **Thought** — LLM reasons about what it knows so far
2. **Action** — LLM outputs one of the three actions above
3. **Observation** — Pipeline executes the action and feeds results back

The agent uses a dedicated system prompt and accumulates chunks across iterations (deduplicated by text).

---

## 12. `rag/agentic_pipeline.py` — Agentic RAG Orchestration

Wraps the standard RAG pipeline with the ReAct agent loop:

```python
def query(self, question, verbose=False):
    if len(store) == 0:
        return build_no_answer_response(status="no_docs", ...)

    for iteration in range(1, max_iterations + 1):
        thought, action = agent.get_next_action(question)
        observation = self._execute_action(action, agent, question)

        if isinstance(action, Answer):
            break  # Agent decided it has enough context

    if agent.accumulated_chunks:
        result = generate_with_sources(question, agent.accumulated_chunks)
    else:
        result = build_no_answer_response(status="no_relevant", ...)

    result["iterations"] = len(agent.steps)
    result["steps"] = agent.steps
    return result
```

The agentic pipeline also includes the `no_docs` guard upfront, and falls back to `build_no_answer_response()` when no chunks were accumulated across all iterations — providing the same standardized miss response as the standard pipeline.

---

## 13. `rag/evaluator.py` — Measuring Quality

Three metrics, each targeting a different part of the pipeline:

| Metric | What It Measures | Implementation |
|--------|-----------------|----------------|
| **Retrieval Precision** | Did we find the right chunks? | Count chunks from expected source files |
| **Answer Faithfulness** | Is the answer grounded in context? | LLM-as-judge scoring |
| **Answer Relevance** | Does the answer address the question? | LLM-as-judge scoring |

---

## Data Flow Summary

```
                    INGEST                              QUERY
                    ======                              =====

  document.pdf ──→ loader ──→ chunker ──→ embedder     "What is X?" ──→ embedder
                                              │                            │
                                              ▼                            ▼
                                         vector_store ◄──────────── retriever
                                         + bm25_index                   │
                                              │                    ┌─────┴─────┐
                                              │                    │ Quality   │
                                         (saved as                │ Gate:     │
                                          JSON file)               │ no_docs?  │
                                              │                    │ no_rel?   │
                                              │                    │ low_conf? │
                                              │                    └─────┬─────┘
                                              │                          │
                                              │                    ┌─────┴─────┐
                                              │                    │ Build     │
                                              │                    │ Response  │
                                              │                    └─────┬─────┘
                                              │                          ▼
                                              │                     generator
                                              │                          │
                                              ▼                          ▼
                                    (optional: reranker)    "X is defined as
                                              │              ... [source]"
                                              ▼
                                         generator
                                         (standard /
                                          agentic /
                                          miss response)
```

---

## Running the Tests

```bash
# Run all 156+ tests
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_chunker.py -v
python -m pytest tests/test_vector_store.py -v
python -m pytest tests/test_retriever.py -v

# Run the miss-handling tests
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_agentic_pipeline.py -v

# Run evaluation tests
python -m pytest tests/test_evaluator.py -v
```

Tests use mock objects for the embedder (no API key needed), so you can run them freely.
