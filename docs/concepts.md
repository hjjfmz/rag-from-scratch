# Core Concepts — What You Need to Know

This guide explains the key concepts behind RAG in plain language. No PhD required.

---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. Let's break that down backwards:

- **Generation** — An LLM (like GPT-4) generates a text answer
- **Augmented** — We augment (enhance) the LLM's prompt with extra context
- **Retrieval** — That context comes from retrieving relevant parts of YOUR documents

### The Problem RAG Solves

LLMs are trained on public internet data. They don't know about:
- Your company's internal docs
- A PDF you downloaded yesterday
- Your personal notes

If you ask an LLM about your documents, it either says "I don't know" or **hallucinate** — confidently makes up an answer that sounds right but is wrong.

RAG fixes this by giving the LLM your documents as context, right inside the prompt.

### Without RAG
```
User: "What's our refund policy?"
LLM:  "I don't have information about your company's refund policy."
       (or worse, makes something up)
```

### With RAG
```
User: "What's our refund policy?"
System: [retrieves relevant chunks from your policy document]
LLM:  "According to your policy document, refunds are available within
       30 days of purchase with a valid receipt." ← grounded in real data
```

---

## What Are Embeddings?

An **embedding** is a way to represent text as a list of numbers (a **vector**).

```
"The cat sat on the mat" → [0.02, -0.15, 0.83, 0.41, -0.22, ...]
```

These aren't random numbers. They capture the **meaning** of the text. Texts with similar meanings get similar numbers:

```
"The cat sat on the mat"       → [0.02, -0.15, 0.83, ...]
"A kitten was resting on a rug" → [0.03, -0.14, 0.81, ...]  ← similar!
"Stock prices rose 5% today"   → [0.91, 0.22, -0.45, ...]  ← very different!
```

### Why 3072 Numbers?

The embedding model we use (`text-embedding-3-large`) represents each text as **3072 floating-point numbers**. Why so many? Because language is complex — you need many dimensions to capture all the nuances of meaning.

Think of it like coordinates:
- **2D coordinates** (x, y) can locate a point on a flat map
- **3D coordinates** (x, y, z) can locate a point in space
- **3072D coordinates** can locate a point in "meaning space" — each dimension captures some aspect of the text's meaning

### How Are Embeddings Created?

Embedding models are neural networks trained on massive text datasets. During training, they learn to place similar texts near each other in vector space. We don't train the model — we use a pre-trained one (`text-embedding-3-large`) hosted on Azure AI Foundry via API.

---

## What Is Cosine Similarity?

Once we have vectors, we need a way to measure how similar two vectors are. That's what **cosine similarity** does.

### The Intuition

Imagine two arrows pointing from the origin. Cosine similarity measures the **angle** between them:

```
  ↗ B (similar to A — small angle)
 ↗ A
→ C (different from A — large angle)
```

- **Small angle** → similar direction → **high cosine similarity** (close to 1.0)
- **Large angle** → different direction → **low cosine similarity** (close to 0.0)
- **Opposite direction** → **negative cosine similarity** (close to -1.0)

### The Math

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

Where:
- `A · B` = **dot product** = sum of (a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ)
- `|A|` = **magnitude** = √(a₁² + a₂² + ... + aₙ²)

### A Simple Example

```
A = [1, 0]    (points right)
B = [0.7, 0.7] (points upper-right, at 45°)
C = [0, 1]    (points up, at 90°)

cosine_sim(A, B) = 0.707  (similar-ish)
cosine_sim(A, C) = 0.0    (unrelated)
cosine_sim(A, A) = 1.0    (identical!)
```

### Why Cosine and Not Euclidean Distance?

Cosine similarity measures **direction**, not **magnitude**. A short paragraph about cats and a long essay about cats might have vectors of very different magnitudes, but they point in roughly the same direction. Cosine similarity correctly identifies them as similar, while Euclidean distance might not.

---

## What Is Chunking?

LLMs have a **context window** — a maximum amount of text they can process at once (e.g., ~128K tokens for GPT-4o-mini). But more importantly, stuffing an entire 100-page document into the prompt is:

1. **Expensive** — you pay per token
2. **Noisy** — the LLM has to find the needle in a haystack
3. **Less accurate** — LLMs perform worse with very long contexts

**Chunking** splits a document into small pieces (e.g., 500 characters each). We embed and store each chunk separately. When the user asks a question, we retrieve only the 3-5 most relevant chunks — giving the LLM focused, relevant context.

### Why Overlap?

If we split text at exactly every 500 characters, we might cut a sentence in half:

```
Chunk 1: "...Overfitting happens when the model memori"
Chunk 2: "zes the training data instead of learning..."
```

Neither chunk has the complete thought. **Overlap** means chunk boundaries share some text:

```
Chunk 1: "...Overfitting happens when the model memorizes the training data..."
Chunk 2: "...the model memorizes the training data instead of learning..."
          ↑ shared text ↑
```

Now the full sentence appears in at least one chunk.

---

## The Complete RAG Flow

Here's how all these concepts work together:

### Ingestion (one-time setup)
```
1. LOAD:    Read document → plain text
2. CHUNK:   Split text → small overlapping pieces
3. EMBED:   Convert each chunk → 3072-dimension vector
4. STORE:   Save vectors + text + metadata to disk
```

### Query (every time a user asks)
```
1. EMBED:    Convert user's question → vector (same model!)
2. SEARCH:   Find chunks with most similar vectors (cosine similarity)
3. BUILD:    Create prompt = system message + retrieved chunks + question
4. GENERATE: Send prompt to LLM → get answer grounded in your documents
```

### Why This Works

The key insight: **similar meanings produce similar vectors**. So when a user asks "What is overfitting?", the question's vector will be similar to chunks that discuss overfitting — even if they use different words like "memorizing training data" or "failing to generalize."

This is fundamentally better than keyword search, which would miss documents that discuss the concept without using the exact word "overfitting."

---

## Evaluation — How Do You Know Your RAG Is Good?

Building a RAG system is only half the battle. You need to **measure** how well it works — otherwise, every change is a guess. Did switching chunk_size from 500 to 300 help? Did adding overlap improve retrieval? Evaluation tells you.

### The Three Metrics

We measure three things, each testing a different part of the pipeline:

#### 1. Retrieval Precision

**Question it answers:** *Did we find the RIGHT chunks?*

```
Query: "What is the return policy?"
Retrieved 5 chunks:
  ✅ refund-policy.md (chunk about 30-day returns)
  ✅ refund-policy.md (chunk about opened products)
  ❌ shipping-policy.md (not about returns)
  ❌ loyalty-program.md (not about returns)
  ❌ faq.md (mentions returns briefly)

Precision = 2 relevant / 5 total = 0.40
```

A low precision means you're retrieving noise. Fix: tune chunk_size, overlap, or similarity threshold.

#### 2. Answer Faithfulness

**Question it answers:** *Is the answer grounded in the context, or did the LLM hallucinate?*

```
Context says: "GlowBeauty offers 30-day returns."

❌ Bad (unfaithful): "GlowBeauty offers 60-day returns with full refund."
   → Made up "60 days" — not in context! Faithfulness = LOW

✅ Good (faithful): "You can return items within 30 days."
   → Every claim is supported by context. Faithfulness = HIGH
```

A low faithfulness means the LLM is adding information from its training data. Fix: strengthen the system prompt ("answer ONLY from context").

#### 3. Answer Relevance

**Question it answers:** *Does the answer actually address what was asked?*

```
Question: "What is the return policy?"

❌ Bad (irrelevant): "GlowBeauty was founded in 2019 and has 50 stores."
   → Factually correct but doesn't answer the question! Relevance = LOW

✅ Good (relevant): "You can return items within 30 days."
   → Directly answers what was asked. Relevance = HIGH
```

### Why You Need All Three

| Scenario | Precision | Faithfulness | Relevance | Diagnosis |
|----------|-----------|-------------|-----------|-----------|
| Everything works | High | High | High | 🎉 Ship it! |
| Wrong chunks retrieved | Low | High | Low | Fix search/chunking |
| Right chunks, LLM hallucinates | High | Low | Medium | Fix system prompt |
| Right chunks, wrong focus | High | High | Low | Fix prompt template |

### LLM-as-Judge

For faithfulness and relevance, we use the same LLM as a "judge" — we ask it to score the answer on a 0.0–1.0 scale. This is called **LLM-as-Judge** and is standard practice in the industry (used by RAGAS, DeepEval, and other evaluation frameworks). It's not perfect, but it's practical and fast.

### Running Evaluation

```bash
python cli.py eval              # Run with defaults
python cli.py eval --verbose    # See per-question details
```

Results are saved as JSON in `eval/results/` with timestamps, so you can compare runs over time.

---

## Hybrid Search — Best of Both Worlds

### The Problem with Vector-Only Search

Vector search (cosine similarity on embeddings) is great at finding **semantically similar** text. But it can miss **exact keyword matches**:

```
Query: "Silk Skin Foundation price"
Vector search might rank a chunk about "makeup products" higher than
the specific chunk containing "Silk Skin Foundation — $44" because
the embedding captures the general "beauty products" concept.
```

### The Problem with Keyword-Only Search

Keyword search (BM25) finds exact word matches but misses **synonyms and paraphrases**:

```
Query: "How do animals learn from experience?"
BM25 finds NOTHING because the relevant chunk says:
"In reinforcement learning, an agent receives rewards or penalties."
No shared keywords! But semantically, it's about learning from experience.
```

### BM25 — The Algorithm Behind Google

BM25 (Best Matching 25) is what powers Elasticsearch, Apache Lucene, and most search engines. It's smarter than simple "does the word appear?" because it considers:

| Factor | What it does | Example |
|--------|-------------|---------|
| **Term Frequency (TF)** | More mentions = higher score (with diminishing returns) | "return" 3 times > 1 time |
| **Inverse Document Frequency (IDF)** | Rare words matter more | "overfitting" (rare) > "the" (common) |
| **Length Normalization** | Short focused chunks scored fairly | 50-word chunk ≈ 500-word chunk |

### Reciprocal Rank Fusion (RRF) — Merging Two Lists

You can't just add vector scores (0.0–1.0) and BM25 scores (0–25+) — they're on different scales. **RRF uses ranks instead of scores:**

```
Vector results: [chunk_A (rank 1), chunk_C (rank 2), chunk_B (rank 3)]
BM25 results:   [chunk_B (rank 1), chunk_A (rank 2), chunk_D (rank 3)]

RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)    where k=60

chunk_A: 1/61 + 1/62 = 0.0326  ← in BOTH lists → boosted!
chunk_B: 1/63 + 1/61 = 0.0322  ← in BOTH lists → boosted!
chunk_C: 1/62 + 0     = 0.0161  ← only in vector
chunk_D: 0    + 1/63  = 0.0159  ← only in BM25
```

A chunk ranking well in **both** methods gets the highest combined score. This is the same fusion algorithm used by Azure AI Search and Elasticsearch.

### Using Hybrid Search

```bash
python cli.py query "return policy" --search-mode vector   # semantic only
python cli.py query "return policy" --search-mode keyword  # BM25 only
python cli.py query "return policy" --search-mode hybrid   # both + RRF
```

---

## Reranking — The Second, Smarter Pass

### The Problem

Neither vector search nor BM25 actually **reads** the query and chunk together. They score independently:
- Vector search compares embeddings (query vector vs chunk vector — separately computed)
- BM25 matches keywords (no understanding of what the question is really asking)

A chunk about "beauty products" might score high for "What is the return policy?" because they share the beauty domain — even though the chunk says nothing about returns.

### How Reranking Works

Reranking adds a **second stage** that reads each (query, chunk) pair together:

```
Stage 1 (fast, cheap):
  "What is the return policy?" → retrieve top-20 candidates
  Cost: 1 embedding API call

Stage 2 (slow, precise):
  For each of the 20 candidates, ask the LLM:
    "Score 0-10: how relevant is this passage to this question?"
  Keep the top 5 highest-scoring chunks.
  Cost: 20 LLM API calls
```

### Why Two Stages?

Reranking every chunk would be too expensive:

| Approach | API calls | What it does |
|----------|-----------|-------------|
| No rerank | 1 | Embed query, return top-5 |
| Rerank all 305 chunks | 306 | Embed query + score every chunk |
| **Two-stage** (our approach) | 21 | Embed query + score 20 candidates |

The two-stage approach gets most of the benefit at a fraction of the cost.

### LLM-as-Reranker

We use the same gpt-5-mini model as our reranker — no extra model needed. In production, companies use specialized cross-encoder models (Cohere Rerank, fine-tuned BERT), but LLM-as-reranker works well for learning.

### Using Reranking

```bash
python cli.py query "return policy" --rerank                        # vector + rerank
python cli.py query "return policy" --search-mode hybrid --rerank   # hybrid + rerank
```

---

## Retrieval Miss Governance — When RAG Can't Find Relevant Content

### The Problem

Every RAG system faces a fundamental question: *what should happen when the knowledge base doesn't contain relevant information for a user's query?*

Without explicit handling, the LLM will do one of two problematic things:
1. **Hallucinate** — make up an answer from its training data, pretending it came from your documents
2. **Give a vague non-answer** — "I don't know" with no guidance on how to fix it

Neither outcome is useful. The system needs to **detect** the miss, **classify** the reason, and **guide** the user toward a solution.

### Why Simple Thresholding Isn't Enough

A naive approach is "if similarity score < threshold, say you don't know." But this misses important nuances:

- **Empty knowledge base** is a setup problem (run `ingest` first), not a retrieval problem
- **Zero results above threshold** means the question is about a topic not in your documents
- **Weak but non-zero scores** (e.g., 0.35–0.45) is ambiguous — maybe the answer is there but the embedding didn't capture it well, or maybe it's genuinely off-topic

Each scenario needs a different response.

### The Three Miss Categories

This project classifies retrieval failure into three distinct categories:

#### 1. `no_docs` — Knowledge Base Is Empty

```
Status: "no_docs"
Answer: "No documents indexed."
Guidance: "Use 'ingest' to add documents, then ask your question again."
```

**When it happens:** The vector store has zero entries — no documents have been ingested yet. This is a setup/precondition issue.

**What to do:** Run `python cli.py ingest --source <your_file>` to index documents, or place files in `sample_data/` for auto-ingestion.

**Where it's checked:**
```python
if len(self.store) == 0:
    return build_no_answer_response(status="no_docs", user_guidance="...")
```

#### 2. `no_relevant` — No Chunks Pass the Threshold

```
Status: "no_relevant"
Answer: "No relevant documents found for your question."
Guidance: "Try rephrasing your question, switch to --search-mode hybrid, or lower --threshold."
```

**When it happens:** The retriever found zero chunks above the similarity threshold. The question topic likely doesn't exist in the knowledge base, or the threshold is too strict.

**What to do:**
- Lower `--threshold` (e.g., from 0.3 to 0.1)
- Switch to `--search-mode hybrid` to catch keyword matches that embeddings missed
- Rephrase the question with different vocabulary
- Consider whether the answer actually exists in your documents

**Where it's checked:**
```python
results = retriever.retrieve(question)
if not results:
    return build_no_answer_response(status="no_relevant", user_guidance="...")
```

#### 3. `low_confidence` — Chunks Found but Quality Is Weak

```
Status: "low_confidence"
Answer: "I couldn't find high-confidence evidence in the documents to answer this question."
Guidance: "Try --search-mode hybrid, increase --top-k, or rephrase with specific keywords."
Quality: {"top_score": 0.38, "avg_top3_score": 0.35, "num_results": 5,
           "search_mode": "vector"}
```

**When it happens:** Some chunks were retrieved above the threshold, but their scores suggest they're only marginally relevant. The system errs on the side of caution — better to admit uncertainty than to generate a low-quality answer from weak context.

**How it's determined:**

The check differs by search mode because BM25 scores and embedding similarity scores are on different scales:

| Search Mode | Low-Confidence Condition |
|-------------|------------------------|
| `vector` | `top_score < max(threshold + 0.15, 0.45)` **or** `avg_top3_score < max(threshold + 0.1, 0.4)` |
| `keyword` / `hybrid` | `len(results) < min(2, top_k)` |

Why the two strategies?
- **Vector scores** are cosine similarities on a 0–1 scale. The heuristic adds a safety margin above the user's threshold (`+0.15` for top score, `+0.1` for average top-3), with absolute floors of 0.45 and 0.40.
- **BM25/hybrid scores** are RRF-fused ranks, not directly comparable to cosine scores. So we use a simpler check: if we got fewer results than expected (e.g., only 1 result when `top_k=5`), the retrieval is likely weak.

**What to do:**
- Switch to `--search-mode hybrid` to utilize both semantic and keyword matching
- Increase `--top-k` to look at more candidates
- Add `--rerank` to get a second, smarter scoring pass
- Rephrase the question with specific domain keywords from your documents

### The Standardized Response Structure

All three miss types return a consistent dict structure, so upstream code (CLI, evaluation, API) can handle them uniformly:

```python
{
    "answer": "..." ,            # Human-readable answer text
    "sources": [],               # Empty — no sources were used
    "chunks_used": [],           # Empty — no chunks were used
    "status": "no_docs",         # One of: "ok" | "no_docs" | "no_relevant" | "low_confidence"
    "user_guidance": "...",      # Actionable suggestions for the user
    "retrieval_quality": {       # Diagnostic metrics (empty for no_docs/no_relevant)
        "top_score": 0.0,
        "avg_top3_score": 0.0,
        "num_results": 0,
        "search_mode": "vector"
    }
}
```

This is built by `build_no_answer_response()` in `rag/generator.py`, providing a single source of truth for all miss-response formatting.

### CLI Feedback

When running queries via the CLI, retrieval misses produce colored hints:

```bash
$ python cli.py query "quantum field theory" --threshold 0.5
ℹ️  Retrieval status: no_relevant
💡 Suggestion: Try rephrasing your question, switch to --search-mode hybrid,
   or lower --threshold.
```

This gives users immediate, actionable feedback without having to inspect code or logs.

### Don't Over-Adjust Thresholds

A common reaction to low-confidence results is to lower the threshold aggressively. Be careful:

| Threshold | Effect |
|-----------|--------|
| Too high (0.5+) | Rejects genuinely relevant chunks → many `no_relevant` results |
| Moderate (0.3) | Good balance — default in this project |
| Too low (0.1) | Accepts noise → LLM gets irrelevant context → lower faithfulness |

The low-confidence check helps find the sweet spot: set a reasonable threshold (0.3), and let the system tell you when it's unsure rather than forcing it to always return *something*.

### Testing Miss Behavior

The test suite includes dedicated tests for each miss scenario in `tests/test_pipeline.py`:

```python
def test_empty_store_returns_no_docs():
    pipeline = RAGPipeline()
    result = pipeline.query("any question")
    assert result["status"] == "no_docs"

def test_no_relevant_returns_status():
    # ...setup with irrelevant content...
    result = pipeline.query("unrelated topic")
    assert result["status"] == "no_relevant"

def test_low_confidence_returns_status():
    # ...setup with low-score chunks...
    result = pipeline.query("any question")
    assert result["status"] == "low_confidence"
```

These tests ensure the miss-detection logic is reliable and won't silently break during refactoring.
