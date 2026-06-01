# 🤖 Agentic RAG — ReAct Pattern for Iterative Retrieval

## What is Agentic RAG?

Standard RAG follows a fixed pipeline: retrieve → generate. It works well for straightforward questions, but struggles when:
- The initial search query doesn't find the right chunks
- The question is ambiguous and needs multiple angles of retrieval  
- The right answer requires combining information from different searches

**Agentic RAG** wraps the retrieval step in an intelligent agent loop. Instead of retrieving once and hoping for the best, the agent **reasons about the results** and decides what to do next.

## The ReAct Pattern

ReAct (Yao et al., 2022) stands for **Reason + Act**. The agent alternates between:

| Step | What happens | Example |
|------|-------------|---------|
| **Thought** | Agent reasons about the current state | "I need info about ingredient X side effects" |
| **Action** | Agent takes one action | `RequestMoreContext("ingredient X side effects")` |
| **Observation** | System provides feedback | "Retrieved 3 chunks: [chunk previews...]" |

This cycle repeats until the agent decides it has enough context to answer, or hits the maximum iteration limit (default: 5).

## How It Works — Step by Step

```
User: "What are the side effects of ingredient X?"
```

### Iteration 1
```
Thought: I need to find information about ingredient X and its side effects.
Action: RequestMoreContext("ingredient X side effects", 5, "vector")
Observation: Retrieved 3 chunks:
  [1] (score: 0.82, source: products.md) "Ingredient X is used in our moisturizer line..."
  [2] (score: 0.71, source: products.md) "Benefits of ingredient X include hydration..."
  [3] (score: 0.65, source: faq.md) "Common questions about ingredient Y..."
```

### Iteration 2
```
Thought: I found info about ingredient X's benefits but not its side effects.
         Chunk 3 is about ingredient Y, not X. Let me reformulate to focus
         specifically on safety and adverse reactions.
Action: ReformulateQuery("ingredient X adverse reactions safety warnings")
Observation: Retrieved 3 chunks:
  [1] (score: 0.88, source: safety.md) "Ingredient X may cause mild irritation in..."
  [2] (score: 0.79, source: safety.md) "Contraindications: Do not use ingredient X if..."
  [3] (score: 0.72, source: products.md) "All products undergo safety testing..."
```

### Iteration 3
```
Thought: Now I have good context about ingredient X's side effects — irritation
         risks and contraindications from safety.md. I have enough to answer.
Action: Answer("Based on the documents, ingredient X may cause...")
```

**Result**: 3 iterations, 6 unique chunks, accurate answer with proper sources.

## Available Actions

### 1. `RequestMoreContext(query, top_k, search_mode)`
Retrieve document chunks from the knowledge base.
- **query**: What to search for
- **top_k**: How many chunks to retrieve (default: 5)
- **search_mode**: `"vector"`, `"keyword"`, or `"hybrid"` (default: `"vector"`)

### 2. `ReformulateQuery(new_query)`
Rephrase the search query to find different or better results. Use when previous retrieval didn't return relevant chunks.

### 3. `Answer(response)`
Provide the final answer. The agent chooses this when it has enough context. **This ends the loop.**

## How Does the Agent Decide "Good Enough"?

The LLM itself judges retrieval quality during the **Thought** step. After seeing the retrieved chunks, it reasons:

- ✅ "These chunks directly address the question" → **Answer**
- ❌ "Results are about the wrong topic" → **ReformulateQuery**
- ❌ "Only partial information, need more" → **RequestMoreContext** with different parameters

No hardcoded thresholds or scoring heuristics. The model's own comprehension drives the loop.

## When to Use Agentic vs. Standard RAG

| | Standard RAG | Agentic RAG |
|---|---|---|
| **Speed** | Fast (1 retrieval) | Slower (1-5 retrievals) |
| **Simple questions** | ✅ Great | ❌ Overkill |
| **Complex questions** | ⚠️ Hit or miss | ✅ Self-correcting |
| **Ambiguous queries** | ⚠️ Depends on phrasing | ✅ Agent reformulates |
| **Cost** | 1 LLM call | 2-6 LLM calls |

**Rule of thumb**: Use standard RAG for simple factual lookups. Use agentic RAG when question quality varies or when you need higher reliability.

## Architecture

```
User Question
     │
     ▼
┌──────────────────────────────────────────┐
│  AgenticRAGPipeline                      │
│                                          │
│  ┌────────────────────────────┐          │
│  │ ReAct Agent Loop           │          │
│  │ (max 5 iterations)         │          │
│  │                            │          │
│  │ 1. Thought (LLM reasons)  │◄──┐      │
│  │         ▼                  │   │      │
│  │ 2. Action (LLM decides)   │   │      │
│  │    • ReformulateQuery      │   │      │
│  │    • RequestMoreContext    │   │      │
│  │    • Answer ──────────────────────┐   │
│  │         ▼                  │   │  │   │
│  │ 3. Observation (execute)  │───┘  │   │
│  └────────────────────────────┘      │   │
│                                      │   │
│  ┌───────────────────────────────┐   │   │
│  │ Existing RAG Modules          │   │   │
│  │ • Retriever (vector/BM25)     │◄──┘   │
│  │ • Generator (streaming)       │       │
│  │ • Reranker (optional)         │       │
│  └───────────────────────────────┘       │
└──────────────────────────────────────────┘
```

## Key Design Decisions

### Context Accumulation
Each retrieval adds chunks to a growing pool (deduplicated by text). When the agent finally answers, the generator sees ALL unique chunks retrieved across all iterations.

### Search Mode Flexibility
The agent can switch between `vector`, `keyword`, and `hybrid` search modes across iterations. For example, start with semantic search, then try keyword search for exact terms.

### Graceful Degradation
If the LLM produces unparseable output, the pipeline falls back to answering with whatever context has been accumulated. No crashes.

### Max Iterations Safety Net
The iteration limit (default: 5) prevents infinite loops. In practice, most questions answer in 1-2 iterations.

### Retrieval Miss Handling

The agentic pipeline integrates with the same retrieval miss governance system used by the standard pipeline:

| Scenario | How the Agentic Pipeline Handles It |
|----------|-------------------------------------|
| **Empty knowledge base** (`no_docs`) | Immediate return before the agent loop starts — suggests running `ingest` first |
| **No chunks across all iterations** (`no_relevant`) | After the agent loop exhausts all attempts (either by hitting max iterations or choosing Answer), if zero chunks were accumulated, returns a standardized `no_relevant` response with guidance |
| **Parse error during agent loop** | Falls back to answering with whatever context was accumulated up to that point — no crash |

The agentic pipeline uses `build_no_answer_response()` from `rag/generator.py` to produce the same standardized response structure as the standard pipeline:

```python
# Empty knowledge base guard (before agent loop starts)
if len(self.rag_pipeline.store) == 0:
    result = build_no_answer_response(
        status="no_docs",
        user_guidance="Use 'ingest' to add documents, then ask your question again.",
    )
    result["iterations"] = 0
    result["steps"] = []
    return result

# After agent loop — no chunks accumulated at all
if not agent.accumulated_chunks:
    result = build_no_answer_response(
        status="no_relevant",
        user_guidance="Try rephrasing your question with concrete keywords, "
                       "or run with --search-mode hybrid.",
    )
```

Note that the agentic pipeline does **not** apply the same `low_confidence` heuristic as the standard pipeline. The agent's own reasoning (via the LLM) serves as its quality gate — if results are weak, the agent will naturally reformulate the query or search for more context. The agent's judgment replaces the need for a hardcoded score threshold.

## Further Reading
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — The original paper
- `rag/agent.py` — The agent implementation with detailed `📚 LEARN:` comments
- `rag/agentic_pipeline.py` — The orchestration loop
- `docs/concepts.md` — Core RAG concepts (embeddings, chunking, etc.)
