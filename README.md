# RAG From Scratch

用 Python 从零实现的 RAG 系统：不依赖 LangChain、不依赖向量数据库 SDK、不依赖 OpenAI SDK；直接用 HTTP 调 OpenRouter（OpenAI-compatible）接口，便于理解 RAG 的每个环节，并提供 CLI + Web UI 便于测试与迭代。

## 能做什么

- 文档摄取：本地 `.pdf` / `.md` / `.txt`，以及网页 URL
- 切分：重叠 chunk（默认 500 字符，overlap 100）
- 向量化：通过 OpenRouter embeddings（默认 Qwen embedding 模型）
- 检索：向量检索 / BM25 / Hybrid（RRF 融合）+ 可选 LLM Rerank（二阶段检索）
- 生成：流式回答 + 来源归因（sources）
- Agentic RAG：ReAct 循环，多轮“检索-思考-再检索”后再回答
- 评测：Precision / Faithfulness / Relevance + keyword hit rate
- Web UI：FastAPI + 纯静态前端（默认挂载在 `/`，API 在 `/api`）

## Architecture

```
Documents (PDF/MD/Web)
        │
        ▼
  ┌─────────────┐
  │  1. Loader   │  reads raw content from files/URLs
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  2. Chunker  │  splits text into overlapping chunks
  └──────┬──────┘
         ▼
  ┌─────────────┐
       │ 3. Embedder  │  converts chunks → vectors via Qwen API
  └──────┬──────┘
         ▼
  ┌──────────────────┐
  │ 4. Vector Store   │  stores vectors + metadata (JSON file)
  │   (from scratch)  │  with cosine similarity search
  └──────┬───────────┘
         ▼
  ┌──────────────┐   ┌──────────────┐
  │ Vector Search │ + │ BM25 Keyword │  ← Hybrid Search
  └──────┬───────┘   └──────┬───────┘
         └────────┬─────────┘
                  ▼
         ┌────────────────┐
         │  Rank Fusion    │  ← Reciprocal Rank Fusion (RRF)
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │   Reranker      │  ← LLM-as-judge (0-10 scoring)
         └────────┬───────┘
                  ▼
         ┌────────────────┐
         │   Generator     │  ← Qwen3-8B streaming answer
         └────────┬───────┘
                  ▼
               Answer
                  │
                  ▼
         ┌────────────────┐
         │   Evaluator     │  ← 3 metrics (precision, faithfulness, relevance)
         └────────────────┘
```
### 🤖 Agentic RAG (NEW)

For complex questions, the system can use a **ReAct agent** that iteratively
improves retrieval before answering:

```
User Question
     │
     ▼
┌──────────────────────────────────┐
│  ReAct Agent Loop (max 5 iter)   │
│                                  │
│  Thought → Action → Observation  │
│    ↑                    │        │
│    └────────────────────┘        │
│                                  │
│  Actions:                        │
│   • RequestMoreContext            │
│   • ReformulateQuery             │
│   • Answer (stops loop)          │
└──────────────┬───────────────────┘
               ▼
         Final Answer
```
## 快速开始


```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 OpenRouter
copy .env.example .env
# 编辑 .env：设置 OPENROUTER_API_KEY=...

# 3) 放入你的文档（可选）
mkdir sample_data
# 将 .pdf/.md/.txt 放入 sample_data/（该目录已被 gitignore）

# 4) 直接提问（首次运行如 index 为空会自动扫描 sample_data/ 并 ingest）
python cli.py query "这份文档主要讲什么？"
```

## 配置说明（.env）

`.env.example` 已给出所有字段，最少需要：

- `OPENROUTER_API_KEY`
- `CHAT_API_URL` / `EMBEDDING_API_URL`
- `MODEL_CHAT` / `MODEL_JUDGE` / `MODEL_EMBEDDING`

## CLI 用法

### 1) 摄取文档

```bash
# 摄取本地文件
python cli.py ingest --source "C:\path\to\doc.pdf"

# 摄取网页
python cli.py ingest --source https://en.wikipedia.org/wiki/Machine_learning

# 自定义 chunk 参数
python cli.py ingest --source notes.md --chunk-size 300 --overlap 50
```

### 2) 提问（RAG）

```bash
# 默认：vector 检索
python cli.py query "退货政策是什么？"

# 展示检索到的 chunks
python cli.py query "退货政策是什么？" --verbose

# 控制检索参数
python cli.py query "退货政策是什么？" --top-k 3 --threshold 0.4

# 检索模式：vector / keyword / hybrid
python cli.py query "HEART method customer service" --search-mode keyword
python cli.py query "退货政策是什么？" --search-mode hybrid

# hybrid + rerank（更准但更慢）
python cli.py query "退货政策是什么？" --search-mode hybrid --rerank
```

### 3) Agentic 提问（ReAct）

```bash
python cli.py agentic-query "对比产品 A 和 B 的差异" --verbose
python cli.py agentic-query "复杂问题" --max-iterations 3 --search-mode hybrid --rerank
```

### 4) 索引管理

```bash
# 列出已索引的文档来源
python cli.py list

# 彻底重建索引（删除本地 JSON 索引文件）
del rag_index.json
```

## Web UI（FastAPI + 前端静态页）

```bash
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn backend.app:app --reload --port 8000
```

- 打开：http://127.0.0.1:8000/
- API：`/api/*`，交互式文档（FastAPI 默认）：http://127.0.0.1:8000/docs

## 评测（Evaluation）

```bash
# 默认参数运行
python cli.py eval

# 对比检索模式
python cli.py eval --search-mode vector
python cli.py eval --search-mode hybrid
python cli.py eval --search-mode hybrid --rerank

# 输出更详细的逐题信息
python cli.py eval --verbose
```

结果会保存到 `eval/results/`（自动创建目录）。

## 项目结构（当前版本）

```text
rag-from-scratch/
  backend/
    api.py
    app.py
    schemas.py
    session_manager.py
  docs/
    agentic_rag.md
    concepts.md
    how_it_works.md
  eval/
    test_questions.json
  frontend/
    index.html
    script.js
  rag/
    agent.py
    agentic_pipeline.py
    bm25.py
    chunker.py
    embedder.py
    evaluator.py
    generator.py
    loader.py
    pipeline.py
    reranker.py
    retriever.py
    vector_store.py
  tests/
    test_agent.py
    test_agentic_pipeline.py
    test_bm25.py
    test_chunker.py
    test_evaluator.py
    test_pipeline.py
    test_reranker.py
    test_retriever.py
    test_vector_store.py
  .env.example
  cli.py
  requirements.txt
  rag_index.json (运行后生成，默认被 gitignore)
  sample_data/ (自行创建，默认被 gitignore)
```

## 依赖（requirements.txt）

运行时依赖以 `requirements.txt` 为准：

- `requests`
- `python-dotenv`
- `PyPDF2`
- `beautifulsoup4`
- `fastapi`
- `uvicorn`
- `python-multipart`（Web 上传）
- `pydantic`

## 约束与注意事项

- 需要能访问 OpenRouter；embedding / rerank / eval（LLM-as-judge）都会产生外部 API 调用
- 默认索引文件为 `rag_index.json`（JSON 文件向量库，适合本地小中规模文档与学习/验证用）
- `sample_data/` 仅在索引为空时自动 ingest（支持子目录）；已有索引时不会重复 ingest

## 进一步阅读

- [docs/concepts.md](docs/concepts.md)
- [docs/how_it_works.md](docs/how_it_works.md)
- [docs/agentic_rag.md](docs/agentic_rag.md)

## License

MIT
