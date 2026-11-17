# Local Knowledge

本地个人知识库与跨文件问答系统。导入自己的文档，构建带原文证据的实体关系图，结合双层图检索、向量和关键词检索回答问题，并引用来源。

CLI 与网页共用同一套服务栈（GraphService）。存储可插拔：

- **local 模式（默认）**：SQLite 适配器 + 进程内精确向量扫描，无需任何数据库服务，适合开发与演示。
- **services 模式**：PostgreSQL（账本与正文 + BM25 词项统计 + 会话）+ Qdrant server（有界向量召回）+ 可选 Redis（答案缓存），面向单机服务化部署叙事。

PG 是正文和图关系的正本，Qdrant 保存向量副本；召回的编号必须回 PG 读取当前记录，过期编号会被丢弃。两者不属于同一个事务：向量发布失败会返回 `vector_status=pending` 和警告，恢复服务后重新导入同一文件会使用已存向量补发，不重新调用模型。该恢复依赖重新导入，不是自动后台修复。旧向量仍可能挤占召回名额，不能将其描述为完全无影响。

余弦、BM25、RRF 和 local/global 融合由自己的代码实现。六模式对拍测试验证了固定小语料上的一致性，不证明候选截断或近似向量召回后仍与全量检索完全等价。图检索仍扫描实体/关系目录并扩展邻接，local 存储模式仍做全量向量扫描；目前没有所有查询工作的固定规模上限。

## 安装与配置

在项目根目录使用 Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-dev.txt
Copy-Item .env.example .env
```

`.env` 关键开关（完整见 `.env.example`）：

- `RAG_STORAGE=local`（默认）或 `services`。services 模式必须配齐 `RAG_PG_DSN`、`RAG_QDRANT_URL`，缺配置会明确报错，绝不静默回退 SQLite。
- `RAG_OFFLINE=1`：离线启发式模型（英文规则抽取 + 词法哈希向量 + 原文摘录），无需任何 API Key。
- 真实模型：填 `CHAT_API_URL`、`EMBEDDING_API_URL`、`MODEL_CHAT`、`MODEL_EMBEDDING` 和密钥；独立密钥 `CHAT_API_KEY` / `EMBEDDING_API_KEY` 优先，否则依次使用 `LLM_API_KEY`、`OPENAI_API_KEY`、`OPENROUTER_API_KEY`。不要提交 `.env`。

### 一行启动网页端

```powershell
.venv\Scripts\python.exe run.py
# 浏览器打开 http://127.0.0.1:8000 —— 上传、问答、引用、会话全部在页面完成
```

### 三服务模式（可选，面试演示用）

```powershell
docker compose up -d          # 启动 PostgreSQL + Qdrant + Redis
# .env 中设置 RAG_STORAGE=services 及三个服务地址，然后：
.venv\Scripts\python.exe run.py
```

没有 Docker 时也可自行安装三个服务（PostgreSQL 安装包 / Qdrant 单文件 / Redis），只要地址可达即可；不装则一直用 local 模式，功能完整。

### 端到端冒烟（对当前配置的模式）

```powershell
.venv\Scripts\python.exe scripts\smoke.py
```

## CLI 使用

以下路径替换为自己的文件和目录：

```powershell
.venv\Scripts\python.exe cli.py ingest --source 'D:\knowledge\工作笔记.md'
.venv\Scripts\python.exe cli.py sync --source 'D:\knowledge'
.venv\Scripts\python.exe cli.py query '结合项目说明与会议纪要，目前方案有哪些约束？' --verbose
.venv\Scripts\python.exe cli.py search '方案依赖哪些组件？' --search-mode local
.venv\Scripts\python.exe cli.py search '各文档如何描述数据一致性？' --search-mode global
.venv\Scripts\python.exe cli.py agentic-query '对比各文件中的存储方案和依赖关系' --max-iterations 3
.venv\Scripts\python.exe cli.py list
.venv\Scripts\python.exe cli.py graph --output .runtime/graph.json
.venv\Scripts\python.exe cli.py delete --source 'D:\knowledge\工作笔记.md'
```

支持 UTF-8 TXT/MD 和有文本层的 PDF。扫描件需要先做 OCR。输入上限为每文件 16 MiB、提取文本 50 万字符、1024 个片段。只接受本地文件（不采集网页）。

- `ingest`：构建完成后原子替换同一来源；内容与处理配置未变则跳过，`--force` 强制重建。
- `sync`：递归同步目录；加 `--prune` 才删除该目录内已消失的来源。有文件处理失败时不执行清理。
- `search`：查看证据、关系、来源和检索轨迹；`query` 根据证据回答并标注 `[S1]` 等引用。
- `agentic-query`：有界拆解问题、多轮召回、融合证据后回答；各轮共享同一版本锚点（来源版本号），运行中途被更新的来源不会混入结果。
- `list` / `delete` / `graph`：管理来源与图谱，不需要模型密钥。

检索模式包括 `vector`、`keyword`、`hybrid`、`local`、`global`、`hybrid_graph`，默认 `hybrid_graph`。查询可设置 `--top-k 1..100`、`--threshold 0..1`、`--rerank`、`--verbose`。各命令支持 `--json`，stdout 输出单个 JSON，失败返回非零退出码。`RAG_STORAGE=services` 时 CLI 自动走三服务栈；`--data-dir` 显式指定目录时强制本地 SQLite。

| 检索模式 | 使用的通道 |
|---|---|
| `vector` | 片段向量相似度 |
| `keyword` | BM25 关键词检索 |
| `hybrid` | 向量 + BM25，经 RRF 融合 |
| `local` | 实体匹配与一跳关系扩展 |
| `global` | 关系主题匹配与关联片段扩展 |
| `hybrid_graph` | 向量 + BM25 + local + global，经 RRF 融合 |

检索模式 `local` 指实体图检索，与存储模式 `local`（SQLite）是两个不同的设置；`global` 当前没有社区发现或社区摘要。

更换向量模型、端点或抽取配置后应在新的 `RAG_DATA_DIR` 重新建库，避免混用模型指纹或向量维度。

## 数据链路与关键文件

```text
上传/导入（一次性准备）
  → 读取与限额校验 → 中英分块，保留原文字符位置
  → 模型抽取实体、关系（三元组）及逐字证据
  → 生成片段、实体、关系向量
  → PostgreSQL/SQLite 同一事务发布：正文、图谱、BM25 词项统计（总名册）
  → 向量副本写入 Qdrant（services 模式；失败警告，重新导入可补发）

查询（召回窗口 + 图扩展；目录扫描和图扩展仍随库增长）
  → Qdrant 向量召回 top-N / 词项表粗筛关键词窗口
  → 候选 ID 回 PG/SQLite 取原文与一跳图邻接
  → 手写 BM25 公式（全局统计持久化）+ RRF + local/global 融合
  → 跨文件候选分配 → 可选重排 → 有界上下文
  → 带来源引用的回答；无充分证据时拒答
  → 答案写入 Redis 缓存（可选），缓存键包含资料内容、版本及向量发布修订号
```

| 文件 | 在链路中的职责 |
|---|---|
| `cli.py` / `run.py` / `backend/` | CLI 与网页入口；共用 `GraphService` 服务栈 |
| `rag/settings.py` | 存储模式、服务地址、导入预算；配置缺失明确报错 |
| `rag/graph_service.py` | 候选集组装、导入/任务编排、缓存与向量补发 |
| `rag/graph_retrieval.py` | 双层检索、多路融合、来源分配、引用绑定与轨迹 |
| `rag/knowledge_store.py` | SQLite 适配器：事务发布、词项统计、有界读取、精确向量扫描基准 |
| `rag/pg_store.py` + `schema.sql` | PostgreSQL 正本：同接口、同事务语义、词项统计、任务台账 |
| `rag/qdrant_store.py` | Qdrant REST 封装：批量向量、指纹隔离集合、有界召回、按来源清理 |
| `rag/cache.py` | Redis 答案缓存：TTL、资料版本签名键、尽力清理、故障降级 |
| `rag/bm25.py` | 中文分词与 BM25/RRF（公式不变；IDF 提取为共享 `idf_for`） |
| `rag/model_client.py` | 模型调用、抽取证据校验、批量向量、缓存、超时重试与限流 |
| `tests/test_bounded_equivalence.py` | 六模式固定小语料对拍，不代表所有规模/后端等价 |
| `tests/test_storage_regressions.py` | 删除/更新统计、缓存故障恢复、向量补发、SQL 导出与计时回归；不启动服务 |
| `sample_data/` + `docs/eval-questions.json` | 演示语料与 12 题独立标注评测集 |

这里借鉴 LightRAG 的双层检索思想，没有调用官方 LightRAG 包。`hybrid` 只融合向量与关键词；`hybrid_graph` 再加入实体和关系两层。跨文件实体按规范化名称连接，没有独立的实体消歧模型。每条关系都保留所属片段和原文引句，可回溯到文件与版本。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest -q                                   # 回归测试（无需启动服务）
.venv\Scripts\python.exe cli.py eval --test-file docs\eval-questions.json --compare
```

`scripts/smoke.py` 会向配置的数据库写入和删除资料，只能用于独立测试库。离线测试与 SQL 测试替身不能替代 PostgreSQL/Qdrant/Redis 的真实联调。

评测使用自己独立标注的 JSON 数组。每项包含 `question`，可附 `id`、`expected_sources`（文件名列表）、`expected_entities`（实体名列表）、`answerable`（布尔值）。输出默认写入当前库目录的 `evaluation.json`，可用 `--output` 指定位置。

评测报告来源召回率、MRR、实体召回率、无答案问题零召回率和耗时。`elapsed_ms` 覆盖本次检索的存储读取、候选召回、关键词/问题向量处理和后续排序，不包含答案生成；模型缓存是否命中也会影响该耗时。它衡量检索，不自动评判回答事实正确性；有效引用只证明可追溯。答案层的 faithfulness、无依据断言比例和 accuracy 尚未实现，需先明确 rubric、标注和统计口径，不能由离线回归测试推断。

## 工程边界

同源更新先完成抽取与向量生成，再在单个事务内整体发布；失败保留旧版本。查询通过"来源版本锚点"保证一致性：单次查询不混用同一来源的两个版本，agentic 多轮共享同一锚点。模型调用具有超时、有限重试、并发和每进程速率限制。

Redis 清理失败时会降级；缓存键包含来源、内容哈希、文档版本和向量发布修订号。向量发布后在现有 `meta` 表递增修订号，即使只补发向量、正文没变，也会使用新缓存键；生成期间资料或向量发布状态发生变化时不写入缓存。此处不承诺查询期间禁止文档变更或跨服务快照隔离。

当前是单机、单租户系统。services 模式下向量召回交给 Qdrant（上限 1000 候选/通道）、关键词窗口 500，图目录扫描与扩展仍可能增大候选集；local 模式为精确扫描基准。尚未提供多租户权限、分布式部署和高可用保证，也没有真实生产流量成绩。URL 导入入口已随旧管线移除：知识来源只接受显式本地文件。

MIT License.
