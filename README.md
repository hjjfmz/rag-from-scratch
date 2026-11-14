# Local Knowledge

本地个人知识库与跨文件问答系统。导入自己的文档，构建带原文证据的实体关系图，结合双层图检索、向量和关键词检索回答问题，并引用来源。

当前交付重点是 CLI 与核心框架。仓库不附带演示语料、业务场景样例或互联网采集器。现有网页仍走旧流程，暂未接入新知识库。

## 安装与配置

在项目根目录使用 Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-dev.txt
Copy-Item .env.example .env
```

在 `.env` 填写兼容服务的 `CHAT_API_URL`、`EMBEDDING_API_URL`、`MODEL_CHAT`、`MODEL_EMBEDDING` 和密钥。聊天与向量服务可以使用不同供应商；独立密钥 `CHAT_API_KEY` / `EMBEDDING_API_KEY` 优先，否则依次使用 `LLM_API_KEY`、`OPENAI_API_KEY`、`OPENROUTER_API_KEY`。不要提交 `.env`。

```powershell
.venv\Scripts\python.exe cli.py doctor
.venv\Scripts\python.exe cli.py doctor --live
```

`doctor` 检查配置和存储；`--live` 会实际调用聊天、向量接口，可能产生费用。缺少配置会明确报错，不会自动切换离线模式。

## 使用自己的知识库

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

支持 UTF-8 TXT/MD 和有文本层的 PDF。扫描件需要先做 OCR。输入上限为每文件 16 MiB、提取文本 50 万字符、1024 个片段。

- `ingest`：构建完成后原子替换同一来源；内容与处理配置未变则跳过，`--force` 强制重建。
- `sync`：递归同步目录；加 `--prune` 才删除该目录内已消失的来源。有文件处理失败时不执行清理。
- `search`：查看证据、关系、来源和检索轨迹；`query` 根据证据回答并标注 `[S1]` 等引用。
- `agentic-query`：有界拆解问题、多轮召回、融合证据后回答；各轮共享同一已提交快照。
- `list` / `delete` / `graph`：管理来源与图谱，不需要模型密钥。

检索模式包括 `vector`、`keyword`、`hybrid`、`local`、`global`、`hybrid_graph`，默认 `hybrid_graph`。查询可设置 `--top-k 1..100`、`--threshold 0..1`、`--rerank`、`--verbose`。各命令支持 `--json`，stdout 输出单个 JSON，失败返回非零退出码。

默认数据库与模型缓存位于项目下 `.runtime/knowledge/`；可用 `--data-dir` 指定独立目录。更换向量模型、端点或抽取配置后应在新目录重新建库，避免混用模型指纹或向量维度。旧版 `rag_index.json` 不会自动转换成图谱。

`--offline` 仅用于无服务诊断，使用独立的 `.runtime/offline/`：英文规则抽取、词法哈希向量、原文摘录，不代表真实模型的中文抽取、语义检索或回答能力。它只处理显式导入的文件。

## 数据链路与关键文件

```text
自己的本地文件
  → 读取与限额校验 → 中英分块，保留原文字符位置
  → 模型抽取实体、关系（三元组）及逐字证据
  → 分别生成片段、实体、关系向量
  → SQLite 同一事务发布来源、片段、实体、关系

问题
  → 低层实体关键词 → local：实体定位、邻接关系、原文证据
  → 高层主题关键词 → global：关系与主题、端点、原文证据
  → 向量 / BM25 → hybrid_graph：多路 RRF 融合
  → 跨文件候选分配 → 可选重排 → 有界上下文
  → 带来源引用的回答；无充分证据时拒答
```

| 文件 | 在链路中的职责 |
|---|---|
| `cli.py` | 命令入口、参数校验、结构化输出与错误退出 |
| `rag/graph_service.py` | 导入、更新、同步、删除、查询与多轮检索编排 |
| `rag/text_chunks.py` | 中英文分块、长度上限、原文字符区间 |
| `rag/model_client.py` | 模型调用、抽取证据校验、批量向量、缓存、超时重试与限流 |
| `rag/knowledge_store.py` | SQLite 事务、来源修订、并发冲突、级联删除与完整性检查 |
| `rag/graph_retrieval.py` | 独立 local/global 路径、多路融合、来源分配、引用绑定与轨迹 |
| `rag/bm25.py` | 包含中文分词处理的关键词检索 |
| `tests/` | 存储、模型边界、召回、跨文件问答与实际 CLI 进程回归验证 |

这里借鉴 LightRAG 的双层检索思想，没有调用官方 LightRAG 包。`hybrid` 只融合向量与关键词；`hybrid_graph` 再加入实体和关系两层。跨文件实体按规范化名称连接，目前没有独立的实体消歧模型。每条关系都保留所属片段和原文引句，可回溯到文件与版本。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe cli.py eval --test-file 'D:\knowledge-evaluation\questions.json' --compare
```

评测使用自己独立标注的 JSON 数组。每项包含 `question`，可附 `id`、`expected_sources`（文件名列表）、`expected_entities`（实体名列表）、`answerable`（布尔值）。输出默认写入当前库目录的 `evaluation.json`，可用 `--output` 指定位置。

评测报告来源召回率、MRR、实体召回率、无答案问题零召回率和耗时。它衡量检索，不自动评判回答事实正确性；有效引用只证明可追溯。真实效果需在自己的文件和真实模型上验证，不能由离线测试推断。

## 工程边界

同源更新先完成抽取与向量生成，再整体发布；失败保留旧版本。来源修订在删除后继续递增，阻止旧任务覆盖重新导入的数据。SQLite WAL 配合短写事务处理单机并发；查询使用一致快照。模型调用具有超时、有限重试、并发和每进程速率限制。

当前是单机、单租户 CLI 系统，向量采用精确扫描，图谱与向量记录仍加载到内存，适合从中小规模知识库开始。尚未提供多租户权限、分布式部署和高可用保证，也没有真实生产流量成绩。后续网页应复用新服务层，现有 `backend/`、`frontend/` 及其旧 RAG 依赖暂时保留。

MIT License。
