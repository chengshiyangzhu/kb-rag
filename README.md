# kb-rag — 企业级 RAG 知识库系统

> 本项目使用 [Trae](https://trae.cn/)（字节跳动出品的 AI IDE）作为开发工具，全程通过 Trae 的 AI 辅助编程能力完成架构设计、代码编写、调试与文档撰写。

## 项目简介

`kb-rag` 是一个企业级检索增强生成（Retrieval-Augmented Generation）知识库平台。它能读取多种格式的文档（PDF / Word / Excel / Markdown / HTML），将其分块、向量化、存储，并在用户提问时检索最相关的内容，交由大模型生成带引用标注的回答。

### 核心能力

| 能力 | 实现 |
|------|------|
| 多格式文档解析 | PDF（含扫描件 OCR）/ DOCX / XLSX / Markdown / HTML / TXT |
| 四种分块策略 | 固定 token / 递归字符 / 语义 / 结构化 |
| 三种嵌入后端 | 本地 bge-m3 / Ollama / 智谱 & OpenAI API |
| 双向量存储 | Qdrant（生产）/ Chroma（开发） |
| 混合检索 | 向量检索 + BM25 + RRF 融合 |
| 重排精筛 | bge-reranker-v2-m3 cross-encoder |
| 多 LLM 支持 | 智谱 GLM / OpenAI GPT / Ollama 本地 |
| 幻觉治理 | 置信度阈值拒答 + 强制引用 + 引用溯源 |
| 可观测性 | 结构化日志 + OpenTelemetry + Prometheus + Grafana |
| 自动评测 | RAGAS 三指标评测框架 |

## 技术栈

- **语言**：Python 3.10+
- **Web 框架**：FastAPI + Uvicorn
- **前端**：Streamlit
- **向量数据库**：Qdrant / Chroma
- **嵌入模型**：bge-m3（本地）/ embedding-3（智谱）/ text-embedding-3-small（OpenAI）
- **重排模型**：bge-reranker-v2-m3
- **LLM**：智谱 GLM-4 / OpenAI GPT / Ollama（Qwen2.5 等）
- **文档解析**：pypdf + RapidOCR / python-docx / openpyxl / markdown / BeautifulSoup
- **中文分词**：jieba
- **可观测性**：structlog + OpenTelemetry + prometheus_client + Grafana
- **部署**：Docker Compose
- **开发工具**：[Trae](https://trae.cn/) AI IDE

## 快速开始

### 方式一：Docker 一键启动（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/chengshiyangzhu/kb-rag.git
cd kb-rag

# 2. 复制环境配置并填入 API Key
cp .env.example .env
# 编辑 .env，至少配置以下项：
#   LLM_PROVIDER=zhipu
#   LLM_MODEL=glm-4-flash
#   ZHIPU_API_KEY=你的智谱API Key
#   EMBEDDER_PROVIDER=api

# 3. 启动全部服务
docker compose -f infra/docker-compose.yml up --build
```

启动后访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| Streamlit UI | http://localhost:8501 | 上传文档、提问、查看引用 |
| FastAPI | http://localhost:8000 | REST API，自动文档 /docs |
| Qdrant | http://localhost:6333 | 向量数据库 |
| Prometheus | http://localhost:9090 | 监控指标 |
| Grafana | http://localhost:3000 | 可视化面板（admin/admin） |

### 方式二：本地 Python 运行

```bash
# 1. 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. 安装依赖
pip install -e ".[dev]"

# 3. 配置环境
cp .env.example .env
# 编辑 .env（建议用 Chroma + API 嵌入，免装 Qdrant 和本地模型）

# 4. 启动 API
uvicorn backend.main:app --reload --port 8000

# 5. 另开终端启动 UI
streamlit run ui/app.py
```

### 首次使用

1. 打开 http://localhost:8501
2. 在左侧上传一份文档（支持 .pdf / .docx / .xlsx / .md / .txt / .html）
3. 等待摄入完成，看到 `num_chunks: N` 表示成功
4. 在右侧输入框提问，查看回答与引用

## 项目结构

```
kb-rag/
├── app/                        # 核心业务逻辑
│   ├── models/                 # 统一数据模型（Document, Chunk, Metadata）
│   ├── config.py               # 集中配置（pydantic-settings）
│   ├── ingest/                 # 文档摄入
│   │   ├── parsers/            #   6 种格式解析器 + 工厂
│   │   └── cleaner.py          #   文本清洗
│   ├── chunkers/               # 4 种分块器 + 工厂
│   ├── embedders/              # 3 种嵌入器 + 工厂
│   ├── stores/                 # 2 种向量存储 + 工厂
│   ├── retrieval/              # 混合检索（向量 + BM25 + RRF）
│   ├── rerank/                 # Cross-encoder 重排
│   ├── generation/             # LLM 生成 + 幻觉治理 + 引用解析
│   ├── pipeline/               # 端到端编排（IngestPipeline + QueryPipeline）
│   └── observability/          # 日志 / 指标 / 追踪
├── backend/                    # FastAPI REST API
│   ├── main.py                 # 应用工厂
│   ├── schemas.py              # 请求/响应模型
│   └── api/v1/                 # 4 个路由：ingest / query / documents / health
├── ui/                         # Streamlit 前端
├── infra/                      # 部署配置
│   ├── docker-compose.yml      # 5 个服务编排
│   ├── prometheus/             # 监控配置
│   └── grafana/                # 仪表盘配置
├── tests/                      # pytest 测试套件（39 个测试）
├── eval/                       # RAGAS 评测脚本
├── scripts/                    # 批量摄入等工具脚本
├── docs/                       # 详细教程文档
├── .env.example                # 环境变量模板
├── pyproject.toml              # 项目元数据与依赖
├── Makefile                    # 常用命令快捷方式
├── Dockerfile                  # API 容器镜像
└── config.yaml                 # 非敏感默认配置
```

## 配置说明

配置分三层（后者覆盖前者）：

1. `config.yaml` — 非敏感默认值
2. `.env` — 环境相关配置（从 `.env.example` 复制后修改）
3. 环境变量 — 运行时覆盖

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | openai | LLM 提供商：openai / zhipu / ollama |
| `LLM_MODEL` | gpt-4o-mini | 模型名 |
| `LLM_TEMPERATURE` | 0.0 | 生成温度，0=确定性输出（RAG 推荐） |
| `LLM_MAX_TOKENS` | 2048 | 回答最大长度 |
| `EMBEDDER_PROVIDER` | local | 嵌入方式：local / ollama / api |
| `VECTOR_STORE` | qdrant | 向量库：qdrant / chroma |
| `CHUNKER_TYPE` | recursive | 分块器：fixed / recursive / semantic / structural |
| `CHUNK_SIZE` | 512 | 分块大小（token） |
| `CHUNK_OVERLAP` | 64 | 分块重叠（token） |
| `RETRIEVE_TOP_N` | 20 | 检索召回数 |
| `RERANK_TOP_K` | 5 | 重排后保留数 |
| `RERANK_THRESHOLD` | 0.3 | 重排置信度阈值，低于此值拒答 |
| `RRF_K` | 60 | RRF 融合参数 |

完整配置见 [.env.example](.env.example)。

## 文档

项目包含详细的新手教程，逐行讲解每一处代码：

| 文档 | 内容 |
|------|------|
| [docs/TUTORIAL_PART1.md](docs/TUTORIAL_PART1.md) | 架构原理、配置系统、数据模型、6 种文档解析器 |
| [docs/TUTORIAL_PART2.md](docs/TUTORIAL_PART2.md) | 文本清洗、4 种分块器、3 种嵌入器、2 种向量存储 |
| [docs/TUTORIAL_PART3.md](docs/TUTORIAL_PART3.md) | 混合检索+RRF、重排、生成+幻觉治理、管线编排 |
| [docs/TUTORIAL_PART4.md](docs/TUTORIAL_PART4.md) | FastAPI、Streamlit UI、可观测性、Docker 部署、参数调优 |
| [docs/LLM_PARAMS_GUIDE.md](docs/LLM_PARAMS_GUIDE.md) | 大模型参数（temperature/max_tokens/top_p）选择指南 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/ingest` | 上传文档并摄入 |
| POST | `/api/v1/query` | 提问并获取回答 |
| GET | `/api/v1/documents` | 列出所有文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/health` | 健康检查 |

启动后访问 http://localhost:8000/docs 查看交互式 API 文档。

## 测试

```bash
make test    # 或 pytest -v
```

覆盖 5 大模块共 39 个测试：解析器与分块器、嵌入与存储、检索与重排与生成、管线端到端、API 接口。

## 开发

```bash
make install    # 安装开发依赖
make lint       # 代码检查
make format     # 格式化
make test       # 运行测试
make dev        # 启动开发服务器
```

## 致谢

本项目使用 [Trae](https://trae.cn/) AI IDE 作为全流程开发工具。Trae 的 AI 辅助能力在以下环节发挥了关键作用：

- 项目架构设计与技术选型
- 全部核心代码编写与调试
- 39 个单元测试的编写与验证
- 6 万字详细教程文档的撰写
- Docker 部署配置与监控仪表盘搭建

## License

MIT
