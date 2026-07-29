# kb-rag 新手教程 · 第一部分：架构、配置、数据模型与文档解析

> 本教程面向只会 Python 基础语法（`def` / `class` / `import`）的同学，不需要你懂任何 RAG / ML / NLP 概念。
> 我们会一行一行地讲代码，每个参数都解释，并用一份"产品手册"作为贯穿全文的例子。

---

## 目录

- [第 1 章 项目为什么要这样设计——架构原理](#第-1-章-项目为什么要这样设计架构原理)
- [第 2 章 配置系统——项目的"控制面板"](#第-2-章-配置系统项目的控制面板)
- [第 3 章 数据模型——整个项目的"通用语言"](#第-3-章-数据模型整个项目的通用语言)
- [第 4 章 文档解析器——从文件到文本](#第-4-章-文档解析器从文件到文本)

---

## 第 1 章 项目为什么要这样设计——架构原理

### 1.1 这个项目到底在干什么

**一句话**：你有一堆文档（Word、Excel、PDF、Markdown、网页、纯文本），你想让电脑读完之后，能回答你关于这些文档的问题。

举个例子：

- 你有一份《智能水杯 X1 产品手册.pdf》
- 你问："水杯 X1 的电池容量是多少？"
- 电脑应该回答："水杯 X1 电池容量为 2000mAh，可续航 30 天。" 而不是胡编一个数字。

#### 为什么不直接把文档全塞给 ChatGPT？

很多新手会想：直接把整份 PDF 复制粘贴到 ChatGPT 不就行了？不行，原因有三：

1. **上下文窗口放不下**：ChatGPT 一次能读的字数有上限（比如 GPT-4o-mini 大概 128K token）。如果你有 100 份 PDF，每份 50 页，全部塞进去会直接爆。
2. **ChatGPT 不一定有你的私有数据**：你公司的内部产品手册、合同、客户资料，ChatGPT 训练时根本没见过，它不会知道。
3. **每次调用要传大量文本很贵**：每次提问都把几百万字塞过去，API 费用会爆炸，而且响应很慢。

#### RAG 的思路（Retrieval-Augmented Generation，检索增强生成）

RAG 把"问答"拆成两步走，就像你去图书馆查资料：

```
┌─────────────────────────────────────────────────────────────────┐
│  第一阶段：摄入（Ingest）—— 提前把文档"整理好"放进库              │
│                                                                  │
│   文档文件 ──> 解析 ──> 清洗 ──> 分块 ──> 嵌入 ──> 存进向量库      │
│   (PDF/Word)   (转纯文本) (去噪)  (切小段) (算指纹) (写库)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  第二阶段：查询（Query）—— 用户提问时现场检索                    │
│                                                                  │
│   用户问题 ──> 嵌入 ──> 检索 ──> 重排 ──> 拼提示词 ──> LLM 回答    │
│   ("电池?")  (算指纹) (找相关段) (精选) (把片段塞给LLM) (生成)    │
└─────────────────────────────────────────────────────────────────┘
```

- **第一阶段（摄入）**：提前把所有文档读进来，切成小段，给每段算一个"指纹"（向量），存进数据库。这一步只做一次，文档变了才重做。
- **第二阶段（查询）**：用户提问时，先把问题也算个"指纹"，去库里找"指纹最像"的几段文本，再把这几段连同问题一起丢给大模型（LLM），让它组织语言回答。

**打比方**：第一阶段就像图书馆员给每本书做索引卡（按主题分类）；第二阶段就像你拿着主题去前台查索引卡，找到对应的那几页，再坐下来仔细读。

### 1.2 项目分成几大块——为什么这样分

下面这张大图展示了数据在 kb-rag 里是怎么流动的：

```
                        ┌───────────────────────────┐
                        │     用户上传文件 / 提问      │
                        └─────────────┬─────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                            │
        (摄入阶段)                                          (查询阶段)
                │                                            │
                ▼                                            ▼
   ┌─────────────────────┐                      ┌─────────────────────┐
   │ 1. 解析器 Parser     │ <── PDF/Word/Excel   │ 7. 检索器 Retriever  │
   │   把文件转纯文本      │     /HTML/MD/TXT     │   找最相关的几段      │
   └─────────┬───────────┘                      └─────────┬───────────┘
             │                                            │
             ▼                                            ▼
   ┌─────────────────────┐                      ┌─────────────────────┐
   │ 2. 清洗器 Cleaner    │                      │ 8. 重排器 Reranker   │
   │   去掉乱码/多余空白   │                      │   精选最好的 K 段     │
   └─────────┬───────────┘                      └─────────┬───────────┘
             │                                            │
             ▼                                            ▼
   ┌─────────────────────┐                      ┌─────────────────────┐
   │ 3. 分块器 Chunker    │                      │ 9. 生成器 Generator  │
   │   切成 512 字小段     │                      │   LLM 总结回答        │
   └─────────┬───────────┘                      └─────────┬───────────┘
             │                                            │
             ▼                                            │
   ┌─────────────────────┐                               │
   │ 4. 嵌入器 Embedder   │                               │
   │   每段算 1024 维向量  │                               │
   └─────────┬───────────┘                               │
             │                                            │
             ▼                                            │
   ┌─────────────────────┐                               │
   │ 5. 存储器 Store      │ ──────── 检索时查询 ──────>  │
   │   向量库 + 关键词索引 │                               │
   └─────────┬───────────┘                               │
             │                                            │
             ▼                                            │
   ┌─────────────────────┐                               │
   │ 6. 管线 Pipeline     │ 编排上面所有步骤              │
   └─────────┬───────────┘                               │
             │                                            │
             ▼                                            ▼
   ┌─────────────────────┐                      ┌─────────────────────┐
   │ 10. API (FastAPI)   │ <── HTTP 接口        │ 11. UI (Streamlit)  │
   │   对外提供 HTTP      │                      │   网页界面            │
   └─────────────────────┘                      └─────────────────────┘
```

**用图书馆打比方解释每一层**：

| 层 | 比喻 | 作用 |
|---|---|---|
| 解析器 | 把各种语言的书都翻成中文版 | PDF/Word/Excel 等不同格式都转成纯文本 |
| 清洗器 | 把脏页擦干净、撕掉空白页 | 去掉乱码、多余空行、特殊字符 |
| 分块器 | 把厚书拆成一章一章 | 长文本切成 512 字的小段（因为检索要找"最相关的小段"而非整篇） |
| 嵌入器 | 给每段贴一个"主题指纹" | 把文本变成 1024 个数字组成的向量，相似内容指纹相似 |
| 存储器 | 图书馆的索引卡柜子 | 把向量和原文存起来，方便快速查找 |
| 检索器 | 你拿着主题去前台查卡片 | 用户提问时，找到最相关的几段文本 |
| 重排器 | 图书馆员帮你从 10 本里挑 3 本最好的 | 从找到的几段里精选最相关的 K 段 |
| 生成器 | 你坐下来读那 3 本书然后回答 | 把选中的文本和问题一起给 LLM，让它组织语言回答 |
| 管线 | 整个图书馆的运作流程 | 把上面这些步骤串起来的"流水线" |
| API | 图书馆的对外服务窗口 | 对外提供 HTTP 接口让别人能调用 |
| UI | 图书馆的大厅触摸屏 | 网页界面让用户能操作 |

### 1.3 目录结构详解

下面是 `kb-rag/` 的完整目录树（只展示到二级）：

```
kb-rag/
├── app/                  # 核心业务代码（所有 RAG 逻辑都在这）
│   ├── config.py         # 配置系统：从 .env 读参数
│   ├── models/           # 数据模型：Document、Chunk 等通用结构
│   ├── ingest/           # 摄入阶段：解析器 + 清洗器
│   │   └── parsers/      # 各种文件格式解析器
│   ├── chunkers/         # 分块器：把长文本切成小段
│   ├── embedders/        # 嵌入器：文本转向量
│   ├── stores/           # 存储器：向量库（Qdrant / Chroma）
│   ├── retrieval/        # 检索器：向量检索 + BM25 + 混合检索
│   ├── rerank/           # 重排器：精选最相关片段
│   ├── generation/       # 生成器：调用 LLM 生成回答
│   ├── pipeline/         # 管线：编排摄入和查询流程
│   └── observability/    # 可观测性：日志、指标、追踪
├── backend/              # FastAPI HTTP 服务
│   ├── main.py           # FastAPI 入口
│   ├── schemas.py        # API 请求/响应模型
│   └── api/v1/           # API 路由（documents/ingest/query/health）
├── ui/                   # Streamlit 网页界面
│   └── app.py
├── data/                 # 数据目录（raw 原始文件 / processed 处理后文件）
├── docs/                 # 文档（你正在读的就是这里的）
├── eval/                 # 评估脚本（用 ragas 测 RAG 效果）
├── infra/                # 基础设施（docker-compose / Prometheus / Grafana）
├── scripts/              # 脚本（如 seed.py 初始化种子数据）
├── tests/                # 单元测试
├── pyproject.toml        # Python 项目配置 + 依赖列表
├── .env.example          # 环境变量示例（复制成 .env 用）
├── config.yaml           # YAML 配置（备用，本教程以 .env 为准）
├── Dockerfile            # 容器镜像构建
├── Makefile              # 常用命令快捷方式
└── README.md             # 项目说明
```

**重点说明 `app/` 下每个子目录为什么单独存在**：

- **`models/`**：定义"通用语言"。后面所有模块都要交换 `Document` 和 `Chunk` 对象，必须先定义好它们。单独抽出来避免循环依赖。
- **`ingest/parsers/`**：解析器只管"文件转文本"，不关心后面怎么切块、怎么嵌入。职责单一才好维护——比如未来想加 PPT 解析器，只要新建一个文件即可。
- **`chunkers/`**：分块策略有多种（按字符切、按 token 切、按语义切、按结构切）。每种一个文件，用 `factory.py` 统一分发。
- **`embedders/`**：嵌入可以本地跑（`sentence-transformers`）、走 Ollama、走 API。三种实现分开。
- **`stores/`**：向量库可以选 Qdrant 或 Chroma。抽象出 `base.py`，两个实现都遵守同一接口。
- **`retrieval/`**：检索有向量检索（dense）、关键词检索（BM25）、混合检索（hybrid）、结果融合（fusion）。每种一个文件。
- **`rerank/`**：重排是一个独立环节，单独抽出来方便替换模型。
- **`generation/`**：生成阶段除了调 LLM，还要做引用（citation）、安全护栏（guardrail）、提示词管理（prompts）。
- **`pipeline/`**：编排层。`ingest_pipeline.py` 串起摄入阶段所有步骤，`query_pipeline.py` 串起查询阶段。
- **`observability/`**：日志、指标、追踪三件套，单独抽出来不污染业务代码。

---

## 第 2 章 配置系统——项目的"控制面板"

### 2.1 为什么需要配置系统

假设你把 `chunk_size = 512` 写死在代码里。某天你想改成 1024 试试效果，你得改代码、重启服务、还可能忘了改回去。更糟的是，开发环境你想用便宜的本地模型，生产环境要用 OpenAI，难道每次部署都改代码？

**配置系统**就是把这些"会变的参数"从代码里挪到外面，让代码读配置文件即可。Python 里常见的做法：

- 写一个 `.env` 文件（就是一个文本文件，里面是 `KEY=VALUE` 一行一行的）。
- 代码用 `pydantic-settings` 库自动读这个文件，把里面的值变成 Python 变量。

**`.env` 文件是什么**：一个隐藏文件（以点开头），内容形如 `LLM_MODEL=gpt-4o-mini`。它通常**不提交到 git**（里面有 API Key 等敏感信息），只在你本地存在。项目提供 `.env.example` 作为模板，你复制成 `.env` 再改值。

### 2.2 config.py 逐行精读

下面是 `app/config.py` 的完整代码，我们一段一段拆开讲：

```python
"""Application settings loaded from environment variables and .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
```

**逐行解释**：

- 第 1 行：模块的文档字符串（docstring），说明这个文件"从环境变量和 .env 文件加载应用配置"。
- 第 2 行 `from __future__ import annotations`：开启"延迟注解求值"。Python 3.11 默认注解会立即求值，加上这行后注解变成字符串不立即求值。好处是你可以写 `int | None` 这种语法而不报错，并且能避免某些循环引用。
- 第 4 行 `from functools import lru_cache`：导入 `lru_cache` 装饰器。LRU = Least Recently Used（最近最少使用）。`lru_cache(maxsize=1)` 表示"缓存最近 1 次调用的结果"，用于让 `get_settings()` 只读一次配置文件。
- 第 5 行 `from typing import Literal`：导入 `Literal` 类型。`Literal["a", "b"]` 表示"这个值只能是 'a' 或 'b'"。用来限制某些配置项只能选特定值。
- 第 7 行 `from pydantic import Field, field_validator`：
  - `Field`：用来给模型字段加默认值、描述、约束等。
  - `field_validator`：装饰器，用来给字段加"校验逻辑"（比如检查值必须大于 0）。
- 第 8 行 `from pydantic_settings import BaseSettings, SettingsConfigDict`：
  - **`pydantic-settings` 这个库是干什么的**：是 `pydantic` 的扩展，专门用来读配置。它能让你的 `Settings` 类自动从环境变量、`.env` 文件读值，并且做类型校验（比如你写 `port: int`，它读到字符串会自动转 int，转不过就报错）。比手动 `os.getenv("XXX")` 安全得多。
  - `BaseSettings`：所有配置类的基类，继承它就能自动读环境变量。
  - `SettingsConfigDict`：配置这个 Settings 类的行为（从哪个文件读、编码、是否大小写敏感等）。

```python
class Settings(BaseSettings):
    """Central configuration for the kb-rag platform.

    Values are read from environment variables first, then from a ``.env`` file
    located at the project root. Non-sensitive defaults mirror ``config.yaml``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
```

**逐行解释**：

- 第 11 行：定义 `Settings` 类，继承 `BaseSettings`。这个类的每个类变量就是一个配置项。
- 第 12-16 行：类的 docstring，说明"先读环境变量，再读项目根目录的 `.env` 文件"。
- 第 18-23 行：`model_config` 是 pydantic 规定的特殊属性，用来配置这个模型的行为。`SettingsConfigDict` 接受 4 个参数：
  - `env_file=".env"`：指定从项目根目录的 `.env` 文件读配置。**意思**：告诉 pydantic-settings 去找 `.env`。
  - `env_file_encoding="utf-8"`：读 `.env` 文件时用 UTF-8 编码。**为什么**：中文注释或中文值不会乱码。
  - `case_sensitive=False`：环境变量名大小写不敏感。**意思**：`LLM_MODEL` 和 `llm_model` 都能匹配到字段 `llm_model`。
  - `extra="ignore"`：如果 `.env` 里有 Settings 类没定义的字段，直接忽略不报错。**为什么**：避免 `.env` 多写了一行就启动失败。

```python
    # ---- Application ----
    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
```

**字段解释**：

- `app_env: Literal["dev", "prod"] = "dev"`：应用环境，只能是 `"dev"`（开发）或 `"prod"`（生产），默认 `"dev"`。**什么时候改**：上线时改成 `"prod"`，让代码知道自己跑在生产环境（比如开启更严格的日志）。
- `log_level: str = "INFO"`：日志级别，默认 `"INFO"`。可选值通常有 `DEBUG` / `INFO` / `WARNING` / `ERROR`。**什么时候改**：调试时改成 `"DEBUG"` 看详细日志。

```python
    # ---- Embedder ----
    embedder_provider: Literal["local", "ollama", "api"] = "local"
    embedder_model: str = "BAAI/bge-m3"
    embedder_dim: int = 1024
```

**字段解释（嵌入器相关）**：

- `embedder_provider: Literal["local", "ollama", "api"] = "local"`：嵌入器从哪里来，三种选择：
  - `"local"`：用本地 `sentence-transformers` 跑模型（免费但要下载模型）。
  - `"ollama"`：走 Ollama 服务（本地起的模型服务）。
  - `"api"`：走远程 API（如 OpenAI）。
  - 默认 `"local"`。
- `embedder_model: str = "BAAI/bge-m3"`：用什么嵌入模型，默认 `"BAAI/bge-m3"`。**为什么选它**：bge-m3 是智源研究院开源的多语言嵌入模型，中文效果好，1024 维。
- `embedder_dim: int = 1024`：嵌入向量维度，默认 1024。**为什么**：bge-m3 输出就是 1024 维。**注意**：换模型必须同步改这个值，否则向量库存进去和查出来的维度对不上会报错。

```python
    # ---- Ollama ----
    ollama_base_url: str = "http://ollama:11434"

    # ---- OpenAI ----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # ---- Zhipu (BigModel) ----
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
```

**字段解释（各家 API 配置）**：

- `ollama_base_url: str = "http://ollama:11434"`：Ollama 服务地址。**为什么是 `ollama`**：项目用 docker-compose 部署，`ollama` 是容器名，docker 内部 DNS 会解析到对应容器。本地裸跑可改成 `http://localhost:11434`。
- `openai_api_key: str = ""`：OpenAI 的 API Key，默认空字符串。**注意**：用 OpenAI 必须填，从 https://platform.openai.com/ 拿。
- `openai_base_url: str = "https://api.openai.com/v1"`：OpenAI API 基础 URL。**为什么**：默认走官方。如果你用代理或兼容服务（如 Azure OpenAI、各种中转），改这里。
- `zhipu_api_key: str = ""`：智谱 BigModel 的 API Key，默认空。**为什么有它**：项目原生支持智谱（国产大模型，中文好且便宜）。
- `zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"`：智谱 API 地址。

```python
    # ---- LLM ----
    llm_provider: Literal["openai", "zhipu", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0       # RAG 需要确定性输出，0 = 每次回答相同
    llm_max_tokens: int = 2048         # 回答最大长度，2048 token 约 1500 字中文
    llm_top_p: float = 1.0             # 核采样，1.0 = 不限制（用 temperature 控制即可）
```

**字段解释（LLM 相关，最关键）**：

- `llm_provider: Literal["openai", "zhipu", "ollama"] = "openai"`：用哪家大模型，默认 OpenAI。
- `llm_model: str = "gpt-4o-mini"`：具体模型名，默认 `gpt-4o-mini`。**为什么选它**：便宜（比 gpt-4o 便宜约 30 倍）、速度快、效果对 RAG 够用。
- `llm_temperature: float = 0.0`：温度参数，范围 0-2，默认 0。**意思**：控制输出随机性。0 = 完全确定性（同样问题每次回答几乎一样），2 = 很随机。**为什么 RAG 设 0**：RAG 要的是"基于检索到的内容忠实回答"，不需要 LLM 发挥创意，所以温度越低越好。
- `llm_max_tokens: int = 2048`：回答最大 token 数，默认 2048。**意思**：LLM 最多生成 2048 个 token。**为什么是这个值**：2048 token 大约 1500 字中文，对常见问答够用，又不会太贵太慢。
- `llm_top_p: float = 1.0`：核采样（nucleus sampling），范围 (0, 1]，默认 1.0。**意思**：1.0 表示不限制，让 `temperature` 一个参数控制即可。**为什么不调**：通常 `temperature` 和 `top_p` 二选一调，不要同时调。

```python
    # ---- Vector store ----
    vector_store: Literal["qdrant", "chroma"] = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "kb_rag"
    chroma_path: str = "./data/chroma"
```

**字段解释（向量库）**：

- `vector_store: Literal["qdrant", "chroma"] = "qdrant"`：用哪个向量库，默认 Qdrant。**为什么选 Qdrant**：性能好、支持过滤、生产级。Chroma 更轻量适合开发。
- `qdrant_url: str = "http://qdrant:6333"`：Qdrant 服务地址，默认 6333 端口。
- `qdrant_collection: str = "kb_rag"`：Qdrant 里的 collection 名（类似数据库的表名），默认 `"kb_rag"`。
- `chroma_path: str = "./data/chroma"`：如果用 Chroma，数据存哪里，默认 `./data/chroma`。Chroma 是嵌入式向量库，数据存本地文件夹。

```python
    # ---- Chunker ----
    chunker_type: Literal["recursive", "fixed", "semantic", "structural"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
```

**字段解释（分块器）**：

- `chunker_type: Literal["recursive", "fixed", "semantic", "structural"] = "recursive"`：分块策略，四种：
  - `"recursive"`：递归字符切分（默认，按 `\n\n` → `\n` → ` ` 层级切），通用首选。
  - `"fixed"`：固定 token 数切。
  - `"semantic"`：按语义切（句子边界）。
  - `"structural"`：按文档结构切（Markdown 标题、表格等）。
- `chunk_size: int = 512`：每块目标字符数，默认 512。**为什么是 512**：兼顾检索精度（太小信息不全）和效率（太大检索慢且噪声多），512 是社区常用经验值。
- `chunk_overlap: int = 64`：相邻块重叠字符数，默认 64。**为什么需要重叠**：如果一句关键的话正好被切在两块边界，没重叠就会丢失。重叠 64 字保证边界内容在两块都出现。

```python
    # ---- Retrieval / Rerank ----
    retrieve_top_n: int = 20
    rerank_top_k: int = 5
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.3
```

**字段解释（检索 + 重排）**：

- `retrieve_top_n: int = 20`：检索阶段先取多少条候选，默认 20。**意思**：先粗排取 20 条，再交给重排器精选。
- `rerank_top_k: int = 5`：重排后保留多少条给 LLM，默认 5。**为什么是 5**：太少上下文不足，太多 token 超限。5 条是经验平衡点。
- `rerank_model: str = "BAAI/bge-reranker-v2-m3"`：重排模型，默认 bge-reranker-v2-m3。**为什么选它**：和 bge-m3 同系列，中文重排效果好。
- `rerank_threshold: float = 0.3`：重排分数阈值，默认 0.3。**意思**：重排分数低于 0.3 的丢弃，认为不相关。

```python
    # ---- Fusion ----
    rrf_k: int = 60

    # ---- BM25 ----
    bm25_index_path: str = "./data/bm25.pkl"
```

**字段解释**：

- `rrf_k: int = 60`：RRF（Reciprocal Rank Fusion，倒数排名融合）的参数 k，默认 60。**意思**：融合向量检索和 BM25 检索结果时用的平滑常数。**为什么是 60**：RRF 论文经验值，k 越大对排名靠后的项越宽容。
- `bm25_index_path: str = "./data/bm25.pkl"`：BM25 关键词索引存哪里，默认 `./data/bm25.pkl`。`.pkl` 是 Python pickle 文件，把 Python 对象序列化到磁盘。

```python
    # ---- Observability ports ----
    prometheus_port: int = 9090
    grafana_port: int = 3000
    api_port: int = 8000
    ui_port: int = 8501
```

**字段解释（可观测性端口）**：

- `prometheus_port: int = 9090`：Prometheus 监控服务端口，默认 9090。
- `grafana_port: int = 3000`：Grafana 可视化面板端口，默认 3000。
- `api_port: int = 8000`：FastAPI 后端端口，默认 8000。
- `ui_port: int = 8501`：Streamlit 前端端口，默认 8501（Streamlit 默认端口）。

接下来是字段校验器：

```python
    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v
```

**逐行解释**：

- `@field_validator("chunk_size")`：pydantic 的装饰器，告诉 pydantic "对字段 `chunk_size` 加一个校验函数"。当 `chunk_size` 被赋值时，这个函数会被自动调用。
- `@classmethod`：声明这是类方法（不需要实例即可调用）。pydantic 校验器约定要加这个。
- `def _chunk_size_positive(cls, v: int) -> int:`：函数名自定义（只要不冲突即可），参数 `v` 是被校验的值（类型 `int`），返回值也是 `int`（校验通过后返回的值，可以做转换）。
- `if v <= 0: raise ValueError("chunk_size must be positive")`：如果值小于等于 0，抛异常阻止赋值。**为什么**：块大小必须是正数，否则没意义。
- `return v`：校验通过，原样返回。

后面的校验器逻辑类似，逐个说明：

- `_overlap_non_negative`：校验 `chunk_overlap >= 0`。重叠可以是 0（不重叠）但不能是负数。
- `_dim_positive`：校验 `embedder_dim > 0`。向量维度必须是正整数。
- `_positive_int`：同时校验 `retrieve_top_n`、`rerank_top_k`、`rrf_k` 三个字段都必须 > 0。**注意**：一个校验器可以绑定多个字段。
- `_temperature_range`：校验 `llm_temperature` 在 [0.0, 2.0]。**为什么是 2.0**：OpenAI API 的温度上限就是 2.0。
- `_top_p_range`：校验 `llm_top_p` 在 (0.0, 1.0]。**为什么不能 0**：top_p=0 没意义（不选任何 token）。
- `_max_tokens_positive`：校验 `llm_max_tokens > 0`。
- `_normalize_level`：把 `log_level` 转大写。**意思**：用户写 `info` 或 `Info` 都会被规范成 `INFO`。

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()
```

**逐行解释**：

- `@lru_cache(maxsize=1)`：缓存装饰器，`maxsize=1` 表示只缓存最近 1 次结果。**为什么用缓存**：`Settings()` 每次实例化都要读 `.env` 文件、做类型转换、跑校验器，开销不小。整个应用生命周期配置基本不变，所以缓存成单例（singleton）。第一次调用读文件，之后直接返回缓存的对象。
- `def get_settings() -> Settings:`：无参函数，返回一个 `Settings` 实例。
- `return Settings()`：实例化 `Settings`，pydantic-settings 会自动读 `.env` 和环境变量。

**怎么用**：

```python
from app.config import get_settings
settings = get_settings()
print(settings.llm_model)  # gpt-4o-mini
```

### 2.3 .env.example 逐行精读

下面是 `.env.example` 的完整内容：

```bash
# ============ Application ============
APP_ENV=dev
APP_LOG_LEVEL=INFO

# ============ Embedder ============
EMBEDDER_PROVIDER=local
EMBEDDER_MODEL=BAAI/bge-m3
EMBEDDER_DIM=1024

# ============ Ollama ============
OLLAMA_BASE_URL=http://ollama:11434

# ============ OpenAI ============
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1

# ============ Zhipu (BigModel) ============
ZHIPU_API_KEY=
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# ============ LLM ============
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
# 生成参数（详见 docs/LLM_PARAMS_GUIDE.md）
LLM_TEMPERATURE=0.0       # 0=确定性输出（RAG推荐），1=有创意，2=随机
LLM_MAX_TOKENS=2048       # 回答最大长度，2048 token 约 1500 字中文
LLM_TOP_P=1.0             # 核采样，1.0=不限制（用 temperature 控制即可）

# ============ Vector Store ============
VECTOR_STORE=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=kb_rag
CHROMA_PATH=./data/chroma

# ============ Chunker ============
CHUNKER_TYPE=recursive
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# ============ Retrieval / Rerank ============
RETRIEVE_TOP_N=20
RERANK_TOP_K=5
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_THRESHOLD=0.3

# ============ Fusion (Reciprocal Rank Fusion) ============
RRF_K=60

# ============ BM25 ============
BM25_INDEX_PATH=./data/bm25.pkl

# ============ Observability Ports ============
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
API_PORT=8000
UI_PORT=8501
```

**逐行解释**（只讲和 `config.py` 默认值不一样或需要补充的地方，其他参见上文）：

- `APP_ENV=dev`：开发环境。**什么时候改**：上线改 `prod`。
- `APP_LOG_LEVEL=INFO`：注意环境变量名是 `APP_LOG_LEVEL`（多了 `APP_` 前缀），但 `config.py` 里字段是 `app_env` 和 `log_level`。**为什么前缀对不上**：因为 `case_sensitive=False` 且 pydantic-settings 默认会忽略大小写但不忽略前缀——实际上这里 `log_level` 字段对应的环境变量是 `LOG_LEVEL`，`APP_LOG_LEVEL` 会被 `extra="ignore"` 忽略。**这是一个容易踩坑的点**：要改日志级别应该用 `LOG_LEVEL=DEBUG`。
- `OPENAI_API_KEY=`：值为空，使用前必须填。**注意**：值和等号之间不要加空格。
- `OPENAI_BASE_URL=https://api.openai.com/v1`：用国内代理或兼容服务时改这里。
- `ZHIPU_API_KEY=`：用智谱时填，从 https://open.bigmodel.cn/ 拿。

**LLM 相关配置详解**（最重要）：

- `LLM_PROVIDER=openai`：用 OpenAI。**改成 `zhipu`**：用智谱 GLM；**改成 `ollama`**：用本地 Ollama 起的模型（完全离线）。
- `LLM_MODEL=gpt-4o-mini`：模型名。改 provider 时必须同步改这个，比如 `zhipu` 配 `glm-4-flash`，`ollama` 配 `qwen2.5:7b`。
- `LLM_TEMPERATURE=0.0`：**RAG 强烈建议保持 0**。如果你做创意写作场景才调高。
- `LLM_MAX_TOKENS=2048`：如果回答被截断，调大到 4096；如果嫌慢，调小到 1024。
- `LLM_TOP_P=1.0`：和 temperature 二选一调，新手保持 1.0 即可。

### 2.4 pyproject.toml 依赖详解

`pyproject.toml` 里 `dependencies` 部分如下：

```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "qdrant-client>=1.9",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "FlagEmbedding>=1.2.10",
    "rank-bm25>=0.2.2",
    "pypdf>=4.3",
    "python-docx>=1.1",
    "openpyxl>=3.1",
    "rapidocr-onnxruntime>=1.3.24",
    "beautifulsoup4>=4.12",
    "markdown>=3.6",
    "tiktoken>=0.7",
    "openai>=1.30",
    "httpx>=0.27",
    "python-multipart>=0.0.9",
    "structlog>=24.1",
    "opentelemetry-sdk>=1.25",
    "opentelemetry-instrumentation-fastapi>=0.46b0",
    "prometheus-fastapi-instrumentator>=7.0",
    "prometheus-client>=0.20",
    "ragas>=0.1.9",
    "streamlit>=1.35",
    "tenacity>=8.3",
    "numpy>=1.26",
]
```

**每个包解释**（按用途分组）：

**Web 框架**：

- `fastapi>=0.110`：FastAPI，现代 Python Web 框架，自动生成 OpenAPI 文档、支持异步。**为什么用它**：写 API 又快又规范，天然契合 LLM 应用。
- `uvicorn[standard]>=0.27`：ASGI 服务器，用来跑 FastAPI。`[standard]` 表示装上推荐扩展（如 `uvloop`、`httptools`），性能更好。
- `python-multipart>=0.0.9`：处理 `multipart/form-data` 上传，FastAPI 文件上传依赖它。

**数据校验 / 配置**：

- `pydantic>=2.6`：数据校验库，定义 `Document` / `Chunk` 等模型。
- `pydantic-settings>=2.2`：pydantic 的配置扩展，读 `.env`。
- `pyyaml>=6.0`：读 YAML 配置文件（`config.yaml`）。

**向量库**：

- `qdrant-client>=1.9`：Qdrant 向量库的 Python 客户端。
- `chromadb>=0.5`：Chroma 向量库（嵌入式，无需起服务）。

**嵌入模型**：

- `sentence-transformers>=3.0`：跑嵌入模型的库，支持 HuggingFace 模型。**为什么用它**：`BAAI/bge-m3` 通过它加载。
- `FlagEmbedding>=1.2.10`：智源官方的嵌入库，bge 系列推荐用它。
- `numpy>=1.26`：数值计算基础库，向量运算必备。

**检索**：

- `rank-bm25>=0.2.2`：BM25 关键词检索的纯 Python 实现。**为什么用它**：传统关键词检索补充向量检索的不足（向量检索对专有名词、数字不敏感）。

**文件解析**：

- `pypdf>=4.3`：纯 Python 的 PDF 解析库。**为什么用它**：无需系统级依赖（不像 pdfplumber 依赖 C 库），跨平台稳定。
- `python-docx>=1.1`：解析 `.docx` 文件。
- `openpyxl>=3.1`：解析 `.xlsx` 文件。
- `rapidocr-onnxruntime>=1.3.24`：OCR 库，识别扫描版 PDF。**为什么用它**：比 Tesseract 中文效果好，基于 ONNX Runtime 跨平台。
- `beautifulsoup4>=4.12`：HTML 解析库，提取网页正文。
- `markdown>=3.6`：Markdown 解析库。

**LLM 客户端**：

- `openai>=1.30`：OpenAI 官方 Python SDK。**为什么用它**：智谱、Ollama 等都兼容 OpenAI API 协议，一个 SDK 走天下。
- `httpx>=0.27`：现代 HTTP 客户端，支持异步。
- `tiktoken>=0.7`：OpenAI 的 token 计数库，用来估算文本 token 数。

**可观测性**：

- `structlog>=24.1`：结构化日志库，输出 JSON 格式日志方便机器解析。
- `opentelemetry-sdk>=1.25`：分布式追踪 SDK。
- `opentelemetry-instrumentation-fastapi>=0.46b0`：自动给 FastAPI 加追踪。
- `prometheus-fastapi-instrumentator>=7.0`：给 FastAPI 加 Prometheus 指标。
- `prometheus-client>=0.20`：Prometheus 客户端，自定义业务指标。

**评估**：

- `ragas>=0.1.9`：RAG 评估库，自动测试 RAG 系统效果（答案准确率、上下文相关性等）。

**UI**：

- `streamlit>=1.35`：数据应用 UI 框架，几行代码写个网页。

**工具**：

- `tenacity>=8.3`：重试库，调用 API 失败时自动重试。

---

## 第 3 章 数据模型——整个项目的"通用语言"

### 3.1 为什么需要统一数据模型

不同格式的文档解析出来结构天差地别：

- PDF：有页码、可能需要 OCR
- Word：有段落、有表格
- Excel：有多个 sheet、每个 sheet 是二维表
- HTML：有标签嵌套

但后续处理（分块、嵌入、检索）不关心文件来源，它只想拿到"一段文本 + 一些附加信息"。所以需要定义一个**统一的数据模型**，所有解析器都输出这个模型，后续模块只认这个模型。

**打比方**：就像不同国家的货币要换成统一货币（比如欧元）才能比较价格。不同格式的文档都转成 `Document` 对象，才能用同一套流程处理。

### 3.2 document.py 逐行精读

下面是 `app/models/document.py` 的完整代码：

```python
"""Unified data models for the kb-rag project.

These models are the single source of truth flowing through every stage of the
pipeline: ingestion -> chunking -> embedding -> storage -> retrieval -> rerank
-> generation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Metadata attached to a document or chunk.

    Attributes:
        source: Original file path or URI of the content.
        page: 1-indexed page number for paginated documents (PDF).
        sheet: Sheet name for spreadsheet documents (XLSX).
        tag: Free-form tags for categorization / filtering.
        created_at: UTC timestamp of creation.
        doc_id: Identifier of the parent document.
        chunk_index: Index of this chunk within the parent document.
        bbox: Optional bounding box for spatially located content (OCR/PDF).
    """

    source: str
    page: int | None = None
    sheet: str | None = None
    tag: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    doc_id: str
    chunk_index: int = 0
    bbox: dict[str, Any] | None = None
```

**逐行解释**：

- 第 1-6 行：模块 docstring，说明这些模型是整个管线的"单一真相来源"（single source of truth），从摄入到生成都用同一套模型。
- 第 7 行 `from __future__ import annotations`：开启延迟注解求值。
- 第 9 行 `from datetime import datetime, timezone`：导入日期时间类和时区类，用来打时间戳。
- 第 10 行 `from typing import Any`：导入 `Any` 类型，表示"任意类型"。
- 第 11 行 `from uuid import uuid4`：导入 `uuid4` 函数，生成 UUID（通用唯一标识符）。
- 第 13 行 `from pydantic import BaseModel, Field`：
  - **pydantic 这个库是干什么的**：Python 最流行的数据校验库。继承 `BaseModel` 的类会自动获得类型校验、JSON 序列化等功能。
  - `BaseModel`：所有模型的基类。
  - `Field`：给字段加默认值、描述、约束等。
- 第 16 行 `class Metadata(BaseModel):`：定义 `Metadata` 类，继承 `BaseModel`。它存"关于文档的附加信息"（元数据）。
- 第 17-27 行：类 docstring，逐字段说明含义。

**Metadata 字段详解**（每个字段用表格说明）：

| 字段名 | 类型 | 默认值 | 含义 | 如果没有这个字段会怎样 |
|---|---|---|---|---|
| `source` | `str` | （必填） | 原始文件路径或 URI，比如 `"manual.pdf"` | 不知道这段文本来自哪个文件，无法溯源 |
| `page` | `int \| None` | `None` | 1-indexed 页码（PDF 用） | 不知道文本在第几页，无法定位 |
| `sheet` | `str \| None` | `None` | sheet 名（Excel 用） | 不知道文本在哪个工作表 |
| `tag` | `list[str]` | `[]` | 自由标签数组，用于分类过滤 | 无法按业务标签过滤检索结果 |
| `created_at` | `datetime` | 当前 UTC 时间 | 创建时间戳 | 不知道这条数据何时入库，无法按时间排序 |
| `doc_id` | `str` | （必填） | 父文档 ID，同一文件的所有 chunk 共享 | 切块后无法知道哪些 chunk 属于同一原文档 |
| `chunk_index` | `int` | `0` | 该 chunk 在父文档中的序号 | 无法还原原文档顺序 |
| `bbox` | `dict[str, Any] \| None` | `None` | 文本在页面上的边界框（OCR/PDF 坐标），如 `{"x": 100, "y": 200, "w": 50, "h": 20}` | 无法在原文档上高亮显示答案位置 |

**字段定义语法解释**：

- `source: str`：必填字段（无默认值），类型 `str`。创建 `Metadata` 时必须传 `source`。
- `page: int | None = None`：可选字段，默认 `None`。`int | None` 是 Python 3.10+ 语法，等价于 `Optional[int]`，表示"可以是 int 或 None"。
- `tag: list[str] = Field(default_factory=list)`：用 `Field(default_factory=list)` 而不是 `= []`。**为什么**：Python 中可变默认值（如 `[]`）会被所有实例共享（著名的可变默认值陷阱），用 `default_factory=list` 每次创建实例时调用 `list()` 生成新列表，避免共享。
- `created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))`：默认值是当前 UTC 时间。**为什么用 lambda**：同样是为了每次创建实例时才计算时间，而不是类定义时算一次。**为什么用 UTC**：UTC 是世界统一时间，避免时区混乱（服务器可能部署在任何时区）。
- `doc_id: str`：必填，父文档 ID。
- `chunk_index: int = 0`：默认 0（表示整个文档本身）。
- `bbox: dict[str, Any] | None = None`：可选的边界框字典。

继续看 `Document` 类：

```python
class Document(BaseModel):
    """A raw document consisting of text and metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: Metadata

    @classmethod
    def from_text(
        cls,
        text: str,
        source: str,
        doc_id: str | None = None,
        tag: list[str] | None = None,
        page: int | None = None,
        sheet: str | None = None,
        bbox: dict[str, Any] | None = None,
        chunk_index: int = 0,
    ) -> Document:
        """Factory building a :class:`Document` from raw text.

        Args:
            text: Full text content of the document.
            source: Original file path or URI.
            doc_id: Optional document id; generated when omitted.
            tag: Optional list of tags.
            page: Optional page number (paginated documents).
            sheet: Optional sheet name (spreadsheets).
            bbox: Optional bounding box (OCR / spatial content).
            chunk_index: Index within the parent document (0 for a root doc).

        Returns:
            A populated :class:`Document` instance.
        """
        resolved_id = doc_id or str(uuid4())
        metadata = Metadata(
            source=source,
            page=page,
            sheet=sheet,
            tag=list(tag) if tag else [],
            doc_id=resolved_id,
            chunk_index=chunk_index,
            bbox=bbox,
        )
        return cls(id=resolved_id, text=text, metadata=metadata)
```

**逐行解释**：

- `class Document(BaseModel):`：定义 `Document` 类，表示"一个原始文档"。
- `id: str = Field(default_factory=lambda: str(uuid4()))`：文档唯一 ID，默认用 `uuid4()` 生成。`uuid4()` 返回一个 UUID 对象（如 `UUID('12345678-...')`），`str()` 转成字符串。
- `text: str`：必填，文档的文本内容。
- `metadata: Metadata`：必填，关联的元数据对象（就是上面定义的 `Metadata` 类的实例）。
- `@classmethod`：声明类方法。**类方法 vs 实例方法**：类方法第一个参数是类本身（`cls`），不需要实例化即可调用，常用作"工厂方法"（提供另一种创建实例的方式）。
- `def from_text(cls, text, source, doc_id=None, tag=None, page=None, sheet=None, bbox=None, chunk_index=0) -> Document:`：工厂方法，参数全解释：

  | 参数名 | 类型 | 默认值 | 含义 |
  |---|---|---|---|
  | `cls` | `type` | （自动传入） | 类本身，用于实例化 |
  | `text` | `str` | （必填） | 文档全文 |
  | `source` | `str` | （必填） | 原始文件路径 |
  | `doc_id` | `str \| None` | `None` | 文档 ID，不传则自动生成 |
  | `tag` | `list[str] \| None` | `None` | 标签列表 |
  | `page` | `int \| None` | `None` | 页码 |
  | `sheet` | `str \| None` | `None` | sheet 名 |
  | `bbox` | `dict \| None` | `None` | 边界框 |
  | `chunk_index` | `int` | `0` | 块序号 |

- `resolved_id = doc_id or str(uuid4())`：如果传了 `doc_id` 就用它，否则生成新的 UUID。**`or` 的用法**：Python 中 `None or X` 返回 `X`，`"abc" or X` 返回 `"abc"`，所以这是"有则用，无则生成"的简写。
- `metadata = Metadata(...)`：构造 `Metadata` 实例。
- `tag=list(tag) if tag else []`：如果传了 `tag` 就复制一份（避免外部修改影响内部），否则空列表。**为什么 `list(tag)`**：复制列表，避免引用共享。
- `return cls(id=resolved_id, text=text, metadata=metadata)`：用 `cls(...)` 实例化并返回。**为什么用 `cls` 不用 `Document(...)`**：如果有人继承 `Document`，`cls` 会自动是子类，工厂方法仍能正常工作。

**返回值**：返回一个 `Document` 实例，包含 `id`、`text`、`metadata` 三个字段。

最后看 `Chunk` 类：

```python
class Chunk(BaseModel):
    """A chunk of a document with optional embedding vector and relevance score."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    vector: list[float] | None = None
    metadata: Metadata
    score: float | None = None

    def snippet(self, max_chars: int = 200) -> str:
        """Return a truncated preview of the chunk text.

        Args:
            max_chars: Maximum number of characters to keep.

        Returns:
            The original text if shorter than ``max_chars``, otherwise a
            truncated copy terminated with an ellipsis character.
        """
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars].rstrip() + "\u2026"
```

**逐行解释**：

- `class Chunk(BaseModel):`：定义 `Chunk` 类。**Chunk 是什么**：把长文档切成的小段，每段就是一个 Chunk。
- `id: str = Field(default_factory=lambda: str(uuid4()))`：chunk 唯一 ID，自动生成。
- `text: str`：必填，该块的文本内容。
- `vector: list[float] | None = None`：该块的嵌入向量（1024 个 float），默认 `None`。**为什么可空**：刚切完块时还没算向量，要等嵌入器处理后才填上。
- `metadata: Metadata`：必填，复用 `Metadata` 类。**为什么复用**：chunk 也有 source、page 等信息，和文档共用一套元数据结构最简单。
- `score: float | None = None`：检索相关性分数，默认 `None`。**什么时候填**：检索器找到这个 chunk 时给它打分，重排器会更新这个分。
- `def snippet(self, max_chars: int = 200) -> str:`：实例方法，返回文本预览。参数：

  | 参数名 | 类型 | 默认值 | 含义 |
  |---|---|---|---|
  | `self` | `Chunk` | （自动传入） | 实例本身 |
  | `max_chars` | `int` | `200` | 最多保留多少字符 |

- `if len(self.text) <= max_chars: return self.text`：如果文本不超过 `max_chars`，原样返回。
- `return self.text[:max_chars].rstrip() + "\u2026"`：否则切片前 200 字，去掉末尾空白，加省略号 `…`（`\u2026` 是 Unicode 省略号字符）。**为什么用 `rstrip()`**：避免出现 `"文本 …"` 这种末尾带空格的丑陋显示。

**Chunk 字段表**：

| 字段名 | 类型 | 默认值 | 含义 | 如果没有这个字段会怎样 |
|---|---|---|---|---|
| `id` | `str` | 自动生成 | chunk 唯一 ID | 无法在向量库中唯一标识 |
| `text` | `str` | （必填） | 块文本 | 没有内容可检索 |
| `vector` | `list[float] \| None` | `None` | 嵌入向量 | 无法做向量相似度检索 |
| `metadata` | `Metadata` | （必填） | 元数据 | 无法溯源、过滤、排序 |
| `score` | `float \| None` | `None` | 相关性分数 | 无法按相关度排序选最佳 |

### 3.3 数据变化示例

我们用一份"产品手册"作为贯穿全文的例子。假设有一份 `manual.txt` 文件：

```
智能水杯 X1 产品手册

第一章 产品规格
电池容量：2000mAh，续航 30 天。
容量：450ml，材质：316 不锈钢。

第二章 使用说明
长按电源键 3 秒开机，短按切换温度档位。
```

#### 解析成 Document 对象后的 JSON

用 `TxtParser` 解析后，得到的 `Document` 对象序列化成 JSON 长这样：

```json
{
  "id": "a3f9c2e1b4d0",
  "text": "智能水杯 X1 产品手册\n\n第一章 产品规格\n电池容量：2000mAh，续航 30 天。\n容量：450ml，材质：316 不锈钢。\n\n第二章 使用说明\n长按电源键 3 秒开机，短按切换温度档位。",
  "metadata": {
    "source": "manual.txt",
    "page": null,
    "sheet": null,
    "tag": [],
    "created_at": "2026-07-29T08:30:00.000000Z",
    "doc_id": "a3f9c2e1b4d0",
    "chunk_index": 0,
    "bbox": null
  }
}
```

注意：`id` 和 `doc_id` 相同（都是 `a3f9c2e1b4d0`），因为这是文档本身（还没切块）；`page` 是 `null`（TXT 没有页码概念）；`chunk_index` 是 `0`。

#### 切成 Chunk 后的 JSON

经过分块器切成两块后，得到的 `Chunk` 对象列表（这里还没算向量，所以 `vector` 是 `null`）：

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "智能水杯 X1 产品手册\n\n第一章 产品规格\n电池容量：2000mAh，续航 30 天。\n容量：450ml，材质：316 不锈钢。",
    "vector": null,
    "metadata": {
      "source": "manual.txt",
      "page": null,
      "sheet": null,
      "tag": [],
      "created_at": "2026-07-29T08:30:01.000000Z",
      "doc_id": "a3f9c2e1b4d0",
      "chunk_index": 0,
      "bbox": null
    },
    "score": null
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "text": "第二章 使用说明\n长按电源键 3 秒开机，短按切换温度档位。",
    "vector": null,
    "metadata": {
      "source": "manual.txt",
      "page": null,
      "sheet": null,
      "tag": [],
      "created_at": "2026-07-29T08:30:01.000000Z",
      "doc_id": "a3f9c2e1b4d0",
      "chunk_index": 1,
      "bbox": null
    },
    "score": null
  }
]
```

注意两个关键点：

1. 两个 chunk 的 `metadata.doc_id` 都是 `a3f9c2e1b4d0`（指向同一个父文档），但 `chunk_index` 不同（0 和 1）。
2. 两个 chunk 的 `id` 是各自独立的新 UUID（chunk 自己的唯一标识）。

经过嵌入器处理后，`vector` 字段会被填上 1024 个 float（比如 `[0.0123, -0.0456, ...]`）。经过检索器后，`score` 会被填上相关度分数（比如 `0.87`）。

---

## 第 4 章 文档解析器——从文件到文本

### 4.1 解析器的工作原理

解析器的职责很简单：**输入一个文件路径，输出一个或多个 `Document` 对象**。

不同文件格式需要不同的解析库：

| 文件格式 | 解析库 | 输出 Document 数量 |
|---|---|---|
| `.txt` | Python 内置 `read_text` | 1 个（整文件） |
| `.md` | Python 内置 `read_text` | 1 个（整文件） |
| `.html` | BeautifulSoup | 1 个（整文件） |
| `.pdf` | pypdf + RapidOCR | N 个（每页一个） |
| `.docx` | python-docx | 1 个（整文件） |
| `.xlsx` | openpyxl | N 个（每个 sheet 一个） |

**工厂模式：为什么不直接 if-else 而要用工厂**

如果不用工厂，调用方代码会是这样：

```python
# 烂代码示例
if file_path.suffix == ".pdf":
    parser = PdfParser()
elif file_path.suffix == ".docx":
    parser = DocxParser()
# ... 一堆 elif
docs = parser.parse(file_path)
```

问题：

1. 每个调用点都要重复这段 if-else。
2. 加新格式要改所有调用点。
3. 调用方必须知道所有具体解析器类名。

**工厂模式**：把"根据扩展名选哪个解析器"的逻辑集中到 `factory.py` 一个地方。调用方只要：

```python
parser = get_parser(file_path)  # 工厂自动选对
docs = parser.parse(file_path)
```

加新格式只改 `factory.py` 一处。**打比方**：工厂就像快递公司的分拣中心，你把包裹（文件）丢进去，它根据标签（扩展名）自动送到对应流水线（解析器）。

### 4.2 base.py 逐行精读

下面是 `app/ingest/parsers/base.py` 的完整代码：

```python
"""Parser abstract base class and shared utilities.

All concrete parsers implement the :class:`Parser` ABC. The shared helpers
``_make_doc_id`` and ``_now_utc`` provide deterministic identifiers and
timestamps used across the ingestion stage.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from app.models.document import Document


class Parser(ABC):
    """Abstract base class for all document parsers.

    A parser converts a single file on disk into one or more
    :class:`~app.models.document.Document` instances. Subclasses must implement
    :meth:`parse`.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> list[Document]:
        """Parse a file into a list of Documents.

        Args:
            file_path: Path to the file to parse.

        Returns:
            A list of :class:`Document` instances extracted from the file.
        """
        raise NotImplementedError


def _make_doc_id(file_path: Path, page_idx: int | None = None) -> str:
    """Build a deterministic 12-char hex identifier from a file and page index.

    The identifier is derived from the SHA-1 hash of the file's binary content
    combined with the optional page index. When ``page_idx`` is ``None`` the
    returned id is file-level (shared by every page/sheet of the file); when
    provided, the id becomes unique to that page/sheet.

    Args:
        file_path: Path to the source file.
        page_idx: Optional page/sheet index used to disambiguate sub-documents.

    Returns:
        The first 12 characters of the SHA-1 hex digest.
    """
    hasher = hashlib.sha1()
    try:
        with file_path.open("rb") as fh:
            hasher.update(fh.read())
    except OSError:
        # Fall back to hashing the path string if the file cannot be read.
        hasher.update(str(file_path).encode("utf-8"))
    if page_idx is not None:
        hasher.update(f":p{page_idx}".encode("utf-8"))
    return hasher.hexdigest()[:12]


def _now_utc() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        A timezone-aware :class:`~datetime.datetime` in UTC.
    """
    return datetime.now(timezone.utc)
```

**逐行解释**：

- 第 1-6 行：模块 docstring，说明这是"解析器抽象基类和共享工具"。
- 第 7 行 `from __future__ import annotations`：延迟注解求值。
- 第 9 行 `import hashlib`：导入哈希库，用来算 SHA-1。
- 第 10 行 `from abc import ABC, abstractmethod`：
  - **`ABC` 是什么**：Abstract Base Class（抽象基类）。继承 `ABC` 的类不能直接实例化，必须由子类实现所有 `@abstractmethod` 标记的方法后才能实例化。
  - **`abstractmethod` 是什么**：装饰器，标记"子类必须实现这个方法"。
  - **为什么需要抽象基类**：强制所有解析器子类都实现 `parse` 方法。如果某个子类忘了实现，实例化时会直接报错，而不是运行到一半才发现。
- 第 11 行 `from datetime import datetime, timezone`：导入日期时间和时区。
- 第 12 行 `from pathlib import Path`：导入 `Path` 类。**为什么用 Path 不用字符串**：`Path` 提供跨平台的路径操作（`/` 拼接、`.suffix` 取扩展名等），比字符串方便且安全。
- 第 14 行 `from app.models.document import Document`：导入 `Document` 类，作为 `parse` 的返回类型。

```python
class Parser(ABC):
    """Abstract base class for all document parsers.
    ...
    """

    @abstractmethod
    def parse(self, file_path: Path) -> list[Document]:
        """..."""
        raise NotImplementedError
```

- `class Parser(ABC):`：定义抽象基类 `Parser`，继承 `ABC`。
- `@abstractmethod`：标记 `parse` 为抽象方法。
- `def parse(self, file_path: Path) -> list[Document]:`：方法签名，参数全解释：

  | 参数名 | 类型 | 含义 |
  |---|---|---|
  | `self` | `Parser` | 实例本身 |
  | `file_path` | `Path` | 要解析的文件路径 |

- 返回值 `list[Document]`：返回 `Document` 列表。**为什么是列表**：PDF 每页一个 Document，Excel 每个 sheet 一个 Document，所以返回列表更通用。
- `raise NotImplementedError`：方法体只抛异常。**为什么**：抽象方法本来就不该被调用，子类必须覆盖。这里写 `raise` 是防御性编程——如果子类没覆盖又意外调用了，会立即报错而不是返回 `None`。

```python
def _make_doc_id(file_path: Path, page_idx: int | None = None) -> str:
    """..."""
    hasher = hashlib.sha1()
    try:
        with file_path.open("rb") as fh:
            hasher.update(fh.read())
    except OSError:
        # Fall back to hashing the path string if the file cannot be read.
        hasher.update(str(file_path).encode("utf-8"))
    if page_idx is not None:
        hasher.update(f":p{page_idx}".encode("utf-8"))
    return hasher.hexdigest()[:12]
```

- `def _make_doc_id(file_path: Path, page_idx: int | None = None) -> str:`：函数名前缀 `_` 表示"内部函数"（约定不对外暴露）。参数：

  | 参数名 | 类型 | 默认值 | 含义 |
  |---|---|---|---|
  | `file_path` | `Path` | （必填） | 文件路径 |
  | `page_idx` | `int \| None` | `None` | 页/sheet 索引，`None` 表示文件级 |

- 返回值 `str`：12 字符的十六进制字符串（SHA-1 的前 12 位）。
- `hasher = hashlib.sha1()`：创建 SHA-1 哈希对象。**为什么用 SHA-1 不用 uuid**：
  - **确定性**：SHA-1 是"相同输入得到相同输出"。同一个文件每次算出的 ID 都一样，重复解析不会产生重复数据。
  - **uuid4 是随机的**：每次生成都不同，没法去重。
  - **SHA-1 已经够安全**：这里不是用来做加密，只是生成 ID，碰撞概率极低（12 字符 = 48 位，碰撞概率约 1/2^24）。
- `try: with file_path.open("rb") as fh: hasher.update(fh.read())`：以二进制读模式（`"rb"`）打开文件，把全部内容喂给哈希器。**为什么二进制**：避免编码问题，PDF/DOCX 都是二进制文件。
- `except OSError: hasher.update(str(file_path).encode("utf-8"))`：如果文件读不了（比如文件已删除、无权限），退而求其次哈希文件路径字符串。**为什么这么设计**：保证函数永远不抛异常，调用方不用 try。
- `if page_idx is not None: hasher.update(f":p{page_idx}".encode("utf-8"))`：如果传了页码，把 `:p1`、`:p2` 这种字符串也喂给哈希器。**为什么**：让同一文件不同页产生不同 ID。
- `return hasher.hexdigest()[:12]`：返回十六进制摘要的前 12 字符。**为什么 12 位**：够用（48 位空间，碰撞概率极低），又短好读。

```python
def _now_utc() -> datetime:
    """..."""
    return datetime.now(timezone.utc)
```

- `def _now_utc() -> datetime:`：无参函数，返回带时区的当前 UTC 时间。
- `return datetime.now(timezone.utc)`：`datetime.now()` 不传时区会返回本地时间（部署到不同时区的服务器结果不同），传 `timezone.utc` 保证全球一致。**为什么用 UTC**：避免时区混乱，所有时间戳可比较。

### 4.3 txt_parser.py 逐行精读

下面是 `app/ingest/parsers/txt_parser.py` 的完整代码：

```python
"""Plain text parser."""
from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


class TxtParser(Parser):
    """Parse a plain ``.txt`` file into a single :class:`Document`."""

    def parse(self, file_path: Path) -> list[Document]:
        """Read the file as UTF-8 text and wrap it in a single Document.

        Args:
            file_path: Path to the ``.txt`` file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
```

**逐行解释**：

- 第 1 行：模块 docstring。
- 第 2-4 行：导入 `Path`、`Parser` 和 `_make_doc_id`。
- 第 7 行 `from app.models.document import Document, Metadata`：导入数据模型。
- 第 10 行 `class TxtParser(Parser):`：定义 `TxtParser`，继承 `Parser`。
- 第 13 行 `def parse(self, file_path: Path) -> list[Document]:`：实现父类的抽象方法。参数 `file_path` 是 `Path` 类型，返回 `list[Document]`。
- 第 22 行 `text = file_path.read_text(encoding="utf-8", errors="ignore")`：
  - `read_text()`：`Path` 对象的方法，一次性读整个文件为字符串。
  - `encoding="utf-8"`：用 UTF-8 解码。**为什么**：UTF-8 是最通用的编码，中文也能正确读。
  - `errors="ignore"`：遇到无法解码的字节直接跳过。**为什么**：避免一个坏字节就让整个文件读失败。代价是丢失少量字符，但对 RAG 检索影响不大。
- 第 23 行 `doc_id = _make_doc_id(file_path, None)`：调用基类工具函数生成文件级 doc_id（`page_idx=None`）。
- 第 24 行 `metadata = Metadata(source=file_path.name, doc_id=doc_id)`：构造元数据。
  - `source=file_path.name`：`Path.name` 返回文件名（不含目录），比如 `"manual.txt"`。**为什么只取文件名**：完整路径可能含敏感目录信息，文件名足够标识。
  - `doc_id=doc_id`：复用上面生成的 ID。
  - 其他字段（`page`、`sheet`、`tag`、`bbox`）用 `Metadata` 的默认值（`None`、`[]`、`None`）。
- 第 25 行 `return [Document(id=doc_id, text=text, metadata=metadata)]`：返回单元素列表。**为什么是列表**：父类接口规定返回 `list[Document]`，统一接口让下游处理简单。

**用示例文本演示解析结果**：

输入 `manual.txt` 内容：

```
智能水杯 X1 产品手册

第一章 产品规格
电池容量：2000mAh，续航 30 天。
```

解析后得到的 `Document` 对象（伪 JSON 表示）：

```json
{
  "id": "a3f9c2e1b4d0",
  "text": "智能水杯 X1 产品手册\n\n第一章 产品规格\n电池容量：2000mAh，续航 30 天。\n",
  "metadata": {
    "source": "manual.txt",
    "page": null,
    "sheet": null,
    "tag": [],
    "created_at": "2026-07-29T08:30:00Z",
    "doc_id": "a3f9c2e1b4d0",
    "chunk_index": 0,
    "bbox": null
  }
}
```

### 4.4 markdown_parser.py 逐行精读

下面是 `app/ingest/parsers/markdown_parser.py` 的完整代码：

```python
"""Markdown parser."""
from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


class MarkdownParser(Parser):
    """Parse a ``.md`` file into a single :class:`Document`.

    The raw Markdown source is preserved verbatim so downstream structural
    chunkers can still recognise heading and table syntax.
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Read the Markdown file as text and wrap it in a single Document.

        Args:
            file_path: Path to the ``.md`` file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
```

**逐行解释**：

- 整体逻辑和 `txt_parser.py` 几乎一样：读文本 → 生成 doc_id → 构造 metadata → 返回单元素 Document 列表。
- **关键差异**在 docstring 第 13-15 行：**"原始 Markdown 源码原样保留，让下游结构化分块器还能识别标题和表格语法"**。
- **为什么不把 Markdown 转成纯文本/HTML**：因为后面的 `structural` 分块器要靠 `##`、`| --- |` 这种 Markdown 语法来切分（按标题切、按表格切）。如果在这里就转成纯文本，结构信息就丢了。

**讲一下 markdown 库**：项目依赖里有 `markdown>=3.6`，但这个解析器并没用它。`markdown` 库是用来把 Markdown **转 HTML** 的（用于网页渲染）。本解析器刻意不转，保留原样。`markdown` 库在项目其他地方（比如 UI 渲染回答时）可能用到。

### 4.5 html_parser.py 逐行精读

下面是 `app/ingest/parsers/html_parser.py` 的完整代码：

```python
"""HTML parser using BeautifulSoup."""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


class HtmlParser(Parser):
    """Parse an ``.html``/``.htm`` file into a single :class:`Document`.

    The parser strips ``<script>`` and ``<style>`` elements and extracts the
    visible text of the ``<body>`` (falling back to the whole document when no
    ``<body>`` tag is present).
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Extract visible body text from the HTML file.

        Args:
            file_path: Path to the HTML file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        container = soup.body if soup.body is not None else soup
        # Remove non-content elements before extracting text.
        for tag in container(["script", "style"]):
            tag.decompose()
        text = container.get_text(separator="\n").strip()
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
```

**逐行解释**：

- 第 6 行 `from bs4 import BeautifulSoup`：
  - **BeautifulSoup 这个库是干什么的**：Python 最流行的 HTML/XML 解析库。它把 HTML 文档解析成一棵树（DOM 树），让你能方便地查找、修改、提取内容。
  - **为什么选它**：比正则表达式鲁棒得多（HTML 嵌套不规则，正则容易出错），且自带容错（即使 HTML 有语法错误也能解析）。
- 第 29 行 `content = file_path.read_text(encoding="utf-8", errors="ignore")`：读 HTML 文件为字符串。
- 第 30 行 `soup = BeautifulSoup(content, "html.parser")`：用 BeautifulSoup 解析 HTML。
  - 第一个参数 `content`：HTML 字符串。
  - 第二个参数 `"html.parser"`：指定用 Python 内置的 HTML 解析器。**为什么用它**：无需装 C 库（`lxml` 要装 C 扩展），跨平台稳定。性能稍差但对解析文档够用。
- 第 31 行 `container = soup.body if soup.body is not None else soup`：如果有 `<body>` 标签，只取 body 内容（去掉 `<head>` 里的元信息）；否则用整个文档。
- 第 33-34 行 `for tag in container(["script", "style"]): tag.decompose()`：
  - `container(["script", "style"])`：找到所有 `<script>` 和 `<style>` 标签。
  - `tag.decompose()`：从 DOM 树中删除这些标签及其内容。**为什么删**：`<script>` 是 JS 代码（不是正文），`<style>` 是 CSS（不是正文），保留它们会污染文本。
- 第 35 行 `text = container.get_text(separator="\n").strip()`：
  - `get_text(separator="\n")`：提取所有文本内容，用 `\n` 分隔不同标签的文本。**为什么用 `\n`**：避免 `<p>段落1</p><p>段落2</p>` 被拼成 `"段落1段落2"`。
  - `.strip()`：去掉首尾空白。
- 后续生成 doc_id、metadata、Document 和 TxtParser 一样。

### 4.6 pdf_parser.py 逐行精读

下面是 `app/ingest/parsers/pdf_parser.py` 的完整代码（这是最复杂的解析器）：

```python
"""PDF parser using pypdf with a RapidOCR fallback for scanned pages.

For each page the parser first attempts direct text extraction with
``pypdf.PdfReader``. When the extracted text is empty or very short the page is
rendered to an image with ``pdf2image`` and passed through ``rapidocr_onnxruntime``.

Both OCR-only dependencies are imported lazily so the module can be imported on
machines where they (or their native backends such as poppler) are unavailable;
in that case the parser logs a warning and returns an empty string for the page
instead of raising.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Module-level OCR singleton. ``None`` = not initialised, ``False`` = unavailable,
# an instance = ready to use.
_OCR_INSTANCE: Any = None


def _get_ocr() -> Any:
    """Return a lazily-initialised :class:`RapidOCR` instance or ``None``.

    The import is performed on first use so the parser module itself can be
    imported without ``rapidocr_onnxruntime`` installed. If the dependency (or
    its ONNX runtime) is missing, a warning is logged and subsequent calls
    short-circuit to ``None``.

    Returns:
        A ``RapidOCR`` instance, or ``None`` if OCR is unavailable.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _OCR_INSTANCE = RapidOCR()
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning(
                "rapidocr_onnxruntime not available; OCR fallback disabled",
                error=str(exc),
            )
            _OCR_INSTANCE = False
    return _OCR_INSTANCE if _OCR_INSTANCE is not False else None


class PdfParser(Parser):
    """Parse a ``.pdf`` file into one :class:`Document` per page.

    Pages whose direct text extraction yields fewer than 10 characters are
    forwarded to the OCR fallback. When OCR is unavailable the page text is
    left empty rather than raising.
    """

    MIN_TEXT_LEN = 10

    def parse(self, file_path: Path) -> list[Document]:
        """Extract text from each PDF page.

        Args:
            file_path: Path to the ``.pdf`` file.

        Returns:
            A list of :class:`Document` instances, one per page (1-indexed).
        """
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        file_doc_id = _make_doc_id(file_path, None)
        docs: list[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < self.MIN_TEXT_LEN:
                ocr_text = self._ocr_page(file_path, page_idx)
                if ocr_text:
                    text = ocr_text
            metadata = Metadata(
                source=file_path.name,
                page=page_idx,
                doc_id=file_doc_id,
            )
            docs.append(
                Document(
                    id=_make_doc_id(file_path, page_idx),
                    text=text,
                    metadata=metadata,
                )
            )
        return docs

    def _ocr_page(self, file_path: Path, page_idx: int) -> str:
        """Render a single page to an image and run OCR over it.

        Args:
            file_path: Path to the source PDF.
            page_idx: 1-indexed page number to render.

        Returns:
            The OCR-extracted text, or an empty string on any failure.
        """
        ocr = _get_ocr()
        if ocr is None:
            return ""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(file_path),
                first_page=page_idx,
                last_page=page_idx,
                dpi=200,
            )
        except Exception as exc:  # pragma: no cover - depends on poppler
            logger.warning(
                "pdf2image rendering failed; skipping OCR for page",
                page=page_idx,
                error=str(exc),
            )
            return ""
        texts: list[str] = []
        for img in images:
            try:
                result, _elapse = ocr(img)
            except Exception as exc:  # pragma: no cover - depends on OCR runtime
                logger.warning(
                    "OCR inference failed for page",
                    page=page_idx,
                    error=str(exc),
                )
                continue
            if result:
                texts.append("\n".join(line[1] for line in result))
        return "\n".join(texts).strip()
```

**逐段解释**：

#### 模块顶部

- 第 1-11 行：模块 docstring，说明"用 pypdf 提文本，扫描页用 RapidOCR 兜底，OCR 依赖懒加载"。
- 第 14 行 `from pathlib import Path`：导入 Path。
- 第 15 行 `from typing import Any`：导入 Any。
- 第 17-19 行：导入基类、数据模型、日志器。
- 第 21 行 `logger = get_logger(__name__)`：创建模块级日志器。`__name__` 是模块名（如 `app.ingest.parsers.pdf_parser`），日志里能看出是哪个模块输出的。
- 第 24-25 行：模块级变量 `_OCR_INSTANCE: Any = None`，OCR 单例。注释说明了三种状态：`None`（未初始化）、`False`（不可用）、实例对象（可用）。

#### `_get_ocr` 函数

```python
def _get_ocr() -> Any:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_INSTANCE = RapidOCR()
        except Exception as exc:
            logger.warning(...)
            _OCR_INSTANCE = False
    return _OCR_INSTANCE if _OCR_INSTANCE is not False else None
```

- `global _OCR_INSTANCE`：声明修改模块级变量（不是创建局部变量）。
- `if _OCR_INSTANCE is None:`：只有"未初始化"时才尝试加载。
- `from rapidocr_onnxruntime import RapidOCR`：**懒加载**——只有在第一次需要 OCR 时才 import。**为什么懒加载**：
  - `rapidocr_onnxruntime` 依赖 ONNX Runtime，可能装不上（缺 C 库）。
  - 如果在模块顶部 import，模块被 import 时就会失败，导致整个解析器模块不可用。
  - 懒加载后，没有 OCR 也能用 PDF 解析（只是扫描页拿不到文本）。
- `_OCR_INSTANCE = RapidOCR()`：实例化 OCR 模型（首次会加载模型文件，较慢）。
- `except Exception as exc:`：捕获所有异常（导入失败、模型加载失败等）。**为什么用 `Exception` 不用具体异常**：依赖环境复杂，可能抛各种异常，宽泛捕获保证不崩。
- `logger.warning(...)`：记录警告日志。
- `_OCR_INSTANCE = False`：标记"OCR 不可用"，下次调用直接短路返回 `None`，不再重试。
- `return _OCR_INSTANCE if _OCR_INSTANCE is not False else None`：如果是 `False` 就返回 `None`，否则返回实例。

#### `PdfParser` 类

- 第 62 行 `MIN_TEXT_LEN = 10`：类常量，判断是否需要 OCR 的阈值。**为什么是 10**：
  - 太小（如 1）：扫描页可能提取出 1-2 个乱码字符，会漏判需要 OCR。
  - 太大（如 100）：正常的短页面（如只有标题的封面页）会被误判为扫描页触发不必要的 OCR。
  - 10 是经验值：能识别"基本没提取到文本"的情况。
- `def parse(self, file_path: Path) -> list[Document]:`：实现父类抽象方法。
- `from pypdf import PdfReader`：**懒加载 pypdf**。虽然不是必需，但保持和 OCR 一致的风格。
  - **pypdf 这个库是干什么的**：纯 Python 的 PDF 解析库。能提取文本、读元数据、合并分割 PDF。**为什么用它**：无需系统级 C 依赖，跨平台稳定。
- `reader = PdfReader(str(file_path))`：创建 PDF 读取器。`str(file_path)` 是因为 pypdf 接受字符串路径。
- `file_doc_id = _make_doc_id(file_path, None)`：生成文件级 doc_id，所有页面共享。
- `docs: list[Document] = []`：累积结果列表。
- `for page_idx, page in enumerate(reader.pages, start=1):`：遍历每一页。
  - `reader.pages`：所有页对象的列表。
  - `enumerate(..., start=1)`：从 1 开始编号（PDF 页码约定从 1 开始）。**为什么从 1**：用户视角"第 1 页"而不是"第 0 页"。
- `text = (page.extract_text() or "").strip()`：提取该页文本。
  - `page.extract_text()`：pypdf 提取文本的方法，可能返回 `None`。
  - `or ""`：`None` 转空字符串，避免 `None.strip()` 报错。
  - `.strip()`：去掉首尾空白。
- `if len(text) < self.MIN_TEXT_LEN:`：如果提取到的文本少于 10 字符，认为是扫描页或图片页。
- `ocr_text = self._ocr_page(file_path, page_idx)`：调用 OCR 方法。
- `if ocr_text: text = ocr_text`：如果 OCR 拿到文本就用它，否则保持原 `text`（可能是空的）。
- `metadata = Metadata(source=file_path.name, page=page_idx, doc_id=file_doc_id)`：构造元数据，**这里 `page=page_idx` 是关键**——记录页码，方便后续定位。
- `docs.append(Document(id=_make_doc_id(file_path, page_idx), text=text, metadata=metadata))`：
  - `id=_make_doc_id(file_path, page_idx)`：用页码生成每页独立的 ID（同一文件不同页 ID 不同）。
  - 注意：返回的 Document 列表是**每页一个 Document**（不是整个 PDF 一个）。

**返回值**：`list[Document]`，长度等于 PDF 页数。每页的 `metadata.doc_id` 相同（指向同一个父 PDF），但 `metadata.page` 不同。

#### `_ocr_page` 方法

- `def _ocr_page(self, file_path: Path, page_idx: int) -> str:`：私有方法，参数 `file_path` 和 `page_idx`（1-indexed）。
- `ocr = _get_ocr()`：获取 OCR 实例。
- `if ocr is None: return ""`：OCR 不可用直接返回空字符串。
- `from pdf2image import convert_from_path`：**懒加载 pdf2image**。
  - **pdf2image 这个库是干什么的**：把 PDF 页面渲染成图片。**依赖 poppler**（系统级工具），可能没装。
- `images = convert_from_path(str(file_path), first_page=page_idx, last_page=page_idx, dpi=200)`：渲染指定页为图片。
  - `first_page` / `last_page`：只渲染这一页，避免渲染整个 PDF。
  - `dpi=200`：分辨率。**为什么 200**：太低（如 72）OCR 识别率差；太高（如 300）渲染慢且文件大。200 是 OCR 的甜点。
- `except Exception as exc: logger.warning(...); return ""`：渲染失败（如 poppler 没装）记日志返回空。
- `texts: list[str] = []`：累积每张图的 OCR 结果。
- `for img in images:`：遍历图片（一般 1 张，但 `convert_from_path` 返回列表）。
- `result, _elapse = ocr(img)`：调用 OCR。
  - `result`：识别结果，是 `[ [bbox, text, score], ... ]` 的列表。
  - `_elapse`：耗时（用不到，加 `_` 前缀表示忽略）。
- `except Exception as exc: logger.warning(...); continue`：单张图 OCR 失败跳过，继续下一张。
- `if result: texts.append("\n".join(line[1] for line in result))`：
  - `line[1]`：每条结果的第 2 个元素是文本（`line[0]` 是 bbox 坐标）。
  - `"\n".join(...)`：每行文本用换行连接。
- `return "\n".join(texts).strip()`：合并所有图片的文本。

### 4.7 docx_parser.py 逐行精读

下面是 `app/ingest/parsers/docx_parser.py` 的完整代码：

```python
"""DOCX parser using python-docx.

Iterates the document body in order, preserving the relative position of
paragraphs and tables. Tables are serialised as Markdown tables so downstream
structural chunkers can keep them intact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from docx.document import Document as _DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


def _iter_block_items(doc: _DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in the order they appear in the body.

    Args:
        doc: A python-docx :class:`Document` instance.

    Yields:
        :class:`Paragraph` or :class:`Table` objects in document order.
    """
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_markdown(table: Table) -> str:
    """Convert a python-docx table into a Markdown table string.

    Args:
        table: A python-docx :class:`Table` instance.

    Returns:
        A Markdown table representation (header row + separator + body rows).
    """
    rows = list(table.rows)
    if not rows:
        return ""
    ncols = len(rows[0].cells)
    lines: list[str] = []
    lines.append("| " + " | ".join(cell.text.strip() for cell in rows[0].cells) + " |")
    lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for row in rows[1:]:
        cells = list(row.cells)
        # Pad / trim to match header column count.
        if len(cells) < ncols:
            cells = cells + [cells[-1]] * (ncols - len(cells))
        lines.append("| " + " | ".join(cell.text.strip() for cell in cells[:ncols]) + " |")
    return "\n".join(lines)


class DocxParser(Parser):
    """Parse a ``.docx`` file into a single :class:`Document`.

    The text of all paragraphs and tables (rendered as Markdown) is concatenated
    in document order. ``metadata.page`` is ``None`` because DOCX is not
    paginated in a reliable way.
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Extract paragraph and table text from a DOCX file.

        Args:
            file_path: Path to the ``.docx`` file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        from docx import Document as _OpenDoc

        docx_doc = _OpenDoc(str(file_path))
        parts: list[str] = []
        for block in _iter_block_items(docx_doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    parts.append(text)
            elif isinstance(block, Table):
                md = _table_to_markdown(block)
                if md:
                    parts.append(md)
        text = "\n\n".join(parts)
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, page=None, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
```

**逐段解释**：

#### 导入

- 第 12-16 行：从 `python-docx` 导入多个类型：
  - **python-docx 这个库是干什么的**：解析 `.docx` 文件的 Python 库。能读段落、表格、样式等。
  - `Document as _DocxDocument`：docx 的 Document 类（别名避免和项目自己的 Document 冲突）。
  - `CT_Tbl` / `CT_P`：底层 XML 元素类（表格元素 / 段落元素）。
  - `Table` / `Paragraph`：高层封装类。
- 第 19-20 行：导入基类和数据模型。

#### `_iter_block_items` 函数

```python
def _iter_block_items(doc: _DocxDocument) -> Iterator[Paragraph | Table]:
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)
```

- `def _iter_block_items(doc) -> Iterator[Paragraph | Table]:`：返回一个迭代器，依次产出段落或表格。参数 `doc` 是 python-docx 的 Document 对象。
- `parent_elm = doc.element.body`：拿到文档体的 XML 元素。
- `for child in parent_elm.iterchildren():`：遍历 body 的所有直接子元素（按文档顺序）。
- `if isinstance(child, CT_P): yield Paragraph(child, doc)`：如果是段落元素，包装成 `Paragraph` 对象并产出。
- `elif isinstance(child, CT_Tbl): yield Table(child, doc)`：如果是表格元素，包装成 `Table` 对象并产出。

**为什么遍历 body 元素而不是直接用 `doc.paragraphs`**：

- `doc.paragraphs` 只返回段落，**不包含表格**，且**不保证顺序**（在某些情况下会漏掉表格之间的段落）。
- 直接遍历 body XML 能拿到所有元素（段落 + 表格）并保持原始顺序。这对保持文档结构至关重要（比如"段落A → 表格B → 段落C"的顺序不能乱）。

#### `_table_to_markdown` 函数

```python
def _table_to_markdown(table: Table) -> str:
    rows = list(table.rows)
    if not rows:
        return ""
    ncols = len(rows[0].cells)
    lines: list[str] = []
    lines.append("| " + " | ".join(cell.text.strip() for cell in rows[0].cells) + " |")
    lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for row in rows[1:]:
        cells = list(row.cells)
        if len(cells) < ncols:
            cells = cells + [cells[-1]] * (ncols - len(cells))
        lines.append("| " + " | ".join(cell.text.strip() for cell in cells[:ncols]) + " |")
    return "\n".join(lines)
```

- `def _table_to_markdown(table: Table) -> str:`：把表格转成 Markdown 表格字符串。
- `rows = list(table.rows)`：拿到所有行。
- `if not rows: return ""`：空表返回空字符串。
- `ncols = len(rows[0].cells)`：用第一行的单元格数作为列数。
- `lines.append("| " + " | ".join(cell.text.strip() for cell in rows[0].cells) + " |")`：构造表头行，比如 `| 列1 | 列2 | 列3 |`。
- `lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")`：构造分隔行 `| --- | --- | --- |`。**Markdown 表格语法要求**表头和正文之间必须有这一行。
- `for row in rows[1:]:`：遍历剩余行（数据行）。
- `cells = list(row.cells)`：拿到该行的单元格列表。
- `if len(cells) < ncols: cells = cells + [cells[-1]] * (ncols - len(cells))`：如果该行单元格数少于列数，用最后一个单元格填充。**为什么这么处理**：Word 表格可能有合并单元格，导致某些行 cells 数量不一致。补齐保证 Markdown 表格不变形。
- `lines.append("| " + " | ".join(cell.text.strip() for cell in cells[:ncols]) + " |")`：构造数据行。`cells[:ncols]` 确保不超过列数（多了截断）。
- `return "\n".join(lines)`：用换行连接所有行。

**表格转 Markdown 算法总结**：表头 → 分隔行 → 数据行，每行用 `|` 分隔单元格，单元格内容用 `.strip()` 去空白。

#### `DocxParser` 类

```python
class DocxParser(Parser):
    def parse(self, file_path: Path) -> list[Document]:
        from docx import Document as _OpenDoc
        docx_doc = _OpenDoc(str(file_path))
        parts: list[str] = []
        for block in _iter_block_items(docx_doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    parts.append(text)
            elif isinstance(block, Table):
                md = _table_to_markdown(block)
                if md:
                    parts.append(md)
        text = "\n\n".join(parts)
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, page=None, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
```

- `from docx import Document as _OpenDoc`：懒加载 python-docx。
- `docx_doc = _OpenDoc(str(file_path))`：打开 docx 文件。
- `parts: list[str] = []`：累积文本片段。
- `for block in _iter_block_items(docx_doc):`：按文档顺序遍历段落和表格。
- `if isinstance(block, Paragraph):`：如果是段落：
  - `text = block.text.strip()`：取段落文本并去空白。
  - `if text: parts.append(text)`：非空才加入。**为什么检查空**：Word 里可能有空段落（只是空行），跳过避免污染。
- `elif isinstance(block, Table):`：如果是表格：
  - `md = _table_to_markdown(block)`：转 Markdown。
  - `if md: parts.append(md)`：非空才加入。
- `text = "\n\n".join(parts)`：用双换行连接所有片段。**为什么双换行**：Markdown 用双换行分隔段落，保持可读性。
- `metadata = Metadata(source=file_path.name, page=None, doc_id=doc_id)`：注意 `page=None`——**DOCX 不分页**。docstring 说"DOCX is not paginated in a reliable way"，因为 Word 分页是渲染时才决定的（取决于字体、打印机），文档结构里没有可靠的页码。
- 返回单元素列表（整个 docx 一个 Document）。

### 4.8 xlsx_parser.py 逐行精读

下面是 `app/ingest/parsers/xlsx_parser.py` 的完整代码：

```python
"""XLSX parser using openpyxl.

Each worksheet becomes its own :class:`Document`. The sheet content is
serialised as a Markdown table so downstream structural chunkers can treat it as
an atomic block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


def _rows_to_markdown(rows: list[tuple[Any, ...]]) -> str:
    """Serialise a sequence of worksheet rows into a Markdown table.

    Args:
        rows: A list of row tuples as returned by ``ws.iter_rows(values_only=True)``.

    Returns:
        A Markdown table string. Empty input yields an empty string.
    """
    if not rows:
        return ""
    ncols = max(len(row) for row in rows)
    if ncols == 0:
        return ""
    lines: list[str] = []
    header = list(rows[0]) + [None] * (ncols - len(rows[0]))
    lines.append("| " + " | ".join(_cell_str(c) for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for row in rows[1:]:
        cells = list(row) + [None] * (ncols - len(row))
        lines.append("| " + " | ".join(_cell_str(c) for c in cells[:ncols]) + " |")
    return "\n".join(lines)


def _cell_str(value: Any) -> str:
    """Render a cell value as a concise string.

    Args:
        value: The raw cell value (may be ``None``).

    Returns:
        The string representation of the value, or an empty string for ``None``.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class XlsxParser(Parser):
    """Parse an ``.xlsx``/``.xls`` file into one :class:`Document` per sheet."""

    def parse(self, file_path: Path) -> list[Document]:
        """Extract each worksheet as a Markdown-table Document.

        Args:
            file_path: Path to the spreadsheet file.

        Returns:
            A list of :class:`Document` instances, one per worksheet.
        """
        from openpyxl import load_workbook

        wb = load_workbook(str(file_path), data_only=True, read_only=True)
        doc_id = _make_doc_id(file_path, None)
        docs: list[Document] = []
        try:
            for sheet_idx, sheet_name in enumerate(wb.sheetnames):
                ws = wb[sheet_name]
                rows = list(ws.iter_rows(values_only=True))
                text = _rows_to_markdown(rows)
                metadata = Metadata(
                    source=file_path.name,
                    page=None,
                    sheet=sheet_name,
                    doc_id=doc_id,
                )
                docs.append(
                    Document(
                        id=_make_doc_id(file_path, sheet_idx),
                        text=text,
                        metadata=metadata,
                    )
                )
        finally:
            wb.close()
        return docs
```

**逐段解释**：

#### `_rows_to_markdown` 函数

- `def _rows_to_markdown(rows: list[tuple[Any, ...]]) -> str:`：把行数据转成 Markdown 表格。参数 `rows` 是元组列表（每个元组是一行）。
- `if not rows: return ""`：空数据返回空。
- `ncols = max(len(row) for row in rows)`：用所有行中最大列数作为表格列数。**为什么用 max**：Excel 行长度可能不一致（前几行有 5 列，后面只有 3 列），取最大值保证不丢数据。
- `if ncols == 0: return ""`：所有行都空，返回空。
- `header = list(rows[0]) + [None] * (ncols - len(rows[0]))`：第一行作为表头，如果列数不足用 `None` 补齐。
- `lines.append("| " + " | ".join(_cell_str(c) for c in header) + " |")`：构造表头行。
- `lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")`：构造分隔行。
- `for row in rows[1:]:`：遍历数据行。
- `cells = list(row) + [None] * (ncols - len(row))`：补齐列数。
- `lines.append("| " + " | ".join(_cell_str(c) for c in cells[:ncols]) + " |")`：构造数据行，截断到 `ncols` 防止超长。
- `return "\n".join(lines)`：连接所有行。

#### `_cell_str` 函数

- `def _cell_str(value: Any) -> str:`：把单元格值转成字符串。
- `if value is None: return ""`：空值返回空字符串。
- `if isinstance(value, float) and value.is_integer(): return str(int(value))`：**关键**——如果是浮点数但实际是整数（如 `2000.0`），转成 int 再转字符串（`"2000"` 而不是 `"2000.0"`）。**为什么这么处理**：Excel 里数字常被读成浮点（openpyxl 行为），但显示 `"2000.0"` 不符合习惯。
- `return str(value)`：其他类型直接 `str()`。

#### `XlsxParser` 类

- 第 70 行 `from openpyxl import load_workbook`：懒加载 openpyxl。
  - **openpyxl 这个库是干什么的**：解析 `.xlsx` 文件的 Python 库。能读单元格、公式、样式等。
- 第 72 行 `wb = load_workbook(str(file_path), data_only=True, read_only=True)`：打开工作簿。参数：
  - `data_only=True`：只读数据值（不读公式）。**为什么**：RAG 只关心显示的值，不关心公式本身。
  - `read_only=True`：只读模式。**为什么**：不修改文件，只读模式内存占用更低、速度更快（适合大文件）。
- `doc_id = _make_doc_id(file_path, None)`：文件级 doc_id。
- `docs: list[Document] = []`：累积结果。
- `try: ... finally: wb.close()`：用 try/finally 确保工作簿一定关闭。**为什么**：openpyxl 在 read_only 模式下会持有文件句柄，不关闭会泄漏。
- `for sheet_idx, sheet_name in enumerate(wb.sheetnames):`：遍历所有 sheet 名。
- `ws = wb[sheet_name]`：拿到工作表对象。
- `rows = list(ws.iter_rows(values_only=True))`：读取所有行。`values_only=True` 表示只返回值（不返回 Cell 对象）。
- `text = _rows_to_markdown(rows)`：转 Markdown 表格。
- `metadata = Metadata(source=file_path.name, page=None, sheet=sheet_name, doc_id=doc_id)`：**关键**——`sheet=sheet_name` 记录 sheet 名，方便后续知道这段文本来自哪个工作表。
- `docs.append(Document(id=_make_doc_id(file_path, sheet_idx), text=text, metadata=metadata))`：每个 sheet 一个 Document，ID 用 sheet 索引区分。

**为什么每个 sheet 一个 Document**：

- 不同 sheet 内容主题可能完全不同（比如"销售数据"和"员工名单"）。
- 如果合并成一个 Document，切块时可能把两个 sheet 的内容混在一起，检索精度下降。
- 分开存让检索时能按 sheet 过滤，且每个 sheet 的元数据（`sheet` 字段）独立。

### 4.9 factory.py 逐行精读

下面是 `app/ingest/parsers/factory.py` 的完整代码：

```python
"""Parser factory dispatching on file extension."""
from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.base import Parser
from app.ingest.parsers.docx_parser import DocxParser
from app.ingest.parsers.html_parser import HtmlParser
from app.ingest.parsers.markdown_parser import MarkdownParser
from app.ingest.parsers.pdf_parser import PdfParser
from app.ingest.parsers.txt_parser import TxtParser
from app.ingest.parsers.xlsx_parser import XlsxParser

#: Mapping of supported file extensions (lower-case, without dot) to parsers.
_PARSER_REGISTRY: dict[str, type[Parser]] = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "xlsx": XlsxParser,
    "xls": XlsxParser,
    "md": MarkdownParser,
    "markdown": MarkdownParser,
    "html": HtmlParser,
    "htm": HtmlParser,
    "txt": TxtParser,
}


def get_parser(file_path: Path) -> Parser:
    """Return a :class:`Parser` instance appropriate for the file extension.

    Args:
        file_path: Path to the file whose extension determines the parser.

    Returns:
        A concrete :class:`Parser` instance.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.suffix.lower().lstrip(".")
    parser_cls = _PARSER_REGISTRY.get(ext)
    if parser_cls is None:
        raise ValueError(
            f"Unsupported file extension: '{file_path.suffix}' for {file_path}"
        )
    return parser_cls()
```

**逐行解释**：

- 第 1 行：模块 docstring，"按文件扩展名分发的解析器工厂"。
- 第 4 行 `from pathlib import Path`：导入 Path。
- 第 6-12 行：导入基类 `Parser` 和所有具体解析器。注意这里**导入所有具体类**——因为工厂需要知道所有可选解析器。**会不会有性能问题**：不会，因为导入是懒的（模块第一次被 import 时才执行），且只会执行一次。
- 第 15 行 `_PARSER_REGISTRY: dict[str, type[Parser]] = {...}`：定义"扩展名 → 解析器类"的映射字典。
  - 键：扩展名字符串（小写、无点），如 `"pdf"`、`"docx"`。
  - 值：解析器**类**本身（不是实例），如 `PdfParser`、`DocxParser`。`type[Parser]` 表示"Parser 的子类"。
  - **为什么要存类不存实例**：实例化解析器几乎没有开销（它们没有状态），每次按需实例化更干净，避免共享实例的潜在状态污染。
- 字典内容：支持 9 种扩展名，映射到 6 个解析器类（`xls` 复用 `XlsxParser`，`markdown` 复用 `MarkdownParser`，`htm` 复用 `HtmlParser`）。
- 第 28 行 `def get_parser(file_path: Path) -> Parser:`：工厂函数，参数 `file_path`，返回 `Parser` 实例。
- 第 40 行 `ext = file_path.suffix.lower().lstrip(".")`：
  - `file_path.suffix`：取扩展名（含点），如 `".PDF"`。
  - `.lower()`：转小写，如 `".pdf"`。**为什么**：用户可能上传 `.PDF` 或 `.pdf`，统一处理。
  - `.lstrip(".")`：去掉前导点，如 `"pdf"`。**为什么**：字典键没有点。
- 第 41 行 `parser_cls = _PARSER_REGISTRY.get(ext)`：从字典查找对应的解析器类。`.get()` 找不到返回 `None`（不会抛 KeyError）。
- 第 42-45 行 `if parser_cls is None: raise ValueError(...)`：找不到对应解析器就抛 `ValueError`。**为什么抛 ValueError**：这是"用户输入错误"（上传了不支持的格式），应该明确告诉用户，而不是静默跳过。错误信息里包含原始扩展名和文件路径，方便排查。
- 第 46 行 `return parser_cls()`：实例化并返回解析器。

**工厂模式的好处**：

1. **集中管理**：所有"扩展名 → 解析器"的映射在一个地方，加新格式只改这一个文件。
2. **解耦**：调用方不需要知道具体有哪些解析器，只要调用 `get_parser()`。
3. **可测试**：测试时可以 mock 字典或替换实现。
4. **可扩展**：未来加 PPT 解析器，只要新建 `ppt_parser.py`，在字典里加一行 `"ppt": PptParser` 即可，调用方代码完全不用改。

**扩展名分发逻辑总结**：

```
file_path = Path("manual.PDF")
  ↓ suffix.lower()  → ".pdf"
  ↓ lstrip(".")     → "pdf"
  ↓ _PARSER_REGISTRY["pdf"] → PdfParser
  ↓ PdfParser()     → 实例
```

---

## 全文总结

本教程第一部分覆盖了 kb-rag 项目的四大基础模块：

1. **架构原理**：用图书馆比喻讲清了 RAG 的两阶段（摄入 + 查询）流程，以及项目为什么分成解析器/清洗器/分块器/嵌入器/存储器/检索器/重排器/生成器/管线/API/UI 这 11 层。给出了完整目录树和每个目录的存在理由。

2. **配置系统**：逐行精读了 `config.py`（pydantic-settings 库的作用、`Settings` 类的 30+ 个字段、7 个 `field_validator` 校验器、`lru_cache` 单例模式）；逐行解释了 `.env.example` 的所有配置项（特别详解了 LLM 相关参数）；逐个说明了 `pyproject.toml` 里 30 个依赖包的用途和选型理由。

3. **数据模型**：逐行精读了 `document.py`，用表格详解了 `Metadata`（8 字段）、`Document`（3 字段 + 工厂方法 `from_text`）、`Chunk`（5 字段 + `snippet` 方法）每个字段的类型、默认值、含义和缺失影响。用"产品手册"示例展示了从文本到 Document 再到 Chunk 的 JSON 变化。

4. **文档解析器**：逐行精读了 `base.py`（抽象基类 + `_make_doc_id` SHA-1 工具 + `_now_utc`）、`txt_parser.py`、`markdown_parser.py`、`html_parser.py`（BeautifulSoup）、`pdf_parser.py`（pypdf + RapidOCR 懒加载 + pdf2image）、`docx_parser.py`（python-docx + 表格转 Markdown）、`xlsx_parser.py`（openpyxl + 每 sheet 一个 Document）、`factory.py`（工厂模式 + 扩展名分发）。

贯穿全文的"智能水杯 X1 产品手册"示例展示了每一步数据形态，所有代码均来自项目真实文件，未作编造。

下一部分将讲解清洗器、分块器、嵌入器、存储器等模块。
