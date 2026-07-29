# kb-rag 新手教程 · 第四部分：对外服务 → UI → 部署 → 监控 → 调优

> 本教程面向只会 Python 基础语法、对 FastAPI / Docker / Prometheus 一无所知的同学。
> 我们会逐行精读项目里所有"对外暴露"和"运维相关"的代码，并打比方帮助理解。
>
> 阅读建议：第一遍跟着章节顺序读，遇到看不懂的代码段不要慌，注释会逐行解释；
> 第二遍可以对照真实文件 `backend/`、`ui/`、`infra/` 目录里的代码一起看。

---

## 第 13 章 · FastAPI 后端——对外提供服务

### 13.1 为什么需要 API

在前三部分教程里，我们已经写好了 `app/` 目录下的所有核心逻辑：
- `app/cleaning/` 负责清洗文本
- `app/chunking/` 负责分块
- `app/embedding/` 负责把文本变向量
- `app/stores/` 负责存进 Qdrant
- `app/retrieval/` 负责检索
- `app/generation/` 负责让 LLM 生成答案
- `app/pipeline/` 把这些步骤串成 `IngestPipeline` 和 `QueryPipeline`

但是，**外部世界**怎么调用这些功能？

- 用户在网页上点"上传文件"按钮，谁来接收这个文件？
- 用户在网页里输入问题，谁来把问题转给 `QueryPipeline`？
- 别的程序员想用 Python 脚本批量摄入文档，能不能不写 UI？
- 运维想知道系统健不健康，能不能查个状态？

这些需求都得靠一个 **"对外接口"** 来满足。这就是 `backend/` 目录的职责。

**打比方**：
```
app/       = 厨房（做菜的地方，有厨师、食材、锅碗瓢盆）
backend/   = 服务员（接订单、传菜、收钱，不动手做菜）
ui/        = 菜单（顾客看菜点单，与服务员沟通）
```

服务员不动手做菜，但能让顾客点到菜。`backend/` 不实现 RAG 算法，但能让外界调用到 RAG 算法。

### 13.2 FastAPI 是什么

**FastAPI** 是 Python 的现代 Web 框架。它的作用就是：**把你的 Python 函数变成 HTTP 接口**。

打个比方，你写了一个函数：

```python
def add(a, b):
    return a + b
```

如果你想让人通过网络调用这个函数， traditionally 你得自己写一堆"监听端口、解析 HTTP 请求、提取参数、调用函数、构造响应、返回响应"的代码。FastAPI 帮你把这些脏活全包了：

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}
```

跑起来之后，访问 `http://localhost:8000/add?a=1&b=2` 就能拿到 `{"result": 3}`。

**FastAPI vs Flask（最经典的另一个 Python Web 框架）**：

| 对比项 | Flask | FastAPI |
|--------|-------|---------|
| 异步支持 | 需要额外插件 | 原生 async/await |
| 类型提示 | 不强制 | 强制（用 Pydantic） |
| 自动 API 文档 | 需要插件 | 自带（Swagger UI + ReDoc） |
| 性能 | 中等 | 接近 Node.js |
| 参数校验 | 手写 | 自动 |
| 上手难度 | 简单 | 简单 |

**为什么本项目选 FastAPI**：
1. RAG 系统会调用 LLM API，是 I/O 密集型任务，异步能显著提升并发
2. Pydantic 类型提示让请求/响应结构清晰，少写校验代码
3. 自动生成 `/docs` 交互文档，前端联调方便
4. 性能足够支撑企业级负载

### 13.3 backend/schemas.py 逐行精读

`schemas.py` 定义了 API 用的"数据模型"。所谓"模型"，就是约定请求体和响应体的字段结构。

> 用 `Pydantic` 库（`from pydantic import BaseModel, Field`）。
> Pydantic 是 Python 里最流行的数据校验库，你继承 `BaseModel` 写一个类，
> 它就会自动校验字段类型、默认值、必填可选等。

完整代码如下：

```python
"""Pydantic request/response models for the kb-rag API (Stage 9).

These models are used as ``response_model`` and request body types for the
REST endpoints exposed under :mod:`backend.api.v1`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Response model for ``POST /api/v1/ingest``.

    Attributes:
        doc_id: Identifier of the ingested document.
        num_chunks: Number of chunks produced and stored.
        file_type: File extension (without dot) of the source file.
        trace_id: Trace identifier correlating logs/metrics across the request.
        errors: Non-fatal errors encountered during ingestion.
    """

    doc_id: str
    num_chunks: int
    file_type: str
    trace_id: str
    errors: list[str] = Field(default_factory=list)


class QueryFilters(BaseModel):
    """Optional metadata filters for the query endpoint.

    Attributes:
        source: Filter by source file path/name (string or list of strings).
        tag: Filter by tag (string or list of strings).
        doc_id: Filter by document identifier.
    """

    source: str | list[str] | None = None
    tag: str | list[str] | None = None
    doc_id: str | None = None


class QueryRequest(BaseModel):
    """Request body for ``POST /api/v1/query``.

    Attributes:
        question: Natural-language question.
        filters: Optional metadata filters forwarded to the retriever.
        top_n: Override for the number of candidates to retrieve.
    """

    question: str
    filters: QueryFilters | None = None
    top_n: int | None = None


class ReferenceOut(BaseModel):
    """A citation reference mapped back to a source chunk.

    Attributes:
        chunk_id: Identifier of the source chunk.
        source: Original file path or URI of the chunk.
        page: 1-indexed page number (if applicable).
        score: Relevance score from the retrieval/rerank stage.
        snippet: Truncated preview of the chunk text.
    """

    chunk_id: str
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    """Response model for ``POST /api/v1/query``.

    Attributes:
        answer: Generated natural-language answer (or the no-result fallback).
        references: Citation references mapped to source chunks.
        trace_id: Trace identifier correlating logs/metrics across the request.
        no_result: ``True`` when retrieval yielded no confident result.
        retrieval_latency: Retrieval stage latency in seconds.
        generation_latency: Generation stage latency in seconds.
    """

    answer: str
    references: list[ReferenceOut] = Field(default_factory=list)
    trace_id: str
    no_result: bool = False
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0


class DocumentOut(BaseModel):
    """Response model for an ingested document.

    Attributes:
        doc_id: Identifier of the document.
        file_type: File extension (without dot) of the source file.
        num_chunks: Number of chunks stored for the document.
        ingested_at: ISO-8601 timestamp of ingestion.
    """

    doc_id: str
    file_type: str
    num_chunks: int
    ingested_at: str


class HealthResponse(BaseModel):
    """Response model for ``GET /health``.

    Attributes:
        status: Liveness status (``"ok"`` or error).
        vector_store: Vector store backend name (``"qdrant"`` or ``"chroma"``).
        chunks: Total number of vectors stored in the collection.
        version: API version string.
    """

    status: str
    vector_store: str
    chunks: int
    version: str


class ErrorResponse(BaseModel):
    """Standard error response payload.

    Attributes:
        detail: Human-readable error description.
        trace_id: Trace identifier for correlation (when available).
    """

    detail: str
    trace_id: str | None = None
```

**逐段解释**：

#### 13.3.1 顶部导入

```python
from __future__ import annotations
```
- `from __future__ import annotations`：把类型注解变成"延迟求值字符串"，让我们可以写 `str | list[str] | None` 这种新语法，而 Python 3.9 以下也能跑（被当成字符串看待）。

```python
from pydantic import BaseModel, Field
```
- `BaseModel`：所有数据模型的父类。继承它，类就具备自动校验能力。
- `Field`：用来给字段加额外配置，比如默认值、描述、约束。

#### 13.3.2 `IngestResponse`——摄入接口的响应

```python
class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    file_type: str
    trace_id: str
    errors: list[str] = Field(default_factory=list)
```

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `doc_id` | `str` | 必填 | 文档唯一 ID（UUID） |
| `num_chunks` | `int` | 必填 | 摄入后切成了多少块 |
| `file_type` | `str` | 必填 | 文件扩展名（不带点，如 `"pdf"`） |
| `trace_id` | `str` | 必填 | 这次请求的追踪 ID，可以查日志 |
| `errors` | `list[str]` | `[]` | 非致命错误列表（如某页解析失败） |

`Field(default_factory=list)` 是个关键点：**不能用 `errors: list[str] = []`**！
因为 Python 里可变默认值会被所有实例共享，是个经典坑。`default_factory=list` 告诉 Pydantic "每次新建对象时调用 `list()` 生成一个新空列表"。

#### 13.3.3 `QueryFilters` 与 `QueryRequest`——查询请求

```python
class QueryFilters(BaseModel):
    source: str | list[str] | None = None
    tag: str | list[str] | None = None
    doc_id: str | None = None


class QueryRequest(BaseModel):
    question: str
    filters: QueryFilters | None = None
    top_n: int | None = None
```

`QueryFilters` 字段：

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `source` | `str` 或 `list[str]` 或 `None` | `None` | 按来源文件名过滤 |
| `tag` | `str` 或 `list[str]` 或 `None` | `None` | 按标签过滤 |
| `doc_id` | `str` 或 `None` | `None` | 按文档 ID 过滤 |

`str | list[str] | None` 意思是：可以是单个字符串、字符串列表、或什么都没有。Pydantic 会自动根据 JSON 内容推断。

`QueryRequest` 字段：

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `question` | `str` | 必填 | 用户问题 |
| `filters` | `QueryFilters` 或 `None` | `None` | 可选过滤条件 |
| `top_n` | `int` 或 `None` | `None` | 检索前 N 个候选；不填用配置默认值 |

#### 13.3.4 `ReferenceOut` 与 `QueryResponse`——查询响应

```python
class ReferenceOut(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    references: list[ReferenceOut] = Field(default_factory=list)
    trace_id: str
    no_result: bool = False
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
```

`ReferenceOut` 表示一条引用（一段被检索到的源文本）：

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `chunk_id` | `str` | 必填 | 分块 ID |
| `source` | `str` | 必填 | 源文件路径 |
| `page` | `int` 或 `None` | `None` | PDF 等文档的页码（1 开始） |
| `score` | `float` 或 `None` | `None` | 相关性分数（越高越相关） |
| `snippet` | `str` | `""` | 截断后的预览文本 |

`QueryResponse` 是整个查询接口的响应：

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `answer` | `str` | 必填 | LLM 生成的答案 |
| `references` | `list[ReferenceOut]` | `[]` | 引用列表 |
| `trace_id` | `str` | 必填 | 追踪 ID |
| `no_result` | `bool` | `False` | 是否没检索到内容 |
| `retrieval_latency` | `float` | `0.0` | 检索耗时（秒） |
| `generation_latency` | `float` | `0.0` | 生成耗时（秒） |

#### 13.3.5 `DocumentOut`、`HealthResponse`、`ErrorResponse`

```python
class DocumentOut(BaseModel):
    doc_id: str
    file_type: str
    num_chunks: int
    ingested_at: str
```
用于 `GET /api/v1/documents`，每条对应一个已摄入文档。

```python
class HealthResponse(BaseModel):
    status: str
    vector_store: str
    chunks: int
    version: str
```
用于 `GET /health`，告诉调用方"我还活着，向量库里一共有多少 chunk"。

```python
class ErrorResponse(BaseModel):
    detail: str
    trace_id: str | None = None
```
所有出错时的标准响应格式，`trace_id` 可选（因为有些错误在绑定 trace_id 前就发生了）。

### 13.4 backend/__init__.py 逐行精读

完整代码：

```python
"""backend package for the kb-rag API.

Exposes shared FastAPI dependency accessors that wrap :class:`app.pipeline.Container`.
Defining them here (rather than in :mod:`backend.main`) breaks what would
otherwise be a circular import: route modules need the accessors at module
load time, but :mod:`backend.main` loads the routers after defining them.
"""
from __future__ import annotations

from app.pipeline import Container, IngestPipeline, QueryPipeline
from app.stores import VectorStore

__all__ = [
    "get_ingest_pipeline_dep",
    "get_query_pipeline_dep",
    "get_vector_store_dep",
]


def get_ingest_pipeline_dep() -> IngestPipeline:
    """Return the shared :class:`IngestPipeline` singleton.

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_ingest_pipeline()


def get_query_pipeline_dep() -> QueryPipeline:
    """Return the shared :class:`QueryPipeline` singleton.

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_query_pipeline()


def get_vector_store_dep() -> VectorStore:
    """Return the shared :class:`VectorStore` (sourced from the ingest pipeline).

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_ingest_pipeline().vector_store
```

**逐行解释**：

#### 13.4.1 模块文档字符串已经说明了核心问题
"Defining them here (rather than in `backend.main`) breaks what would otherwise be a circular import"。
**翻译**：把这些访问器函数定义在 `backend/__init__.py` 里，而不是 `backend/main.py` 里，是为了打破**循环导入**。

循环导入是怎么产生的？
- `backend/main.py` 想导入 `backend/api/v1/ingest.py`
- `backend/api/v1/ingest.py` 又想用 `backend.main` 里的 `get_ingest_pipeline_dep`
- 结果就是：main 等 ingest，ingest 等 main，谁也跑不起来

**解决方法**：把访问器函数放到独立的 `backend/__init__.py` 里，两边都从这里拿。

#### 13.4.2 三个访问器函数

```python
def get_ingest_pipeline_dep() -> IngestPipeline:
    return Container.get_ingest_pipeline()
```
- 参数：无
- 返回值：`IngestPipeline` 实例（来自 `app.pipeline.Container`）
- 用途：让 FastAPI 路由函数能拿到摄入流水线

```python
def get_query_pipeline_dep() -> QueryPipeline:
    return Container.get_query_pipeline()
```
- 参数：无
- 返回值：`QueryPipeline` 实例
- 用途：让查询路由能拿到查询流水线

```python
def get_vector_store_dep() -> VectorStore:
    return Container.get_ingest_pipeline().vector_store
```
- 参数：无
- 返回值：`VectorStore` 实例（向量库客户端）
- 用途：健康检查需要查向量库存活状态

#### 13.4.3 什么是 `Depends()`

这三个函数之所以以 `_dep` 结尾，是为了表达**它们将作为 FastAPI 依赖**使用。在路由里你会看到：

```python
def ingest_file(
    file: Annotated[UploadFile, File(...)],
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> IngestResponse:
```

`Depends(get_ingest_pipeline_dep)` 告诉 FastAPI：
> "调用这个路由前，先调用 `get_ingest_pipeline_dep()`，把它的返回值作为 `pipeline` 参数传进来。"

**打比方**：Depends 就像饭店的"服务员给你递上菜单"。
你写菜单（路由函数）的时候说"我需要一本菜单"，服务员（FastAPI）就自动给你递过来。
你不用关心菜单是从哪来的，FastAPI 帮你拿。

**为什么这么设计？**
1. **解耦**：路由不直接 `import` Pipeline，方便测试时换成假的 Pipeline
2. **生命周期管理**：FastAPI 知道每个请求该注入什么，不需要全局变量
3. **可测试**：测试时可以用 `app.dependency_overrides[get_ingest_pipeline_dep] = fake_func` 替换

### 13.5 backend/main.py 逐行精读

完整代码：

```python
"""FastAPI application entry point for kb-rag (Stage 9).

Wires the ingest/query pipelines and vector store into a REST API exposed
under ``/api/v1`` plus ``/health`` and ``/metrics`` at the root level.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import init_tracer

# Dependency accessors live in the backend package (__init__.py) to avoid
# circular imports between backend.main and the route modules.  Re-exported
# here for backward compatibility (``backend.main.get_ingest_pipeline_dep``).
from backend import (  # noqa: F401
    get_ingest_pipeline_dep,
    get_query_pipeline_dep,
    get_vector_store_dep,
)

# Configure logging once on import so module-level log calls are structured.
_settings_initial = get_settings()
configure_logging(level=_settings_initial.log_level)
logger = get_logger(__name__)

# Import routers.  Safe to do at this point: route modules pull the dependency
# accessors from the already-initialized ``backend`` package.
from backend.api.v1 import router as v1_router  # noqa: E402
from backend.api.v1.health import router as health_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hook.

    On startup: configure logging, initialize the OpenTelemetry tracer, and
    log the active environment.  Container initialization is lazy (the
    pipelines are built on first request), so no eager warm-up is performed.
    """
    s = get_settings()
    configure_logging(level=s.log_level)
    init_tracer(service_name="kb-rag-api")
    logger.info("kb-rag API starting", env=s.app_env)
    yield
    logger.info("kb-rag API shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    s = get_settings()
    app = FastAPI(
        title="kb-rag API",
        version="0.1.0",
        description="Enterprise RAG knowledge base API.",
        lifespan=lifespan,
    )

    # CORS: allow the Streamlit UI origin (default localhost:8501); in dev mode
    # also allow "*" so a browser running on any origin can call the API.
    origins: list[str] = ["http://localhost:8501"]
    if s.app_env == "dev":
        origins.append("*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics: automatically exposes GET /metrics.
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
    ).instrument(app).expose(app, endpoint="/metrics")

    # Mount the v1 router under /api/v1 and the health router at root.
    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(health_router, tags=["health"])

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler returning a 500 with a fresh trace_id."""
        trace_id = uuid.uuid4().hex
        logger.error(
            "unhandled.exception",
            error=str(exc),
            trace_id=trace_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "trace_id": trace_id},
        )

    logger.info("kb-rag API initialized", env=s.app_env)
    return app


app = create_app()


def run() -> None:
    """Entry point for the ``kb-rag-api`` console script."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=s.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
```

**逐段解释**：

#### 13.5.1 顶部导入

```python
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
```
- `uuid`：生成 UUID，给异常分配 trace_id
- `AsyncIterator`：lifespan 函数返回值的类型提示
- `asynccontextmanager`：把 async 函数变成"上下文管理器"的装饰器

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
```
- `FastAPI`：核心应用类
- `Request`：HTTP 请求对象，异常处理器里要拿请求路径
- `CORSMiddleware`：跨域中间件
- `JSONResponse`：返回 JSON 响应

```python
from prometheus_fastapi_instrumentator import Instrumentator
```
- 第三方库 `prometheus-fastapi-instrumentator`：自动给 FastAPI 加上 Prometheus 监控指标（请求数、延迟、状态码等）。

```python
from app.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import init_tracer
```
- 拿配置、配置日志、初始化追踪器。

```python
from backend import (
    get_ingest_pipeline_dep,
    get_query_pipeline_dep,
    get_vector_store_dep,
)
```
- 从 `backend/__init__.py` 拿访问器，并且 re-export 出来（保持向后兼容）。

#### 13.5.2 模块加载时配置日志

```python
_settings_initial = get_settings()
configure_logging(level=_settings_initial.log_level)
logger = get_logger(__name__)
```
- `get_settings()`：拿到 `.env` 解析后的配置对象
- `configure_logging(level=...)`：初始化 structlog 日志系统（详见第 15 章）
- `get_logger(__name__)`：拿到当前模块的 logger
- **注意**：这是在模块导入时就执行的，保证后面所有 `logger.info(...)` 都能正常输出

#### 13.5.3 导入路由

```python
from backend.api.v1 import router as v1_router
from backend.api.v1.health import router as health_router
```
- `v1_router`：聚合了 `/ingest`、`/query`、`/documents` 三个子路由
- `health_router`：单独的健康检查路由
- 注释明确说"现在导入是安全的，因为访问器已经初始化好了"

#### 13.5.4 lifespan——启动与关闭钩子

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_logging(level=s.log_level)
    init_tracer(service_name="kb-rag-api")
    logger.info("kb-rag API starting", env=s.app_env)
    yield
    logger.info("kb-rag API shutting down")
```

- 参数：`app: FastAPI`，应用实例本身
- 返回值：`AsyncIterator[None]`，因为是 async 生成器
- 启动时：重新配置日志、初始化 tracer、记录一条日志
- `yield`：这一行表示"应用运行中"
- 关闭时：`yield` 之后的部分会执行，记录关闭日志

**打比方**：lifespan 像饭店的"开门/打烊仪式"。
- 开门前，店长检查桌子、点灯、广播"我们开业了"（启动逻辑）
- 营业期间让顾客进进出出（`yield`）
- 关门时，店长广播"我们打烊了"（关闭逻辑）

注释说"Container 初始化是 lazy 的"——意思是 `IngestPipeline` 这些大对象不是在启动时创建的，而是第一次有请求进来才创建。这样启动快，第一次请求会稍慢。

#### 13.5.5 `create_app()`——工厂函数

```python
def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="kb-rag API",
        version="0.1.0",
        description="Enterprise RAG knowledge base API.",
        lifespan=lifespan,
    )
```

- 参数：无
- 返回值：`FastAPI` 应用实例
- 这是个"工厂函数"——每次调用返回一个新的 app 实例，方便测试时创建独立的应用
- `FastAPI(...)` 参数：
  - `title`：API 标题（显示在 `/docs` 文档里）
  - `version`：API 版本号
  - `description`：API 描述
  - `lifespan`：上面定义的 lifespan 函数

#### 13.5.6 CORS 中间件

```python
origins: list[str] = ["http://localhost:8501"]
if s.app_env == "dev":
    origins.append("*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**为什么需要 CORS**：浏览器有"同源策略"——网页来自 `localhost:8501`，想调 `localhost:8000` 的接口，浏览器会拦截。CORS（跨域资源共享）就是后端告诉浏览器"我允许哪些来源访问我"。

- `origins`：允许的来源列表
  - 默认放行 Streamlit UI（`http://localhost:8501`）
  - dev 模式额外放行 `*`（任何来源，方便本地调试）
- `allow_origins`：允许的来源
- `allow_credentials=False`：不允许带 Cookie（RAG API 不需要登录态）
- `allow_methods=["*"]`：所有 HTTP 方法都允许（GET/POST/DELETE 等）
- `allow_headers=["*"]`：所有请求头都允许

#### 13.5.7 Instrumentator——自动监控

```python
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
).instrument(app).expose(app, endpoint="/metrics")
```

- `should_group_status_codes=True`：把 200、201、202 这些 2xx 合并成一组指标
- `should_ignore_untemplated=True`：忽略未模板化的路由
- `should_respect_env_var=False`：不通过环境变量开关，强制启用
- `.instrument(app)`：给 app 加监控钩子
- `.expose(app, endpoint="/metrics")`：暴露 `/metrics` 端点给 Prometheus 抓取

#### 13.5.8 挂载路由

```python
app.include_router(v1_router, prefix="/api/v1")
app.include_router(health_router, tags=["health"])
```
- `v1_router` 挂在 `/api/v1` 前缀下，所以完整路径是 `/api/v1/ingest`、`/api/v1/query`、`/api/v1/documents`
- `health_router` 挂在根下，所以是 `/health`
- `tags` 是 Swagger 文档里给接口分组的标签

#### 13.5.9 全局异常处理器

```python
@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    trace_id = uuid.uuid4().hex
    logger.error(
        "unhandled.exception",
        error=str(exc),
        trace_id=trace_id,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "trace_id": trace_id},
    )
```

- 参数：`request: Request`（出错的请求）、`exc: Exception`（捕获的异常）
- 返回值：`JSONResponse`（HTTP 500）
- 作用：所有未被其他处理器捕获的异常都会落到这里，统一返回 500 + `trace_id`
- **为什么这样做**：默认 FastAPI 会把异常堆栈直接吐给调用方，不安全也不友好。这里固定返回 `{"detail": "internal server error", "trace_id": "..."}`，让用户拿 trace_id 找运维查日志。

#### 13.5.10 `app = create_app()` 和 `run()`

```python
app = create_app()


def run() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=s.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
```

- `app = create_app()`：模块级别创建一个 app 实例，被 `uvicorn backend.main:app` 命令使用
- `run()`：命令行入口
  - `uvicorn` 是 ASGI 服务器（FastAPI 的运行容器）
  - `"backend.main:app"`：字符串路径，告诉 uvicorn 去 `backend/main.py` 找 `app` 变量
  - `host="0.0.0.0"`：监听所有网卡（容器内必须用 0.0.0.0 才能被外部访问）
  - `port=s.api_port`：默认 8000
  - `reload=False`：生产模式不热重载

### 13.6 backend/api/v1/ingest.py 逐行精读

完整代码：

```python
"""``POST /api/v1/ingest`` endpoint for uploading and ingesting documents."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.observability.logging import get_logger
from app.pipeline import IngestPipeline
from backend import get_ingest_pipeline_dep
from backend.api.v1.documents import DocRegistry, get_doc_registry
from backend.schemas import IngestResponse

logger = get_logger(__name__)

router = APIRouter()

# Whitelist of supported file extensions (lowercase, no leading dot).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"pdf", "docx", "xlsx", "xls", "md", "html", "htm", "txt"}
)

# Maximum accepted upload size (50 MB).
MAX_FILE_SIZE: int = 50 * 1024 * 1024


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document file",
)
def ingest_file(
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> IngestResponse:
    """Ingest a single document file into the RAG knowledge base.

    The uploaded file is saved to ``data/raw/{uuid}_{filename}`` and then
    processed end-to-end by the ingest pipeline.  Supported types: pdf, docx,
    xlsx, xls, md, html, htm, txt.  Maximum file size: 50 MB.  Unknown
    extensions are rejected with HTTP 400.
    """
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type: .{extension}",
        )

    # Read content and enforce size limit.
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large: {len(content)} bytes (max {MAX_FILE_SIZE})",
        )

    # Persist to data/raw/{uuid}_{filename}
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = raw_dir / saved_name
    saved_path.write_bytes(content)
    logger.info("ingest.saved", path=str(saved_path), size=len(content))

    # Run the ingest pipeline.
    result = pipeline.ingest_file(saved_path)

    # Record in the doc registry.
    registry.add(
        doc_id=result.doc_id,
        file_type=result.file_type,
        num_chunks=result.num_chunks,
    )

    return IngestResponse(
        doc_id=result.doc_id,
        num_chunks=result.num_chunks,
        file_type=result.file_type,
        trace_id=result.trace_id,
        errors=list(result.errors),
    )
```

**逐段解释**：

#### 13.6.1 顶部导入

```python
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
```
- `uuid`：给保存的文件加 UUID 前缀，避免重名
- `Path`：面向对象的路径操作
- `Annotated`：给类型加额外元数据（FastAPI 用它知道该注入什么）
- `APIRouter`：子路由器
- `Depends`：依赖注入
- `File`：标记参数为文件上传字段
- `HTTPException`：抛 HTTP 异常
- `UploadFile`：FastAPI 的文件上传类型
- `status`：HTTP 状态码常量

#### 13.6.2 白名单和大小限制

```python
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"pdf", "docx", "xlsx", "xls", "md", "html", "htm", "txt"}
)

MAX_FILE_SIZE: int = 50 * 1024 * 1024
```

- `ALLOWED_EXTENSIONS`：用 `frozenset`（不可变集合），支持 8 种格式
  - `pdf`：PDF 文档
  - `docx`：Word 2007+
  - `xlsx`：Excel 2007+
  - `xls`：旧 Excel
  - `md`：Markdown
  - `html`/`htm`：网页
  - `txt`：纯文本
- `MAX_FILE_SIZE`：50 MB = 50 × 1024 × 1024 字节
  - **为什么限制 50MB**：太大文件会撑爆内存、占用 LLM token、拖慢响应

#### 13.6.3 路由定义

```python
@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document file",
)
```
- `@router.post("")`：注册 POST 路由，路径为空（实际路径由 main.py 的前缀 `/api/v1/ingest` 决定）
- `response_model=IngestResponse`：告诉 FastAPI 返回值会被 `IngestResponse` 校验和序列化
- `status_code=201`：创建资源成功用 201（不是 200）
- `summary`：API 文档里显示的标题

#### 13.6.4 函数签名

```python
def ingest_file(
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> IngestResponse:
```

三个参数：
- `file`：类型 `UploadFile`，FastAPI 看到这个类型会自动从 multipart 表单里提取文件
- `pipeline`：通过 `Depends(get_ingest_pipeline_dep)` 注入
- `registry`：通过 `Depends(get_doc_registry)` 注入

返回值：`IngestResponse`

**`UploadFile` 是什么**：FastAPI 提供的"上传文件"包装类，包含：
- `file.filename`：原始文件名
- `file.file`：底层文件对象（SpooledTemporaryFile），可以 `.read()`
- `file.content_type`：MIME 类型

#### 13.6.5 函数体——校验、保存、摄入

```python
filename = file.filename or "unknown"
extension = Path(filename).suffix.lower().lstrip(".")
if extension not in ALLOWED_EXTENSIONS:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"unsupported file type: .{extension}",
    )
```

- 拿到文件名（防御性默认 `"unknown"`）
- 提取扩展名：`Path("report.PDF").suffix` → `".PDF"`，`.lower()` → `".pdf"`，`.lstrip(".")` → `"pdf"`
- 不在白名单就抛 `HTTPException`（FastAPI 会自动转成 400 响应）

```python
content = file.file.read()
if len(content) > MAX_FILE_SIZE:
    raise HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"file too large: {len(content)} bytes (max {MAX_FILE_SIZE})",
    )
```

- 读出所有字节
- 超过 50MB 抛 413（413 = Request Entity Too Large，标准 HTTP 状态码）

```python
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
saved_name = f"{uuid.uuid4().hex}_{filename}"
saved_path = raw_dir / saved_name
saved_path.write_bytes(content)
logger.info("ingest.saved", path=str(saved_path), size=len(content))
```

- `data/raw/` 目录不存在就创建（`parents=True` 连父目录一起建，`exist_ok=True` 已存在不报错）
- 文件名加 UUID 前缀，防止重名覆盖
- 写入磁盘
- 记录日志（带路径和大小）

```python
result = pipeline.ingest_file(saved_path)
```

- 调用 `IngestPipeline.ingest_file()`，传入保存的文件路径
- 这一步会触发：解析 → 清洗 → 分块 → 嵌入 → 存储

```python
registry.add(
    doc_id=result.doc_id,
    file_type=result.file_type,
    num_chunks=result.num_chunks,
)
```

- 把文档元数据写入 `doc_registry.json`（详见 13.8）

```python
return IngestResponse(
    doc_id=result.doc_id,
    num_chunks=result.num_chunks,
    file_type=result.file_type,
    trace_id=result.trace_id,
    errors=list(result.errors),
)
```

- 用 `result` 构造 `IngestResponse` 返回
- `list(result.errors)`：把元组/列表转成普通 list（Pydantic 校验更严）

### 13.7 backend/api/v1/query.py 逐行精读

完整代码：

```python
"""``POST /api/v1/query`` endpoint for running RAG queries."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.pipeline import QueryPipeline
from backend import get_query_pipeline_dep
from backend.schemas import QueryRequest, QueryResponse, ReferenceOut

router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG knowledge base",
)
def query(
    request: QueryRequest,
    pipeline: Annotated[QueryPipeline, Depends(get_query_pipeline_dep)],
) -> QueryResponse:
    """Run a RAG query end-to-end.

    Retrieves relevant chunks via hybrid retrieval, reranks them, checks the
    guardrail, and generates a natural-language answer with citation
    references.
    """
    filters_dict: dict | None = None
    if request.filters is not None:
        filters_dict = {
            k: v
            for k, v in request.filters.model_dump().items()
            if v is not None
        }
        if not filters_dict:
            filters_dict = None

    result = pipeline.query(
        question=request.question,
        filters=filters_dict,
        top_n=request.top_n,
    )

    return QueryResponse(
        answer=result.answer,
        references=[
            ReferenceOut(
                chunk_id=ref.chunk_id,
                source=ref.source,
                page=ref.page,
                score=ref.score,
                snippet=ref.snippet,
            )
            for ref in result.references
        ],
        trace_id=result.trace_id,
        no_result=result.no_result,
        retrieval_latency=result.retrieval_latency,
        generation_latency=result.generation_latency,
    )
```

**逐行解释**：

#### 13.7.1 路由注册

```python
@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG knowledge base",
)
```
- POST 方法
- 路径为空（外层会拼成 `/api/v1/query`）
- 响应模型 `QueryResponse`

#### 13.7.2 函数签名

```python
def query(
    request: QueryRequest,
    pipeline: Annotated[QueryPipeline, Depends(get_query_pipeline_dep)],
) -> QueryResponse:
```

- `request: QueryRequest`：**特殊参数**。FastAPI 看到 Pydantic 模型类型会自动从请求体（JSON body）解析
- `pipeline`：通过依赖注入拿到 `QueryPipeline` 单例
- 返回值：`QueryResponse`

#### 13.7.3 过滤条件处理

```python
filters_dict: dict | None = None
if request.filters is not None:
    filters_dict = {
        k: v
        for k, v in request.filters.model_dump().items()
        if v is not None
    }
    if not filters_dict:
        filters_dict = None
```

- `request.filters` 是 `QueryFilters` 对象，可能为 `None`
- `model_dump()`：Pydantic v2 方法，把模型转成字典
- 字典推导过滤掉值为 `None` 的字段（用户没传的过滤条件不要传给 pipeline）
- 如果过滤后字典为空，就转回 `None`（让 pipeline 知道"没过滤"）

#### 13.7.4 调用 pipeline 并构造响应

```python
result = pipeline.query(
    question=request.question,
    filters=filters_dict,
    top_n=request.top_n,
)
```

三个参数：
- `question`：`str`，用户问题
- `filters`：`dict | None`，过滤条件
- `top_n`：`int | None`，可选的 top_n 覆盖

返回 `result` 是 pipeline 内部结果对象，含 `answer`、`references`、`trace_id` 等字段。

```python
return QueryResponse(
    answer=result.answer,
    references=[
        ReferenceOut(
            chunk_id=ref.chunk_id,
            source=ref.source,
            page=ref.page,
            score=ref.score,
            snippet=ref.snippet,
        )
        for ref in result.references
    ],
    trace_id=result.trace_id,
    no_result=result.no_result,
    retrieval_latency=result.retrieval_latency,
    generation_latency=result.generation_latency,
)
```

- 用列表推导把 `result.references`（内部对象列表）转成 `ReferenceOut`（API 响应对象列表）
- 这种"内部模型 → API 模型"的转换是典型分层模式，避免内部结构泄露给外部

### 13.8 backend/api/v1/documents.py 逐行精读

完整代码：

```python
"""GET/DELETE ``/api/v1/documents`` endpoints with a JSON-backed doc registry.

The :class:`DocRegistry` persists a list of ingested document metadata to
``data/processed/doc_registry.json`` so the API can list and delete documents
without modifying :mod:`app.stores`.  All file operations are guarded by a
class-level :class:`threading.Lock` to remain safe under concurrent requests.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.observability.logging import get_logger
from app.pipeline import IngestPipeline
from backend import get_ingest_pipeline_dep
from backend.schemas import DocumentOut

logger = get_logger(__name__)

router = APIRouter()

REGISTRY_PATH = Path("data/processed/doc_registry.json")


class DocRegistry:
    """Thread-safe JSON-backed document registry.

    Stores a list of ingested document metadata at the configured path.  All
    read/write operations are serialized via a class-level lock so concurrent
    requests do not corrupt the file.

    Args:
        path: Optional override for the registry file location (used in tests).
    """

    _lock: threading.Lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the registry with an optional path override.

        Args:
            path: Path to the registry JSON file.  Defaults to
                :data:`REGISTRY_PATH`.
        """
        self.path = path or REGISTRY_PATH

    # ------------------------------------------------------------------
    # Internal I/O (callers must already hold the lock)
    # ------------------------------------------------------------------
    def _read(self) -> list[dict[str, Any]]:
        """Read and return the registry list (empty if missing/corrupt)."""
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            logger.warning("registry.read.failed", error=str(exc), path=str(self.path))
        return []

    def _write(self, docs: list[dict[str, Any]]) -> None:
        """Write the registry list to disk, creating parent dirs as needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(docs, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    def list_all(self) -> list[dict[str, Any]]:
        """Return a copy of all registered documents."""
        with self._lock:
            return self._read()

    def add(self, doc_id: str, file_type: str, num_chunks: int) -> None:
        """Register a newly ingested document.

        Args:
            doc_id: Identifier of the document.
            file_type: File extension (without dot) of the source file.
            num_chunks: Number of chunks produced for the document.
        """
        with self._lock:
            docs = self._read()
            docs.append(
                {
                    "doc_id": doc_id,
                    "file_type": file_type,
                    "num_chunks": num_chunks,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._write(docs)

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the registry.

        Args:
            doc_id: Identifier of the document to remove.

        Returns:
            ``True`` if the document was found and removed, ``False`` otherwise.
        """
        with self._lock:
            docs = self._read()
            new_docs = [d for d in docs if d.get("doc_id") != doc_id]
            removed = len(new_docs) < len(docs)
            if removed:
                self._write(new_docs)
            return removed


def get_doc_registry() -> DocRegistry:
    """Return a :class:`DocRegistry` instance (overridable in tests)."""
    return DocRegistry()


@router.get(
    "",
    response_model=list[DocumentOut],
    summary="List ingested documents",
)
def list_documents(
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> list[DocumentOut]:
    """Return the list of documents recorded in the doc registry."""
    docs = registry.list_all()
    return [
        DocumentOut(
            doc_id=d["doc_id"],
            file_type=d.get("file_type", ""),
            num_chunks=d.get("num_chunks", 0),
            ingested_at=d.get("ingested_at", ""),
        )
        for d in docs
    ]


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an ingested document",
)
def delete_document(
    doc_id: str,
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> None:
    """Delete a document and all its chunks from every store and the registry.

    Args:
        doc_id: Identifier of the document to remove.
    """
    pipeline.delete_document(doc_id)
    registry.remove(doc_id)
```

**逐段解释**：

#### 13.8.1 为什么单独搞个 `DocRegistry`

向量库（Qdrant）只管"向量 + payload"，不管"你摄入过哪些文件、什么时候摄入的"。
但用户需要在 UI 上看"已摄入文档列表"。所以这里搞一个简单的 JSON 文件 `data/processed/doc_registry.json` 来记。

**打比方**：Qdrant 像仓库货架，`doc_registry.json` 像仓库门口的进出登记本。登记本不影响货架，但方便你查"今天进了哪些货"。

#### 13.8.2 `DocRegistry` 类

**类级锁**：
```python
_lock: threading.Lock = threading.Lock()
```

- `threading.Lock()`：线程锁
- **为什么是类级（不是实例级）**：所有 `DocRegistry` 实例共享同一把锁。因为它们都读写同一个文件，必须串行化。
- **打比方**：仓库门只有一个，所有进出人员都得排一个队，不能两个登记本同时往门里挤。

**`__init__`**：
```python
def __init__(self, path: Path | None = None) -> None:
    self.path = path or REGISTRY_PATH
```
- 参数：`path: Path | None = None`，可选的文件路径覆盖
- 默认值：`None`
- 含义：测试时可以传一个临时路径，生产用默认的 `REGISTRY_PATH`

**`_read`（内部方法，调用方必须先持锁）**：
```python
def _read(self) -> list[dict[str, Any]]:
    if not self.path.exists():
        return []
    try:
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("registry.read.failed", error=str(exc), path=str(self.path))
    return []
```
- 返回值：`list[dict[str, Any]]`，文档元数据列表
- 文件不存在就返回空列表
- 文件损坏（JSON 解析失败）就记日志、返回空列表（不抛异常，让请求继续）

**`_write`（内部方法，调用方必须先持锁）**：
```python
def _write(self, docs: list[dict[str, Any]]) -> None:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)
```
- 参数：`docs`，要写入的列表
- 自动创建父目录
- `ensure_ascii=False`：保留中文字符（不转义成 `\uXXXX`）
- `indent=2`：缩进 2 空格，方便人读

**`list_all`（公开方法，加锁）**：
```python
def list_all(self) -> list[dict[str, Any]]:
    with self._lock:
        return self._read()
```
- `with self._lock:`：进入时获取锁，离开时自动释放
- 拿到锁后调用内部 `_read`

**`add`**：
```python
def add(self, doc_id: str, file_type: str, num_chunks: int) -> None:
    with self._lock:
        docs = self._read()
        docs.append({
            "doc_id": doc_id,
            "file_type": file_type,
            "num_chunks": num_chunks,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
        self._write(docs)
```
- 参数：`doc_id: str`、`file_type: str`、`num_chunks: int`
- 返回值：`None`
- `datetime.now(timezone.utc).isoformat()`：当前 UTC 时间，ISO 8601 格式（如 `2025-01-01T12:34:56+00:00`）

**`remove`**：
```python
def remove(self, doc_id: str) -> bool:
    with self._lock:
        docs = self._read()
        new_docs = [d for d in docs if d.get("doc_id") != doc_id]
        removed = len(new_docs) < len(docs)
        if removed:
            self._write(new_docs)
        return removed
```
- 参数：`doc_id: str`
- 返回值：`bool`，是否真的删除了
- 列表推导过滤掉匹配项，靠长度变化判断是否删除

#### 13.8.3 GET /documents

```python
@router.get(
    "",
    response_model=list[DocumentOut],
    summary="List ingested documents",
)
def list_documents(
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> list[DocumentOut]:
    docs = registry.list_all()
    return [
        DocumentOut(
            doc_id=d["doc_id"],
            file_type=d.get("file_type", ""),
            num_chunks=d.get("num_chunks", 0),
            ingested_at=d.get("ingested_at", ""),
        )
        for d in docs
    ]
```

- `response_model=list[DocumentOut]`：返回值是 `DocumentOut` 列表
- 通过依赖注入拿到 `DocRegistry` 实例
- 拿到字典列表，转成 `DocumentOut` 对象列表

#### 13.8.4 DELETE /documents/{doc_id}

```python
@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an ingested document",
)
def delete_document(
    doc_id: str,
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> None:
    pipeline.delete_document(doc_id)
    registry.remove(doc_id)
```

- `/{doc_id}`：路径参数，FastAPI 自动从 URL 提取
- `status_code=204`：删除成功用 204（No Content，没有响应体）
- 参数：`doc_id: str`（路径参数）、`pipeline` 和 `registry`（依赖注入）
- 返回值：`None`（对应 204 状态码）
- 先从向量库删（`pipeline.delete_document`），再从注册表删（`registry.remove`）

### 13.9 backend/api/v1/health.py 逐行精读

完整代码：

```python
"""``GET /health`` endpoint for liveness and readiness probing."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.observability.logging import get_logger
from app.stores import VectorStore
from backend import get_vector_store_dep
from backend.schemas import HealthResponse

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness and readiness probe",
)
def health(
    store: Annotated[VectorStore, Depends(get_vector_store_dep)],
) -> HealthResponse:
    """Check API liveness and vector store connectivity.

    Calls ``vector_store.count()`` to verify the store is reachable.  Returns
    HTTP 503 when the store is unavailable.
    """
    settings = get_settings()
    try:
        chunks = store.count()
    except Exception as exc:  # noqa: BLE001 - surface as 503
        logger.error("health.check.failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="vector store unavailable",
        )
    return HealthResponse(
        status="ok",
        vector_store=settings.vector_store,
        chunks=chunks,
        version="0.1.0",
    )
```

**逐行解释**：

#### 13.9.1 路由注册

```python
@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness and readiness probe",
)
```
- GET 方法
- **完整路径就是 `/health`**（这个路由在 main.py 里没加前缀，直接挂根）
- `tags=["health"]`：Swagger 文档里归到 "health" 分组

#### 13.9.2 函数签名

```python
def health(
    store: Annotated[VectorStore, Depends(get_vector_store_dep)],
) -> HealthResponse:
```
- 通过依赖注入拿到 `VectorStore` 实例

#### 13.9.3 健康检查逻辑

```python
settings = get_settings()
try:
    chunks = store.count()
except Exception as exc:
    logger.error("health.check.failed", error=str(exc))
    raise HTTPException(
        status_code=503,
        detail="vector store unavailable",
    )
return HealthResponse(
    status="ok",
    vector_store=settings.vector_store,
    chunks=chunks,
    version="0.1.0",
)
```

- `store.count()`：调向量库的 count 接口，返回总 chunk 数
  - 这一步**真正去连了 Qdrant**，所以能验证连通性
- 出任何异常都返回 503（Service Unavailable）
- 正常返回 `HealthResponse`，含：
  - `status="ok"`：活着
  - `vector_store=settings.vector_store`：当前用的什么向量库（qdrant/chroma）
  - `chunks=chunks`：向量总数
  - `version="0.1.0"`：API 版本

**为什么这样设计**：
- "liveness"=进程活着（能响应 `/health` 就算活着）
- "readiness"=依赖可用（Qdrant 能 `count()` 才算 ready）
- 这两种探针都被 Docker / Kubernetes 用来判断容器是否健康

### 13.10 API 使用示例

假设后端跑在 `localhost:8000`。

#### 健康检查

```bash
curl http://localhost:8000/health
```

返回：
```json
{
  "status": "ok",
  "vector_store": "qdrant",
  "chunks": 42,
  "version": "0.1.0"
}
```

#### 上传文件摄入

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@report.pdf"
```

返回（HTTP 201）：
```json
{
  "doc_id": "abc-123-def",
  "num_chunks": 18,
  "file_type": "pdf",
  "trace_id": "9b8a...",
  "errors": []
}
```

#### 查询

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "X-2025 续航多久？"}'
```

返回：
```json
{
  "answer": "X-2025 续航时间为 12 小时。",
  "references": [
    {
      "chunk_id": "chunk-001",
      "source": "data/raw/abc_report.pdf",
      "page": 2,
      "score": 0.823,
      "snippet": "续航时间 12 小时，充电时间 2 小时..."
    }
  ],
  "trace_id": "abc123",
  "no_result": false,
  "retrieval_latency": 0.342,
  "generation_latency": 1.856
}
```

#### 列出文档

```bash
curl http://localhost:8000/api/v1/documents
```

返回：
```json
[
  {
    "doc_id": "abc-123-def",
    "file_type": "pdf",
    "num_chunks": 18,
    "ingested_at": "2025-01-01T12:34:56+00:00"
  }
]
```

#### 删除文档

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/abc-123-def
```

返回：HTTP 204（无响应体）

#### 查看 Prometheus 指标

```bash
curl http://localhost:8000/metrics
```

返回 Prometheus 文本格式的指标。

#### 查看交互式文档

浏览器打开 `http://localhost:8000/docs` —— Swagger UI，可以直接在网页上点按钮调用接口。

---

## 第 14 章 · Streamlit UI——用户界面

### 14.1 Streamlit 是什么

**Streamlit** 是一个 Python 库，让你**用纯 Python 代码写网页**，不需要写 HTML、CSS、JavaScript。

**典型用法**：
```python
import streamlit as st
st.title("我的网页")
name = st.text_input("你叫什么？")
if st.button("打招呼"):
    st.write(f"你好，{name}！")
```

跑起来就是一个网页：有标题、输入框、按钮、输出。

**适合什么场景**：
- 内部工具、原型、数据看板
- 不需要复杂交互逻辑的应用
- 数据科学家、算法工程师快速做 demo

**不适合**：
- 复杂的电商网站、社交平台
- 需要细粒度 UI 控制的应用
- 高并发生产环境

RAG 知识库管理后台正好是 Streamlit 擅长的场景：上传文件、看回答、查引用，简单够用。

### 14.2 ui/app.py 逐段精读

文件较长，分段精读。

#### 14.2.1 头部和导入

```python
"""Streamlit UI entry point for kb-rag.

Stage 10: full chat / ingestion UI. Talks to the FastAPI backend exposed
under ``/api/v1`` (ingest, query, documents). Document metadata and recent
Q&A history are kept in :mod:`streamlit` ``session_state``.

The API base URL defaults to ``http://localhost:8000`` and can be overridden
either via the sidebar input or the ``KB_RAG_API_URL`` environment variable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests
import streamlit as st
```

- `requests`：HTTP 客户端库，UI 通过它调后端 API
- `streamlit as st`：约定俗成的别名
- `dataclasses`：Python 标准库，用 `@dataclass` 装饰器快速定义数据类

#### 14.2.2 页面配置

```python
st.set_page_config(
    page_title="kb-rag 知识库",
    page_icon="📚",
    layout="wide",
)
```

- `page_title`：浏览器标签页标题
- `page_icon`：标签页图标（emoji）
- `layout="wide"`：宽屏布局，占满浏览器宽度
- **必须最先调用**：Streamlit 要求 `set_page_config` 是第一个 Streamlit 调用，否则报错

#### 14.2.3 常量配置

```python
DEFAULT_API_URL = "http://localhost:8000"
ACCEPTED_FILE_TYPES = ["pdf", "docx", "xlsx", "md", "txt", "html"]
MAX_HISTORY = 10
API_TIMEOUT = 120  # seconds; bounded so st.spinner never blocks forever.
```

| 常量 | 值 | 含义 |
|------|----|------|
| `DEFAULT_API_URL` | `"http://localhost:8000"` | 默认 API 地址 |
| `ACCEPTED_FILE_TYPES` | 6 种格式 | UI 上传组件允许的扩展名 |
| `MAX_HISTORY` | 10 | 最多保留 10 条历史问答 |
| `API_TIMEOUT` | 120 秒 | 单次 API 调用最多等 2 分钟 |

#### 14.2.4 数据结构（dataclass）

```python
@dataclass
class DocumentInfo:
    """Document metadata returned by ``GET /api/v1/documents``."""
    doc_id: str
    file_type: str
    num_chunks: int
    ingested_at: str


@dataclass
class Reference:
    """A single retrieval reference returned by ``POST /api/v1/query``."""
    chunk_id: str
    source: str
    page: int
    score: float
    snippet: str


@dataclass
class QueryAnswer:
    """Parsed result of a ``POST /api/v1/query`` call."""
    answer: str
    references: list[Reference] = field(default_factory=list)
    trace_id: str = "-"
    no_result: bool = False
    retrieval_latency: float | None = None
    generation_latency: float | None = None
```

**`@dataclass` 是什么**：Python 标准库提供的"自动生成 `__init__`/`__repr__` 等方法"装饰器。

```python
@dataclass
class DocumentInfo:
    doc_id: str
    file_type: str
    num_chunks: int
    ingested_at: str
```

等价于手写：

```python
class DocumentInfo:
    def __init__(self, doc_id: str, file_type: str, num_chunks: int, ingested_at: str):
        self.doc_id = doc_id
        self.file_type = file_type
        self.num_chunks = num_chunks
        self.ingested_at = ingested_at
```

省去一大堆样板代码。

三个类的字段含义和 `schemas.py` 里的 Pydantic 模型一一对应，只是这里用 dataclass 而不是 Pydantic。

`QueryAnswer.references` 用 `field(default_factory=list)`：和 Pydantic 一样，可变默认值不能用 `[]`，得用工厂函数。

#### 14.2.5 `_api_call` 函数——统一调用 API

```python
def _api_call(
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float = API_TIMEOUT,
    **kwargs: Any,
) -> tuple[int, Any]:
    """Call the kb-rag API and return ``(status_code, data)``."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout:
        return 0, {"error": f"请求超时（{timeout}s）：{url}"}
    except requests.exceptions.ConnectionError as exc:
        return 0, {"error": f"无法连接 API（{url}）：{exc}"}
    except requests.exceptions.RequestException as exc:
        return 0, {"error": f"请求异常：{exc}"}

    try:
        data: Any = resp.json()
    except ValueError:
        data = resp.text
    return resp.status_code, data
```

**逐参数解释**：

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `method` | `str` | 必填 | HTTP 方法（"GET"/"POST"/"DELETE"） |
| `base_url` | `str` | 必填 | API 根地址 |
| `path` | `str` | 必填 | 接口路径（如 `/api/v1/documents`） |
| `timeout` | `float` | `120` | 超时秒数 |
| `**kwargs` | `Any` | - | 透传给 `requests.request` 的额外参数（如 `json=`、`files=`） |

**返回值**：`tuple[int, Any]`，元组 `(状态码, 数据)`。状态码 0 表示请求根本没发出去（网络错误）。

**关键设计**：
1. **从不抛异常**：所有异常都被捕获，转成 `(0, {"error": "..."})`，让调用方用 `st.error()` 友好展示
2. **状态码 0 是哨兵值**：表示"网络层失败"，区别于任何 HTTP 状态码
3. **`base_url.rstrip('/')`**：去掉末尾斜杠，避免 `http://...:8000//api/v1/...`

#### 14.2.6 文件大小格式化

```python
def _format_filesize(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
```

- 参数：`num_bytes: int`
- 返回值：`str`，如 `"1.5 MB"`
- 算法：循环除以 1024，直到小于 1024 或单位用完

#### 14.2.7 加载文档列表

```python
def _load_documents(base_url: str) -> list[DocumentInfo]:
    """Fetch the list of ingested documents from the API."""
    status, data = _api_call("GET", base_url, "/api/v1/documents")
    if status != 200 or not isinstance(data, list):
        return []
    docs: list[DocumentInfo] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            docs.append(
                DocumentInfo(
                    doc_id=str(item.get("doc_id", "")),
                    file_type=str(item.get("file_type", "")),
                    num_chunks=int(item.get("num_chunks", 0) or 0),
                    ingested_at=str(item.get("ingested_at", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return docs
```

- 参数：`base_url: str`
- 返回值：`list[DocumentInfo]`
- 调用 `GET /api/v1/documents`
- 任何异常都跳过（防御式编程），保证 UI 不会因为某个字段类型不对就崩溃
- `or 0` 处理 `None`：`int(None or 0) → int(0)`

#### 14.2.8 侧边栏

```python
def _render_sidebar() -> str:
    """Render the sidebar (title, API URL, upload, document list).

    Returns the currently configured API base URL.
    """
    st.sidebar.title("📚 kb-rag 知识库")

    default_url = os.environ.get("KB_RAG_API_URL", DEFAULT_API_URL)
    base_url = st.sidebar.text_input(
        "API Base URL",
        value=default_url,
        key="api_base_url",
        help="FastAPI 后端地址，例如 http://localhost:8000",
    )
```

- 函数签名：无参数，返回 `str`（当前 API URL）
- `st.sidebar.X`：所有组件都会渲染到左侧侧边栏
- `st.sidebar.text_input`：文本输入框
  - `value=default_url`：默认值（优先读环境变量）
  - `key="api_base_url"`：组件 ID，存进 session_state

#### 14.2.9 上传组件

```python
st.sidebar.divider()
st.sidebar.subheader("📤 文件上传")
uploaded = st.sidebar.file_uploader(
    "选择文件（可多选）",
    type=ACCEPTED_FILE_TYPES,
    accept_multiple_files=True,
    key="file_uploader",
)

if st.sidebar.button("上传并摄入", type="primary", use_container_width=True):
    if not uploaded:
        st.sidebar.warning("请先选择至少一个文件")
    else:
        _ingest_files(base_url, uploaded)
        st.session_state["docs"] = _load_documents(base_url)
```

- `st.sidebar.file_uploader`：文件上传组件
  - `type=ACCEPTED_FILE_TYPES`：只接受这些扩展名
  - `accept_multiple_files=True`：允许多选
- `st.sidebar.button`：按钮
  - `type="primary"`：主按钮样式（蓝色高亮）
  - `use_container_width=True`：占满容器宽度
- 点击后调 `_ingest_files`，然后刷新文档列表

#### 14.2.10 文档列表

```python
st.sidebar.divider()
st.sidebar.subheader("📋 已摄入文档")

if "docs" not in st.session_state:
    st.session_state["docs"] = _load_documents(base_url)

if st.sidebar.button("🔄 刷新列表", use_container_width=True):
    st.session_state["docs"] = _load_documents(base_url)

docs: list[DocumentInfo] = st.session_state.get("docs", []) or []
if not docs:
    st.sidebar.caption("暂无文档")
else:
    for doc in docs:
        cols = st.sidebar.columns([4, 1])
        with cols[0]:
            st.markdown(f"**`{doc.doc_id[:8]}…`**")
            st.caption(
                f"`{doc.file_type or '-'}` · {doc.num_chunks} chunks  \n"
                f"_{doc.ingested_at}_"
            )
        with cols[1]:
            if st.button("🗑️", key=f"del_{doc.doc_id}", help=f"删除文档 {doc.doc_id}"):
                _delete_doc(base_url, doc.doc_id)
                st.session_state["docs"] = _load_documents(base_url)
                st.rerun()
```

- `st.session_state`：Streamlit 的"会话状态"，跨请求保存数据。每个用户有独立的状态。
- `st.session_state["docs"]` 缓存文档列表，避免每次刷新都重新调 API
- `st.sidebar.columns([4, 1])`：把侧边栏分成 4:1 两列。左边显示信息，右边放删除按钮
- `doc.doc_id[:8]`：只显示前 8 位（UUID 太长）
- `st.rerun()`：手动触发 Streamlit 重新运行整个脚本（删除后立刻刷新 UI）

#### 14.2.11 上传逻辑

```python
def _ingest_files(base_url: str, files: list[Any]) -> None:
    """Upload every file via ``POST /api/v1/ingest`` and report per-file results."""
    progress = st.sidebar.progress(0.0, text="开始上传…")
    total = len(files)

    for idx, file in enumerate(files, start=1):
        label = f"摄入中 ({idx}/{total})：{file.name}"
        progress.progress(idx / total, text=label)

        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/api/v1/ingest",
                files={"file": (file.name, file.getvalue(), "application/octet-stream")},
                timeout=API_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            st.sidebar.error(f"❌ {file.name}：上传失败 - {exc}")
            continue

        if resp.status_code in (200, 201):
            data = resp.json() if resp.text else {}
            chunks = int(data.get("num_chunks", 0) or 0)
            errors = data.get("errors") or []
            trace = data.get("trace_id", "-")
            size_label = _format_filesize(file.size)
            if errors:
                st.sidebar.warning(
                    f"⚠️ {file.name} ({size_label})：{chunks} chunks · "
                    f"{len(errors)} 个错误  \ntrace: `{trace}`"
                )
                for err in errors[:3]:
                    st.sidebar.caption(f"  · {err}")
            else:
                st.sidebar.success(
                    f"✅ {file.name} ({size_label})：{chunks} chunks  \n"
                    f"trace: `{trace}`"
                )
        else:
            try:
                err_data = resp.json()
                msg = err_data.get("detail") or err_data
            except ValueError:
                msg = resp.text[:200] or f"HTTP {resp.status_code}"
            st.sidebar.error(f"❌ {file.name}：HTTP {resp.status_code} - {msg}")

    progress.empty()
```

- 参数：`base_url: str`、`files: list[Any]`（Streamlit 上传文件对象列表）
- 返回值：`None`
- **进度条**：`st.sidebar.progress(0.0, text="...")` 创建进度条，`progress.progress(idx/total, ...)` 更新进度
- 每个文件单独 POST 到 `/api/v1/ingest`
- 用 `files={"file": (filename, content, mime)}` 的格式（`requests` 库的标准用法）
- `file.getvalue()`：拿到 Streamlit 文件的字节内容
- `application/octet-stream`：通用的二进制 MIME 类型
- 成功：显示 chunks 数 + trace_id
- 有错误：显示 warning + 前 3 个错误
- HTTP 错误：显示状态码和详情

#### 14.2.12 主区域——问答

```python
def _render_main(base_url: str) -> None:
    """Render the main Q&A panel: input, filters, answer, references, history."""
    st.title("💬 知识库问答")
    st.caption(f"后端 API：`{base_url}`")

    with st.expander("🔍 过滤条件（可选）", expanded=False):
        col_src, col_tag, col_top = st.columns([2, 2, 1])
        with col_src:
            filter_source = st.text_input(
                "来源过滤 (source)", key="filter_source", placeholder="例如：report.pdf"
            )
        with col_tag:
            filter_tag = st.text_input(
                "标签过滤 (tag)", key="filter_tag", placeholder="例如：finance"
            )
        with col_top:
            top_n = st.number_input(
                "Top N", min_value=1, max_value=50, value=5, step=1, key="top_n"
            )

    question = st.text_input(
        "🧠 输入你的问题",
        key="question",
        placeholder="例如：这份文档主要讲了什么？",
        label_visibility="visible",
    )

    if st.button("🚀 提问", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请输入问题")
        else:
            _do_query(base_url, question.strip(), filter_source, filter_tag, int(top_n))

    st.divider()

    _render_current_answer()
    _render_history()
```

- 三列布局：source、tag、top_n
- `st.number_input`：数字输入框，限定 1~50，默认 5
- `st.text_input`：问题输入框
- 点击"提问"后调 `_do_query`
- 渲染当前回答 + 历史

#### 14.2.13 提问逻辑

```python
def _do_query(
    base_url: str,
    question: str,
    filter_source: str,
    filter_tag: str,
    top_n: int,
) -> None:
    """Execute a query and store the result in ``session_state``."""
    filters: dict[str, str] = {}
    if filter_source.strip():
        filters["source"] = filter_source.strip()
    if filter_tag.strip():
        filters["tag"] = filter_tag.strip()

    payload: dict[str, Any] = {"question": question, "top_n": top_n}
    if filters:
        payload["filters"] = filters

    with st.spinner("正在检索与生成…"):
        status, data = _api_call("POST", base_url, "/api/v1/query", json=payload)

    if status != 200 or not isinstance(data, dict):
        msg = data.get("error") if isinstance(data, dict) else str(data)
        st.error(f"查询失败：{msg}")
        return

    try:
        qa = QueryAnswer(
            answer=str(data.get("answer", "")),
            references=[
                Reference(
                    chunk_id=str(r.get("chunk_id", "")) if isinstance(r, dict) else "",
                    source=str(r.get("source", "")) if isinstance(r, dict) else "",
                    page=int(r.get("page", 0) or 0) if isinstance(r, dict) else 0,
                    score=float(r.get("score", 0.0) or 0.0) if isinstance(r, dict) else 0.0,
                    snippet=str(r.get("snippet", "")) if isinstance(r, dict) else "",
                )
                for r in data.get("references", []) or []
            ],
            trace_id=str(data.get("trace_id", "-")),
            no_result=bool(data.get("no_result", False)),
            retrieval_latency=data.get("retrieval_latency"),
            generation_latency=data.get("generation_latency"),
        )
    except (TypeError, ValueError) as exc:
        st.error(f"解析返回结果失败：{exc}")
        return

    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])
    prev = st.session_state.get("current_result")
    if prev is not None:
        history.insert(0, prev)
        st.session_state["history"] = history[:MAX_HISTORY]

    st.session_state["current_result"] = {
        "question": question,
        "answer": qa.answer,
        "references": [vars(r) for r in qa.references],
        "trace_id": qa.trace_id,
        "no_result": qa.no_result,
        "retrieval_latency": qa.retrieval_latency,
        "generation_latency": qa.generation_latency,
    }
```

- 5 个参数：API 地址、问题、source 过滤、tag 过滤、top_n
- 构造 payload，调 `POST /api/v1/query`
- `st.spinner`：旋转动画 + 文字提示
- 把返回的 dict 解析成 `QueryAnswer` 对象（每个字段都用 `data.get(...)` 防御性取值）
- **历史归档**：把之前的"当前结果"塞进 history 列表头部，限制最多 10 条
- **新当前结果**：把新问答存到 `session_state["current_result"]`
- `vars(r)`：把 dataclass 实例转成 dict（用于序列化）

#### 14.2.14 渲染当前回答与引用

```python
def _render_qa_block(item: dict[str, Any], *, expanded: bool) -> None:
    """Render a single Q&A record (answer + references + metadata)."""
    question = item.get("question", "")
    answer = item.get("answer", "")
    refs = item.get("references", []) or []
    no_result = bool(item.get("no_result", False))

    st.markdown(f"**❓ 问题：** {question}")

    if no_result:
        st.warning("⚠️ 未检索到相关内容（no_result=True）")

    st.markdown("**💡 回答：**")
    if answer:
        st.markdown(answer)
    else:
        st.caption("_(空回答)_")

    st.markdown("---")
    st.markdown(f"**📎 引用来源**（{len(refs)} 条）")
    if not refs:
        st.caption("无引用")
    else:
        for i, ref in enumerate(refs, start=1):
            source = str(ref.get("source", "") or "")
            page = ref.get("page", 0)
            try:
                page_i = int(page or 0)
            except (TypeError, ValueError):
                page_i = 0
            try:
                score_f = float(ref.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score_f = 0.0
            snippet = str(ref.get("snippet", "") or "")
            title = f"[{i}] {source} (p.{page_i}) score={score_f:.3f}"
            with st.expander(title, expanded=expanded and i == 1):
                if snippet:
                    st.info(snippet)
                else:
                    st.caption("_(无 snippet)_")
```

- 把回答用 Markdown 渲染（支持列表、加粗等）
- 每个引用做成可展开块（`st.expander`），第一块默认展开
- 防御性解析：所有字段都用 try/except 兜底

### 14.3 UI 与 API 的交互流程

```
┌─────────────┐                          ┌─────────────┐
│   用户浏览器  │                          │  Streamlit  │
│             │                          │  UI 进程    │
│             │                          │  :8501      │
└──────┬──────┘                          └──────┬──────┘
       │ 点"上传并摄入"                          │
       ├──────────────────────────────────────►│
       │                                       │ POST /api/v1/ingest
       │                                       ├───────────────────────┐
       │                                       │                       ▼
       │                                       │             ┌──────────────┐
       │                                       │             │  FastAPI     │
       │                                       │             │  :8000       │
       │                                       │             └──────┬───────┘
       │                                       │                    │ pipeline.ingest_file
       │                                       │                    ├─────────────► Qdrant
       │                                       │                    │
       │                                       │             ◄──────┤ IngestResponse
       │                                       │ ◄──────────┤ 201 JSON
       │                                       │ 显示成功 ✅
       │ ◄─────────────────────────────────────┤
       │                                       │
       │ 输入问题"X-2025 续航多久？"             │
       ├──────────────────────────────────────►│
       │                                       │ POST /api/v1/query
       │                                       ├───────────────────────┐
       │                                       │                       ▼
       │                                       │             ┌──────────────┐
       │                                       │             │  FastAPI    │
       │                                       │             └──────┬───────┘
       │                                       │                    │ pipeline.query
       │                                       │                    ├─► 检索 → 重排 → 护栏 → LLM
       │                                       │             ◄──────┤ QueryResponse
       │                                       │ ◄──────────┤ 200 JSON
       │                                       │ 渲染回答 + 引用
       │ ◄─────────────────────────────────────┤
```

UI 只是"传话员"，自己不实现 RAG 逻辑。

---

## 第 15 章 · 可观测性——日志、指标、追踪

### 15.1 为什么需要可观测性

线上系统出了问题，你怎么排查？
- 用户说"我提问没回答"，你怎么知道是 LLM 挂了、检索没找到、还是 embedding 服务超时？
- 流量突然涨 10 倍，你怎么知道是哪个接口被压爆了？
- 一次请求跨了 5 个服务，怎么把它们串起来？

**三大支柱**：
1. **日志（Logging）**：每一步发生了什么（事件级别）
2. **指标（Metrics）**：聚合数据，比如 QPS、延迟分位数（数值级别）
3. **追踪（Tracing）**：一个请求贯穿所有服务的链路（请求级别）

**打比方**：
- 日志 = 你写的日记（今天干了什么）
- 指标 = 你的体检报告（一年跑了多少公里、平均心率多少）
- 追踪 = 你的出差单（哪天去哪、坐了什么车、住了哪个酒店，全链路）

### 15.2 logging.py 逐行精读

完整代码：

```python
"""Structured logging configuration built on structlog.

Produces JSON-formatted log events and bridges structlog with the standard
``logging`` module so third-party libraries are captured. A ``trace_id`` can be
bound to the current context so that every log line within a request span is
correlatable.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import structlog

# ContextVar holding the active trace_id for the current async/task context.
_trace_id_var: ContextVar[str | None] = ContextVar("kb_rag_trace_id", default=None)

# Guard against repeated configuration (e.g. on re-import in tests).
_CONFIGURED: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Initialize structlog with JSON rendering and bridge to standard logging.

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_trace_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge: route stdlib logging through structlog so library logs are JSON too.
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    _CONFIGURED = True


def _inject_trace_id(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor that injects the current ``trace_id`` into the event dict."""
    trace_id = _trace_id_var.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logical logger name (e.g. module ``__name__``).

    Returns:
        A bound structlog logger. Configures logging on first call.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[str]:
    """Bind a ``trace_id`` to the current context for the duration of the block.

    Args:
        trace_id: Identifier of the active trace/span.

    Yields:
        The provided ``trace_id``.

    Example:
        >>> with bind_trace_id("abc123"):
        ...     get_logger().info("handling request")
    """
    token = _trace_id_var.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id_var.reset(token)
```

**逐段解释**：

#### 15.2.1 structlog 是什么

`structlog` 是 Python 的结构化日志库。和标准 `logging` 不同，它输出的是 JSON，方便机器解析。

**对比**：

标准 logging 输出：
```
2025-01-01 12:34:56 INFO mymodule: handling request user=alice
```

structlog 输出：
```json
{"event": "handling request", "level": "info", "timestamp": "2025-01-01T12:34:56Z", "user": "alice", "trace_id": "abc"}
```

JSON 格式可以丢给 ELK、Loki、Datadog 等日志系统，能直接查询和聚合。

#### 15.2.2 ContextVar

```python
_trace_id_var: ContextVar[str | None] = ContextVar("kb_rag_trace_id", default=None)
```

- `ContextVar`：Python 3.7+ 的"上下文变量"，类似线程本地变量（ThreadLocal），但支持 asyncio
- **为什么用 ContextVar 而不是 ThreadLocal**：FastAPI 是异步的，一个线程可能跑多个请求，ThreadLocal 会串号
- 默认值 `None`：没有 trace_id 时为空

#### 15.2.3 `configure_logging`

```python
def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
```

- 参数：`level: str = "INFO"`，日志级别名
- `_CONFIGURED` 全局变量防止重复配置（多次 import 不会重复初始化）
- `getattr(logging, level.upper(), logging.INFO)`：把字符串 `"INFO"` 转成数字常量（`logging.INFO = 20`）

```python
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _inject_trace_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
```

**processors 是什么**：structlog 的"流水线"，每条日志经过一系列处理器加工。每个处理器接收并返回一个 `event_dict`。

按顺序：
1. `merge_contextvars`：合并上下文变量（如绑定的 trace_id）
2. `add_log_level`：加上 `level` 字段
3. `TimeStamper(fmt="iso", utc=True)`：加 ISO 8601 UTC 时间戳
4. `_inject_trace_id`：自定义处理器，把当前 trace_id 注入
5. `StackInfoRenderer`：加上调用栈信息（如果开启）
6. `format_exc_info`：格式化异常信息
7. `JSONRenderer`：最后转成 JSON 字符串

```python
logging.basicConfig(
    level=log_level,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
```

- **桥接**：让标准 `logging` 库的日志也走 structlog（第三方库如 uvicorn、requests 的日志也能被结构化）
- `format="%(message)s"`：直接输出 message（因为 message 已经是 JSON 了）
- `StreamHandler()`：输出到 stderr

#### 15.2.4 `_inject_trace_id` 自定义处理器

```python
def _inject_trace_id(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    trace_id = _trace_id_var.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict
```

- 三个参数：`_logger`（不用，加下划线）、`_method_name`（不用）、`event_dict`（事件字典）
- 返回值：`dict[str, Any]`，加工后的字典
- 作用：把当前上下文的 trace_id 加到每条日志里

#### 15.2.5 `get_logger`

```python
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
```

- 参数：`name: str | None = None`，logger 名字（一般传 `__name__`）
- 返回值：`BoundLogger`，可调用的 logger 对象
- 没配置过就先配置（懒初始化）

#### 15.2.6 `bind_trace_id` 上下文管理器

```python
@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[str]:
    token = _trace_id_var.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id_var.reset(token)
```

- 参数：`trace_id: str`
- 返回值：`Iterator[str]`（因为是上下文管理器）
- 用法：

```python
with bind_trace_id("abc123"):
    logger.info("handling request")  # 这条日志会带上 trace_id="abc123"
# 离开 with 块后，trace_id 自动恢复成 None
```

- `set()` 返回一个 `token`，`reset(token)` 恢复之前的值
- **打比方**：你戴上"工牌 abc123"开始工作，所有动作都被记到 abc123 名下；下班时摘下工牌，恢复成"无工牌"状态

### 15.3 metrics.py 逐行精读

完整代码：

```python
"""Prometheus metrics for the kb-rag pipeline.

Exposes counters and histograms covering query volume, retrieval/generation
latency, no-result occurrences and ingestion throughput. Use the ``record_*``
helpers rather than touching the instruments directly so labels stay consistent.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---- Instruments ----

rag_query_total = Counter(
    "rag_query_total",
    "Total number of RAG queries processed.",
    ["status"],
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of the retrieval stage in seconds.",
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Latency of the generation stage in seconds.",
)

rag_no_result_total = Counter(
    "rag_no_result_total",
    "Number of queries that yielded no relevant context.",
)

rag_ingest_total = Counter(
    "rag_ingest_total",
    "Number of documents ingested.",
    ["file_type"],
)


# ---- Recording helpers ----


def record_query(status: str) -> None:
    """Increment :data:`rag_query_total` for the given status.

    Args:
        status: Outcome label, e.g. ``"ok"``, ``"error"``.
    """
    rag_query_total.labels(status=status).inc()


def record_retrieval_latency(seconds: float) -> None:
    """Observe a retrieval latency in seconds."""
    rag_retrieval_latency_seconds.observe(seconds)


def record_generation_latency(seconds: float) -> None:
    """Observe a generation latency in seconds."""
    rag_generation_latency_seconds.observe(seconds)


def record_no_result() -> None:
    """Increment the no-result counter."""
    rag_no_result_total.inc()


def record_ingest(file_type: str) -> None:
    """Increment the ingestion counter for a given file type.

    Args:
        file_type: File extension or type label, e.g. ``"pdf"``, ``"docx"``.
    """
    rag_ingest_total.labels(file_type=file_type).inc()
```

**逐段解释**：

#### 15.3.1 prometheus_client 库

`prometheus_client` 是 Prometheus 官方 Python 客户端。Prometheus 是开源监控系统，把指标存起来供查询。

**两种核心指标类型**：

- **Counter（计数器）**：只增不减。比如"总请求数""总错误数"。
- **Histogram（直方图）**：分布统计。比如"延迟分布在 0~0.1s 的有多少个，0.1~0.5s 的有多少个"。

#### 15.3.2 五个指标

```python
rag_query_total = Counter(
    "rag_query_total",
    "Total number of RAG queries processed.",
    ["status"],
)
```

- 名字：`rag_query_total`
- 描述：处理过的 RAG 查询总数
- 标签：`["status"]`（按状态分组，如 "ok"、"error"）

```python
rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of the retrieval stage in seconds.",
)
```

- 名字：`rag_retrieval_latency_seconds`
- 描述：检索阶段耗时
- 没有标签（不需要分组）

```python
rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Latency of the generation stage in seconds.",
)
```

- 同上，但记录生成阶段耗时

```python
rag_no_result_total = Counter(
    "rag_no_result_total",
    "Number of queries that yielded no relevant context.",
)
```

- 没检索到内容的查询数（"未命中"）

```python
rag_ingest_total = Counter(
    "rag_ingest_total",
    "Number of documents ingested.",
    ["file_type"],
)
```

- 摄入的文档数，按文件类型分组

#### 15.3.3 助手函数

每个指标都对应一个 `record_*` 函数：

```python
def record_query(status: str) -> None:
    rag_query_total.labels(status=status).inc()
```
- 参数：`status: str`
- `.labels(status=status)`：选择标签组合（创建一个带标签的"子计数器"）
- `.inc()`：自增 1

```python
def record_retrieval_latency(seconds: float) -> None:
    rag_retrieval_latency_seconds.observe(seconds)
```
- 参数：`seconds: float`
- `.observe(value)`：Histogram 专用方法，记录一次观测值

```python
def record_ingest(file_type: str) -> None:
    rag_ingest_total.labels(file_type=file_type).inc()
```
- 参数：`file_type: str`

**为什么包一层函数**：
- 防止直接操作指标对象时拼错标签名
- 未来加新标签时只需改函数，不用改调用方
- 函数名清晰表达意图

### 15.4 tracing.py 逐行精读

完整代码：

```python
"""OpenTelemetry tracing utilities for the kb-rag pipeline.

Provides a tracer provider initializer, a helper to read the current trace id,
and a decorator / context manager for creating spans.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

F = TypeVar("F", bound=Callable[..., Any])

_INITIALIZED: bool = False


def init_tracer(service_name: str = "kb-rag-api") -> TracerProvider:
    """Initialize and register a global :class:`TracerProvider`.

    Args:
        service_name: Service name reported on every span.

    Returns:
        The configured :class:`TracerProvider`.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return trace.get_tracer_provider()  # type: ignore[return-value]

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return provider


def get_current_trace_id() -> str | None:
    """Return the current span's trace id as a zero-padded hex string.

    Returns:
        The 32-char hex trace id, or ``None`` when no valid span is active.
    """
    span = trace.get_current_span()
    context = span.get_span_context()
    if context is not None and context.is_valid:
        return format(context.trace_id, "032x")
    return None


@contextmanager
def start_span(name: str) -> Iterator[Any]:
    """Context manager that starts a child span.

    Args:
        name: Span name.

    Yields:
        The active span.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        yield span


def trace_span(name: str) -> Callable[[F], F]:
    """Decorator wrapping a callable in a span named ``name``.

    Args:
        name: Span name.

    Returns:
        A decorator.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
```

**逐段解释**：

#### 15.4.1 OpenTelemetry 是什么

OpenTelemetry（OTel）是 CNCF 维护的可观测性标准，支持日志、指标、追踪三大支柱。

**核心概念**：
- **Span**：一段工作单元，有开始时间、结束时间、属性、事件
- **Trace**：由多个 Span 组成的链路树
- **TracerProvider**：全局 tracer 工厂
- **Exporter**：把 Span 数据发到后端（Jaeger、Zipkin、OTLP 等）

**打比方**：trace 是一次"出差"全程，span 是出差里的"打车去机场""过安检""登机"等步骤。

#### 15.4.2 顶部导入

```python
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
```

- `wraps`：装饰器工具，保留原函数的元数据
- `Resource`：OTel 的"资源"对象，标识服务身份
- `TracerProvider`：tracer 提供者
- `BatchSpanProcessor`：批量处理 span（提升性能）
- `ConsoleSpanExporter`：把 span 导出到控制台（生产应该换成 OTLP/Jaeger）

```python
F = TypeVar("F", bound=Callable[..., Any])
```
- 类型变量，用于装饰器的类型注解

#### 15.4.3 `init_tracer`

```python
def init_tracer(service_name: str = "kb-rag-api") -> TracerProvider:
    global _INITIALIZED
    if _INITIALIZED:
        return trace.get_tracer_provider()

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return provider
```

- 参数：`service_name: str = "kb-rag-api"`，服务名
- 返回值：`TracerProvider`
- 流程：
  1. 已初始化就返回当前 provider
  2. 创建 `TracerProvider`，资源带 `service.name` 标签
  3. 加一个 `BatchSpanProcessor`，用 `ConsoleSpanExporter` 把 span 打到 stdout
  4. 注册成全局 provider
  5. 标记已初始化

#### 15.4.4 `get_current_trace_id`

```python
def get_current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()
    if context is not None and context.is_valid:
        return format(context.trace_id, "032x")
    return None
```

- 参数：无
- 返回值：`str | None`，32 字符的十六进制 trace_id
- `format(context.trace_id, "032x")`：转成 32 位 hex（不足补 0）
- 用途：把 OTel 的 trace_id 暴露给日志和 API 响应，实现"日志 ↔ 追踪"互通

#### 15.4.5 `start_span` 上下文管理器

```python
@contextmanager
def start_span(name: str) -> Iterator[Any]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        yield span
```

- 参数：`name: str`，span 名字
- 返回值：`Iterator[Any]`
- 用法：

```python
with start_span("retrieval"):
    # 这块代码被一个名为 "retrieval" 的 span 包裹
    # 自动记录开始/结束时间
    docs = retrieve(query)
```

#### 15.4.6 `trace_span` 装饰器

```python
def trace_span(name: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

- 参数：`name: str`
- 返回值：装饰器函数
- 用法：

```python
@trace_span("my_function")
def my_function(x):
    return x * 2
```

每次调用 `my_function` 都会自动创建一个 span。

---

## 第 16 章 · Docker 部署——一键启动全套

### 16.1 Docker 是什么（大白话）

**问题**：在你电脑上能跑的程序，搬到服务器上常常报错"找不到依赖""版本不对""操作系统不兼容"。

**Docker 的解决方案**：把程序 + 它的所有依赖 + 操作系统环境打包成一个"集装箱"（叫 image 镜像），然后任何机器只要装了 Docker，就能跑这个集装箱，效果一模一样。

**打比方**：
- 传统部署 = 把家具拆散了搬到新家再组装（经常少螺丝）
- Docker 部署 = 把整个房间原封不动搬过去（连墙纸都一起搬）

**docker-compose**：一个 Docker 一次只能跑一个集装箱。但 RAG 项目要跑 5 个服务（API、UI、Qdrant、Prometheus、Grafana）。`docker-compose.yml` 就是"一群集装箱的清单"，一条命令把它们全启动。

### 16.2 docker-compose.yml 逐行精读

完整代码：

```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:v1.11.0
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - kb-rag-net

  api:
    build:
      context: ..
      dockerfile: Dockerfile
    env_file:
      - ../.env
    restart: unless-stopped
    depends_on:
      - qdrant
    ports:
      - "8000:8000"
    healthcheck:
      test:
        - "CMD"
        - "python"
        - "-c"
        - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)"
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    networks:
      - kb-rag-net

  ui:
    build:
      context: ../ui
      dockerfile: Dockerfile
    env_file:
      - ../.env
    restart: unless-stopped
    depends_on:
      - api
    ports:
      - "8501:8501"
    healthcheck:
      test:
        - "CMD"
        - "python"
        - "-c"
        - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status==200 else 1)"
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    networks:
      - kb-rag-net

  prometheus:
    image: prom/prometheus:v2.51.0
    restart: unless-stopped
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    networks:
      - kb-rag-net

  grafana:
    image: grafana/grafana:10.4.0
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - ./grafana/dashboards/dashboards.yml:/etc/grafana/provisioning/dashboards/dashboards.yml:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    networks:
      - kb-rag-net

volumes:
  qdrant_data:

networks:
  kb-rag-net:
    driver: bridge
```

**逐个服务解释**：

#### 16.2.1 顶部

```yaml
version: "3.9"
```
- Compose 文件版本（3.9 是较新的稳定版）

#### 16.2.2 qdrant——向量数据库

```yaml
qdrant:
  image: qdrant/qdrant:v1.11.0
  restart: unless-stopped
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_data:/qdrant/storage
  networks:
    - kb-rag-net
```

| 字段 | 值 | 含义 |
|------|----|------|
| `image` | `qdrant/qdrant:v1.11.0` | 用 Qdrant 官方镜像，版本 1.11.0 |
| `restart` | `unless-stopped` | 容器崩了自动重启，除非手动 stop |
| `ports` | `6333:6333`、`6334:6334` | 把容器内 6333/6334 端口映射到宿主机同号端口 |
| `volumes` | `qdrant_data:/qdrant/storage` | 把容器的 `/qdrant/storage` 挂到 named volume `qdrant_data` |
| `networks` | `kb-rag-net` | 加入自定义网络 |

- **6333**：Qdrant HTTP 接口（REST API）
- **6334**：Qdrant gRPC 接口（更高性能）
- **volumes 作用**：容器删了，数据保留在 named volume 里（生产必备）
- **networks 作用**：让 api 容器能用域名 `qdrant` 访问它

#### 16.2.3 api——FastAPI 后端

```yaml
api:
  build:
    context: ..
    dockerfile: Dockerfile
  env_file:
    - ../.env
  restart: unless-stopped
  depends_on:
    - qdrant
  ports:
    - "8000:8000"
  healthcheck:
    test:
      - "CMD"
      - "python"
      - "-c"
      - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)"
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 30s
  networks:
    - kb-rag-net
```

| 字段 | 值 | 含义 |
|------|----|------|
| `build.context` | `..` | 构建上下文是 docker-compose.yml 的上级目录（项目根） |
| `build.dockerfile` | `Dockerfile` | 用项目根的 Dockerfile |
| `env_file` | `../.env` | 加载根目录的 `.env` 文件作为环境变量 |
| `depends_on` | `qdrant` | 等 qdrant 启动后再启动 api |
| `ports` | `8000:8000` | FastAPI 监听 8000 |
| `healthcheck.test` | Python 脚本调 `/health` | 健康检查命令 |
| `healthcheck.interval` | `30s` | 每 30 秒检查一次 |
| `healthcheck.timeout` | `5s` | 单次检查超时 5 秒 |
| `healthcheck.retries` | `3` | 连续 3 次失败才标记不健康 |
| `healthcheck.start_period` | `30s` | 启动后 30 秒内失败不算数（给应用预热时间） |

**健康检查命令详解**：
```python
import urllib.request, sys
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)
```
- 用 Python 标准库（不需要装 curl）请求 `/health`
- 返回 200 退出码 0（健康），否则退出码 1（不健康）

#### 16.2.4 ui——Streamlit 前端

```yaml
ui:
  build:
    context: ../ui
    dockerfile: Dockerfile
  env_file:
    - ../.env
  restart: unless-stopped
  depends_on:
    - api
  ports:
    - "8501:8501"
  healthcheck:
    test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).status==200 else 1)"]
    interval: 30s
    timeout: 5s
    retries: 3
    start_period: 30s
  networks:
    - kb-rag-net
```

- 构建上下文是 `ui/` 目录（用 `ui/Dockerfile`）
- 依赖 `api`（先启动 api 再启动 ui）
- Streamlit 默认 8501 端口
- 健康检查路径是 `/_stcore/health`（Streamlit 内置健康端点）

#### 16.2.5 prometheus——监控抓取

```yaml
prometheus:
  image: prom/prometheus:v2.51.0
  restart: unless-stopped
  volumes:
    - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
  ports:
    - "9090:9090"
  networks:
    - kb-rag-net
```

- 用 Prometheus 官方镜像 v2.51.0
- 把 `prometheus/prometheus.yml` 挂载到容器的 `/etc/prometheus/prometheus.yml`（Prometheus 默认配置路径）
- `:ro` 表示 read-only（容器内不能改这个文件，安全）
- 暴露 9090 端口（Prometheus UI）

#### 16.2.6 grafana——可视化面板

```yaml
grafana:
  image: grafana/grafana:10.4.0
  restart: unless-stopped
  environment:
    - GF_SECURITY_ADMIN_USER=admin
    - GF_SECURITY_ADMIN_PASSWORD=admin
  volumes:
    - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    - ./grafana/dashboards/dashboards.yml:/etc/grafana/provisioning/dashboards/dashboards.yml:ro
    - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
  ports:
    - "3000:3000"
  networks:
    - kb-rag-net
```

- Grafana 官方镜像 10.4.0
- 环境变量配置默认账号密码（admin/admin）
- 三个 volumes：
  - 数据源配置（让 Grafana 知道 Prometheus 在哪）
  - 仪表盘加载配置（让 Grafana 自动加载仪表盘文件）
  - 仪表盘文件目录（具体的 JSON 仪表盘）
- 暴露 3000 端口

#### 16.2.7 顶级 volumes 和 networks

```yaml
volumes:
  qdrant_data:

networks:
  kb-rag-net:
    driver: bridge
```

- `volumes.qdrant_data`：声明 named volume（Docker 管理，持久化数据）
- `networks.kb-rag-net`：声明自定义网络
  - `driver: bridge`：用桥接网络（默认，单机部署）
  - 同一网络的容器可以用服务名互相访问（如 `api` 容器内 `http://qdrant:6333`）

### 16.3 启动和验证

#### 启动命令

```bash
cd infra
docker compose up --build
```

- `--build`：每次都重新构建镜像（首次必加，之后改代码也加）
- 加 `-d` 可以后台运行：`docker compose up -d --build`

#### 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 应返回
# {"status":"ok","vector_store":"qdrant","chunks":0,"version":"0.1.0"}

# API 文档
# 浏览器打开 http://localhost:8000/docs

# Streamlit UI
# 浏览器打开 http://localhost:8501

# Prometheus
# 浏览器打开 http://localhost:9090

# Grafana
# 浏览器打开 http://localhost:3000  账号 admin / admin
```

#### 常见问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `qdrant` 容器日志报"port already in use" | 宿主机 6333 被占 | 改 compose 端口映射 |
| `api` 容器启动后立刻退出 | `.env` 缺失或字段错 | 检查根目录 `.env` 文件 |
| `api` 健康检查一直失败 | Qdrant 还没起来或网络不通 | `docker compose logs qdrant` |
| UI 报"无法连接 API" | 浏览器跨域 / API 没起 | 检查 `api` 容器状态 |
| Grafana 没数据 | Prometheus 没抓到指标 | 访问 `http://localhost:9090/targets` 看抓取状态 |
| 上传大文件失败 | 50MB 限制 | 改 `MAX_FILE_SIZE` |

#### 停止

```bash
docker compose down            # 停止并删除容器，保留数据
docker compose down -v         # 同时删除 qdrant_data 数据卷（小心！）
```

---

## 第 17 章 · 监控配置

### 17.1 prometheus.yml 逐行精读

完整代码：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: "kb-rag-monitor"

scrape_configs:
  - job_name: "kb-rag-api"
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["api:8000"]
```

**逐行解释**：

#### global 区段
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: "kb-rag-monitor"
```

| 字段 | 值 | 含义 |
|------|----|------|
| `scrape_interval` | `15s` | 默认每 15 秒抓一次指标 |
| `evaluation_interval` | `15s` | 每 15 秒评估一次告警规则 |
| `external_labels.monitor` | `"kb-rag-monitor"` | 给所有时间序列加这个标签（多集群时区分） |

#### scrape_configs 区段
```yaml
scrape_configs:
  - job_name: "kb-rag-api"
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["api:8000"]
```

- `job_name`：任务名（标识一组抓取目标）
- `metrics_path: /metrics`：抓取的 HTTP 路径（FastAPI 的 Instrumentator 默认就暴露在这里）
- `scrape_interval: 15s`：本任务单独的抓取间隔
- `static_configs.targets: ["api:8000"]`：静态目标列表
  - `api` 是 docker-compose 里的服务名（容器间 DNS 解析）
  - `8000` 是 API 端口
  - **关键**：Prometheus 容器自己也在 `kb-rag-net` 网络里，所以能用服务名访问 api

### 17.2 Grafana 配置

#### 17.2.1 datasource.yml——数据源

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    uid: PBFA97CFB590B2093
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

| 字段 | 值 | 含义 |
|------|----|------|
| `apiVersion` | `1` | provisioning API 版本 |
| `name` | `Prometheus` | 数据源显示名 |
| `uid` | `PBFA97CFB590B2093` | 数据源 UID（仪表盘 JSON 里引用它） |
| `type` | `prometheus` | 数据源类型 |
| `access` | `proxy` | 通过 Grafana 后端代理（避免 CORS） |
| `url` | `http://prometheus:9090` | Prometheus 地址（容器网络内） |
| `isDefault` | `true` | 设为默认数据源 |
| `editable` | `true` | 允许在 UI 编辑 |

#### 17.2.2 dashboards.yml——仪表盘加载

```yaml
apiVersion: 1

providers:
  - name: "kb-rag dashboards"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

| 字段 | 值 | 含义 |
|------|----|------|
| `name` | `"kb-rag dashboards"` | 加载器名字 |
| `orgId` | `1` | Grafana 组织 ID（默认主组织） |
| `folder` | `""` | 仪表盘放在根目录 |
| `type` | `file` | 从文件系统加载 |
| `disableDeletion` | `false` | 允许在 UI 删除 |
| `editable` | `true` | 允许在 UI 编辑 |
| `updateIntervalSeconds` | `30` | 每 30 秒扫描一次目录（自动发现新仪表盘） |
| `options.path` | `/var/lib/grafana/dashboards` | 仪表盘 JSON 文件目录（被 volume 挂载） |

#### 17.2.3 rag.json——4 个监控面板

rag.json 文件结构（这里只看 4 个 panel 的标题和查询表达式）：

**Panel 1：RAG QPS（按状态码）**
- 标题：`"RAG QPS (by status)"`
- 类型：`timeseries`（时间序列折线图）
- 查询表达式：`sum(rate(rag_query_total[1m])) by (status)`
  - `rag_query_total[1m]`：过去 1 分钟的查询总数序列
  - `rate(...)`：算每秒速率（QPS）
  - `sum(...) by (status)`：按 status 标签分组求和
- 单位：`reqps`（请求数/秒）

**Panel 2：Retrieval latency（检索延迟）**
- 标题：`"Retrieval latency (s)"`
- 两条线：
  - p95：`histogram_quantile(0.95, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (le))`
  - p50：`histogram_quantile(0.50, ...)`
- 含义：
  - p50 = 中位数延迟（一半请求低于这个值）
  - p95 = 95% 的请求低于这个值（衡量"长尾"）
  - `histogram_quantile` 是 Prometheus 计算 Histogram 分位数的标准函数

**Panel 3：Generation latency（生成延迟）**
- 同上，但用 `rag_generation_latency_seconds_bucket`
- 生成延迟通常远大于检索延迟（因为要调 LLM）

**Panel 4：No-result queries（未命中速率）**
- 标题：`"No-result queries (rate)"`
- 查询：`rate(rag_no_result_total[5m])`
- 含义：每秒有多少查询没检索到内容
- 这个指标突增说明：要么用户问的问题离知识库太远，要么 embedding 模型坏了

**仪表盘全局设置**：
- `refresh: "15s"`：每 15 秒自动刷新
- `time.from: "now-1h"`：默认显示最近 1 小时
- `tags: ["kb-rag", "rag"]`：仪表盘标签

---

## 第 18 章 · 实用脚本

### 18.1 scripts/seed.py 逐行精读

完整代码：

```python
"""Batch ingestion script for the kb-rag API.

Walks a directory and uploads every matching file to the ``/api/v1/ingest``
endpoint, printing per-file outcomes and a final summary. The ``--clean`` flag
first deletes existing documents via ``GET/DELETE /api/v1/documents``.

Usage::

    python scripts/seed.py --dir data/raw --api http://localhost:8000

The script uses the ``requests`` library with a 10s timeout per call. It is
imported lazily so that ``--help`` works even when ``requests`` is not yet
installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

INGEST_PATH = "/api/v1/ingest"
DOCUMENTS_PATH = "/api/v1/documents"
TIMEOUT = 10.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the kb-rag knowledge base by uploading files via the ingest API.",
    )
    parser.add_argument(
        "--dir",
        default="data/raw",
        help="Directory of files to ingest (default: data/raw).",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="Base URL of the kb-rag API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for files to ingest (default: *).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing documents before ingesting.",
    )
    return parser.parse_args(argv)


def clean_existing(api_base: str, session: Any) -> int:
    """Delete existing documents via the API. Returns the count removed."""
    url = api_base.rstrip("/") + DOCUMENTS_PATH
    deleted = 0
    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[clean] failed to list documents: {exc}", file=sys.stderr)
        return 0

    payload = resp.json()
    if isinstance(payload, dict):
        candidates = (
            payload.get("documents")
            or payload.get("items")
            or payload.get("data")
            or []
        )
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []

    for item in candidates:
        doc_id = item.get("doc_id") if isinstance(item, dict) else item
        if not doc_id:
            continue
        try:
            r = session.delete(f"{url}/{doc_id}", timeout=TIMEOUT)
        except Exception as exc:
            print(f"[clean] error deleting {doc_id}: {exc}", file=sys.stderr)
            continue
        if r.status_code in (200, 204):
            deleted += 1
        elif r.status_code == 405:
            print(
                "[clean] DELETE not supported by API; skipping remaining docs.",
                file=sys.stderr,
            )
            break
        else:
            print(
                f"[clean] could not delete {doc_id}: HTTP {r.status_code}",
                file=sys.stderr,
            )
    return deleted


def ingest_file(api_base: str, file_path: Path, session: Any) -> dict[str, Any]:
    """Upload a single file to the ingest endpoint. Returns parsed JSON."""
    url = api_base.rstrip("/") + INGEST_PATH
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh)}
        resp = session.post(url, files=files, timeout=TIMEOUT)
    if resp.status_code >= 400:
        return {
            "doc_id": None,
            "num_chunks": 0,
            "errors": [f"HTTP {resp.status_code}: {resp.text[:200]}"],
        }
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, optionally clean, then ingest every file."""
    args = parse_args(argv)

    try:
        import requests
    except ImportError:
        print(
            "error: the 'requests' package is required. "
            "Install with: pip install requests",
            file=sys.stderr,
        )
        return 2

    root = Path(args.dir)
    if not root.exists():
        print(f"[seed] directory does not exist: {root}", file=sys.stderr)
        return 2

    files = sorted(p for p in root.glob(args.pattern) if p.is_file())
    if not files:
        print(f"[seed] no files matched {args.pattern!r} in {root}")
        return 0

    session = requests.Session()
    if args.clean:
        n = clean_existing(args.api, session)
        print(f"[seed] cleaned {n} existing document(s)")

    success = 0
    failure = 0
    total_chunks = 0
    for fp in files:
        try:
            result = ingest_file(args.api, fp, session)
        except Exception as exc:
            print(f"[seed] {fp.name}: ERROR {exc}")
            failure += 1
            continue

        doc_id = result.get("doc_id")
        num_chunks = int(result.get("num_chunks", 0) or 0)
        errors = result.get("errors") or []
        if errors or doc_id is None:
            failure += 1
            status = "FAIL"
        else:
            success += 1
            total_chunks += num_chunks
            status = "OK"
        print(
            f"[seed] {fp.name}: {status} doc_id={doc_id} "
            f"chunks={num_chunks} errors={len(errors)}"
        )
        for err in errors:
            print(f"        - {err}")

    print(
        f"\n[summary] success={success} failure={failure} "
        f"total_chunks={total_chunks} files_scanned={len(files)}"
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

**逐段解释**：

#### 18.1.1 顶部常量

```python
INGEST_PATH = "/api/v1/ingest"
DOCUMENTS_PATH = "/api/v1/documents"
TIMEOUT = 10.0
```

- 两个 API 路径常量
- 每次请求最多 10 秒（批量脚本不需要等太久）

#### 18.1.2 `parse_args`

```python
parser.add_argument("--dir", default="data/raw", ...)
parser.add_argument("--api", default="http://localhost:8000", ...)
parser.add_argument("--pattern", default="*", ...)
parser.add_argument("--clean", action="store_true", ...)
```

四个命令行参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--dir` | `data/raw` | 待摄入文件的目录 |
| `--api` | `http://localhost:8000` | API 地址 |
| `--pattern` | `*` | glob 模式（如 `*.pdf`） |
| `--clean` | `False` | 是否先清空已有文档 |

#### 18.1.3 `clean_existing`

```python
def clean_existing(api_base: str, session: Any) -> int:
```

- 参数：`api_base: str`、`session: Any`（requests.Session）
- 返回值：`int`，删除的文档数

流程：
1. GET `/api/v1/documents` 拿现有文档列表
2. 兼容多种响应结构（list / `{"documents": ...}` / `{"items": ...}` / `{"data": ...}`）
3. 对每个文档 DELETE
4. 遇到 405（DELETE 不支持）就停止
5. 任何异常都不中断，返回成功删除数

#### 18.1.4 `ingest_file`

```python
def ingest_file(api_base: str, file_path: Path, session: Any) -> dict[str, Any]:
    url = api_base.rstrip("/") + INGEST_PATH
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh)}
        resp = session.post(url, files=files, timeout=TIMEOUT)
    if resp.status_code >= 400:
        return {
            "doc_id": None,
            "num_chunks": 0,
            "errors": [f"HTTP {resp.status_code}: {resp.text[:200]}"],
        }
    return resp.json()
```

- 参数：`api_base: str`、`file_path: Path`、`session: Any`
- 返回值：`dict[str, Any]`，API 响应
- 以二进制读文件，用 multipart 上传
- 失败时返回一个"假响应"（保持接口一致），方便后续统一处理

#### 18.1.5 `main` 主流程

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import requests
    except ImportError:
        print("error: the 'requests' package is required. ...", file=sys.stderr)
        return 2
```

- **懒导入 requests**：`--help` 不需要 requests 也能跑（用户没装也能看帮助）
- 没装就退出码 2

```python
root = Path(args.dir)
if not root.exists():
    print(f"[seed] directory does not exist: {root}", file=sys.stderr)
    return 2

files = sorted(p for p in root.glob(args.pattern) if p.is_file())
```

- 检查目录存在
- `root.glob(args.pattern)`：用 glob 模式匹配文件
- `p.is_file()`：过滤掉子目录
- `sorted`：保证顺序可重现

```python
session = requests.Session()
if args.clean:
    n = clean_existing(args.api, session)
    print(f"[seed] cleaned {n} existing document(s)")
```

- 用 Session 复用 TCP 连接（更快）
- 需要 clean 就先清空

```python
for fp in files:
    try:
        result = ingest_file(args.api, fp, session)
    except Exception as exc:
        print(f"[seed] {fp.name}: ERROR {exc}")
        failure += 1
        continue
    ...
```

- 遍历所有文件
- 单文件失败不中断整个批次
- 统计成功/失败/总 chunks

```python
print(
    f"\n[summary] success={success} failure={failure} "
    f"total_chunks={total_chunks} files_scanned={len(files)}"
)
return 0 if failure == 0 else 1
```

- 打印汇总
- 全成功返回 0，否则返回 1（让 shell 知道脚本失败）

### 18.2 eval/rag_eval.py 逐行精读

完整代码（节选关键部分）：

```python
"""RAGAS-based evaluation harness for kb-rag.

Runs offline evaluation of the kb-rag query API against a JSONL testset using
RAGAS metrics (faithfulness, answer_relevancy, context_precision). Falls back
to lightweight heuristics when the ``ragas`` package is unavailable, marking
the report as ``heuristic`` mode.
"""

QUERY_PATH = "/api/v1/query"
TIMEOUT = 30.0
TEMPLATE_PATH = Path(__file__).parent / "report_template.md"

DEFAULT_TEMPLATE = """\
# kb-rag 评测报告

> 评测模式: {{MODE}}

## 整体指标

| 指标 | 数值 |
|------|------|
| Faithfulness | {{OVERALL_FAITHFULNESS}} |
| Answer Relevancy | {{OVERALL_ANSWER_RELEVANCY}} |
| Context Precision | {{OVERALL_CONTEXT_PRECISION}} |

## 逐条详情

{{DETAILS_TABLE}}
"""
```

**3 个核心常量**：
- `QUERY_PATH`：查询接口路径
- `TIMEOUT`：30 秒（比 seed 长，因为查询要调 LLM）
- `TEMPLATE_PATH`：报告模板文件路径（同目录的 `report_template.md`，不存在就用内置模板）

**3 个 RAGAS 指标含义**：
| 指标 | 含义 | 范围 |
|------|------|------|
| `faithfulness` | 忠实度：答案是否完全基于检索到的上下文（不编造） | 0~1，越高越好 |
| `answer_relevancy` | 答案相关性：答案是否切题 | 0~1，越高越好 |
| `context_precision` | 上下文精度：检索的片段是否真的相关 | 0~1，越高越好 |

#### 18.2.1 `parse_args`——命令行参数

```python
parser.add_argument("--testset", default="eval/testset.jsonl", ...)
parser.add_argument("--api", default="http://localhost:8000", ...)
parser.add_argument("--output", default="eval/report.md", ...)
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--testset` | `eval/testset.jsonl` | 测试集文件路径 |
| `--api` | `http://localhost:8000` | API 地址 |
| `--output` | `eval/report.md` | 报告输出路径 |

#### 18.2.2 `load_testset`——加载测试集

```python
def load_testset(path: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "ground_truth" not in obj:
                raise ValueError(
                    f"testset line {lineno}: missing question/ground_truth"
                )
            items.append(obj)
    return items
```

- 参数：`path: str`
- 返回值：`list[dict[str, str]]`
- 测试集格式：JSONL（每行一个 JSON 对象），每个对象含 `question` 和 `ground_truth`（标准答案）
- 缺字段就抛错，并指明第几行

#### 18.2.3 `call_query` 与 `build_contexts`

```python
def call_query(api_base: str, question: str, session: Any) -> dict[str, Any]:
    url = api_base.rstrip("/") + QUERY_PATH
    resp = session.post(url, json={"question": question}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def build_contexts(result: dict[str, Any]) -> list[str]:
    refs = result.get("references") or []
    contexts: list[str] = []
    for ref in refs:
        snippet = (
            (ref.get("snippet") or "").strip() if isinstance(ref, dict) else str(ref)
        )
        if snippet:
            contexts.append(snippet)
    return contexts
```

- `call_query`：POST 一个问题，返回完整响应
- `build_contexts`：从响应里提取非空 snippet 列表，作为 RAGAS 的 `contexts` 字段

#### 18.2.4 RAGAS 路径 vs 启发式路径

`_run_ragas` 函数尝试导入 `ragas` 库并跑评估，失败就返回 `({}, [], False)`。

`_run_heuristic` 函数用纯 Python 实现简化的指标：

```python
def _heuristic_score(sample: dict[str, Any]) -> dict[str, float]:
    answer = sample.get("answer", "") or ""
    contexts = sample.get("contexts", []) or []
    ground_truth = sample.get("ground_truth", "") or ""

    gt_tokens = set(_tokenize(ground_truth))
    ans_tokens = _tokenize(answer)
    ans_token_set = set(ans_tokens)
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens.update(_tokenize(c))

    if ans_token_set:
        faithfulness = len(ans_token_set & ctx_tokens) / len(ans_token_set)
    else:
        faithfulness = 0.0

    if gt_tokens:
        context_precision = len(gt_tokens & ctx_tokens) / len(gt_tokens)
    else:
        context_precision = 0.0

    if gt_tokens and ans_token_set:
        answer_relevancy = len(ans_token_set & gt_tokens) / max(len(gt_tokens), 1)
    else:
        answer_relevancy = 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
    }
```

- 用正则 `\W+` 分词（按非单词字符切）
- faithfulness：答案 token 中有多少能在上下文里找到
- context_precision：标准答案 token 中有多少能在上下文里找到
- answer_relevancy：答案 token 与标准答案 token 的重合度
- 这是简化版，比 RAGAS 粗糙，但不需要装额外库

#### 18.2.5 `render_report`——生成 Markdown 报告

```python
def render_report(
    overall: dict[str, float],
    per_row: list[dict[str, Any]],
    mode: str,
) -> str:
    if TEMPLATE_PATH.exists():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = DEFAULT_TEMPLATE

    detail_lines = [
        "| # | Question | Faithfulness | Answer Relevancy | Context Precision |",
        "|---|----------|--------------|------------------|-------------------|",
    ]
    for i, row in enumerate(per_row, 1):
        q = row.get("question", "").replace("|", "\\|")
        if len(q) > 80:
            q = q[:77] + "..."
        detail_lines.append(
            f"| {i} | {q} | {row.get('faithfulness', 0.0):.4f} | "
            f"{row.get('answer_relevancy', 0.0):.4f} | "
            f"{row.get('context_precision', 0.0):.4f} |"
        )
    details_table = "\n".join(detail_lines)

    return (
        template.replace("{{MODE}}", mode)
        .replace("{{OVERALL_FAITHFULNESS}}", f"{overall.get('faithfulness', 0.0):.4f}")
        .replace("{{OVERALL_ANSWER_RELEVANCY}}", f"{overall.get('answer_relevancy', 0.0):.4f}")
        .replace("{{OVERALL_CONTEXT_PRECISION}}", f"{overall.get('context_precision', 0.0):.4f}")
        .replace("{{DETAILS_TABLE}}", details_table)
    )
```

- 参数：`overall`（整体指标）、`per_row`（每行详情）、`mode`（"ragas" 或 "heuristic"）
- 返回值：`str`，完整 Markdown 文本
- 模板用 `{{XXX}}` 占位符，用字符串 replace 填充
- 长问题截断到 80 字符
- `|` 转义成 `\|`（Markdown 表格分隔符冲突）

#### 18.2.6 `main` 主流程

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    testset = load_testset(args.testset)
    # ... 调 API 收集 samples ...
    overall, per_row, ragas_ok = _run_ragas(samples)
    if ragas_ok:
        mode = "ragas"
    else:
        overall, per_row = _run_heuristic(samples)
        mode = "heuristic"
    report = render_report(overall, per_row, mode)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[eval] report written to {out_path} (mode={mode})")
    return 0
```

- 加载测试集 → 逐条调查询 API → 尝试 RAGAS → 失败回退启发式 → 渲染报告 → 写文件
- 输出 Markdown 报告，可以直接在 GitHub/IDE 里查看

**运行示例**：

```bash
python eval/rag_eval.py --testset eval/testset.jsonl --api http://localhost:8000
```

---

## 第 19 章 · 参数调优指南

### 19.1 所有参数总览表

下表汇总 kb-rag 项目里所有可调参数。配置位置指 `.env` 文件里的环境变量名。

| 参数名 | 配置位置 | 默认值 | 含义 | 什么时候调 |
|--------|----------|--------|------|----------|
| `LLM_PROVIDER` | `.env` | `zhipu` | LLM 提供商 | 想换模型时（openai/zhipu/ollama） |
| `LLM_MODEL` | `.env` | `glm-4-flash` | LLM 模型名 | 想换更猛的模型（glm-4、gpt-4o-mini） |
| `LLM_TEMPERATURE` | `.env` | `0.0` | 生成随机性 | RAG 用 0；聊天用 0.3~0.7 |
| `LLM_MAX_TOKENS` | `.env` | `2048` | 最大生成长度 | 答案被截断→调大；想省钱→调小 |
| `LLM_TOP_P` | `.env` | `1.0` | 核采样 | 想过滤低概率词→0.9 |
| `EMBEDDING_PROVIDER` | `.env` | `zhipu` | embedding 提供商 | 想换嵌入模型 |
| `EMBEDDING_MODEL` | `.env` | `embedding-3` | 嵌入模型名 | 想换更好的嵌入模型 |
| `EMBEDDING_DIM` | `.env` | `1024` | 嵌入维度 | 改模型时配套调 |
| `VECTOR_STORE` | `.env` | `qdrant` | 向量库类型 | 改成 chroma 跑单机 |
| `QDRANT_URL` | `.env` | `http://localhost:6333` | Qdrant 地址 | 远程部署 Qdrant 时改 |
| `QDRANT_COLLECTION` | `.env` | `kb_rag` | 集合名 | 多套知识库用不同集合 |
| `CHUNK_SIZE` | `.env` | `500` | 分块字符数 | 答案太碎→调大；不相关多→调小 |
| `CHUNK_OVERLAP` | `.env` | `50` | 分块重叠 | 上下文断裂→调大 |
| `TOP_K` | `.env` | `5` | 检索召回数 | 检索不到→调大；太慢→调小 |
| `RERANK_TOP_N` | `.env` | `3` | 重排后取前 N | 答案跑偏→调小；想多引用→调大 |
| `MIN_SCORE` | `.env` | `0.3` | 最低相关性分 | 不相关多→调高；检索少→调低 |
| `MAX_FILE_SIZE` | `ingest.py` 常量 | `52428800` | 上传大小上限（字节） | 想上传大文件→调大 |
| `ALLOWED_EXTENSIONS` | `ingest.py` 常量 | 8 种 | 允许的文件类型 | 想加 pptx 等→加扩展名 |
| `API_PORT` | `.env` | `8000` | API 端口 | 端口冲突时改 |
| `LOG_LEVEL` | `.env` | `INFO` | 日志级别 | 调试时改 `DEBUG` |
| `APP_ENV` | `.env` | `dev` | 环境名 | 生产用 `prod`，关 CORS 通配 |

### 19.2 常见场景调优

#### 场景 1：回答不准确（编造、答非所问）

**可能原因 + 调整**：
1. **LLM 编造** → `LLM_TEMPERATURE=0.0`（确定性输出）
2. **检索到不相关内容** → 调高 `MIN_SCORE=0.5`（过滤掉低分片段）
3. **检索到的片段太多混杂** → `RERANK_TOP_N=2`（只取最相关的 2 条）
4. **Prompt 不够约束** → 修改 `app/generation/prompts.py`（加"只基于以下上下文回答，不知道就说不知道"）
5. **分块太大包含多个主题** → `CHUNK_SIZE=300`（小块更聚焦）

#### 场景 2：检索不到内容（no_result=True）

**可能原因 + 调整**：
1. **知识库里真没有** → 上传更多相关文档
2. **嵌入模型不行** → 换更好的嵌入模型（如 `embedding-3` 升级到更大版本）
3. **召回太少** → `TOP_K=10`（多召回几个）
4. **分数阈值太高** → `MIN_SCORE=0.2` 或 `0.1`
5. **问题措辞和文档差异大** → 上传一份"问答对"文档让模型对齐
6. **分块太小信息不全** → `CHUNK_SIZE=800, CHUNK_OVERLAP=100`

#### 场景 3：太慢（请求超过 30 秒）

**可能原因 + 调整**：
1. **LLM 太慢** → 换更快的模型（如 `glm-4-flash` 而不是 `glm-4`）
2. **`max_tokens` 太大** → `LLM_MAX_TOKENS=1024` 或 `512`
3. **检索太多** → `TOP_K=3`（少检索几个）
4. **嵌入太慢** → 换更快的 embedding API
5. **Qdrant 远程延迟** → 把 Qdrant 部署到同机房
6. **并发太高** → 加水平扩容（多开几个 API 容器）

#### 场景 4：表格被切碎（信息断裂）

**可能原因 + 调整**：
1. **分块没考虑表格边界** → 用支持表格感知的分块器（项目 `app/chunking/` 里可能需要扩展）
2. **`CHUNK_SIZE` 太小** → 调到 `1000` 或 `2000`
3. **`CHUNK_OVERLAP` 不够** → 调到 `100` 或 `200`（让表格部分重叠）
4. **解析时表格错位** → 检查 `app/parsing/` 是否正确处理表格
5. **临时方案** → 把 Excel 转成 Markdown 表格再上传

#### 场景 5：回答被截断（半句话就结束）

**可能原因 + 调整**：
1. **`max_tokens` 不够** → `LLM_MAX_TOKENS=4096` 或 `8192`
2. **模型上限** → 检查模型支持的最大输出（如 `glm-4-flash` 可能限制 2048）
3. **prompt 太长挤占输出** → 减少 context（`RERANK_TOP_N=2`）或缩短片段（`CHUNK_SIZE=400`）
4. **网络超时** → 检查 `API_TIMEOUT=120` 是否够用

### 19.3 LLM 参数选择

详细内容请看 [`docs/LLM_PARAMS_GUIDE.md`](./LLM_PARAMS_GUIDE.md)，这里给出速查表：

| 场景 | temperature | max_tokens | top_p | 说明 |
|------|-------------|------------|-------|------|
| **RAG 知识问答（默认）** | **0.0** | **2048** | **1.0** | **本项目默认**，确定性输出 |
| 简短问答（只要一句话） | 0.0 | 256 | 1.0 | 省钱省时间 |
| 客服对话（要自然） | 0.3 | 1024 | 1.0 | 略带变化但不跑偏 |
| 多文档综合分析 | 0.0 | 4096 | 1.0 | 需要更长回答 |
| 头脑风暴（非 RAG） | 0.9 | 2048 | 1.0 | 鼓励创意 |
| 严格不跑题 | 0.1 | 1024 | 0.85 | 双重保险 |

**三个参数的关系**：
- `temperature`：控制随机性（最重要）
- `max_tokens`：控制回答长度（按 token 计费，影响成本）
- `top_p`：另一种控制随机性的方式（一般不动，让 temperature 主导）

**经验法则**：不要同时调 `temperature` 和 `top_p`，选一个控制即可。

**不同 LLM 提供商差异**：
- **智谱 GLM**：temperature 上限 1.0，max_tokens 上限 4096
- **OpenAI**：temperature 上限 2.0，max_tokens 视模型而定（gpt-4o-mini 最大 16384）
- **Ollama 本地**：参数名不同，`max_tokens` 叫 `num_predict`，要放在 `options` 里

---

## 第 20 章 · 快速上手

### 20.1 用智谱 GLM 跑起来

智谱 GLM 提供免费版 `glm-4-flash`，注册就送 token，适合新手试水。

#### 步骤 1：申请智谱 API Key

1. 访问 https://open.bigmodel.cn/
2. 注册账号（手机号即可）
3. 进入"API Keys"页面，点"添加新 API Key"
4. 复制 Key（形如 `xxxxxxxx.xxxxxxxx`）

#### 步骤 2：准备环境

```bash
# 克隆项目（如果还没有）
git clone <你的仓库地址>
cd kb-rag

# 安装依赖
pip install -e ".[dev]"
```

#### 步骤 3：配置 .env

在项目根目录创建 `.env` 文件：

```bash
# === LLM 配置 ===
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0

# === Embedding 配置 ===
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024

# === 智谱 API Key ===
ZHIPU_API_KEY=你刚才复制的Key

# === 向量库 ===
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=kb_rag
QDRANT_API_KEY=

# === 分块 ===
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# === 检索 ===
TOP_K=5
RERANK_TOP_N=3
MIN_SCORE=0.3

# === 服务 ===
API_PORT=8000
LOG_LEVEL=INFO
APP_ENV=dev
```

#### 步骤 4：启动 Qdrant（用 Docker）

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.11.0
```

#### 步骤 5：启动后端 API

```bash
# 方式 1：用 Makefile
make dev

# 方式 2：直接 uvicorn
uvicorn backend.main:app --reload --port 8000
```

看到 `Uvicorn running on http://0.0.0.0:8000` 就成功。

#### 步骤 6：启动 Streamlit UI

```bash
streamlit run ui/app.py
```

浏览器自动打开 `http://localhost:8501`。

#### 步骤 7：验证

```bash
curl http://localhost:8000/health
# 期望：{"status":"ok","vector_store":"qdrant","chunks":0,"version":"0.1.0"}
```

### 20.2 第一次使用

#### 20.2.1 上传文档

在 Streamlit UI 左侧侧边栏：
1. 点"选择文件（可多选）"或"Browse files"
2. 选一个 PDF / DOCX / MD / TXT 文件
3. 点"上传并摄入"按钮
4. 等待进度条走完，看到 ✅ 成功提示

也可以用 curl 上传：

```bash
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@report.pdf"
```

返回示例：
```json
{
  "doc_id": "a1b2c3d4-...",
  "num_chunks": 18,
  "file_type": "pdf",
  "trace_id": "abc123",
  "errors": []
}
```

#### 20.2.2 提问

在主区域输入框输入问题，例如：
```
这份文档主要讲了什么？
```

点"🚀 提问"按钮。等待几秒（取决于 LLM 速度），主区域会显示：
- **❓ 问题**：你刚才输入的问题
- **💡 回答**：LLM 生成的答案
- **📎 引用来源**：N 条，每条可展开看 snippet
- 底部 metadata：trace_id、检索延迟、生成延迟

#### 20.2.3 查看引用

点开引用 1 的展开块，能看到：
- 来源文件路径
- 页码（如果是 PDF）
- 相关性分数
- 截断后的原文片段（snippet）

**判断回答质量**：
- ✅ 引用片段支持答案 → 回答可信
- ⚠️ 引用片段和答案对不上 → 可能幻觉，调低 `LLM_TEMPERATURE`
- ⚠️ `no_result=True` → 没检索到，要么知识库里没有，要么 `MIN_SCORE` 太高

#### 20.2.4 历史问答

每次新提问，之前的问答会自动归档到底部"历史"区。最多保留 10 条（`MAX_HISTORY=10`）。

#### 20.2.5 删除文档

侧边栏"已摄入文档"列表里，每条文档右侧有个 🗑️ 按钮，点击即删除。删除会同时清掉向量库和注册表里的数据。

#### 20.2.6 用 curl 批量操作

```bash
# 列出所有文档
curl http://localhost:8000/api/v1/documents

# 查询带过滤
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"X-2025 重量是多少？","filters":{"source":"report.pdf"},"top_n":3}'

# 删除指定文档
curl -X DELETE http://localhost:8000/api/v1/documents/a1b2c3d4-...
```

#### 20.2.7 看监控

如果按第 16 章启动了全套 Docker：

- **API 文档**：http://localhost:8000/docs
- **Prometheus**：http://localhost:9090（看 `Targets` 状态、查指标）
- **Grafana**：http://localhost:3000（账号 admin/admin，看 4 个监控面板）
- **API 指标**：http://localhost:8000/metrics（原始 Prometheus 文本）

监控面板能看到：
- 实时 QPS（按成功/失败分组）
- 检索 p50/p95 延迟
- 生成 p50/p95 延迟
- 未命中速率

---

## 总结

第四部分覆盖了 kb-rag 项目"对外暴露 + 运维"的全部代码：

- **第 13 章**：FastAPI 后端，从 schemas、依赖访问器、main.py 到 4 个路由文件（ingest/query/documents/health），逐行精读
- **第 14 章**：Streamlit UI，dataclass 数据结构、API 调用、侧边栏、主区域、会话状态
- **第 15 章**：可观测性三大支柱——structlog 日志、Prometheus 指标、OpenTelemetry 追踪
- **第 16 章**：Docker Compose 部署，5 个容器（qdrant/api/ui/prometheus/grafana）逐个解释
- **第 17 章**：监控配置，Prometheus 抓取 + Grafana 数据源/仪表盘/4 个面板
- **第 18 章**：实用脚本，批量摄入 seed.py + RAGAS 评测 rag_eval.py
- **第 19 章**：参数调优，总览表 + 5 个常见场景 + LLM 参数速查
- **第 20 章**：用智谱 GLM 从零跑起来 + 第一次使用流程

读完这一部分，你应该能：
1. 理解 FastAPI 怎么把 Python 函数变成 HTTP 接口
2. 知道每个 API 接口接受什么参数、返回什么数据
3. 会用 Streamlit UI 或 curl 调用 API
4. 看懂 Docker Compose 文件，能一键启动全套
5. 看懂 Grafana 监控面板，知道哪个指标异常意味着什么
6. 出问题时知道调哪个参数

下一步建议：
- 跑通第一遍后，试着改 `LLM_TEMPERATURE` 看回答变化
- 试着用 `seed.py` 批量摄入几十个文档
- 跑一次 `rag_eval.py` 看评测报告
- 把项目部署到一台云服务器上，让团队成员一起用