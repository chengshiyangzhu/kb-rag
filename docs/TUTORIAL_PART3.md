# kb-rag 新手教程 · 第三部分：检索 → 重排 → 生成 → 编排

> 本教程面向只会 Python 基础语法、对 ML/NLP 一无所知的同学。
>
> 我们继续用一个统一的例子贯穿全文：知识库里有一条这样的"片段"（chunk）：
>
> ```text
> X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。
> ```
>
> 用户提出一个问题：
>
> ```text
> X-2025 续航多久？
> ```
>
> 接下来几章会看到：系统是怎么从一堆 chunk 里把这条捞出来的、
> 又是怎么把它喂给大模型（LLM）让它回答"X-2025 续航 12 小时 [1]"的。

---

## 第 9 章 · 检索——从知识库找到相关内容

### 9.1 为什么要混合检索

在第二部分里我们做了两件事：
1. 把 chunk 转成向量（embedding），存进向量库；
2. 把 chunk 文本也存进 BM25 倒排索引。

为什么两边都存？因为这两种检索方式各有各的"瞎眼区"：

| 检索方式 | 擅长 | 不擅长 |
| --- | --- | --- |
| 向量检索（dense） | 懂"意思"。问"电池能用多久"能找到"续航 12 小时" | 懂不了精确关键词。问"X-2025"时未必能精准匹配上 |
| BM25 检索（sparse） | 懂"关键词"。问"X-2025"能精确命中带这个词的文档 | 不懂"意思"。问"电池能用多久"找不到"续航 12 小时"，因为两个词不重叠 |

打个比方：
- **向量检索**像"按描述找人"——你说"找一个戴黑框眼镜、穿格子衫的程序员"，它能定位到老王，哪怕你不知道老王的名字。
- **BM25 检索**像"按名字查人"——你说"我要找老王"，它能精准命中老王，但你如果说"戴黑框眼镜那个"，它就懵了。

两个一起用，互为补充：既能懂意思，又能精确匹配关键词。这就是"混合检索"（hybrid retrieval）。

kb-rag 的混合检索分三步：
1. **向量检索** 找 top-N；
2. **BM25 检索** 也找 top-N；
3. **RRF 融合** 把两份结果按排名合并出最终 top-N。

本章会逐行精读五个文件：
- `app/retrieval/filters.py`——构造过滤条件；
- `app/retrieval/dense.py`——向量检索；
- `app/retrieval/bm25.py`——BM25 检索；
- `app/retrieval/fusion.py`——RRF 融合；
- `app/retrieval/hybrid.py`——把三者串起来。

---

### 9.2 filters.py 逐行精读

文件路径：`app/retrieval/filters.py`。

完整代码：

```python
"""Filter construction helpers for retrieval.

The ``build_filters`` helper produces a normalized filter dictionary understood
by vector store ``search`` implementations. Filters are intentionally permissive:
``None`` values are dropped so callers can pass partial arguments.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def build_filters(
    source: str | None = None,
    tag: str | None = None,
    doc_id: str | None = None,
    time_range: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """Build a normalized metadata filter dictionary.

    Args:
        source: Exact-match source path/URI.
        tag: Tag that must appear in the chunk's ``metadata.tag`` list.
        doc_id: Exact-match parent document id.
        time_range: ``(start, end)`` inclusive ``created_at`` window.

    Returns:
        A dict with only the keys for which a non-``None`` value was supplied.
        Returns an empty dict when no arguments are provided.
    """
    filters: dict[str, Any] = {}
    if source is not None:
        filters["source"] = source
    if tag is not None:
        filters["tag"] = tag
    if doc_id is not None:
        filters["doc_id"] = doc_id
    if time_range is not None:
        start, end = time_range
        filters["time_range"] = (start, end)
    return filters
```

逐行解释：

- 文件开头 `"""..."""` 是模块级 docstring（文档字符串），告诉别人这个文件是"过滤条件构造器"。
- `from __future__ import annotations`：让类型注解（`str | None` 这种写法）在所有 Python 版本下都能用。`|` 在 Python 3.10+ 才原生支持，加这一行可以兼容更老版本。
- `from datetime import datetime`：导入时间类型，因为 `time_range` 要用 `datetime` 表示起止时间。
- `from typing import Any`：导入 `Any` 类型，表示"什么类型都行"。

接着是 `build_filters` 函数：

- 函数名 `build_filters`：构造过滤条件。
- 四个参数全部默认 `None`，意思是"不传就是不过滤这个字段"：
  - `source: str | None = None`：参数名 `source`，类型"字符串或 None"，默认 None。含义：按来源（比如文件路径 `kb/manuals/x2025.pdf`）精确过滤。为什么过滤？比如你只想在产品手册里搜，不想把论坛帖子也搜出来。
  - `tag: str | None = None`：按标签过滤。一个 chunk 的 metadata 里可能带 `tag = ["音箱", "续航"]`，过滤时只要含这个标签的。
  - `doc_id: str | None = None`：按父文档 id 过滤。比如你只想在 doc_id=`abc123` 这份文档里搜。
  - `time_range: tuple[datetime, datetime] | None = None`：按 `created_at` 时间窗口过滤，是一个 `(start, end)` 二元组，闭区间。
- 返回值：`dict[str, Any]`，一个 dict，**只包含传了非 None 值的字段**。什么都不传就返回空 dict `{}`。

函数体逐行：

- `filters: dict[str, Any] = {}`：先建一个空 dict。
- `if source is not None: filters["source"] = source`：如果传了 source，就加到 dict 里。
- 后面三行同理。
- `if time_range is not None: start, end = time_range; filters["time_range"] = (start, end)`：先把二元组解包成 `start` 和 `end`，再塞回 dict。这一步看似多余，其实是为了确保存进去的一定是二元组（防止调用方传了列表之类的）。
- `return filters`：返回构造好的 dict。

**返回值示例**：

```python
build_filters()                                      # → {}
build_filters(source="kb/manuals/x2025.pdf")         # → {"source": "kb/manuals/x2025.pdf"}
build_filters(tag="续航", doc_id="abc123")            # → {"tag": "续航", "doc_id": "abc123"}
```

**为什么过滤很重要**：假设你的知识库里有 10 万条 chunk，其中 1 万条是已下架产品的资料。如果不过滤，用户问老产品时检索器可能把新产品也混进来；过滤一下 `time_range` 就能把搜索范围限定在某个时间段。

---

### 9.3 dense.py 逐行精读（向量检索）

文件路径：`app/retrieval/dense.py`。这是"按描述找人"那一半。

完整代码：

```python
"""Dense (vector) retriever backed by a vector store.

The vector store and embedder are passed in via duck typing to avoid a hard
dependency on ``app.stores`` and ``app.embedders`` (developed in parallel).

Expected duck-typed interfaces:

* ``store.search(query_vector: list[float], top_n: int, filters: dict | None) -> list[Chunk]``
* ``embedder.embed_query(query: str) -> list[float]``
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.models.document import Chunk
from app.observability.logging import get_logger
from app.observability.metrics import record_retrieval_latency

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieve chunks via dense vector similarity search.

    Args:
        store: Object exposing a ``search(query_vector, top_n, filters)``
            method that returns a list of :class:`Chunk`.
        embedder: Object exposing an ``embed_query(query)`` method returning
            a list of floats.
    """

    def __init__(self, store: Any, embedder: Any) -> None:
        """Initialize the retriever with a store and embedder."""
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Embed the query and search the vector store.

        Args:
            query: Natural-language query string.
            top_n: Maximum number of chunks to return.
            filters: Optional metadata filters forwarded to the store.

        Returns:
            A list of :class:`Chunk` ranked by similarity, with ``score``
            populated by the store.
        """
        logger.info("vector.retrieve.start", query=query[:80], top_n=top_n)
        import time

        start = time.perf_counter()
        try:
            query_vector: list[float] = self._embedder.embed_query(query)
            results = self._store.search(
                query_vector=query_vector,
                top_n=top_n,
                filters=filters,
            )
            logger.info(
                "vector.retrieve.done",
                returned=len(results),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return results
        finally:
            record_retrieval_latency(time.perf_counter() - start)
```

逐行解释：

- 模块 docstring 说明这个文件是"基于向量库的稠密（dense）检索器"，并强调 store 和 embedder 是用 **duck typing**（鸭子类型）传进来的——只要对象有 `search` 方法和 `embed_query` 方法就行，不强制要求是某个具体类。这样开发时可以并行写 store 和 embedder，互不阻塞。
- `from app.models.document import Chunk`：导入 `Chunk` 数据类（一个 chunk 就是知识库的一条目，第二部分讲过）。
- `from app.observability.logging import get_logger`：拿到一个 logger，用来打日志。
- `from app.observability.metrics import record_retrieval_latency`：导入记录"检索耗时"指标的函数，便于监控。
- `if TYPE_CHECKING: from pathlib import Path`：仅在类型检查时才导入 Path，运行时不导入，省内存。
- `logger = get_logger(__name__)`：用当前模块名建一个 logger。

接着是 `VectorRetriever` 类：

- 类名 `VectorRetriever`：向量检索器。
- 类 docstring 解释了两个参数 `store` 和 `embedder` 各自需要什么方法（鸭子类型）。

`__init__` 方法：

- `def __init__(self, store: Any, embedder: Any) -> None:`
  - 参数 `store: Any`：名字叫 store，类型 `Any`（鸭子类型，所以不写具体类型）。含义：向量库对象，必须有 `search(query_vector, top_n, filters)` 方法，返回 `list[Chunk]`。
  - 参数 `embedder: Any`：名字叫 embedder。含义：嵌入器对象，必须有 `embed_query(query)` 方法，返回 `list[float]`（一个向量）。
- 函数体只有两行：`self._store = store` 和 `self._embedder = embedder`，把传进来的对象存到私有属性里。前缀 `_` 表示"内部用，外部别直接碰"。

`retrieve` 方法：

- `def retrieve(self, query: str, top_n: int = 20, filters: dict | None = None) -> list[Chunk]:`
  - 参数 `query: str`：用户问题字符串，比如 `"X-2025 续航多久？"`。
  - 参数 `top_n: int = 20`：最多返回几条 chunk，默认 20。为什么是 20 而不是 5？因为这是"海选"阶段，多捞一些留给后面的 RRF 融合和重排去精挑。
  - 参数 `filters: dict | None = None`：可选的过滤条件，就是 9.2 节 `build_filters` 的产物。不传就不过滤。
- 返回值 `list[Chunk]`：按相似度从高到低排好的 chunk 列表，每个 chunk 的 `score` 字段是相似度分数（由 store 填好）。

函数体逐行：

- `logger.info("vector.retrieve.start", query=query[:80], top_n=top_n)`：打日志说"向量检索开始"，顺便记录查询前 80 字符（截断是为了不让日志过长）。
- `import time`：导入 time 模块（放在函数内导入是个小习惯，避免在模块顶部多一个依赖）。
- `start = time.perf_counter()`：记录开始时间戳。`perf_counter` 比 `time.time()` 精度高，适合测耗时。
- `try:`：用 try/finally 保证不管成功失败都会把耗时上报到 metrics。
- `query_vector: list[float] = self._embedder.embed_query(query)`：调用 embedder 把查询字符串变成向量。比如 `"X-2025 续航多久？"` 变成一个 768 维（取决于嵌入模型）的浮点数列表。
- `results = self._store.search(query_vector=query_vector, top_n=top_n, filters=filters)`：拿这个向量去向量库搜，返回 top_n 个最相似的 chunk。
- `logger.info("vector.retrieve.done", returned=len(results), elapsed_ms=int((time.perf_counter() - start) * 1000))`：打日志说"完成了"，记录返回了多少条、花了多少毫秒。
- `return results`：把结果返回给调用者。
- `finally: record_retrieval_latency(time.perf_counter() - start)`：无论如何都把这次检索耗时上报给 metrics 系统，便于做监控告警。

**返回值含义**：`list[Chunk]`，每个 Chunk 都带 `score`（相似度，越大越相似），按相似度从大到小排好。

---

### 9.4 bm25.py 逐行精读（BM25 检索）

文件路径：`app/retrieval/bm25.py`。这是"按名字查人"那一半。

完整代码：

```python
"""Sparse (BM25) retriever built on rank_bm25.

Tokenization uses :mod:`jieba` for CJK content when available and falls back
to character-level segmentation, while Latin text is split on whitespace.
The index can be persisted to disk via :meth:`BM25Retriever.persist`.
"""
from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Any

from app.models.document import Chunk
from app.observability.logging import get_logger
from app.observability.metrics import record_retrieval_latency

logger = get_logger(__name__)

# Match runs of CJK characters (Chinese/Japanese/Korean).
_CJK_PATTERN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)

try:  # jieba is optional
    import jieba  # type: ignore[import-untyped]

    _JIEBA_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    jieba = None  # type: ignore[assignment]
    _JIEBA_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    """Tokenize a piece of text for BM25 indexing.

    CJK substrings are routed through jieba (with a character-level fallback),
    and Latin substrings are split on whitespace and lowercased. Empty tokens
    are discarded.

    Args:
        text: Input text to tokenize.

    Returns:
        A list of non-empty token strings.
    """
    if not text:
        return []
    tokens: list[str] = []
    # Split the text into CJK and non-CJK segments so each can be handled
    # with the appropriate tokenizer.
    segments: list[tuple[bool, str]] = []
    buf = ""
    is_cjk = False
    for ch in text:
        ch_cjk = bool(_CJK_PATTERN.match(ch))
        if not buf:
            is_cjk = ch_cjk
            buf = ch
            continue
        if ch_cjk == is_cjk:
            buf += ch
        else:
            segments.append((is_cjk, buf))
            buf = ch
            is_cjk = ch_cjk
    if buf:
        segments.append((is_cjk, buf))

    for seg_cjk, seg in segments:
        if not seg.strip():
            continue
        if seg_cjk:
            if _JIEBA_AVAILABLE:
                tokens.extend(t for t in jieba.lcut(seg) if t.strip())
            else:
                tokens.extend(ch for ch in seg if not ch.isspace())
        else:
            tokens.extend(
                w.lower() for w in re.findall(r"\w+", seg, flags=re.UNICODE) if w
            )
    return tokens


class BM25Retriever:
    """BM25 sparse retriever with a pickled on-disk index.

    Args:
        index_path: Path to a pickle file used to persist the index.
    """

    def __init__(self, index_path: Path | str) -> None:
        """Initialize (or load) a BM25 index from ``index_path``."""
        self.index_path: Path = Path(index_path)
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._doc_ids: set[str] = set()
        self._removed_doc_ids: set[str] = set()
        self._bm25: Any | None = None
        self._load()

    # ---- Persistence ----

    def _load(self) -> None:
        """Load the index from disk if it exists, otherwise init empty."""
        if not self.index_path.exists():
            logger.info("bm25.init.empty", path=str(self.index_path))
            return
        try:
            with self.index_path.open("rb") as fh:
                payload = pickle.load(fh)
            self._chunks = payload.get("chunks", [])
            self._corpus = payload.get("corpus", [])
            self._doc_ids = set(payload.get("doc_ids", []))
            self._removed_doc_ids = set(payload.get("removed_doc_ids", []))
            self._rebuild_index()
            logger.info(
                "bm25.load.ok",
                path=str(self.index_path),
                count=len(self._chunks),
            )
        except Exception as exc:  # pragma: no cover - corrupt index
            logger.warning("bm25.load.failed", error=str(exc))
            self._chunks = []
            self._corpus = []
            self._doc_ids = set()
            self._removed_doc_ids = set()
            self._bm25 = None

    def persist(self) -> None:
        """Pickle the index, corpus, and metadata to ``index_path``."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._chunks,
            "corpus": self._corpus,
            "doc_ids": list(self._doc_ids),
            "removed_doc_ids": list(self._removed_doc_ids),
        }
        with self.index_path.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("bm25.persist.ok", path=str(self.index_path), count=len(self._chunks))

    # ---- Mutation ----

    def add(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks and rebuild the BM25 model.

        Args:
            chunks: Chunks to add to the index.
        """
        if not chunks:
            return
        for chunk in chunks:
            if chunk.id in {c.id for c in self._chunks}:
                # Skip duplicates by chunk id.
                continue
            self._chunks.append(chunk)
            self._corpus.append(_tokenize(chunk.text))
            self._doc_ids.add(chunk.metadata.doc_id)
        self._rebuild_index()
        logger.info("bm25.add.ok", added=len(chunks), total=len(self._chunks))

    def remove_by_doc(self, doc_id: str) -> None:
        """Mark all chunks of a document as removed and rebuild the index.

        Args:
            doc_id: Identifier of the document to remove.
        """
        if doc_id not in self._doc_ids and doc_id not in self._removed_doc_ids:
            return
        self._removed_doc_ids.add(doc_id)
        kept_chunks: list[Chunk] = []
        kept_corpus: list[list[str]] = []
        for chunk, tokens in zip(self._chunks, self._corpus, strict=False):
            if chunk.metadata.doc_id == doc_id:
                continue
            kept_chunks.append(chunk)
            kept_corpus.append(tokens)
        self._chunks = kept_chunks
        self._corpus = kept_corpus
        self._doc_ids.discard(doc_id)
        self._rebuild_index()
        logger.info("bm25.remove.ok", doc_id=doc_id, remaining=len(self._chunks))

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    # ---- Search ----

    def search(
        self,
        query: str,
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Score chunks against the query and return the top-N matches.

        Args:
            query: Natural-language query.
            top_n: Maximum number of chunks to return.
            filters: Optional metadata filters (``source``, ``tag``,
                ``doc_id``).

        Returns:
            A list of :class:`Chunk` with ``score`` set to the BM25 score,
            sorted descending.
        """
        start = time.perf_counter()
        try:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            indexed = list(enumerate(scores))
            # Apply filters up-front to skip irrelevant chunks.
            if filters:
                indexed = [
                    (i, s)
                    for i, s in indexed
                    if _matches_filters(self._chunks[i], filters)
                ]
            indexed.sort(key=lambda kv: kv[1], reverse=True)
            top = indexed[:top_n]
            results: list[Chunk] = []
            for i, score in top:
                if score <= 0:
                    continue
                chunk = self._chunks[i].model_copy(deep=True)
                chunk.score = float(score)
                results.append(chunk)
            logger.info(
                "bm25.search.done",
                query=query[:80],
                returned=len(results),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return results
        finally:
            record_retrieval_latency(time.perf_counter() - start)

    # ---- Internal ----

    def _rebuild_index(self) -> None:
        """Rebuild the underlying ``BM25Okapi`` model from the corpus."""
        if not self._corpus:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._corpus)


def _matches_filters(chunk: Chunk, filters: dict[str, Any]) -> bool:
    """Return ``True`` if ``chunk`` matches every entry in ``filters``.

    Supports ``source`` (exact), ``tag`` (membership), ``doc_id`` (exact)
    and ``time_range`` (tuple of datetimes compared against ``created_at``).
    """
    meta = chunk.metadata
    if "source" in filters and filters["source"] is not None:
        if meta.source != filters["source"]:
            return False
    if "tag" in filters and filters["tag"] is not None:
        if filters["tag"] not in (meta.tag or []):
            return False
    if "doc_id" in filters and filters["doc_id"] is not None:
        if meta.doc_id != filters["doc_id"]:
            return False
    if "time_range" in filters and filters["time_range"] is not None:
        start, end = filters["time_range"]
        if meta.created_at < start or meta.created_at > end:
            return False
    return True
```

#### BM25 算法原理（大白话）

BM25 是一种**关键词**检索算法。核心思想用一句话讲：**"罕见词命中比常见词命中更值钱，长文档的命中比短文档的命中稍微打折。"**

它由两部分组成：

- **TF（词频，Term Frequency）**：一个词在某篇文档里出现次数越多，越相关。"续航"在文档 A 里出现 3 次，比出现 1 次更相关。
- **IDF（逆文档频率，Inverse Document Frequency）**：一个词在所有文档里越罕见，越有区分度。"X-2025"这种专属词 IDF 高，"的"这种到处都是的词 IDF 低。

简化公式（不需要死记，看一眼理解即可）：

```
score = IDF × (TF × (k+1)) / (TF + k × (1 - b + b × doc_len/avg_doc_len))
```

- `k`：控制 TF 的"饱和速度"。`k` 越大，TF 涨到一定值后效果继续上升；`k` 越小越早饱和。常用值 1.2~2.0。
- `b`：控制文档长度归一化的强度。`b=1` 完全按长度惩罚长文档，`b=0` 不考虑长度。常用值 0.75。
- `doc_len/avg_doc_len`：当前文档长度 / 平均文档长度，长文档的命中打折。

**用到的库**：`rank_bm25`。这是一个纯 Python 实现的 BM25 库，安装即用，不依赖 C 扩展。我们用它的 `BM25Okapi` 类（Okapi 是 BM25 的一个常用变体，参数 k=1.5, b=0.75 是它默认值）。

**中文分词**：BM25 是基于"词"的算法，对中文需要先分词。kb-rag 用 `jieba`（"结巴"分词，业界最常用的中文分词库）做主分词器，没装 jieba 时回退到"字符级分词"（每个汉字当成一个 token）。

#### 模块顶部逐行

- 模块 docstring 说明：本文件是基于 `rank_bm25` 的稀疏（sparse）检索器；分词对中文用 jieba，没装 jieba 时用字符级回退，拉丁文按空白切；索引可以通过 `persist()` 存到磁盘。
- `import pickle`：Python 自带的序列化库，把 Python 对象存成二进制文件。
- `import re`：正则表达式库。
- `import time`：测耗时。
- `from pathlib import Path`：跨平台的路径对象。
- `from app.models.document import Chunk`：导入 Chunk 数据类。
- `from app.observability.logging import get_logger`：拿 logger。
- `from app.observability.metrics import record_retrieval_latency`：上报检索耗时。

- `_CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")`：编译一个正则，匹配 CJK（中日韩）字符。这几个 Unicode 范围分别覆盖日文假名、CJK 扩展 A、CJK 基本汉字、CJK 兼容汉字、半角假名。`re.compile` 把正则预编译好，重复匹配时更快。

- `try: import jieba; _JIEBA_AVAILABLE = True except Exception: jieba = None; _JIEBA_AVAILABLE = False`：尝试导入 jieba，成功就置标志为 True；失败（没装）就置为 False，不报错。这是"可选依赖"的常见写法。

#### `_tokenize` 函数逐行

```python
def _tokenize(text: str) -> list[str]:
```
- 参数 `text: str`：要分词的字符串。
- 返回 `list[str]`：分词后的 token 列表。

- `if not text: return []`：空字符串直接返回空列表。
- `tokens: list[str] = []`：结果列表。
- `segments: list[tuple[bool, str]] = []`：把输入拆成多段，每段标明是 CJK 还是非 CJK。
- 接下来的 `for ch in text:` 循环：扫描每个字符，把连续的 CJK 字符串成一段，连续的非 CJK 字符串成另一段。`buf` 是当前缓冲区，`is_cjk` 是当前段是不是 CJK。一旦遇到不同类型，就把 `buf` 存进 segments，开启新段。

**举例**：输入 `"X-2025 续航多久"`，扫描后得到 segments：
- `(False, "X-2025 ")`（非 CJK 段，含字母数字空格）
- `(True, "续航多久")`（CJK 段）

- `for seg_cjk, seg in segments:`：遍历每段。
- `if not seg.strip(): continue`：跳过纯空白段。
- `if seg_cjk:`：如果是 CJK 段：
  - `if _JIEBA_AVAILABLE: tokens.extend(t for t in jieba.lcut(seg) if t.strip())`：装了 jieba 就用 `jieba.lcut` 全切模式，把每个非空 token 加进 tokens。`jieba.lcut("续航多久")` 返回 `["续航", "多久"]`。
  - `else: tokens.extend(ch for ch in seg if not ch.isspace())`：没装 jieba 就按字符切，"续航多久" → `["续", "航", "多", "久"]`。
- `else:`（非 CJK 段）：
  - `tokens.extend(w.lower() for w in re.findall(r"\w+", seg, flags=re.UNICODE) if w)`：用正则 `\w+` 抓出所有"字母数字串"，每个转小写。`"X-2025 "` → `["x", "2025"]`。

**最终**：`_tokenize("X-2025 续航多久")` ≈ `["x", "2025", "续航", "多久"]`。

#### `BM25Retriever` 类逐行

```python
class BM25Retriever:
    def __init__(self, index_path: Path | str) -> None:
        self.index_path: Path = Path(index_path)
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._doc_ids: set[str] = set()
        self._removed_doc_ids: set[str] = set()
        self._bm25: Any | None = None
        self._load()
```

- 参数 `index_path: Path | str`：索引文件路径。类型"Path 或字符串"，传字符串也行，因为下面会 `Path(index_path)` 转一下。含义：BM25 索引会以 pickle 格式存在这个文件里。
- `self.index_path: Path = Path(index_path)`：把传入的路径转成 Path 对象存起来。
- `self._chunks: list[Chunk] = []`：存原始 chunk 对象列表。
- `self._corpus: list[list[str]] = []`：存每个 chunk 的分词结果列表，与 `_chunks` 一一对应。
- `self._doc_ids: set[str] = set()`：当前索引里所有文档 id 的集合（去重用）。
- `self._removed_doc_ids: set[str] = set()`：记录被删除的文档 id 集合，避免重复删除。
- `self._bm25: Any | None = None`：BM25 模型对象，初始为 None。
- `self._load()`：构造函数最后调用 `_load()`，从磁盘加载已有索引（如果有）。

#### `_load` 方法逐行

- `if not self.index_path.exists(): logger.info(...); return`：索引文件不存在就直接返回空索引，打条日志。
- `try:`：尝试加载索引，加载失败要恢复成空索引防止崩溃。
- `with self.index_path.open("rb") as fh: payload = pickle.load(fh)`：以二进制读模式打开文件，用 pickle 反序列化出 dict。
- `self._chunks = payload.get("chunks", [])`：从 dict 取出 chunks，没有就空列表。下面三行同理取 corpus、doc_ids、removed_doc_ids。
- `self._rebuild_index()`：基于加载的 corpus 重建 BM25Okapi 模型（因为模型本身不可序列化，必须重建）。
- `logger.info("bm25.load.ok", path=..., count=...)`：打日志说加载成功，加载了多少条。
- `except Exception as exc:`：万一文件损坏或反序列化失败：
  - `logger.warning("bm25.load.failed", error=str(exc))`：警告日志。
  - 把所有字段重置为空，避免半损坏的状态影响后续。

#### `persist` 方法逐行

- `self.index_path.parent.mkdir(parents=True, exist_ok=True)`：先确保父目录存在，没有就建。
- `payload = {"chunks": ..., "corpus": ..., "doc_ids": list(...), "removed_doc_ids": list(...)}`：把要存的字段打包成 dict。注意 `set` 不能直接 pickle，所以先转成 `list`。
- `with self.index_path.open("wb") as fh: pickle.dump(payload, fh)`：以二进制写模式打开，pickle 序列化写入。
- `logger.info(...)`：打日志说存好了。

#### `add` 方法逐行

```python
def add(self, chunks: list[Chunk]) -> None:
```
- 参数 `chunks: list[Chunk]`：要加入索引的 chunk 列表，无返回值。

- `if not chunks: return`：空列表直接返回。
- `for chunk in chunks:`：遍历每个 chunk。
- `if chunk.id in {c.id for c in self._chunks}: continue`：**去重**——如果这个 chunk id 已经在索引里，跳过。每次都扫一遍所有 chunk 是 O(N²) 的，量小没事；量大可优化成 set 查找。
- `self._chunks.append(chunk)`：把 chunk 加到列表末尾。
- `self._corpus.append(_tokenize(chunk.text))`：把 chunk 文本分词后加到 corpus。
- `self._doc_ids.add(chunk.metadata.doc_id)`：登记该文档 id。
- `self._rebuild_index()`：调用下面要讲的 `_rebuild_index`，重新构造 BM25 模型。**每次 add 都会重建**，量小没事；大批量摄入应该攒一批再 add。
- `logger.info("bm25.add.ok", added=len(chunks), total=len(self._chunks))`：打日志说加了 N 条，总共有 M 条。

#### `remove_by_doc` 方法逐行

```python
def remove_by_doc(self, doc_id: str) -> None:
```
- 参数 `doc_id: str`：要删的文档 id，无返回值。

- `if doc_id not in self._doc_ids and doc_id not in self._removed_doc_ids: return`：如果这个 doc_id 既不在当前索引里，也不在已删除集合里，直接返回，啥也不做。
- `self._removed_doc_ids.add(doc_id)`：登记到"已删除"集合。
- `kept_chunks: list[Chunk] = []` 和 `kept_corpus: list[list[str]] = []`：准备两个新列表装"保留下来的"。
- `for chunk, tokens in zip(self._chunks, self._corpus, strict=False):`：把 chunk 和它的分词结果配对遍历。
- `if chunk.metadata.doc_id == doc_id: continue`：跳过要删的文档的 chunk。
- `kept_chunks.append(chunk); kept_corpus.append(tokens)`：保留其余的。
- `self._chunks = kept_chunks; self._corpus = kept_corpus`：替换。
- `self._doc_ids.discard(doc_id)`：从 doc_ids 集合里移除该 id。`discard` 不存在也不报错（区别于 `remove`）。
- `self._rebuild_index()`：重建索引。

#### `count` 方法

- 一行：`return len(self._chunks)`，返回当前索引的 chunk 数量。

#### `search` 方法逐行

```python
def search(self, query: str, top_n: int = 20, filters: dict | None = None) -> list[Chunk]:
```
- 参数 `query: str`：查询字符串。
- 参数 `top_n: int = 20`：最多返回几条，默认 20。
- 参数 `filters: dict | None = None`：可选过滤条件。
- 返回 `list[Chunk]`：按 BM25 分数从高到低排好的 chunk 列表，每个 chunk 的 `score` 是 BM25 分数。

- `start = time.perf_counter()`：开始计时。
- `try:`：try/finally 保证上报耗时。
- `if self._bm25 is None or not self._chunks: return []`：索引还没建或没数据，直接返回空。
- `tokens = _tokenize(query)`：把查询分词。比如 `"X-2025 续航多久"` → `["x", "2025", "续航", "多久"]`。
- `if not tokens: return []`：分词结果是空（比如纯标点）就返回空。
- `scores = self._bm25.get_scores(tokens)`：调用 `BM25Okapi.get_scores`，对每篇文档算一个分数，返回 `numpy.ndarray`，长度等于语料库大小。
- `indexed = list(enumerate(scores))`：把分数列表变成 `[(0, score0), (1, score1), ...]`，i 是文档在语料库里的下标。
- `if filters: indexed = [(i, s) for i, s in indexed if _matches_filters(self._chunks[i], filters)]`：**先过滤再排序**，把不符合过滤条件的 chunk 直接踢掉。
- `indexed.sort(key=lambda kv: kv[1], reverse=True)`：按分数降序排。`reverse=True` 表示从大到小。
- `top = indexed[:top_n]`：取前 top_n 个。
- `results: list[Chunk] = []`：准备结果列表。
- `for i, score in top:`：遍历 top_n。
- `if score <= 0: continue`：跳过分数 ≤ 0 的（完全没匹配到关键词的）。
- `chunk = self._chunks[i].model_copy(deep=True)`：**深拷贝**一份 chunk。为什么要深拷贝？因为下面要改 `score`，不深拷贝就会改到索引里的原始对象，下次再搜就拿到错的分数。
- `chunk.score = float(score)`：把 BM25 分数写进 chunk。
- `results.append(chunk)`：加进结果。
- `logger.info("bm25.search.done", ...)`：打日志。
- `return results`：返回。
- `finally: record_retrieval_latency(...)`：上报耗时。

#### `_rebuild_index` 方法逐行

- `if not self._corpus: self._bm25 = None; return`：语料库为空就把模型置 None。
- `from rank_bm25 import BM25Okapi`：函数内导入，避免在没装 rank_bm25 的环境里整个模块就 import 失败。
- `self._bm25 = BM25Okapi(self._corpus)`：用当前 corpus 构造一个 BM25Okapi 实例。`BM25Okapi(corpus)` 的 `corpus` 参数是 `list[list[str]]`——每个文档是一个 token 列表。构造时会计算 IDF、平均文档长度等统计量。

#### `_matches_filters` 函数逐行

- 参数 `chunk: Chunk` 和 `filters: dict[str, Any]`。
- 返回 `bool`：是否通过过滤。
- `meta = chunk.metadata`：拿到 chunk 的元数据。
- `if "source" in filters and filters["source"] is not None: if meta.source != filters["source"]: return False`：如果要求 source 等于某值，但 chunk 的 source 不等，就 return False。
- `if "tag" in filters ...`：tag 是 **成员检查**（`in`），只要 chunk 的 tag 列表里有这个 tag 就行。
- `if "doc_id" in filters ...`：doc_id 是**精确匹配**。
- `if "time_range" in filters ...`：`if meta.created_at < start or meta.created_at > end: return False`——时间在 [start, end] 闭区间内才算通过。
- `return True`：所有条件都通过才 return True。

---

### 9.5 fusion.py 逐行精读（RRF 融合）

文件路径：`app/retrieval/fusion.py`。这一步把两份检索结果合并成一份。

完整代码：

```python
"""Reciprocal Rank Fusion (RRF) for combining multiple result lists.

RRF assigns each candidate a score of ``1 / (k + rank)`` from every list it
appears in, then sums the contributions. It is a robust, parameter-light way
to fuse dense and sparse retrievers.
"""
from __future__ import annotations

from app.models.document import Chunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RRFFusion:
    """Fuse multiple ranked chunk lists via Reciprocal Rank Fusion."""

    def fuse(
        self,
        result_lists: list[list[Chunk]],
        k: int = 60,
        top_n: int = 20,
    ) -> list[Chunk]:
        """Fuse ranked lists with RRF and return the top-N chunks.

        Args:
            result_lists: A list of ranked chunk lists. Within each list,
                rank 0 is the most relevant item.
            k: RRF smoothing constant (default 60, per the original paper).
            top_n: Maximum number of fused chunks to return.

        Returns:
            A list of :class:`Chunk` sorted by summed RRF score descending,
            with ``score`` set to the fused score. Chunks sharing the same
            ``id`` are merged across lists.
        """
        scores: dict[str, float] = {}
        # Keep one representative Chunk per id (the first occurrence).
        representatives: dict[str, Chunk] = {}

        for result_list in result_lists:
            for rank, chunk in enumerate(result_list):
                cid = chunk.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in representatives:
                    representatives[cid] = chunk

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        results: list[Chunk] = []
        for cid in ranked_ids[:top_n]:
            chunk = representatives[cid].model_copy(deep=True)
            chunk.score = scores[cid]
            results.append(chunk)
        logger.info(
            "rrf.fuse.done",
            input_lists=len(result_lists),
            output=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return results
```

#### RRF 公式

RRF（Reciprocal Rank Fusion，倒数排名融合）的核心公式：

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

- `d`：某个候选文档（chunk）。
- `i`：第 i 份结果列表（向量检索结果是一份、BM25 是另一份）。
- `rank_i(d)`：文档 d 在第 i 份列表里的排名，从 0 开始（rank=0 是最相关）。
- `k`：平滑常数，默认 **60**。

**为什么 k=60**：这个值来自 RRF 原始论文（Cormack 等 2009）。k 越大，排名差距的影响越被"压平"（第 1 名和第 10 名差距变小）；k 越小，靠前的名次优势越大。60 是论文实验得出的"鲁棒性强、对各种数据集都还行"的折中值。kb-rag 沿用这个默认。

#### 手算例子

假设向量检索 top3 和 BM25 top3 都是这些 chunk（按排名从 0 开始）：

向量检索结果：
- rank 0: chunk A（"X-2025 智能音箱，续航 12 小时..."）
- rank 1: chunk B（"蓝牙 5.0 双模连接"）
- rank 2: chunk C（"长按电源键 3 秒开机"）

BM25 检索结果：
- rank 0: chunk A（"X-2025 智能音箱..."，因为 query 里 "X-2025" 精确命中）
- rank 1: chunk D（"续航 12 小时，充电 2 小时"）
- rank 2: chunk B（"蓝牙 5.0"）

取 k=60，计算每个 chunk 的 RRF 分数：

- chunk A：1/(60+0) [来自向量] + 1/(60+0) [来自 BM25] = 1/60 + 1/60 = **0.0333**
- chunk B：1/(60+1) [向量] + 1/(60+2) [BM25] = 1/61 + 1/62 ≈ **0.0325**
- chunk C：1/(60+2) [向量] = 1/62 ≈ **0.0161**
- chunk D：1/(60+1) [BM25] = 1/61 ≈ **0.0164**

排序：A > B > D > C。

**关键观察**：chunk A 在两份列表里都排第一，所以它的 RRF 分数最高（被两边"投票"），融合后稳居第一。这就是 RRF 的妙处——**两份检索都认同的 chunk 会冒到最前面**。

#### `fuse` 方法逐行

- `def fuse(self, result_lists: list[list[Chunk]], k: int = 60, top_n: int = 20) -> list[Chunk]:`
  - 参数 `result_lists: list[list[Chunk]]`：多份已排序的 chunk 列表，比如 `[向量检索结果, BM25检索结果]`。每份列表内部 rank 0 是最相关。
  - 参数 `k: int = 60`：RRF 平滑常数，默认 60，理由见上。
  - 参数 `top_n: int = 20`：最终返回前 top_n 个。
- 返回 `list[Chunk]`：按 RRF 分数降序排好的 chunk 列表，每个 chunk 的 `score` 是融合后的 RRF 分数。**同 id 的 chunk 会被合并**（两份列表里同 id 的 chunk 视为同一个）。

函数体：

- `scores: dict[str, float] = {}`：dict，key 是 chunk id，value 是累加的 RRF 分数。
- `representatives: dict[str, Chunk] = {}`：dict，key 是 chunk id，value 是"代表 chunk"（每个 id 只保留第一次出现的 chunk 对象）。
- `for result_list in result_lists:`：遍历每份结果列表。
- `for rank, chunk in enumerate(result_list):`：`enumerate` 给每个 chunk 配上它的排名 rank（0, 1, 2, ...）。
- `cid = chunk.id`：拿 chunk 的唯一 id。
- `scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)`：累加 RRF 分数。`scores.get(cid, 0.0)` 是"取 cid 对应的分数，没有就返回 0.0"。每个列表贡献 `1/(k+rank)`。
- `if cid not in representatives: representatives[cid] = chunk`：第一次见到这个 id 就存进 representatives，后面再见同 id 不覆盖。
- `ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)`：按 RRF 分数从大到小排序所有 chunk id。
- `results: list[Chunk] = []`：结果列表。
- `for cid in ranked_ids[:top_n]:`：取前 top_n 个 id。
- `chunk = representatives[cid].model_copy(deep=True)`：深拷贝代表 chunk，避免修改原对象。
- `chunk.score = scores[cid]`：把 RRF 分数写进去。
- `results.append(chunk)`：加进结果。
- `logger.info("rrf.fuse.done", input_lists=len(result_lists), output=len(results), top_score=...)`：打日志说合并了几份列表、产出几条、最高分多少。
- `return results`：返回。

---

### 9.6 hybrid.py 逐行精读

文件路径：`app/retrieval/hybrid.py`。把上面三块串成一个完整的混合检索器。

完整代码：

```python
"""Hybrid retriever combining dense and sparse retrieval with RRF fusion.

Both retrievers run concurrently via :mod:`concurrent.futures`, and their
results are merged with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.models.document import Chunk
from app.observability.logging import get_logger
from app.observability.metrics import record_retrieval_latency
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import VectorRetriever
from app.retrieval.fusion import RRFFusion

logger = get_logger(__name__)


class HybridRetriever:
    """Combine :class:`VectorRetriever` and :class:`BM25Retriever` via RRF.

    Args:
        vector_retriever: Dense retriever instance.
        bm25_retriever: Sparse BM25 retriever instance.
        rrf_k: RRF smoothing constant (default 60).
        rrf: Optional pre-constructed :class:`RRFFusion` instance.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        rrf: RRFFusion | None = None,
    ) -> None:
        """Initialize the hybrid retriever."""
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k
        self._rrf = rrf or RRFFusion()

    def retrieve(
        self,
        query: str,
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Run dense and sparse retrieval concurrently and fuse via RRF.

        Args:
            query: Natural-language query.
            top_n: Desired final result count.
            filters: Optional metadata filters forwarded to both retrievers.

        Returns:
            A fused, ranked list of :class:`Chunk` of length ``<= top_n``.
        """
        start = time.perf_counter()
        try:
            # Each retriever pulls its own top_n so fusion has room to reorder.
            per_retriever_n = max(top_n, 20)
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_future = pool.submit(
                    self._vector.retrieve,
                    query=query,
                    top_n=per_retriever_n,
                    filters=filters,
                )
                sparse_future = pool.submit(
                    self._bm25.search,
                    query=query,
                    top_n=per_retriever_n,
                    filters=filters,
                )
                dense_results = dense_future.result()
                sparse_results = sparse_future.result()
            fused = self._rrf.fuse(
                [dense_results, sparse_results],
                k=self._rrf_k,
                top_n=top_n,
            )
            logger.info(
                "hybrid.retrieve.done",
                query=query[:80],
                dense=len(dense_results),
                sparse=len(sparse_results),
                fused=len(fused),
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return fused
        finally:
            record_retrieval_latency(time.perf_counter() - start)
```

#### `__init__` 方法逐行

- 参数 `vector_retriever: VectorRetriever`：向量检索器实例。
- 参数 `bm25_retriever: BM25Retriever`：BM25 检索器实例。
- 参数 `rrf_k: int = 60`：RRF 平滑常数，默认 60。理由见 9.5。
- 参数 `rrf: RRFFusion | None = None`：可选，调用方可以传一个预先构造好的 RRFFusion 实例（比如想做 mock 测试）；不传就 `RRFFusion()` 自己 new 一个。

- `self._vector = vector_retriever`：存向量检索器。
- `self._bm25 = bm25_retriever`：存 BM25 检索器。
- `self._rrf_k = rrf_k`：存 k 值。
- `self._rrf = rrf or RRFFusion()`：`A or B` 在 Python 里是"如果 A 是 falsy（None、空等）就用 B"。所以没传 rrf 就 new 一个。

#### `retrieve` 方法逐行

- 参数 `query: str`：查询字符串。
- 参数 `top_n: int = 20`：最终想要的结果数。
- 参数 `filters: dict | None = None`：过滤条件，会同时转发给两个检索器。
- 返回 `list[Chunk]`：长度不超过 top_n 的融合结果。

- `start = time.perf_counter()`：开始计时。
- `per_retriever_n = max(top_n, 20)`：每个检索器各自至少取 20 条。**为什么？** 因为融合阶段会重排，候选越多重排空间越大。如果只要 top_n=5 条最终结果，但每个检索器只取 5 条，那 RRF 几乎没东西可融合。
- `with ThreadPoolExecutor(max_workers=2) as pool:`：开一个**线程池**，最多 2 个工作线程。
  - 为什么用线程池？因为向量检索和 BM25 检索互相独立，可以**并发执行**，省一半时间。`max_workers=2` 因为就两个任务。
  - 为什么用线程而不是协程？因为这两个操作大部分时间在等 I/O 或 numpy 计算，Python 线程在 I/O 等待时会释放 GIL，能让另一个线程跑。
- `dense_future = pool.submit(self._vector.retrieve, query=query, top_n=per_retriever_n, filters=filters)`：把向量检索任务提交给线程池，返回一个 `Future` 对象（异步结果的占位符）。
- `sparse_future = pool.submit(self._bm25.search, query=query, top_n=per_retriever_n, filters=filters)`：把 BM25 任务也提交。
- `dense_results = dense_future.result()`：`Future.result()` 会**阻塞**直到这个任务完成，返回结果。两个任务此时是并行跑的，所以总耗时 ≈ max(向量耗时, BM25 耗时)，而不是两者之和。
- `sparse_results = sparse_future.result()`：取 BM25 结果。
- `fused = self._rrf.fuse([dense_results, sparse_results], k=self._rrf_k, top_n=top_n)`：调 RRF 融合，传两份结果、k 值、top_n。
- `logger.info("hybrid.retrieve.done", ...)`：打日志记录两份结果数和最终融合数。
- `return fused`：返回。
- `finally: record_retrieval_latency(...)`：上报耗时。

**`with ThreadPoolExecutor(...) as pool:` 的含义**：`with` 退出时会自动调用 `pool.shutdown(wait=True)`，等所有任务结束、释放线程池资源。

---

### 9.7 检索过程示例

输入：知识库有一条 chunk：
- chunk A（id=`c001`，doc_id=`d001`，source=`manuals/x2025.pdf`）：
  > X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。

用户问："X-2025 续航多久？"

#### 第 1 步：向量检索

- 查询 `"X-2025 续航多久？"` 经 `embed_query` 变成 768 维向量。
- 向量库 `search` 返回 top3：
  | rank | chunk | 相似度 |
  | --- | --- | --- |
  | 0 | A（X-2025 智能音箱，续航 12 小时...） | 0.81 |
  | 1 | B（蓝牙 5.0 双模连接） | 0.62 |
  | 2 | C（充电时间 2 小时） | 0.55 |

#### 第 2 步：BM25 检索

- 查询分词 `_tokenize("X-2025 续航多久？")` ≈ `["x", "2025", "续航", "多久"]`。
- BM25 算分。chunk A 含 "X-2025" 和 "续航"，分高；chunk B/C 不含 "X-2025"。
- 返回 top3：
  | rank | chunk | BM25 分 |
  | --- | --- | --- |
  | 0 | A（X-2025 智能音箱...） | 6.2 |
  | 1 | D（续航 12 小时，充电 2 小时） | 3.1 |
  | 2 | B（蓝牙 5.0） | 1.4 |

#### 第 3 步：RRF 融合（k=60）

- chunk A：1/60（向量 rank 0）+ 1/60（BM25 rank 0）= 0.0333
- chunk B：1/61（向量 rank 1）+ 1/62（BM25 rank 2）= 0.0325
- chunk C：1/62（向量 rank 2）= 0.0161
- chunk D：1/61（BM25 rank 1）= 0.0164

排序后 top3：A > B > D。

#### 最终输出

```python
[
    Chunk(id="c001", text="X-2025 智能音箱，续航 12 小时...", score=0.0333, metadata=...),
    Chunk(id="c002", text="蓝牙 5.0 双模连接", score=0.0325, metadata=...),
    Chunk(id="c004", text="续航 12 小时，充电 2 小时", score=0.0164, metadata=...),
]
```

可以看到，**chunk A 因为在两份检索里都排第一，被 RRF 抬到了第一**——这就是混合检索的核心价值。

---

## 第 10 章 · 重排——精挑细选

### 10.1 为什么检索后还要重排

向量检索和 BM25 检索都有一个共同特点：**它们都是"粗筛"**。

- 向量检索用 bi-encoder（双编码器）：问题和文档**分别**编码成向量，再算余弦相似度。这种"先各自编码再比较"的方式速度快（1 万篇文档只需 1 万次编码 + 1 万次余弦计算），但不够精细——它看不到问题和文档**结合起来**的语义。
- BM25 检索只看关键词统计，连语义都不懂。

**重排（rerank）是"精排"**，用 cross-encoder（交叉编码器）：把问题和文档**拼成一句话**送进模型，让模型判断"这段文档到底能不能回答这个问题"。它更准，但慢——1 万篇文档要算 1 万次模型前向。

打个比方：
- 向量检索 / BM25 = **海选**。几千份简历 5 分钟扫完，挑出 50 份还行的。
- 重排 = **决赛**。对这 50 份逐一细看，挑出最匹配的 5 份。

**为什么不直接用 cross-encoder 检索**？太慢。假设知识库 1 万篇文档，每篇用 cross-encoder 算一次要 50 毫秒，1 万次就是 500 秒（8 分钟），用户根本等不起。所以必须先用快速方法海选，再用 cross-encoder 精排。

kb-rag 的默认流程是：
1. 混合检索 top 20 →
2. 重排取 top 5 →
3. 把这 5 个 chunk 喂给 LLM 生成答案。

本章精读三个文件：
- `app/rerank/base.py`——抽象接口；
- `app/rerank/bge_reranker.py`——基于 BGE 的 cross-encoder 实现；
- `app/rerank/factory.py`——工厂方法。

---

### 10.2 base.py 逐行精读

文件路径：`app/rerank/base.py`。

完整代码：

```python
"""Abstract reranker interface for kb-rag."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.document import Chunk


class Reranker(ABC):
    """Abstract base class for rerankers.

    A reranker refines an initial set of retrieved chunks by computing a
    query-aware relevance score for each candidate and returning the top-K.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Re-score ``candidates`` for ``query`` and return the top-K.

        Args:
            query: The user query.
            candidates: Chunks retrieved by the first-stage retriever.
            top_k: Maximum number of chunks to return.

        Returns:
            A list of :class:`Chunk` sorted by rerank score descending, with
            ``score`` set to the rerank score. Length is ``<= top_k``.
        """
        raise NotImplementedError
```

逐行解释：

- 模块 docstring：抽象重排器接口。
- `from abc import ABC, abstractmethod`：导入抽象基类工具。`ABC` 是 Abstract Base Class 的缩写，`abstractmethod` 是装饰器，标记"子类必须实现这个方法"。
- `from app.models.document import Chunk`：导入 Chunk 数据类。

- `class Reranker(ABC):`：定义抽象基类 `Reranker`，继承 `ABC`。继承 ABC 表示这个类不能直接实例化，必须由子类继承并实现抽象方法。
- 类 docstring：重排器对一批候选 chunk 重新打分，返回 top-K。
- `@abstractmethod`：装饰器，标记下面的方法是抽象方法，子类必须实现，否则子类也不能实例化。
- `def rerank(self, query: str, candidates: list[Chunk], top_k: int = 5) -> list[Chunk]:`
  - 参数 `query: str`：用户查询。
  - 参数 `candidates: list[Chunk]`：第一阶段检索器返回的候选 chunk 列表。
  - 参数 `top_k: int = 5`：最多返回几条。**为什么默认 5**？这是工程经验值：5 条既足够给 LLM 提供上下文，又不会塞爆 LLM 的 context window（每条 chunk 几百字，5 条就是 1~2k token，刚好）。少了 LLM 信息不足，多了既贵又可能"淹没"关键信息。
- 返回 `list[Chunk]`：按重排分数降序，长度不超过 top_k，每个 chunk 的 `score` 字段被更新为重排分数。
- `raise NotImplementedError`：抽象方法的占位实现，子类不实现就调用会抛错。

**返回值含义**：`list[Chunk]`，长度 ≤ top_k，按重排分数从大到小排好，每个 chunk 的 `score` 是 0~1 之间的相关性概率（越大越相关）。

---

### 10.3 bge_reranker.py 逐行精读

文件路径：`app/rerank/bge_reranker.py`。这是 `Reranker` 的具体实现，用 BGE 重排模型。

完整代码：

```python
"""BGE-based cross-encoder reranker.

Prefers :class:`FlagEmbedding.FlagReranker` and transparently falls back to
:class:`sentence_transformers.CrossEncoder` when FlagEmbedding is unavailable.
The model is loaded lazily on the first ``rerank`` call so the module can be
imported in environments without the heavy ML dependencies.
"""
from __future__ import annotations

import time
from typing import Any

from app.models.document import Chunk
from app.observability.logging import get_logger
from app.rerank.base import Reranker

logger = get_logger(__name__)


class BgeReranker(Reranker):
    """Cross-encoder reranker backed by BGE-reranker-v2-m3.

    Args:
        model_name: Hugging Face model id (default
            ``"BAAI/bge-reranker-v2-m3"``).
        use_fp16: Forwarded to ``FlagReranker`` to enable half-precision
            inference. Ignored when falling back to CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
    ) -> None:
        """Initialize the reranker without loading the model yet."""
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model: Any | None = None
        self._backend: str | None = None

    # ---- Model loading ----

    def _load_model(self) -> None:
        """Lazily load the reranker model.

        Tries FlagEmbedding first, then falls back to sentence-transformers.
        Raises ``ImportError`` if neither is available.
        """
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
            self._backend = "flag_embedding"
            logger.info("reranker.load.flag_embedding", model=self.model_name)
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "reranker.flag_embedding.unavailable",
                error=str(exc),
            )
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self.model_name)
            self._backend = "sentence_transformers"
            logger.info("reranker.load.sentence_transformers", model=self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.error("reranker.load.failed", error=str(exc))
            raise ImportError(
                "Neither FlagEmbedding nor sentence-transformers is available; "
                "install one of them to use BgeReranker."
            ) from exc

    # ---- Public API ----

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Score each candidate against the query and return the top-K.

        Args:
            query: The user query.
            candidates: Chunks to re-score.
            top_k: Maximum number of chunks to return.

        Returns:
            Top-K chunks sorted by rerank score descending. The returned
            :class:`Chunk` instances have ``score`` set to the rerank score.
        """
        if not candidates:
            return []
        self._load_model()
        start = time.perf_counter()
        pairs = [(query, c.text) for c in candidates]
        scores = self._compute_scores(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        results: list[Chunk] = []
        for chunk, score in ranked[:top_k]:
            new_chunk = chunk.model_copy(deep=True)
            new_chunk.score = float(score)
            results.append(new_chunk)
        logger.info(
            "rerank.done",
            candidates=len(candidates),
            top_k=top_k,
            backend=self._backend,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        return results

    # ---- Internal ----

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute rerank scores for query/text pairs.

        Args:
            pairs: List of ``(query, text)`` tuples.

        Returns:
            A list of float scores, one per pair. FlagReranker applies
            sigmoid internally; CrossEncoder outputs are passed through a
            sigmoid to normalize them to ``[0, 1]``.
        """
        assert self._model is not None and self._backend is not None
        if self._backend == "flag_embedding":
            # FlagReranker.compute_score already applies sigmoid.
            raw = self._model.compute_score(pairs, normalize=True)
            if isinstance(raw, float):
                return [float(raw)]
            return [float(s) for s in raw]
        # CrossEncoder returns logits; apply sigmoid for normalization.
        raw = self._model.predict(pairs)
        import numpy as np  # local import keeps module import light

        probs = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
        return [float(p) for p in probs]
```

#### 背景知识：FlagReranker 和 CrossEncoder

- **FlagReranker**：来自 `FlagEmbedding` 库（BAAI 出品，BGE 系列模型的官方库）。它是 BGE 重排模型的封装，输入 `[(query, text), ...]` 列表，输出每对的"相关概率"（已经过 sigmoid 归一化到 0~1）。**为什么优先用它**？因为它是 BGE 模型的官方实现，性能最优，且支持 `use_fp16=True` 用半精度推理省显存。
- **CrossEncoder**：来自 `sentence_transformers` 库（非常流行的通用 sentence embedding 库）。它的 `predict` 接口接受同样的输入列表，但输出的是 **logits**（未归一化的原始分），需要自己套 sigmoid 转成概率。

**为什么两种都支持**：因为不同环境装的库不一样。FlagEmbedding 依赖较新，sentence_transformers 更通用。代码先试 FlagEmbedding，没装或失败就回退到 sentence_transformers，两者都没有就抛 `ImportError`。

**`compute_score([[query, text], ...])` 输入格式**：列表的列表，每个内层列表是 `[query, text]` 两个字符串。代码里写的是 `pairs = [(query, c.text) for c in candidates]`，是 tuple 列表 `[(query, text), ...]`——对 FlagReranker 和 CrossEncoder 来说，tuple 和 list 都行（都按顺序解构成两个元素）。

**返回分数含义**：0~1 之间的概率，越大表示 query 和 text 越相关。0.9 表示非常相关，0.1 表示基本无关。

#### `__init__` 方法逐行

- 参数 `model_name: str = "BAAI/bge-reranker-v2-m3"`：Hugging Face 模型 id。默认值 `"BAAI/bge-reranker-v2-m3"` 是 BAAI 出的 BGE 重排器 v2-m3 版本，**多语言**（中英都行）+ **轻量**（约 568MB），是当前业界最常用的开源重排模型之一。
- 参数 `use_fp16: bool = False`：是否用半精度浮点（FP16）推理。`False` 表示用 FP32（精度高但慢一倍、显存翻倍）；`True` 用 FP16（精度几乎无损，速度更快、显存省一半）。默认 False 是为了精度优先；线上推理建议设 True。
- `self.model_name = model_name`、`self.use_fp16 = use_fp16`：存起来。
- `self._model: Any | None = None`：模型对象，**初始为 None**。这就是"懒加载"——构造 `BgeReranker` 时不加载模型，等真正调 `rerank` 时才加载。
- `self._backend: str | None = None`：记录用哪个后端（"flag_embedding" 或 "sentence_transformers"），便于日志和分支判断。

#### `_load_model` 方法逐行（懒加载）

- `if self._model is not None: return`：已经加载过就直接返回，避免重复加载。
- `try: from FlagEmbedding import FlagReranker; self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16); self._backend = "flag_embedding"; logger.info(...); return`
  - 在函数内 import，避免模块顶部 import 失败影响整个模块。
  - 用 model_name 和 use_fp16 构造 FlagReranker 实例。
  - 记录后端标志。
  - 打日志。
  - 成功就 return。
- `except Exception as exc: logger.warning(...)`：FlagEmbedding 没装或加载失败，打个 warning，继续往下试 CrossEncoder。
- `try: from sentence_transformers import CrossEncoder; self._model = CrossEncoder(self.model_name); self._backend = "sentence_transformers"; logger.info(...)`
  - 注意：CrossEncoder 不支持 use_fp16 参数，所以构造时不传。
- `except Exception as exc: logger.error(...); raise ImportError(...) from exc`
  - 两个都没装就抛 `ImportError`，提示用户装一个。`from exc` 保留原始异常链，便于排查。

#### `rerank` 方法逐行

- 参数同 base.py：`query`、`candidates`、`top_k=5`。
- 返回 `list[Chunk]`：按重排分数降序，长度 ≤ top_k。

- `if not candidates: return []`：空列表直接返回空。
- `self._load_model()`：确保模型已加载（第一次调用时加载，后续直接返回）。
- `start = time.perf_counter()`：开始计时。
- `pairs = [(query, c.text) for c in candidates]`：构造 `[("X-2025 续航多久？", "X-2025 智能音箱，续航 12 小时..."), ...]` 这种对列表。
- `scores = self._compute_scores(pairs)`：调用下面要讲的 `_compute_scores`，返回每对的分数列表。
- `ranked = sorted(zip(candidates, scores, strict=False), key=lambda kv: float(kv[1]), reverse=True)`
  - `zip(candidates, scores)`：把 chunk 和分数配对成 `[(chunk1, score1), (chunk2, score2), ...]`。
  - `strict=False`：长度不等也不报错（Python 3.10+ 的参数）。
  - `key=lambda kv: float(kv[1])`：按分数排序。
  - `reverse=True`：从大到小。
- `results: list[Chunk] = []`：结果列表。
- `for chunk, score in ranked[:top_k]:`：取前 top_k 个。
- `new_chunk = chunk.model_copy(deep=True)`：深拷贝。
- `new_chunk.score = float(score)`：写入重排分数。
- `results.append(new_chunk)`：加入结果。
- `logger.info("rerank.done", candidates=..., top_k=..., backend=..., elapsed_ms=...)`：打日志。
- `return results`：返回。

#### `_compute_scores` 方法逐行

- 参数 `pairs: list[tuple[str, str]]`：query-text 对列表。
- 返回 `list[float]`：每对的分数。

- `assert self._model is not None and self._backend is not None`：断言模型和后端都已就绪。`assert` 在生产环境通常不依赖（可以用 `-O` 关掉），这里只是防御性检查。
- `if self._backend == "flag_embedding":`
  - `raw = self._model.compute_score(pairs, normalize=True)`：调用 FlagReranker，`normalize=True` 让它内部做 sigmoid 归一化。
  - `if isinstance(raw, float): return [float(raw)]`：如果只有一对，FlagReranker 可能返回单个 float 而不是列表，统一包成列表。
  - `return [float(s) for s in raw]`：把 numpy 数组转成 Python float 列表。
- `# CrossEncoder returns logits; apply sigmoid for normalization.`
  - `raw = self._model.predict(pairs)`：CrossEncoder.predict 返回 logits（任意实数）。
  - `import numpy as np`：函数内导入 numpy，避免在模块顶部增加依赖。
  - `probs = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))`：套 sigmoid 函数 `1/(1+e^-x)`，把任意实数压到 0~1 区间。
  - `return [float(p) for p in probs]`：转 Python float 列表。

**为什么两种后端归一化方式不一样**：FlagReranker 的 `compute_score(normalize=True)` 内部已经做了 sigmoid，再套一次会让所有分数都接近 0.5；CrossEncoder 返回 logits，必须自己套 sigmoid 才能转成概率。

---

### 10.4 factory.py 逐行精读

文件路径：`app/rerank/factory.py`。

完整代码：

```python
"""Factory for constructing rerankers from settings."""
from __future__ import annotations

from typing import Any

from app.rerank.base import Reranker
from app.rerank.bge_reranker import BgeReranker


def get_reranker(settings: Any) -> Reranker:
    """Build a :class:`Reranker` from the application settings.

    Currently always returns a :class:`BgeReranker` configured with
    ``settings.rerank_model``.

    Args:
        settings: Application settings (duck-typed; must expose
            ``rerank_model``).

    Returns:
        A :class:`Reranker` instance.
    """
    model_name = getattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")
    return BgeReranker(model_name=model_name)
```

逐行解释：

- 模块 docstring：根据 settings 构造 reranker 的工厂。
- `from app.rerank.base import Reranker`：导入抽象基类作为返回类型。
- `from app.rerank.bge_reranker import BgeReranker`：导入具体实现。
- `def get_reranker(settings: Any) -> Reranker:`
  - 参数 `settings: Any`：鸭子类型，只要对象有 `rerank_model` 属性就行。为什么用 `Any` 而不是具体类型？因为 settings 来自配置层，配置层和 rerank 层要解耦，不互相 import。
  - 返回 `Reranker`：一个 Reranker 实例。
- `model_name = getattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")`：`getattr(obj, name, default)` 是"取属性，没有就用 default"。这里读 `settings.rerank_model`，没配置就用默认模型 `"BAAI/bge-reranker-v2-m3"`。
- `return BgeReranker(model_name=model_name)`：构造并返回 BgeReranker 实例。

**当前实现只支持 BGE 一种**——这就是工厂模式的好处：将来想支持别的重排器（比如 Cohere Rerank），只需要在 factory 里加一个 if 分支，调用方代码完全不用改。

---

### 10.5 重排示例

接 9.7 的混合检索结果，假设 top5 是：

| rank | chunk | RRF 分 |
| --- | --- | --- |
| 0 | A（X-2025 智能音箱，续航 12 小时...） | 0.0333 |
| 1 | B（蓝牙 5.0 双模连接） | 0.0325 |
| 2 | D（续航 12 小时，充电 2 小时） | 0.0164 |
| 3 | E（4GB 内存与 32GB 存储） | 0.0150 |
| 4 | C（长按电源键 3 秒开机） | 0.0161 |

注意：RRF 分数都很小（0.01~0.03），因为它们是排名倒数之和，不是相似度。这些数字本身没太大意义，**只有相对大小有意义**。

#### 重排输入

```python
query = "X-2025 续航多久？"
candidates = [A, B, D, E, C]   # 5 个 chunk
```

#### 重排过程

构造 pairs：
```python
[
    ("X-2025 续航多久？", "X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。"),
    ("X-2025 续航多久？", "蓝牙 5.0 双模连接"),
    ("X-2025 续航多久？", "续航 12 小时，充电 2 小时"),
    ("X-2025 续航多久？", "4GB 内存与 32GB 存储"),
    ("X-2025 续航多久？", "长按电源键 3 秒开机"),
]
```

BGE 模型对每对算一个相关概率（cross-encoder 看到问题和文档拼在一起，能理解"X-2025"和"续航"在问题里很重要，"续航 12 小时"在文档里直接命中）。

#### 重排后（top_k=5）

| rank | chunk | 重排分（概率） |
| --- | --- | --- |
| 0 | A（X-2025 智能音箱，续航 12 小时...） | 0.98 |
| 1 | D（续航 12 小时，充电 2 小时） | 0.85 |
| 2 | B（蓝牙 5.0 双模连接） | 0.12 |
| 3 | E（4GB 内存与 32GB 存储） | 0.05 |
| 4 | C（长按电源键 3 秒开机） | 0.03 |

**关键变化**：
- chunk A 从 RRF 第 1 升到重排第 1（分数 0.98），因为它最直接回答"续航多久"。
- chunk D 从 RRF 第 3 升到重排第 2，因为它含"续航 12 小时"。
- chunk B 从 RRF 第 2 掉到重排第 3（分数仅 0.12），因为"蓝牙 5.0"和"续航"无关。
- E、C 直接被压到 0.05 以下，虽然还在 top5 里，但分数告诉后面的护栏"这俩其实没 answering the question"。

重排分数比 RRF 分数更"诚实"——0.98 和 0.12 的对比远比 0.033 和 0.032 的对比明显。这也为下一章的"置信度护栏"提供了依据。

---

## 第 11 章 · 生成与幻觉治理——回答问题

### 11.1 Prompt Engineering 在 RAG 中的关键性

到目前为止我们已经把"最相关的 5 个 chunk"挑出来了。但**直接把 chunk 塞给 LLM 让它自由发挥**会有大问题：

1. **幻觉（hallucination）**：LLM 可能基于自己的预训练知识瞎编，而不是基于我们给的 chunk。比如知识库里没有的内容，LLM 可能"脑补"出来。
2. **不可追溯**：LLM 给出答案后，用户不知道哪句话来自哪个文档，没法核对。
3. **答非所问**：LLM 可能跑题，把无关的 chunk 内容也写进答案。

**好的 RAG prompt 设计要解决三件事**：
- **强约束**：明确告诉 LLM"只能基于上下文回答"，没有就说"不知道"。
- **强制引用**：要求 LLM 在每个事实后面标 `[1]` `[2]` 这样的引用编号，对应到我们提供的 chunk 顺序。
- **可拒答**：上下文不足时，让 LLM 输出一个固定字符串（比如"未在知识库中找到相关内容"），方便下游识别。

kb-rag 的策略是：
- `prompts.py` 构造一个**强约束 + 强制引用**的 system prompt；
- `guardrail.py` 在生成前先看检索分数够不够，不够直接拒答，连 LLM 都不调（省钱省时）；
- `citation.py` 在生成后从答案里解析 `[n]` 引用编号，映射回具体 chunk；
- `llm.py` 提供三种 LLM 后端（OpenAI、智谱、Ollama）的统一接口。

本章精读六个文件：`prompts.py`、`base.py`、`llm.py`、`guardrail.py`、`citation.py`、`factory.py`。

---

### 11.2 prompts.py 逐行精读

文件路径：`app/generation/prompts.py`。

完整代码：

```python
"""RAG prompt construction.

Builds the chat messages handed to the LLM: a strict system prompt that
constrains the model to the retrieved context and requires citation, plus a
user message that numbers each context fragment for reference.
"""
from __future__ import annotations

from app.models.document import Chunk

SYSTEM_PROMPT = (
    "你是知识库问答助手。只能基于下方检索到的上下文回答问题。"
    "每个片段前有编号 [1] [2]...。回答时必须在对应句子末尾用 [编号] 标注引用来源。"
    "若上下文不足以回答，请回复：未在知识库中找到相关内容。"
)


def build_rag_prompt(query: str, contexts: list[Chunk]) -> list[dict]:
    """Build chat messages for a RAG generation call.

    Args:
        query: The user's natural-language query.
        contexts: Chunks to include as numbered context. Each chunk is
            rendered as ``[n] <text>``.

    Returns:
        A list of ``{"role", "content"}`` message dicts of length 2:
        the system message first, then the user message.
    """
    numbered: list[str] = []
    for idx, chunk in enumerate(contexts, start=1):
        snippet = chunk.text.strip()
        numbered.append(f"[{idx}] {snippet}")
    context_block = "\n\n".join(numbered) if numbered else "（无可用上下文）"
    user_content = f"问题：{query}\n\n上下文：\n{context_block}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
```

#### SYSTEM_PROMPT 逐句解析

`SYSTEM_PROMPT` 是一个三句话的字符串（用括号分成多行方便看，Python 会自动拼成一个字符串）：

1. `"你是知识库问答助手。只能基于下方检索到的上下文回答问题。"`
   - 第一句**设定角色**：你是知识库问答助手，不是通用 AI。
   - 第二句**强约束**：只能用下面给的上下文答，别动用你自己的预训练知识。这一句是反幻觉的关键。
2. `"每个片段前有编号 [1] [2]...。回答时必须在对应句子末尾用 [编号] 标注引用来源。"`
   - 第一句告诉 LLM 上下文长什么样（每个片段有 `[n]` 编号）。
   - 第二句**强制引用**：每个事实后面必须标 `[编号]`。这样后面 citation.py 能用正则把引用抓出来。
3. `"若上下文不足以回答，请回复：未在知识库中找到相关内容。"`
   - **拒答指令**：上下文不够时不要瞎编，输出固定字符串。这个字符串后面 guardrail.py 也会用，两边对齐。

#### `build_rag_prompt` 函数逐行

- 参数 `query: str`：用户问题。
- 参数 `contexts: list[Chunk]`：要塞进 prompt 的 chunk 列表（重排后的 top_k）。
- 返回 `list[dict]`：长度为 2 的消息列表 `[system_message, user_message]`，每个消息是 `{"role": ..., "content": ...}` dict。

- `numbered: list[str] = []`：准备一个列表装"编号好的片段"。
- `for idx, chunk in enumerate(contexts, start=1):`：`enumerate(..., start=1)` 给每个 chunk 配上一个从 1 开始的编号 idx。
- `snippet = chunk.text.strip()`：取 chunk 的文本，去掉首尾空白。
- `numbered.append(f"[{idx}] {snippet}")`：拼成 `"[1] X-2025 智能音箱，续航 12 小时..."` 这种格式。
- `context_block = "\n\n".join(numbered) if numbered else "（无可用上下文）"`：把所有片段用空行连起来。如果 contexts 为空，给一个"（无可用上下文）"占位。
- `user_content = f"问题：{query}\n\n上下文：\n{context_block}"`：拼出 user 消息内容，包含问题和上下文。
- `return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]`：返回两条消息，system 在前，user 在后。这是 OpenAI Chat Completions API 的标准格式。

#### 构建后的完整 messages JSON 示例

假设：
- query = "X-2025 续航多久？"
- contexts = [chunk_A, chunk_D]（重排后 top2）

构建后的 messages：

```json
[
  {
    "role": "system",
    "content": "你是知识库问答助手。只能基于下方检索到的上下文回答问题。每个片段前有编号 [1] [2]...。回答时必须在对应句子末尾用 [编号] 标注引用来源。若上下文不足以回答，请回复：未在知识库中找到相关内容。"
  },
  {
    "role": "user",
    "content": "问题：X-2025 续航多久？\n\n上下文：\n[1] X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。\n\n[2] 续航 12 小时，充电 2 小时"
  }
]
```

---

### 11.3 base.py 逐行精读

文件路径：`app/generation/base.py`。

完整代码：

```python
"""Abstract generator interface and shared result model."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.models.document import Chunk


class GenerationResult(BaseModel):
    """Output of a generation call.

    Attributes:
        answer: Generated natural-language answer.
        citations: 1-based citation numbers extracted from the answer.
        used_chunk_ids: Ids of the chunks that contributed to the answer.
    """

    answer: str
    citations: list[int] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)


class Generator(ABC):
    """Abstract base class for LLM-backed answer generators."""

    @abstractmethod
    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Generate an answer grounded in ``contexts`` for ``query``.

        Args:
            query: The user query.
            contexts: Reranked chunks to ground the answer.

        Returns:
            A :class:`GenerationResult` with the answer, extracted citations
            and the list of used chunk ids.
        """
        raise NotImplementedError
```

#### `GenerationResult` 逐字段

继承 `pydantic.BaseModel`——pydantic 是数据校验库，继承 BaseModel 后字段会自动校验类型。

- `answer: str`：LLM 生成的自然语言答案，比如 `"X-2025 续航 12 小时 [1] [2]"`。
- `citations: list[int] = Field(default_factory=list)`：从答案里解析出的引用编号列表（1-based）。`Field(default_factory=list)` 表示"默认值是新建一个空列表"（不能用 `[]`，因为可变默认值会被所有实例共享）。
- `used_chunk_ids: list[str] = Field(default_factory=list)`：被答案引用的 chunk 的 id 列表，与 citations 一一对应（但已经从编号映射成了 chunk id）。

#### `Generator` 抽象基类

- `class Generator(ABC):`：抽象基类，所有生成器都要继承它。
- `@abstractmethod`：标记抽象方法。
- `def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:`
  - 参数 `query: str`：用户问题。
  - 参数 `contexts: list[Chunk]`：重排后的 chunk 列表，作为生成答案的依据。
  - 返回 `GenerationResult`：包含答案、引用编号、引用的 chunk id。
- `raise NotImplementedError`：占位实现。

---

### 11.4 llm.py 逐行精读

文件路径：`app/generation/llm.py`。这是三个具体 Generator 的实现：OpenAI、智谱、Ollama。

完整代码（分块讲解）：

```python
"""LLM-backed generators for the RAG pipeline.

Three concrete implementations of :class:`app.generation.base.Generator`:

* :class:`OpenAIGenerator`  - OpenAI-compatible Chat Completions API.
* :class:`ZhipuGenerator`   - Zhipu (BigModel) via the OpenAI client.
* :class:`OllamaGenerator`  - Local Ollama ``/api/chat`` REST endpoint.

All generators wrap their network calls in tenacity retries (3 attempts).
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.generation.base import GenerationResult, Generator
from app.generation.citation import CitationParser
from app.generation.prompts import build_rag_prompt
from app.models.document import Chunk
from app.observability.logging import get_logger
from app.observability.metrics import record_generation_latency

logger = get_logger(__name__)
```

逐行：

- 模块 docstring：说明三个生成器和它们共用的重试策略。
- `import time`：测耗时。
- `from typing import Any`：鸭子类型。
- `import httpx`：**httpx** 是一个现代的 HTTP 客户端库（比 `requests` 更现代，支持 HTTP/2 和 async）。这里用于 Ollama 的 REST API 调用。
- `from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential`：**tenacity** 是 Python 重试库，能优雅地给函数加"失败重试"。我们用它的四个组件：
  - `retry`：重试装饰器。
  - `retry_if_exception_type`：遇到指定类型的异常才重试。
  - `stop_after_attempt`：最多重试几次。
  - `wait_exponential`：指数退避（每次等更久）。
- 后面 import 自己项目里的模块：`GenerationResult` 和 `Generator`、`CitationParser`、`build_rag_prompt`、`Chunk`、logger、metrics。

#### `_RETRY_DECORATOR` 逐行

```python
_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (
            httpx.HTTPError,
            httpx.TimeoutException,
            httpx.NetworkError,
            RuntimeError,
        )
    ),
    reraise=True,
)
```

- `stop=stop_after_attempt(3)`：最多尝试 3 次（1 次原始 + 2 次重试）。
- `wait=wait_exponential(multiplier=1, min=1, max=10)`：指数退避。`wait = min(max, multiplier × 2^(attempt-1))` 秒：
  - 第 1 次失败后等 1 秒重试（`1 × 2^0 = 1`，但 min=1 兜底）。
  - 第 2 次失败后等 2 秒重试（`1 × 2^1 = 2`）。
  - 第 3 次失败后不再重试，因为 stop=3。
  - `max=10` 是上限，避免指数爆炸（重试多了也等不及）。
- `retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException, httpx.NetworkError, RuntimeError))`：**只在遇到这些异常时才重试**。这些是网络相关异常和 RuntimeError。如果遇到逻辑错误（比如 KeyError）不重试。
- `reraise=True`：重试用完后，**把原始异常重新抛出**（而不是 tenacity 自己包装的 RetryError）。这样上层捕获到的是干净的 HTTPError，方便排查。

#### `_extract_answer` 函数

```python
def _extract_answer(response_obj: Any) -> str:
    try:
        return response_obj.choices[0].message.content or ""
    except Exception as exc:
        logger.error("llm.parse.failed", error=str(exc))
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc
```

- 参数 `response_obj: Any`：OpenAI 风格的响应对象（有 `.choices[0].message.content` 这种结构）。
- 返回 `str`：模型回复的文本。
- `response_obj.choices[0].message.content or ""`：取第一个候选的 message 内容，如果是 None 就返回空串。
- 异常处理：解析失败就抛 `RuntimeError`，会被上面的 `_RETRY_DECORATOR` 重试。

#### `OpenAIGenerator` 类

```python
class OpenAIGenerator(Generator):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._client: Any | None = None
        self._citation_parser = CitationParser()
```

`__init__` 参数逐个：

- `api_key: str`：OpenAI API 密钥。必填，没默认值。
- `base_url: str | None = None`：API 基址，默认 None 用 OpenAI 官方地址。可改成 Azure OpenAI 或兼容代理（比如本地 vLLM）。
- `model: str = "gpt-4o-mini"`：模型 id，默认 `gpt-4o-mini`。**为什么默认这个**：gpt-4o-mini 是 OpenAI 当前性价比最高的模型，速度快、便宜，对 RAG 这种结构化任务足够。生产可换成 `gpt-4o`、`gpt-4-turbo` 等。
- `temperature: float = 0.0`：温度，控制随机性。0.0 = 完全确定性（每次输出都一样），1.0 = 比较随机，>1.0 = 很混乱。**RAG 推荐 0.0**，因为我们要的是"基于事实的稳定答案"，不要模型每次都给不一样的话。
- `max_tokens: int = 2048`：生成的最大 token 数。2048 ≈ 1500~3000 个汉字，对一般问答够用。设太大会增加成本（按输出 token 计费）。
- `top_p: float = 1.0`：核采样阈值。1.0 = 禁用（用 temperature 控制），<1.0 = 只从概率前 p 的 token 里采样。一般 temperature 和 top_p 二选一，不要同时调。

- `self._client: Any | None = None`：OpenAI 客户端对象，初始 None（懒加载）。
- `self._citation_parser = CitationParser()`：构造一个引用解析器实例。

#### `_get_client` 方法（懒加载）

```python
def _get_client(self) -> Any:
    if self._client is None:
        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
    return self._client
```

- 如果 `_client` 是 None，就构造一个 OpenAI 客户端。
- `from openai import OpenAI`：函数内导入，避免在没装 openai 库的环境里整个模块 import 失败。
- `kwargs = {"api_key": self.api_key}`：先放 api_key。
- `if self.base_url: kwargs["base_url"] = self.base_url`：传了 base_url 才加到 kwargs。
- `self._client = OpenAI(**kwargs)`：构造客户端存起来。
- 下次再调 `_get_client` 直接返回已构造的实例，避免重复创建。

#### `generate` 方法

```python
def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
    messages = build_rag_prompt(query, contexts)
    start = time.perf_counter()
    try:
        answer = self._call_with_retry(messages)
        citations = self._citation_parser.parse(answer)
        used = self._citation_parser.map_to_references(citations, contexts)
        used_ids = [c.id for c in used]
        logger.info(
            "openai.generate.done",
            model=self.model,
            citations=citations,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        return GenerationResult(
            answer=answer,
            citations=citations,
            used_chunk_ids=used_ids,
        )
    finally:
        record_generation_latency(time.perf_counter() - start)
```

- `messages = build_rag_prompt(query, contexts)`：构造 RAG prompt（11.2 节讲过）。
- `start = time.perf_counter()`：计时。
- `try:`：保证上报耗时。
- `answer = self._call_with_retry(messages)`：调 LLM（带重试），返回模型回复文本。
- `citations = self._citation_parser.parse(answer)`：从答案里解析 `[1] [2]` 引用编号。
- `used = self._citation_parser.map_to_references(citations, contexts)`：把编号映射回 chunk。
- `used_ids = [c.id for c in used]`：取出被引用的 chunk id。
- `logger.info(...)`：打日志。
- `return GenerationResult(answer=answer, citations=citations, used_chunk_ids=used_ids)`：组装返回。
- `finally: record_generation_latency(...)`：上报耗时。

#### `_call_with_retry` 方法

```python
@_RETRY_DECORATOR
def _call_with_retry(self, messages: list[dict]) -> str:
    client = self._get_client()
    response = client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        top_p=self.top_p,
    )
    return _extract_answer(response)
```

- `@_RETRY_DECORATOR`：用上面的重试装饰器装饰这个方法。这样方法内的网络异常会自动重试 3 次。
- `client = self._get_client()`：拿到 OpenAI 客户端（懒加载）。
- `response = client.chat.completions.create(model=..., messages=..., temperature=..., max_tokens=..., top_p=...)`：调 OpenAI Chat Completions API。
  - `model`：用哪个模型。
  - `messages`：消息列表（system + user）。
  - `temperature`：随机性。
  - `max_tokens`：最大生成长度。
  - `top_p`：核采样。
- `return _extract_answer(response)`：从响应里提取答案文本。

#### `ZhipuGenerator` 类

```python
class ZhipuGenerator(Generator):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        ...
    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client
```

**与 OpenAIGenerator 的差异**：
- `base_url` 默认值是 `"https://open.bigmodel.cn/api/paas/v4"`（智谱的 API 地址），而不是 None。
- `model` 默认值是 `"glm-4"`（智谱 GLM-4 模型），而不是 `"gpt-4o-mini"`。
- 智谱的 API **完全兼容 OpenAI 协议**，所以直接用 `openai.OpenAI` 客户端，只是把 `base_url` 指向智谱。这是智谱官方推荐的接入方式。
- `generate` 和 `_call_with_retry` 与 OpenAIGenerator 几乎一模一样，只是日志 key 从 `"openai.generate.done"` 换成 `"zhipu.generate.done"`。

#### `OllamaGenerator` 类

```python
class OllamaGenerator(Generator):
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "qwen2.5",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._citation_parser = CitationParser()
```

**与 OpenAI/Zhipu 的关键差异**：
- 没有 `api_key`——Ollama 是本地部署的，不需要鉴权。
- `base_url` 默认 `"http://ollama:11434"`——这是 Docker 容器里访问 Ollama 的默认地址（容器名 ollama，端口 11434）。本地开发可能要改成 `http://localhost:11434`。
- `model` 默认 `"qwen2.5"`——通义千问 2.5，Ollama 上最常用的中文模型之一。
- `self.base_url = base_url.rstrip("/")`：去掉末尾斜杠，避免拼 URL 时变成 `//api/chat`。

`_call_with_retry` 方法：

```python
@_RETRY_DECORATOR
def _call_with_retry(self, messages: list[dict]) -> str:
    url = f"{self.base_url}/api/chat"
    payload = {
        "model": self.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "top_p": self.top_p,
        },
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data.get("message", {}).get("content", "") or ""
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc
```

- 不用 OpenAI SDK，直接用 **httpx** 发 HTTP POST 请求到 Ollama 的 `/api/chat` REST 端点。
- `payload` 是请求体：
  - `model`：模型名。
  - `messages`：消息列表（与 OpenAI 同结构）。
  - `stream: False`：不要流式响应，一次性返回完整结果。简化解析。
  - `options`：Ollama 把生成参数放在 `options` 子对象里。
    - `temperature`：温度。
    - `num_predict`：**Ollama 里 max_tokens 叫 num_predict**，是 Ollama 私有命名。这里把外部传入的 `max_tokens` 映射到 `num_predict`。
    - `top_p`：核采样。
- `with httpx.Client(timeout=120.0) as client:`：构造 httpx 客户端，超时 120 秒。
  - **为什么 timeout=120**：本地大模型推理比云端慢得多（取决于机器配置，7B 模型在 CPU 上生成 2000 token 可能要 30~60 秒），给足 2 分钟避免超时。
- `resp = client.post(url, json=payload)`：发 POST 请求。
- `resp.raise_for_status()`：如果状态码不是 2xx，抛 HTTPError（会被重试装饰器捕获重试）。
- `data = resp.json()`：解析 JSON 响应。
- `return data.get("message", {}).get("content", "") or ""`：从 Ollama 响应的 `message.content` 字段取回复文本，没有就空串。
- 异常处理：解析失败抛 RuntimeError（会被重试）。

---

### 11.5 guardrail.py 逐行精读

文件路径：`app/generation/guardrail.py`。

完整代码：

```python
"""Hallucination guardrail for the generation stage.

The :class:`Guardrail` decides whether the top retrieval score is high
enough to trust the LLM with answering, and provides a fixed fallback
answer when retrieval yields nothing relevant.
"""
from __future__ import annotations

from app.observability.logging import get_logger
from app.observability.metrics import record_no_result

logger = get_logger(__name__)

NO_RESULT_ANSWER = "未在知识库中找到相关内容。"


class Guardrail:
    """Confidence-based guardrail for the RAG generation stage."""

    def check_confidence(self, top_score: float, threshold: float = 0.3) -> bool:
        """Return ``True`` if the top retrieval score clears ``threshold``.

        Args:
            top_score: Best score from the retrieval/rerank stage.
            threshold: Minimum score required to attempt generation.

        Returns:
            ``True`` when ``top_score >= threshold``; ``False`` (and a
            no-result counter increment) otherwise.
        """
        if top_score is None or top_score < threshold:
            logger.info(
                "guardrail.reject",
                top_score=top_score,
                threshold=threshold,
            )
            record_no_result()
            return False
        return True

    def build_no_result_answer(self) -> str:
        """Return the canonical answer used when retrieval fails."""
        return NO_RESULT_ANSWER
```

#### `Guardrail` 类逐行

- `NO_RESULT_ANSWER = "未在知识库中找到相关内容。"`：模块级常量。注意它和 `prompts.SYSTEM_PROMPT` 里"若上下文不足以回答，请回复：未在知识库中找到相关内容。"对齐——LLM 拒答时输出的就是这个字符串，guardrail 主动拒答时也用同一个字符串，下游不需要区分是谁拒答的。

- `check_confidence(self, top_score: float, threshold: float = 0.3) -> bool`
  - 参数 `top_score: float`：重排阶段最高分（重排后 top1 chunk 的 score，0~1 之间的概率）。
  - 参数 `threshold: float = 0.3`：阈值，默认 0.3。**为什么是 0.3**：BGE 重排分数是 0~1 的概率。低于 0.3 意味着模型认为"这段文档和问题不太相关"，再让 LLM 答可能就是幻觉了。0.3 是经验值，比 0.5 宽松（避免误杀），比 0.1 严格（避免放水）。
  - 返回 `bool`：True 表示"有信心，可以答"；False 表示"没信心，应该拒答"。

- 函数体：
  - `if top_score is None or top_score < threshold:`：分数为 None 或低于阈值。
  - `logger.info("guardrail.reject", top_score=..., threshold=...)`：打日志说"被护栏拒绝"。
  - `record_no_result()`：上报一个"无结果"指标，便于监控告警。
  - `return False`：返回 False，告诉调用方"别让 LLM 答了"。
  - `return True`：分数够，返回 True。

- `build_no_result_answer(self) -> str`：返回 `NO_RESULT_ANSWER` 常量。这个方法封装的好处是：将来想改拒答话术（比如多语言、加错误码），只改一个地方。

---

### 11.6 citation.py 逐行精读

文件路径：`app/generation/citation.py`。

完整代码：

```python
"""Citation parsing and mapping utilities.

The :class:`CitationParser` extracts ``[n]`` citation markers from the LLM
answer and maps them back to the :class:`Chunk` instances that were supplied
to the prompt (1-based indexing, matching :func:`build_rag_prompt`).
"""
from __future__ import annotations

import re

from app.models.document import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationParser:
    """Parse ``[n]`` citation markers and map them to source chunks."""

    def parse(self, answer: str) -> list[int]:
        """Extract unique citation numbers from ``answer``, preserving order.

        Args:
            answer: LLM-generated answer potentially containing ``[n]``
                citation markers.

        Returns:
            A deduplicated, order-preserving list of 1-based citation ints.
        """
        if not answer:
            return []
        seen: set[int] = set()
        result: list[int] = []
        for match in _CITATION_RE.finditer(answer):
            num = int(match.group(1))
            if num in seen:
                continue
            seen.add(num)
            result.append(num)
        return result

    def map_to_references(
        self,
        citations: list[int],
        contexts: list[Chunk],
    ) -> list[Chunk]:
        """Map 1-based citation numbers back to ``contexts``.

        Args:
            citations: Citation numbers (1-based) extracted from the answer.
            contexts: The chunks that were passed to the prompt, in the same
                order (so citation ``n`` maps to ``contexts[n-1]``).

        Returns:
            The subset of ``contexts`` referenced by ``citations``, in
            citation order. Out-of-range citations are skipped.
        """
        result: list[Chunk] = []
        for num in citations:
            if num <= 0 or num > len(contexts):
                continue
            result.append(contexts[num - 1])
        return result
```

#### 模块顶部

- `_CITATION_RE = re.compile(r"\[(\d+)\]")`：预编译正则。`\[(\d+)\]` 拆开看：
  - `\[`：匹配字面的 `[`（要转义，因为 `[` 在正则里有特殊含义）。
  - `(\d+)`：匹配一个或多个数字，**加括号表示捕获组**，可以后续用 `match.group(1)` 取出。
  - `\]`：匹配字面的 `]`。
  - 整个正则匹配 `[1]`、`[12]`、`[345]` 这种引用标记。

#### `parse` 方法逐行

- 参数 `answer: str`：LLM 生成的答案。
- 返回 `list[int]`：去重后的引用编号列表，**保持首次出现顺序**。

- `if not answer: return []`：空答案直接返回空。
- `seen: set[int] = set()`：去重用的集合。
- `result: list[int] = []`：结果列表。
- `for match in _CITATION_RE.finditer(answer):`：`finditer` 按顺序找出所有匹配，每个 match 是一个 Match 对象。
- `num = int(match.group(1))`：`group(1)` 取第一个捕获组（也就是数字部分），转成 int。
- `if num in seen: continue`：已经见过就跳过。
- `seen.add(num); result.append(num)`：没见过就加入集合和结果。
- `return result`：返回。

**举例**：`parse("X-2025 续航 12 小时 [1] [2]，蓝牙 5.0 [1]")` → `[1, 2]`（第二个 `[1]` 被去重）。

#### `map_to_references` 方法逐行

- 参数 `citations: list[int]`：引用编号列表（1-based）。
- 参数 `contexts: list[Chunk]`：传给 prompt 的 chunk 列表（**与 prompt 里的 `[n]` 编号一一对应**）。
- 返回 `list[Chunk]`：被引用的 chunk 列表，按引用顺序。

- `result: list[Chunk] = []`：结果列表。
- `for num in citations:`：遍历引用编号。
- `if num <= 0 or num > len(contexts): continue`：**越界检查**。如果 LLM 输出了 `[99]` 但 contexts 只有 5 个 chunk，就跳过这个编号，不报错。
- `result.append(contexts[num - 1])`：`num` 是 1-based，转成 0-based 取 chunk。比如 `num=1` 取 `contexts[0]`。
- `return result`：返回。

**为什么是 1-based**：因为 `build_rag_prompt` 用 `enumerate(contexts, start=1)` 给 chunk 编号，prompt 里就是 `[1] [2] ...`。这里映射要保持一致。

---

### 11.7 factory.py 逐行精读

文件路径：`app/generation/factory.py`。

完整代码：

```python
"""Factory for constructing generators from settings."""
from __future__ import annotations

from typing import Any

from app.generation.base import Generator
from app.generation.llm import OllamaGenerator, OpenAIGenerator, ZhipuGenerator


def get_generator(settings: Any) -> Generator:
    """Build a :class:`Generator` based on ``settings.llm_provider``.

    Args:
        settings: Application settings (duck-typed). Must expose
            ``llm_provider``, ``llm_model`` and the credentials/URLs for the
            requested provider.

    Returns:
        One of :class:`OpenAIGenerator`, :class:`ZhipuGenerator`,
        :class:`OllamaGenerator`.

    Raises:
        ValueError: If ``llm_provider`` is not one of the supported values.
    """
    provider = getattr(settings, "llm_provider", "openai").lower()
    model = getattr(settings, "llm_model", "gpt-4o-mini")
    temperature = getattr(settings, "llm_temperature", 0.0)
    max_tokens = getattr(settings, "llm_max_tokens", 2048)
    top_p = getattr(settings, "llm_top_p", 1.0)
    if provider == "openai":
        return OpenAIGenerator(
            api_key=getattr(settings, "openai_api_key", ""),
            base_url=getattr(settings, "openai_base_url", None),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    if provider == "zhipu":
        return ZhipuGenerator(
            api_key=getattr(settings, "zhipu_api_key", ""),
            base_url=getattr(settings, "zhipu_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    if provider == "ollama":
        return OllamaGenerator(
            base_url=getattr(settings, "ollama_base_url", "http://ollama:11434"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    raise ValueError(f"Unsupported llm_provider: {provider!r}")
```

#### 逐行解释

- 参数 `settings: Any`：鸭子类型，必须有 `llm_provider` 等属性。
- 返回 `Generator`：一个生成器实例。

- `provider = getattr(settings, "llm_provider", "openai").lower()`：读 `llm_provider`，没配置默认 `"openai"`，转小写（用户可能写 `OpenAI` 或 `OPENAI`）。
- `model = getattr(settings, "llm_model", "gpt-4o-mini")`：读模型 id，默认 `gpt-4o-mini`。
- `temperature = getattr(settings, "llm_temperature", 0.0)`：读温度，默认 0.0（确定性输出，RAG 推荐）。
- `max_tokens = getattr(settings, "llm_max_tokens", 2048)`：读最大 token，默认 2048。
- `top_p = getattr(settings, "llm_top_p", 1.0)`：读 top_p，默认 1.0（禁用）。

- `if provider == "openai": return OpenAIGenerator(api_key=..., base_url=..., model=..., ...)`：openai 分支。
- `if provider == "zhipu": return ZhipuGenerator(api_key=..., base_url=..., model=..., ...)`：zhipu 分支。
- `if provider == "ollama": return OllamaGenerator(base_url=..., model=..., ...)`：ollama 分支（无 api_key）。
- `raise ValueError(f"Unsupported llm_provider: {provider!r}")`：都不是就抛 ValueError。`!r` 表示用 `repr` 格式化（带引号），方便看到 provider 的真实值。

**工厂模式的好处**：调用方只要传 settings，不用管哪个 provider，工厂根据配置返回正确实例。要加新 provider（比如 Anthropic），只需要在这里加一个 if 分支。

---

### 11.8 生成过程示例

接 10.5 的重排结果，假设重排后 top5 是：

| rank | chunk | 重排分 |
| --- | --- | --- |
| 0 | A（X-2025 智能音箱，续航 12 小时...） | 0.98 |
| 1 | D（续航 12 小时，充电 2 小时） | 0.85 |
| 2 | B（蓝牙 5.0 双模连接） | 0.12 |
| ... | ... | ... |

#### 第 1 步：构造 messages

```json
[
  {
    "role": "system",
    "content": "你是知识库问答助手。只能基于下方检索到的上下文回答问题。每个片段前有编号 [1] [2]...。回答时必须在对应句子末尾用 [编号] 标注引用来源。若上下文不足以回答，请回复：未在知识库中找到相关内容。"
  },
  {
    "role": "user",
    "content": "问题：X-2025 续航多久？\n\n上下文：\n[1] X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。\n\n[2] 续航 12 小时，充电 2 小时\n\n[3] 蓝牙 5.0 双模连接\n\n[4] 4GB 内存与 32GB 存储\n\n[5] 长按电源键 3 秒开机"
  }
]
```

#### 第 2 步：调用 LLM

假设用 OpenAI `gpt-4o-mini`，`temperature=0.0`，模型返回的 answer：

```text
X-2025 智能音箱的续航时间为 12 小时 [1] [2]。
```

#### 第 3 步：解析引用

```python
answer = "X-2025 智能音箱的续航时间为 12 小时 [1] [2]。"
citations = parser.parse(answer)
# → [1, 2]
```

#### 第 4 步：映射到 chunks

```python
contexts = [chunk_A, chunk_D, chunk_B, chunk_E, chunk_C]
referenced = parser.map_to_references([1, 2], contexts)
# → [chunk_A, chunk_D]
```

#### 第 5 步：组装 GenerationResult

```python
GenerationResult(
    answer="X-2025 智能音箱的续航时间为 12 小时 [1] [2]。",
    citations=[1, 2],
    used_chunk_ids=["c001", "c004"],
)
```

#### 最终给用户的回答

```text
X-2025 智能音箱的续航时间为 12 小时 [1] [2]。

参考文献：
[1] X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。（来源：manuals/x2025.pdf，第 1 页，相关度：0.98）
[2] 续航 12 小时，充电 2 小时。（来源：manuals/x2025.pdf，第 2 页，相关度：0.85）
```

用户点击 `[1]` 就能跳到原始文档的对应位置——这就是 RAG 的"可追溯"特性。

---

## 第 12 章 · 管线编排——把所有步骤串起来

### 12.1 为什么需要 Pipeline

前面几章我们讲了 6 个独立组件：
1. **解析器**（parser）——把文件变成文档；
2. **清洗器**（cleaner）——清洗文本；
3. **分块器**（chunker）——切成 chunk；
4. **嵌入器**（embedder）——chunk 变向量；
5. **向量库 + BM25 索引**——存向量 + 存倒排；
6. **检索器 + 重排器 + 生成器**——查 → 排 → 答。

每个组件单独能用，但用户只想"丢个文件进来""问个问题出去"，不想关心中间这么多步。**Pipeline（管线）就是把这些组件按顺序串起来的"流水线"**，对外暴露两个简单方法：
- `ingest_file(path)`——把文件吃进去，自动完成"解析 → 清洗 → 分块 → 嵌入 → 存储"。
- `query(question)`——回答一个问题，自动完成"嵌入查询 → 检索 → 重排 → 生成"。

**关键设计：惰性初始化（lazy init）**

为什么不在 Pipeline 启动时就加载所有模型？
1. **启动慢**：BGE 嵌入模型 + BGE 重排模型 + LLM 客户端加起来要好几十秒，启动时全加载用户等不起。
2. **占内存**：每个模型上 GB 显存/内存，全加载可能 OOM。
3. **没必要**：如果用户只摄入不查询，加载查询相关的模型就是浪费。

kb-rag 的做法是：构造 Pipeline 时**只存 settings**，所有组件用 `@property` 装饰器**第一次访问时才构造**，构造后缓存在实例属性里，下次访问直接返回。这样：
- 启动快（只读配置）。
- 用啥加载啥（不用的模型永远不加载）。
- 重复用不重复加载（懒加载 + 缓存）。

本章精读三个文件：
- `app/pipeline/ingest_pipeline.py`——摄入管线；
- `app/pipeline/query_pipeline.py`——查询管线；
- `app/pipeline/container.py`——单例容器。

---

### 12.2 ingest_pipeline.py 逐行精读

文件路径：`app/pipeline/ingest_pipeline.py`。

完整代码：

```python
"""Ingestion pipeline orchestrating parse -> clean -> chunk -> embed -> store.

The :class:`IngestPipeline` ties together every upstream stage (parsing,
cleaning, chunking, embedding, vector storage and BM25 indexing) into a single
end-to-end flow for file ingestion.  Heavy components (chunker, embedder,
vector store, BM25 retriever) are lazily initialised on first use so that
constructing the pipeline is cheap and model loading is deferred until the
first ``ingest_file`` call.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from app.chunkers import get_chunker
from app.embedders import get_embedder
from app.ingest import clean_documents, get_parser
from app.observability.logging import bind_trace_id, get_logger
from app.observability.metrics import record_ingest
from app.observability.tracing import start_span
from app.retrieval import BM25Retriever
from app.stores import get_vector_store

logger = get_logger(__name__)


class IngestResult(BaseModel):
    """Outcome of ingesting a single file.

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


class IngestPipeline:
    """End-to-end document ingestion pipeline.

    The pipeline parses a file into documents, cleans the text, chunks it,
    embeds the chunks, persists them to the vector store, and updates the BM25
    index.  All heavy components are lazily initialised on first use.

    Args:
        settings: Application :class:`Settings` instance.  Only stored in the
            constructor; components are built on demand.
    """

    def __init__(self, settings: object) -> None:
        """Store settings; components are initialised lazily.

        Args:
            settings: Application :class:`Settings` instance.
        """
        self.settings = settings
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        self._bm25_retriever: BM25Retriever | None = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    @property
    def chunker(self):
        """Lazily build and return the chunker."""
        if self._chunker is None:
            self._chunker = get_chunker(self.settings)
        return self._chunker

    @property
    def embedder(self):
        """Lazily build and return the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def vector_store(self):
        """Lazily build and return the vector store."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.settings)
        return self._vector_store

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Lazily build and return the BM25 retriever."""
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.settings.bm25_index_path)
        return self._bm25_retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: Path) -> IngestResult:
        """Ingest a single file end-to-end.

        The flow is: parse -> clean -> chunk -> embed (batch) -> store.upsert
        -> bm25.add -> bm25.persist.  A ``trace_id`` is generated and bound to
        the logging context so every log line within the request is
        correlatable.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            An :class:`IngestResult` describing the outcome.  Non-fatal errors
            are recorded in ``result.errors`` rather than raised.
        """
        trace_id = uuid4().hex
        file_type = file_path.suffix.lower().lstrip(".") or "unknown"
        errors: list[str] = []
        doc_id = uuid4().hex
        num_chunks = 0

        with bind_trace_id(trace_id), start_span("ingest"):
            logger.info(
                "ingest.start",
                file=str(file_path),
                file_type=file_type,
            )

            # 1. Parse
            try:
                parser = get_parser(file_path)
                docs = parser.parse(file_path)
            except Exception as exc:  # noqa: BLE001 - record and continue
                logger.error(
                    "ingest.parse.failed",
                    file=str(file_path),
                    error=str(exc),
                )
                errors.append(f"parse: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            if not docs:
                errors.append("parse: no documents produced")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            doc_id = docs[0].metadata.doc_id

            # 2. Clean
            try:
                docs = clean_documents(docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest.clean.failed", error=str(exc))
                errors.append(f"clean: {exc}")

            # 3. Chunk
            try:
                chunks = self.chunker.chunk_documents(docs)
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.chunk.failed", error=str(exc))
                errors.append(f"chunk: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            if not chunks:
                errors.append("chunk: no chunks produced")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            num_chunks = len(chunks)

            # 4. Embed (batch)
            try:
                vectors = self.embedder.embed_texts([c.text for c in chunks])
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.embed.failed", error=str(exc))
                errors.append(f"embed: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            # 5. Store
            try:
                self.vector_store.upsert(chunks, vectors)
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.store.failed", error=str(exc))
                errors.append(f"store: {exc}")

            # 6. BM25 index
            try:
                self.bm25_retriever.add(chunks)
                self.bm25_retriever.persist()
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.bm25.failed", error=str(exc))
                errors.append(f"bm25: {exc}")

            record_ingest(file_type)
            logger.info(
                "ingest.done",
                file=str(file_path),
                doc_id=doc_id,
                num_chunks=num_chunks,
                errors=len(errors),
            )

            return IngestResult(
                doc_id=doc_id,
                num_chunks=num_chunks,
                file_type=file_type,
                trace_id=trace_id,
                errors=errors,
            )

    def ingest_directory(
        self,
        dir_path: Path,
        pattern: str = "**/*",
    ) -> list[IngestResult]:
        """Ingest every file under ``dir_path`` matching ``pattern``.

        Each file is processed independently via :meth:`ingest_file`; a failure
        on one file does not abort the batch.

        Args:
            dir_path: Root directory to scan.
            pattern: Glob pattern (default ``"**/*"`` for recursive scan).

        Returns:
            A list of :class:`IngestResult`, one per file encountered.
        """
        results: list[IngestResult] = []
        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                result = self.ingest_file(file_path)
            except Exception as exc:  # noqa: BLE001 - never crash the batch
                logger.error(
                    "ingest.file.unexpected",
                    file=str(file_path),
                    error=str(exc),
                )
                result = IngestResult(
                    doc_id=uuid4().hex,
                    num_chunks=0,
                    file_type=file_path.suffix.lower().lstrip(".") or "unknown",
                    trace_id=uuid4().hex,
                    errors=[f"unexpected: {exc}"],
                )
            results.append(result)
        return results

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks from every store.

        Removes chunks from the vector store and the BM25 index, then
        persists the BM25 index.

        Args:
            doc_id: Identifier of the document to remove.
        """
        logger.info("ingest.delete", doc_id=doc_id)
        try:
            self.vector_store.delete_by_doc(doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingest.delete.store.failed",
                doc_id=doc_id,
                error=str(exc),
            )
        try:
            self.bm25_retriever.remove_by_doc(doc_id)
            self.bm25_retriever.persist()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingest.delete.bm25.failed",
                doc_id=doc_id,
                error=str(exc),
            )
```

#### 顶部 import

- `from pathlib import Path`、`from uuid import uuid4`：路径对象和 UUID 生成。
- `from pydantic import BaseModel, Field`：数据模型。
- `from app.chunkers import get_chunker`、`from app.embedders import get_embedder`、`from app.ingest import clean_documents, get_parser`、`from app.stores import get_vector_store`：从各子模块的 factory 拿组件构造函数。
- `from app.observability.logging import bind_trace_id, get_logger`：`bind_trace_id` 是上下文管理器，把 trace_id 绑到日志上下文，让本次请求的所有日志都带同一个 trace_id。
- `from app.observability.metrics import record_ingest`：上报摄入指标。
- `from app.observability.tracing import start_span`：开启一个 trace span（用于分布式追踪）。
- `from app.retrieval import BM25Retriever`：BM25 检索器。

#### `IngestResult` 字段

- `doc_id: str`：摄入的文档 id。
- `num_chunks: int`：本次产生的 chunk 数。
- `file_type: str`：文件扩展名（不带点），比如 `"pdf"`、`"docx"`。
- `trace_id: str`：本次请求的追踪 id，便于在日志系统里串起整个调用链。
- `errors: list[str] = Field(default_factory=list)`：非致命错误列表（致命错误会直接 return，不进 errors）。

#### `IngestPipeline.__init__` 方法

- 参数 `settings: object`：应用配置对象。
- 函数体只做两件事：存 settings，把所有组件属性初始化为 None（待懒加载）。

#### 懒加载属性

四个 `@property`：`chunker`、`embedder`、`vector_store`、`bm25_retriever`。模式都一样：
- 如果对应私有属性是 None，就调对应的 factory 构造实例并缓存。
- 否则直接返回缓存的实例。

`@property` 装饰器让方法像属性一样访问：`pipeline.chunker` 而不是 `pipeline.chunker()`。

`bm25_retriever` 略不同：它直接 `BM25Retriever(self.settings.bm25_index_path)` 而不是走 factory，因为 BM25Retriever 构造时要从磁盘加载索引。

#### `ingest_file` 方法（6 步流程）

签名：
- 参数 `file_path: Path`：要摄入的文件路径。
- 返回 `IngestResult`：摄入结果。

函数体：

- `trace_id = uuid4().hex`：生成一个 32 位 hex 字符串作为本次请求的唯一 id。
- `file_type = file_path.suffix.lower().lstrip(".") or "unknown"`：取扩展名小写去点，没扩展名就 `"unknown"`。比如 `Path("x.pdf").suffix` → `".pdf"`，`.lstrip(".")` → `"pdf"`。
- `errors: list[str] = []`：错误列表。
- `doc_id = uuid4().hex`：先预生成一个 doc_id（万一解析失败没拿到，就用这个）。
- `num_chunks = 0`：初始化 chunk 计数。

- `with bind_trace_id(trace_id), start_span("ingest"):`：进入两个上下文管理器：
  - `bind_trace_id(trace_id)`：把 trace_id 绑到日志上下文，本次请求的所有日志都带这个 id。
  - `start_span("ingest")`：开一个名为 "ingest" 的追踪 span，用于分布式追踪系统（如 Jaeger）看耗时。

- `logger.info("ingest.start", file=..., file_type=...)`：打日志说摄入开始。

**第 1 步：Parse（解析）**

- `try: parser = get_parser(file_path); docs = parser.parse(file_path)`
  - `get_parser(file_path)`：根据扩展名选解析器（pdf → pdfplumber，docx → python-docx 等）。
  - `parser.parse(file_path)`：解析文件，返回 Document 列表。
- `except Exception as exc:`：解析失败：
  - 打 error 日志。
  - errors 加 `f"parse: {exc}"`，记录失败原因。
  - 直接 return 一个 `IngestResult(doc_id=..., num_chunks=0, ...)`，告诉调用方"这次摄入失败，0 条 chunk"。**注意：解析失败是致命错误，后面步骤都跳过。**

- `if not docs:`：解析成功但没产出文档（比如空文件）：
  - errors 加 `"parse: no documents produced"`。
  - 同样直接 return。

- `doc_id = docs[0].metadata.doc_id`：从解析出的第一个文档拿真正的 doc_id（覆盖前面预生成的）。后续都用这个 id。

**第 2 步：Clean（清洗）**

- `try: docs = clean_documents(docs)`：调用清洗器，清洗后**覆盖**原 docs。
- `except Exception as exc:`：清洗失败：
  - 打 warning 日志（不是 error，因为清洗失败不致命，可以用原脏文本继续）。
  - errors 加 `f"clean: {exc}"`。
  - **不 return**，继续往下走——用清洗前的 docs 进入下一步。这就是"非致命错误"的处理方式。

**第 3 步：Chunk（分块）**

- `try: chunks = self.chunker.chunk_documents(docs)`：调分块器，把 docs 切成 chunk 列表。
- `except Exception as exc:`：分块失败：
  - 打 error 日志。
  - errors 加 `f"chunk: {exc}"`。
  - 直接 return（分块失败致命，没 chunk 后面嵌入也没法做）。

- `if not chunks:`：分块成功但没产出 chunk（比如分块器配置异常）：
  - errors 加 `"chunk: no chunks produced"`。
  - return。

- `num_chunks = len(chunks)`：记录 chunk 数量，最后要塞进 IngestResult。

**第 4 步：Embed（批量嵌入）**

- `try: vectors = self.embedder.embed_texts([c.text for c in chunks])`：
  - `[c.text for c in chunks]`：把每个 chunk 的文本抽出来变列表。
  - `embed_texts(...)`：**批量嵌入**，一次性把所有 chunk 文本变成向量列表。比循环调 `embed_one` 快得多。
- `except Exception as exc:`：嵌入失败：
  - 打 error 日志。
  - errors 加 `f"embed: {exc}"`。
  - return（嵌入失败，没向量没法存进向量库）。

**第 5 步：Store（存进向量库）**

- `try: self.vector_store.upsert(chunks, vectors)`：把 chunk 和对应的向量一起 upsert 进向量库。`upsert` = update or insert，存在就更新，不存在就插入。
- `except Exception as exc:`：存储失败：
  - 打 error 日志。
  - errors 加 `f"store: {exc}"`。
  - **不 return**——继续往下做 BM25 索引。因为向量库和 BM25 是两个独立存储，一个失败不影响另一个。

**第 6 步：BM25 索引**

- `try: self.bm25_retriever.add(chunks); self.bm25_retriever.persist()`：
  - `add(chunks)`：把 chunk 加进 BM25 索引（9.4 节讲过）。
  - `persist()`：把更新后的索引存到磁盘，下次启动能加载。
- `except Exception as exc:`：BM25 失败：
  - 打 error 日志。
  - errors 加 `f"bm25: {exc}"`。

**收尾**

- `record_ingest(file_type)`：上报一个摄入成功指标（按 file_type 分类），便于监控。
- `logger.info("ingest.done", ...)`：打日志说摄入完成，记录 doc_id、num_chunks、errors 数。
- `return IngestResult(doc_id=..., num_chunks=num_chunks, file_type=..., trace_id=..., errors=errors)`：返回结果。

#### trace_id 注入与 start_span

- **`bind_trace_id(trace_id)`**：这是个上下文管理器（contextmanager），进入时把 trace_id 推到日志上下文（contextvars），退出时弹出。这样本次请求内所有 `logger.info(...)` 都会自动带上 trace_id，便于在 ELK/Loki 等日志系统里按 trace_id 过滤出整条调用链。

- **`start_span("ingest")`**：开启一个 OpenTelemetry 风格的 trace span，名字叫 "ingest"。span 是分布式追踪的基本单元，记录起止时间、属性，便于在 Jaeger/Tempo 里看每一步耗时。

两个上下文管理器一起用 `with A(), B():`，进入时同时进入 A 和 B，退出时反序退出。

#### 错误处理策略

整个 `ingest_file` 用了多层 `try/except`，每层都不让单步失败崩掉整个流程：
- **致命错误**（解析、分块、嵌入失败）→ 记录错误、立即 return，跳过后续步骤。
- **非致命错误**（清洗、存储、BM25 失败）→ 记录错误、继续往下走，能做多少做多少。
- **意外异常**（`ingest_directory` 里的 try/except）→ 整个文件处理崩了，也返回一个失败的 IngestResult，不影响下一个文件。

这种"尽力而为"的设计在批量摄入时特别重要——100 个文件里有 3 个损坏，不能让整个批次失败。

#### `ingest_directory` 方法

- 参数 `dir_path: Path`：要扫描的目录。
- 参数 `pattern: str = "**/*"`：glob 模式，默认 `"**/*"` 递归扫描所有文件。可改成 `"*.pdf"` 只扫 PDF。
- 返回 `list[IngestResult]`：每个文件一个 IngestResult。

- `results: list[IngestResult] = []`：结果列表。
- `for file_path in sorted(dir_path.glob(pattern)):`：
  - `dir_path.glob(pattern)`：glob 扫描，返回 Path 列表。
  - `sorted(...)`：按文件名排序，保证处理顺序确定（不传 sorted 的话不同 OS 顺序可能不一样）。
- `if not file_path.is_file(): continue`：跳过目录（glob `**/*` 会把子目录也带出来）。
- `try: result = self.ingest_file(file_path)`：调单文件摄入。
- `except Exception as exc:`：万一 `ingest_file` 自己没接住的异常：
  - 打 error 日志。
  - 构造一个失败的 IngestResult（num_chunks=0，errors=["unexpected: ..."]）。
  - **不让异常逃出去**，保证批次不中断。
- `results.append(result)`：加进结果。
- `return results`：返回所有文件的结果。

#### `delete_document` 方法

- 参数 `doc_id: str`：要删的文档 id，无返回值。

- `logger.info("ingest.delete", doc_id=doc_id)`：打日志。
- **第 1 个 try**：`self.vector_store.delete_by_doc(doc_id)`——从向量库删该文档的所有 chunk。失败就打 error 日志（不让一个存储失败影响另一个）。
- **第 2 个 try**：`self.bm25_retriever.remove_by_doc(doc_id)` + `self.bm25_retriever.persist()`——从 BM25 索引删并落盘。失败就打 error 日志。

注意：删除是"双删"——向量库和 BM25 都要删，否则检索时一边有数据一边没有，会出问题。

---

### 12.3 query_pipeline.py 逐行精读

文件路径：`app/pipeline/query_pipeline.py`。查询管线是摄入管线的镜像：摄入是"写"，查询是"读 + 答"。

完整代码：

```python
"""Query pipeline orchestrating retrieve -> rerank -> guardrail -> generate.

The :class:`QueryPipeline` ties together the embedding, hybrid retrieval,
reranking, guardrail and generation stages into a single end-to-end RAG query
flow.  Heavy components (embedder, vector store, BM25 index, hybrid retriever,
reranker, generator) are lazily initialised on first use so that constructing
the pipeline is cheap and model loading is deferred until the first ``query``
call.
"""
from __future__ import annotations

import time
from uuid import uuid4

from pydantic import BaseModel, Field

from app.embedders import get_embedder
from app.generation import CitationParser, Guardrail, get_generator
from app.observability.logging import bind_trace_id, get_logger
from app.observability.metrics import (
    record_generation_latency,
    record_no_result,
    record_query,
    record_retrieval_latency,
)
from app.observability.tracing import start_span
from app.rerank import get_reranker
from app.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from app.stores import get_vector_store

logger = get_logger(__name__)


class Reference(BaseModel):
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


class QueryResult(BaseModel):
    """Result of a RAG query.

    Attributes:
        answer: Generated natural-language answer (or the no-result fallback).
        references: Citation references mapped to source chunks.
        trace_id: Trace identifier correlating logs/metrics across the request.
        no_result: ``True`` when retrieval yielded no confident result.
        retrieval_latency: Retrieval stage latency in seconds.
        generation_latency: Generation stage latency in seconds.
    """

    answer: str
    references: list[Reference] = Field(default_factory=list)
    trace_id: str
    no_result: bool = False
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0


class QueryPipeline:
    """End-to-end RAG query pipeline.

    The pipeline embeds the query, retrieves candidates via hybrid retrieval,
    reranks them, checks the guardrail, and generates an answer with
    citations.  All heavy components are lazily initialised on first use.

    Args:
        settings: Application :class:`Settings` instance.  Only stored in the
            constructor; components are built on demand.
    """

    def __init__(self, settings: object) -> None:
        """Store settings; components are initialised lazily.

        Args:
            settings: Application :class:`Settings` instance.
        """
        self.settings = settings
        self._embedder = None
        self._vector_store = None
        self._bm25_retriever: BM25Retriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker = None
        self._generator = None
        self._guardrail: Guardrail | None = None
        self._citation_parser: CitationParser | None = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    @property
    def embedder(self):
        """Lazily build and return the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def vector_store(self):
        """Lazily build and return the vector store."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.settings)
        return self._vector_store

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Lazily build and return the BM25 retriever."""
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.settings.bm25_index_path)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Lazily build and return the hybrid retriever.

        The hybrid retriever wraps a :class:`VectorRetriever` (which needs the
        embedder and vector store) and the :class:`BM25Retriever`.
        """
        if self._hybrid_retriever is None:
            vector_retriever = VectorRetriever(
                store=self.vector_store,
                embedder=self.embedder,
            )
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                rrf_k=getattr(self.settings, "rrf_k", 60),
            )
        return self._hybrid_retriever

    @property
    def reranker(self):
        """Lazily build and return the reranker."""
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    @property
    def generator(self):
        """Lazily build and return the generator."""
        if self._generator is None:
            self._generator = get_generator(self.settings)
        return self._generator

    @property
    def guardrail(self) -> Guardrail:
        """Return the guardrail (created on first access)."""
        if self._guardrail is None:
            self._guardrail = Guardrail()
        return self._guardrail

    @property
    def citation_parser(self) -> CitationParser:
        """Return the citation parser (created on first access)."""
        if self._citation_parser is None:
            self._citation_parser = CitationParser()
        return self._citation_parser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict | None = None,
        top_n: int | None = None,
    ) -> QueryResult:
        """Run a RAG query end-to-end.

        Flow: embed_query -> hybrid retrieve -> rerank -> guardrail check ->
        generate -> parse citations -> map references.

        A ``trace_id`` is generated and bound to the logging context so every
        log line within the request is correlatable.

        Args:
            question: Natural-language question.
            filters: Optional metadata filters forwarded to the retriever.
            top_n: Override for the number of candidates to retrieve.  Falls
                back to ``settings.retrieve_top_n`` when ``None``.

        Returns:
            A :class:`QueryResult` with the answer, references and latencies.
        """
        trace_id = uuid4().hex
        effective_top_n = top_n or getattr(self.settings, "retrieve_top_n", 20)
        threshold = getattr(self.settings, "rerank_threshold", 0.3)
        rerank_top_k = getattr(self.settings, "rerank_top_k", 5)

        with bind_trace_id(trace_id), start_span("query"):
            logger.info("query.start", question=question[:80])

            # 1-2. Retrieve candidates via hybrid retrieval.
            retrieval_start = time.perf_counter()
            try:
                candidates = self.hybrid_retriever.retrieve(
                    question,
                    top_n=effective_top_n,
                    filters=filters,
                )
            except Exception as exc:  # noqa: BLE001 - treat as empty result
                logger.error("query.retrieve.failed", error=str(exc))
                candidates = []
            retrieval_latency = time.perf_counter() - retrieval_start
            record_retrieval_latency(retrieval_latency)

            # 3. No-result path: empty candidates.
            if not candidates:
                record_no_result()
                record_query("no_result")
                logger.info("query.no_result", reason="empty_candidates")
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=0.0,
                )

            # 4. Rerank.
            try:
                reranked = self.reranker.rerank(
                    question,
                    candidates,
                    top_k=rerank_top_k,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to raw candidates
                logger.error("query.rerank.failed", error=str(exc))
                reranked = candidates[:rerank_top_k]

            top_score = reranked[0].score if reranked else 0.0

            # 5. Guardrail: reject low-confidence results.
            #    check_confidence() already calls record_no_result() on reject,
            #    so we must not call it again here to avoid double counting.
            if not self.guardrail.check_confidence(top_score, threshold=threshold):
                record_query("no_result")
                logger.info(
                    "query.no_result",
                    reason="low_confidence",
                    top_score=top_score,
                )
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=0.0,
                )

            # 6. Generate answer.
            generation_start = time.perf_counter()
            try:
                result = self.generator.generate(question, reranked)
            except Exception as exc:  # noqa: BLE001
                logger.error("query.generate.failed", error=str(exc))
                generation_latency = time.perf_counter() - generation_start
                record_generation_latency(generation_latency)
                record_query("no_result")
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=generation_latency,
                )
            generation_latency = time.perf_counter() - generation_start
            record_generation_latency(generation_latency)

            # 7. Parse citations and map back to source chunks.
            citations = self.citation_parser.parse(result.answer)
            referenced_chunks = self.citation_parser.map_to_references(
                citations,
                reranked,
            )
            references = [
                Reference(
                    chunk_id=chunk.id,
                    source=chunk.metadata.source,
                    page=chunk.metadata.page,
                    score=chunk.score,
                    snippet=chunk.snippet(),
                )
                for chunk in referenced_chunks
            ]

            record_query("ok")
            logger.info(
                "query.done",
                answer_len=len(result.answer),
                references=len(references),
                retrieval_ms=int(retrieval_latency * 1000),
                generation_ms=int(generation_latency * 1000),
            )

            return QueryResult(
                answer=result.answer,
                references=references,
                trace_id=trace_id,
                no_result=False,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
            )
```

#### `Reference` 字段

- `chunk_id: str`：被引用的 chunk 的 id。
- `source: str`：chunk 的原始来源（文件路径或 URI），方便用户跳转。
- `page: int | None = None`：1-based 页码（PDF 才有，其他文档为 None）。
- `score: float | None = None`：相关性分数（重排分数），便于前端展示"相关度 95%"。
- `snippet: str = ""`：chunk 文本的前若干字符预览，让用户在点击前先看一眼内容。

#### `QueryResult` 字段

- `answer: str`：LLM 生成的答案（或拒答字符串）。
- `references: list[Reference] = Field(default_factory=list)`：引用列表。
- `trace_id: str`：本次请求的追踪 id。
- `no_result: bool = False`：是否为"无结果"（拒答）。前端可以据此显示不同 UI。
- `retrieval_latency: float = 0.0`：检索阶段耗时（秒），便于性能监控。
- `generation_latency: float = 0.0`：生成阶段耗时（秒）。

#### `QueryPipeline.__init__` 方法

- 参数 `settings: object`：应用配置。
- 函数体：存 settings，把 8 个组件属性初始化为 None（embedder、vector_store、bm25_retriever、hybrid_retriever、reranker、generator、guardrail、citation_parser），全部懒加载。

#### 懒加载属性

八个 `@property`，模式同 IngestPipeline。值得专门讲的是 `hybrid_retriever`：

```python
@property
def hybrid_retriever(self) -> HybridRetriever:
    if self._hybrid_retriever is None:
        vector_retriever = VectorRetriever(
            store=self.vector_store,    # ← 复用 self.vector_store
            embedder=self.embedder,     # ← 复用 self.embedder
        )
        self._hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=self.bm25_retriever,  # ← 复用 self.bm25_retriever
            rrf_k=getattr(self.settings, "rrf_k", 60),
        )
    return self._hybrid_retriever
```

注意这里**复用了已经懒加载的 embedder、vector_store、bm25_retriever**——通过 `self.xxx` 访问，会触发对应的 `@property`，第一次访问时构造，后续直接返回缓存。这样保证整个 pipeline 只有一份 embedder、vector_store、bm25_retriever 实例，不会重复加载模型。

`rrf_k=getattr(self.settings, "rrf_k", 60)`：从配置读 RRF 的 k 值，没配默认 60。

#### `query` 方法（7 步流程）

签名：
- 参数 `question: str`：用户问题。
- 参数 `filters: dict | None = None`：可选过滤条件。
- 参数 `top_n: int | None = None`：候选数覆盖。None 时用 settings 默认值。
- 返回 `QueryResult`。

函数体：

- `trace_id = uuid4().hex`：生成追踪 id。
- `effective_top_n = top_n or getattr(self.settings, "retrieve_top_n", 20)`：用传入的 top_n，没传就读 `settings.retrieve_top_n`，再没配默认 20。
- `threshold = getattr(self.settings, "rerank_threshold", 0.3)`：读护栏阈值，默认 0.3。
- `rerank_top_k = getattr(self.settings, "rerank_top_k", 5)`：读重排 top_k，默认 5。

- `with bind_trace_id(trace_id), start_span("query"):`：进入追踪上下文。
- `logger.info("query.start", question=question[:80])`：打日志。

**第 1-2 步：混合检索**

- `retrieval_start = time.perf_counter()`：检索开始计时。
- `try: candidates = self.hybrid_retriever.retrieve(question, top_n=effective_top_n, filters=filters)`：调混合检索器（向量 + BM25 + RRF 融合）。
- `except Exception as exc: logger.error(...); candidates = []`：检索失败就当空结果处理，不让异常逃出去。
- `retrieval_latency = time.perf_counter() - retrieval_start`：算检索耗时。
- `record_retrieval_latency(retrieval_latency)`：上报指标。

**第 3 步：空候选拒答**

- `if not candidates:`：如果检索没拿到任何候选：
  - `record_no_result()`：上报无结果。
  - `record_query("no_result")`：上报查询类型为"no_result"。
  - `logger.info("query.no_result", reason="empty_candidates")`：打日志说"空候选"。
  - return 一个 QueryResult：answer 是拒答字符串，references 空，no_result=True，generation_latency=0.0（没调 LLM）。

**第 4 步：重排**

- `try: reranked = self.reranker.rerank(question, candidates, top_k=rerank_top_k)`：调重排器，取 top_k 个。
- `except Exception as exc: logger.error(...); reranked = candidates[:rerank_top_k]`：重排失败就**降级**——直接取检索前 top_k 个，不让重排失败导致整个查询挂掉。
- `top_score = reranked[0].score if reranked else 0.0`：拿重排后第一名的分数。

**第 5 步：护栏检查**

- `if not self.guardrail.check_confidence(top_score, threshold=threshold):`：调护栏检查置信度。
- 注意注释：`check_confidence()` 内部已经调过 `record_no_result()`，所以这里**不再调**，避免重复计数。
- 不通过就：
  - `record_query("no_result")`：上报 no_result。
  - `logger.info("query.no_result", reason="low_confidence", top_score=...)`：打日志说"置信度不足"。
  - return 拒答 QueryResult，generation_latency=0.0。

**第 6 步：生成答案**

- `generation_start = time.perf_counter()`：生成开始计时。
- `try: result = self.generator.generate(question, reranked)`：调生成器，传入问题和重排后的 chunks。
- `except Exception as exc:`：生成失败（比如 LLM API 挂了、超时）：
  - `logger.error("query.generate.failed", error=str(exc))`：打 error 日志。
  - `generation_latency = time.perf_counter() - generation_start`：算生成耗时（哪怕失败了也要上报）。
  - `record_generation_latency(generation_latency)`：上报。
  - `record_query("no_result")`：上报 no_result。
  - return 拒答 QueryResult，但 generation_latency 是真实耗时（用来监控 LLM 的故障响应时间）。
- `generation_latency = time.perf_counter() - generation_start`：成功路径，算耗时。
- `record_generation_latency(generation_latency)`：上报。

**第 7 步：解析引用 + 映射**

- `citations = self.citation_parser.parse(result.answer)`：从答案里解析 `[1] [2]` 引用编号。
- `referenced_chunks = self.citation_parser.map_to_references(citations, reranked)`：把编号映射回 reranked 列表里的 chunk。
- `references = [Reference(chunk_id=chunk.id, source=chunk.metadata.source, page=chunk.metadata.page, score=chunk.score, snippet=chunk.snippet()) for chunk in referenced_chunks]`：构造 Reference 列表，每个引用对应一个 Reference 对象，含 chunk_id、source、page、score、snippet 五个字段。

**收尾**

- `record_query("ok")`：上报查询成功。
- `logger.info("query.done", ...)`：打日志，记录答案长度、引用数、检索耗时、生成耗时。
- `return QueryResult(answer=result.answer, references=references, trace_id=trace_id, no_result=False, retrieval_latency=..., generation_latency=...)`：返回完整结果。

#### guardrail 拒答逻辑总结

整个 `query` 方法有 **三处** 拒答路径：
1. **空候选**（candidates 为空）——检索阶段就没拿到东西。
2. **低置信度**（top_score < threshold）——检索拿到了但重排分太低。
3. **生成失败**（generator.generate 抛异常）——检索和重排都通过，但 LLM 调用失败。

每种路径都返回 `no_result=True` 和拒答字符串，但 `retrieval_latency` / `generation_latency` 字段如实记录已发生的耗时——便于监控区分"检索没找到"和"LLM 出了问题"。

#### 延迟记录

`QueryResult` 里特意分了 `retrieval_latency` 和 `generation_latency` 两个字段（而不是只给一个 `total_latency`），原因：
- 检索慢通常是向量库/BM25 索引问题，要优化索引或换硬件。
- 生成慢通常是 LLM API 慢或 prompt 太长，要换模型或缩短 context。
- 两个字段分开能快速定位瓶颈在哪一阶段，便于针对性优化。

---

### 12.4 container.py 逐行精读

文件路径：`app/pipeline/container.py`。这是 DI（依赖注入）容器，用单例模式管理 settings 和两个 pipeline。

完整代码：

```python
"""Dependency-injection container caching shared pipeline instances.

The :class:`Container` provides singleton access to the application settings
and the two top-level pipelines (:class:`IngestPipeline` and
:class:`QueryPipeline`).  Using :func:`functools.lru_cache` on the accessor
methods ensures that repeated calls return the *same* instance, avoiding
re-loading of ML models and re-initialisation of stores on every request.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings

from app.pipeline.ingest_pipeline import IngestPipeline
from app.pipeline.query_pipeline import QueryPipeline


class Container:
    """Singleton container caching settings and pipeline instances.

    All accessors are class methods backed by :func:`functools.lru_cache` so
    that the first call constructs the instance and every subsequent call
    returns the cached singleton.

    Typical usage::

        pipeline = Container.get_query_pipeline()
        result = pipeline.query("What is RAG?")
    """

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls):
        """Return the cached :class:`Settings` singleton.

        Returns:
            The application :class:`Settings` instance (loaded once).
        """
        return get_settings()

    @classmethod
    @lru_cache(maxsize=1)
    def get_ingest_pipeline(cls) -> IngestPipeline:
        """Return the cached :class:`IngestPipeline` singleton.

        Returns:
            A shared :class:`IngestPipeline` instance (constructed once).
        """
        return IngestPipeline(cls.get_settings())

    @classmethod
    @lru_cache(maxsize=1)
    def get_query_pipeline(cls) -> QueryPipeline:
        """Return the cached :class:`QueryPipeline` singleton.

        Returns:
            A shared :class:`QueryPipeline` instance (constructed once).
        """
        return QueryPipeline(cls.get_settings())

    @classmethod
    def reset(cls) -> None:
        """Clear all cached singletons.

        Primarily useful in tests where a fresh pipeline is needed.
        """
        cls.get_settings.cache_clear()
        cls.get_ingest_pipeline.cache_clear()
        cls.get_query_pipeline.cache_clear()
```

#### `@lru_cache` 单例模式逐行

- `from functools import lru_cache`：导入 `lru_cache` 装饰器。LRU = Least Recently Used，"最近最少使用"。`lru_cache(maxsize=1)` 表示"只缓存最近 1 次调用的结果"——这就是单例：第一次调用构造实例并缓存，第二次调用直接返回缓存，不会再次构造。

- `class Container:`：容器类，**没有 `__init__`**——所有方法都是 classmethod，不需要实例化就能用。

- `@classmethod` + `@lru_cache(maxsize=1)` **两个装饰器叠加**：
  - `@classmethod`：让方法是类方法，可以用 `Container.get_xxx()` 调用，不需要 `Container().get_xxx()`。
  - `@lru_cache(maxsize=1)`：缓存最近一次调用的返回值。两个装饰器叠加的顺序很关键：`@lru_cache` 在内层，`@classmethod` 在外层，意思是"先把方法变成带缓存的函数，再包成类方法"。

#### `get_settings` 方法

- `return get_settings()`：调配置层的 `get_settings()` 拿 Settings 实例。第一次调用时构造，后续调用直接返回缓存。

#### `get_ingest_pipeline` 方法

- `return IngestPipeline(cls.get_settings())`：用 settings 构造 IngestPipeline。
- 注意 `cls.get_settings()` 也是 `@lru_cache` 的——所以这里不会重复读配置，而是拿到同一个 Settings 实例。
- 第一次调用时构造 IngestPipeline，后续直接返回缓存。这样 BGE 模型只会加载一次，向量库连接只建立一次。

#### `get_query_pipeline` 方法

- `return QueryPipeline(cls.get_settings())`：同理，构造 QueryPipeline 并缓存。

#### `reset` 方法

- `cls.get_settings.cache_clear()`：清空 `get_settings` 的 lru_cache。
- 后面两行同理清空另外两个的缓存。
- 用途：**测试时**想用一个全新的 pipeline 实例（比如换了 mock settings），调 `Container.reset()` 后再 `get_xxx()` 就会重新构造。生产环境一般不调。

#### 为什么不直接用全局变量？

有些人会问：直接写 `pipeline = QueryPipeline(get_settings())` 不就行了吗？为什么要这么绕？

理由有三：
1. **延迟初始化**：全局变量在模块导入时就构造了，但导入时未必想加载模型。`@lru_cache` 是**第一次调用时**才构造，符合懒加载理念。
2. **可重置**：全局变量没法清缓存，`lru_cache` 可以 `cache_clear()`，方便测试。
3. **可测试**：测试时可以 mock `get_settings` 或 reset cache，灵活替换组件。

#### 典型用法

```python
from app.pipeline.container import Container

# 拿到 query pipeline 单例
pipeline = Container.get_query_pipeline()

# 跑一次查询
result = pipeline.query("X-2025 续航多久？")
print(result.answer)
print([r.source for r in result.references])
```

整个应用从 API 层（FastAPI 路由）到底层 pipeline，都通过 `Container` 拿单例。这样：
- 同一个进程内所有请求共享同一份 pipeline（共享模型、共享连接池）。
- 不会每次请求都重新加载模型（每次加载 BGE 要好几秒，根本扛不住高 QPS）。

---

### 12.5 完整流程示例

把"摄入"和"查询"两条线的数据流串起来走一遍，对照示例文档"X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0"和问题"X-2025 续航多久？"。

#### 摄入数据流

输入：文件 `manuals/x2025.pdf`（含那段"X-2025 智能音箱..."的文字）。

| 步骤 | 组件 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1. Parse | pdfplumber 解析器 | `manuals/x2025.pdf` | `[Document(text="X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。...", metadata=DocumentMetadata(doc_id="d001", source="manuals/x2025.pdf", page=1, ...))]` |
| 2. Clean | clean_documents | 上面那个 Document 列表 | 清洗后的 Document 列表（去多余空行、规范化空白等） |
| 3. Chunk | chunker.chunk_documents | 清洗后 Document | `[Chunk(id="c001", text="X-2025 智能音箱，续航 12 小时，支持蓝牙 5.0。", metadata=...), Chunk(id="c002", text="蓝牙 5.0 双模连接", ...), ...]` |
| 4. Embed | embedder.embed_texts | `[c.text for c in chunks]` | `[[0.12, -0.34, ...768 维...], [0.45, 0.21, ...], ...]` |
| 5. Store | vector_store.upsert | chunks + vectors | 向量库多了 5 条记录 |
| 6. BM25 | bm25_retriever.add + persist | chunks | BM25 索引多了 5 条记录，pickle 文件落盘 |

返回：`IngestResult(doc_id="d001", num_chunks=5, file_type="pdf", trace_id="abc...", errors=[])`。

#### 查询数据流

输入：问题 `"X-2025 续航多久？"`。

| 步骤 | 组件 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1-2. Retrieve | hybrid_retriever.retrieve | 问题 + top_n=20 + filters=None | `[Chunk(c001, score=0.0333), Chunk(c002, score=0.0325), Chunk(c004, score=0.0164), ...]`（向量 + BM25 并发检索后 RRF 融合） |
| 3.（无空候选跳过） | — | — | — |
| 4. Rerank | reranker.rerank | 问题 + 上面 5 个候选 + top_k=5 | `[Chunk(c001, score=0.98), Chunk(c004, score=0.85), Chunk(c002, score=0.12), ...]`（BGE cross-encoder 重排） |
| 5. Guardrail | guardrail.check_confidence | top_score=0.98, threshold=0.3 | `True`（通过，进入生成阶段） |
| 6. Generate | generator.generate | 问题 + 重排后 5 个 chunk | `GenerationResult(answer="X-2025 智能音箱的续航时间为 12 小时 [1] [2]。", citations=[1, 2], used_chunk_ids=["c001", "c004"])` |
| 7. Citation | citation_parser.parse + map_to_references | answer + reranked | `[Chunk(c001, ...), Chunk(c004, ...)]` → 转成 `[Reference(chunk_id="c001", source="manuals/x2025.pdf", page=1, score=0.98, snippet="X-2025 智能音箱，续航..."), Reference(chunk_id="c004", source="manuals/x2025.pdf", page=2, score=0.85, snippet="续航 12 小时，充电...")]` |

返回 `QueryResult`：

```python
QueryResult(
    answer="X-2025 智能音箱的续航时间为 12 小时 [1] [2]。",
    references=[
        Reference(chunk_id="c001", source="manuals/x2025.pdf", page=1, score=0.98, snippet="X-2025 智能音箱，续航 12 小时..."),
        Reference(chunk_id="c004", source="manuals/x2025.pdf", page=2, score=0.85, snippet="续航 12 小时，充电 2 小时"),
    ],
    trace_id="def...",
    no_result=False,
    retrieval_latency=0.18,    # 检索花了 180 毫秒
    generation_latency=1.24,    # 生成花了 1.24 秒
)
```

#### 端到端视角

把上面所有章节的代码串起来，一个完整的 RAG 系统长这样：

```
[用户上传文件] ──► IngestPipeline ──► [chunks + vectors + BM25 索引]
                       │
                       ├─ parser     (解析文件)
                       ├─ cleaner    (清洗文本)
                       ├─ chunker    (切片)
                       ├─ embedder   (向量化)
                       ├─ vector_store  (存向量)
                       └─ bm25_retriever (存倒排索引)

[用户提问]    ──► QueryPipeline    ──► [answer + references]
                       │
                       ├─ hybrid_retriever ──┬─ vector_retriever (向量检索)
                       │                     └─ bm25_retriever   (BM25 检索)
                       │                     └─ rrf_fusion       (RRF 融合)
                       ├─ reranker    (cross-encoder 重排)
                       ├─ guardrail   (置信度护栏)
                       ├─ generator   (LLM 生成)
                       └─ citation_parser (引用解析)
```

每个方框都是前面某一章精读过的文件里的类。所有这些组件由 `Container` 单例管理，由 `IngestPipeline` / `QueryPipeline` 编排，对用户只暴露两个简单方法：`ingest_file` 和 `query`。

#### 关键设计回顾

整个第三部分涵盖了五大设计原则，回头看一眼：

1. **混合检索**（9 章）：向量懂意思，BM25 懂关键词，RRF 把两份结果融合，互补互补。
2. **粗筛 + 精排**（10 章）：先用快速的 bi-encoder 海选 top20，再用 cross-encoder 精排到 top5，兼顾速度和精度。
3. **强约束 Prompt**（11 章）：system prompt 限定只能用上下文、必须引用、不够就说不知道，三管齐下防幻觉。
4. **多层护栏**（11-12 章）：检索前不过滤、检索后看分数、生成失败兜底拒答，每一步都有降级路径。
5. **懒加载 + 单例**（12 章）：组件第一次用到才构造，构造后缓存复用；整个应用通过 Container 共享实例，避免重复加载模型。

掌握这五条，你就理解了 kb-rag 的核心架构。剩下的细节（具体模型选择、向量库选型、配置项调参）都是工程优化，可以慢慢摸索。

---

**第三部分完。**

下一部分（如果有）会讲部署：Docker、FastAPI 路由、可观测性（日志/指标/追踪）、性能调优等。


