# kb-rag 新手教程 · 第二部分：清洗 → 分块 → 嵌入 → 存储

> 本教程面向只会 Python 基础语法、对 ML/NLP 一无所知的同学。
> 我们会用一个统一的例子贯穿全文：一份《X-2025 智能音箱产品手册》。
>
> ```text
> # X-2025 智能音箱产品手册
>
> ## 1. 产品概述
> X-2025 是一款支持语音交互的智能音箱，搭载四核 ARM A55 处理器，
> 配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。
>
> ## 2. 技术规格
> 续航时间 12 小时，充电时间 2 小时，扬声器功率 20W，重量 1.2kg。
>
> ## 3. 使用说明
> 首次使用时，请先充电至 80% 以上。长按电源键 3 秒开机。
> ```

---

## 第 5 章 · 文本清洗——把"脏"文本变干净

### 5.1 为什么需要清洗

在真实世界里，你拿到的文本往往是从 PDF、Word、Excel、HTML 里"抠"出来的。
这些抽取工具（pdfplumber、python-docx、BeautifulSoup 等）并不完美，
抠出来的文本里常常夹带下面这些"垃圾"：

| 问题 | 例子 | 危害 |
| --- | --- | --- |
| 多余空行 | 一段话后面跟 5 个 `\n` | 浪费 token，分块时被误判为多段 |
| 单词中间断行 | `process\nor` 实际是 `processor` | 检索"processor"找不到 |
| 全角半角混用 | `，` 和 `,` 混着用 | 同义词检索失败 |
| 不可见控制字符 | `\x00`、`\x07` | 嵌入模型报错或结果异常 |
| 行尾多余空格 | `续行时间 12 小时。   \n` | 看着干净其实不干净 |
| Windows 换行符 | `\r\n` | 在不同 OS 上行为不一致 |

**清洗前 vs 清洗后**对比示例：

清洗前（一段从 PDF 抠出来的"脏"文本）：

```text
# X-2025 智能音箱产品手册\r
\r
\r
## 1. 产品概述\r
X-2025 是一款支持语音交互的智能音箱，搭载四核 ARM A55 处理\n器，\r
配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。\r
```

清洗后（cleaner.py 输出）：

```text
# X-2025 智能音箱产品手册

## 1. 产品概述
X-2025 是一款支持语音交互的智能音箱,搭载四核 ARM A55 处理器,
配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。
```

变化点：
1. `\r\n` 统一变成 `\n`；
2. 三个连续空行压成一段段落分隔 `\n\n`；
3. `处理\n器` 这种"单词中间断行"被拼回 `处理器`；
4. 行尾的 `\r` 没了；
5. （注：项目里 `_PUNCT_MAP` 会把 `，` 换成 `,`、`。` 换成 `.`，但 `处理器` 后面的逗号是中文全角，本项目设计上会变成半角，方便统一处理中英文混合检索。）

### 5.2 cleaner.py 逐行精读

文件路径：`app/ingest/cleaner.py`。我们一段一段来看。

#### 5.2.1 文件头与 import

```python
"""Text cleaning utilities for the ingestion stage.

The cleaner normalises text extracted from heterogeneous file formats so
downstream chunkers and embedders operate on consistent input. It handles:

* removal of control characters,
* normalisation of full-width (CJK) punctuation to half-width equivalents,
* collapsing excessive blank lines into paragraph breaks,
* removing spurious in-word line breaks while preserving paragraph boundaries,
* collapsing runs of spaces into a single space.
"""
from __future__ import annotations

import re
import unicodedata

from app.models.document import Document
```

逐行解释：

- `"""..."""`：模块的文档字符串（docstring），说明这个文件干啥的。新手记住：每个 Python 文件第一行通常写一段说明，方便别人读。
- `from __future__ import annotations`：开启"延迟注解求值"。意思是 `def f(x: "Document") -> "list[Document]"` 这种类型注解不会在定义时立刻被解析，方便你写 `Document` 还没真正 import 完时的递归引用。Python 3.10+ 项目常用。
- `import re`：导入内置正则表达式库。`re` 就是 regular expression。
- `import unicodedata`：导入内置 Unicode 工具库，用来查询字符的分类（比如判断一个字符是不是"控制字符"）。
- `from app.models.document import Document`：从项目里导入 `Document` 数据类。`Document` 长这样（简化版）：
  - `id: str` 文档唯一 id
  - `text: str` 文档正文
  - `metadata: Metadata` 元信息（来源、页码、标签等）

#### 5.2.2 全角→半角标点映射表

```python
_PUNCT_MAP: dict[str, str] = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "「": "'",
    "」": "'",
    "『": '"',
    "』": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
    "～": "~",
}
```

解释：

- `_PUNCT_MAP` 是一个模块级常量（前缀 `_` 表示"模块内部用，别在外部 import"）。
- 类型注解 `dict[str, str]`：key 是字符串、value 也是字符串。
- 作用：把中文全角标点（如 `，`）替换成英文半角标点（如 `,`）。
- 打个比方：你写文档时一会儿用全角逗号一会儿用半角，搜"续航时间，12 小时"和"续航时间,12 小时"会被系统当作两段不同的文字。统一成半角后，检索就不会漏。
- 注意：`……`、`——` 这种有语义的中文符号**故意没替换**，因为它们本身有特殊含义（省略号、破折号），换成英文反而读不懂。

#### 5.2.3 预编译正则表达式

```python
# Pre-compiled regexes used by :func:`clean_text`.
_RE_CRLF = re.compile(r"\r\n")
_RE_CR = re.compile(r"\r")
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")
# A single \n flanked by non-space characters is a broken in-word line break.
_RE_MIDWORD_NL = re.compile(r"(?<=\S)\n(?=\S)")
# A single \n with surrounding spaces should collapse to a single space.
_RE_NL_WITH_SPACES = re.compile(r"(?<=\S)\n[ ]+(?=\S)")
# Trailing spaces before a newline.
_RE_SPACES_BEFORE_NL = re.compile(r"[ ]+\n")
# Leading spaces after a newline.
_RE_NL_THEN_SPACES = re.compile(r"\n[ ]+")
# Runs of two or more spaces.
_RE_MANY_SPACES = re.compile(r"[ ]{2,}")
```

逐个解释：

- `re.compile(...)`：把正则表达式预先编译成一个 `Pattern` 对象。如果同一段正则要用很多次，预先编译比每次 `re.sub` 现编译要快。
- `_RE_CRLF = re.compile(r"\r\n")`：匹配 Windows 风格的换行符 `\r\n`。
  - `r"..."`：raw string，反斜杠不被转义，所以 `\r` 是"回车符"而不是字符串里的反斜杠+r。
- `_RE_CR = re.compile(r"\r")`：匹配单独的回车符 `\r`（老 Mac 风格）。
- `_RE_MANY_NEWLINES = re.compile(r"\n{3,}")`：匹配**至少 3 个**连续 `\n`。
  - `\n` 是换行符；`{3,}` 表示"出现 3 次或更多次"。
- `_RE_MIDWORD_NL = re.compile(r"(?<=\S)\n(?=\S)")`：匹配"单词中间的断行"。
  - `(?<=\S)`：lookbehind（向后看），要求 `\n` 前面是一个非空白字符（`\S` 是 non-whitespace）。
  - `(?=\S)`：lookahead（向前看），要求 `\n` 后面也是一个非空白字符。
  - 效果：只有像 `处理\n器` 这种"前后都是文字"的 `\n` 才匹配；段落之间的空行（`\n\n`）不会匹配。
- `_RE_NL_WITH_SPACES = re.compile(r"(?<=\S)\n[ ]+(?=\S)")`：匹配"前后是文字、中间夹着空格的换行"。
  - `[ ]+`：一个或多个空格。
  - 例：`处理 \n 器` 会被匹配，并替换成单个空格 ` `。
- `_RE_SPACES_BEFORE_NL = re.compile(r"[ ]+\n")`：行尾的空格。
  - 例：`续航 12 小时。   \n` 会把 `   \n` 替换成 `\n`。
- `_RE_NL_THEN_SPACES = re.compile(r"\n[ ]+")`：行首的空格（缩进）。
- `_RE_MANY_SPACES = re.compile(r"[ ]{2,}")`：连续 2 个或更多空格。
  - 例：`支持   Wi-Fi` 会被替换成 `支持 Wi-Fi`。

#### 5.2.4 主函数 `clean_text`

```python
def clean_text(text: str) -> str:
    """Normalise whitespace and punctuation in a text string.

    Args:
        text: Raw text potentially containing control characters, full-width
            punctuation, excessive blank lines, broken in-word line breaks, and
            runs of spaces.

    Returns:
        Cleaned text with paragraph breaks (``\\n\\n``) preserved, in-word line
        breaks removed, and runs of spaces collapsed to a single space.
    """
    if not text:
        return text

    # 1. Strip control characters (category Cc) except \n, \r, \t.
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\r", "\t") or unicodedata.category(ch) != "Cc"
    )

    # 2. Normalise line endings to \n.
    text = _RE_CRLF.sub("\n", text)
    text = _RE_CR.sub("\n", text)

    # 3. Normalise full-width punctuation to half-width.
    for full, half in _PUNCT_MAP.items():
        if full in text:
            text = text.replace(full, half)

    # 4. Collapse 3+ consecutive newlines into a paragraph break.
    text = _RE_MANY_NEWLINES.sub("\n\n", text)

    # 5. Remove in-word single newlines (keep \n\n paragraph breaks intact).
    text = _RE_MIDWORD_NL.sub("", text)

    # 6. Replace newlines surrounded by spaces with a single space.
    text = _RE_NL_WITH_SPACES.sub(" ", text)

    # 7. Trim spaces around newlines.
    text = _RE_SPACES_BEFORE_NL.sub("\n", text)
    text = _RE_NL_THEN_SPACES.sub("\n", text)

    # 8. Collapse runs of spaces into one.
    text = _RE_MANY_SPACES.sub(" ", text)

    # 9. Strip trailing whitespace on each line and overall.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
```

**函数签名**：

- 函数名：`clean_text`
- 参数：`text: str`，输入是要清洗的原始字符串。
- 返回值：`str`，清洗后的字符串。

**逐行解释**：

- `if not text: return text`：如果传入空字符串 `""` 或 `None`（其实 `None` 会让 `not text` 为 True，但类型注解上写的是 `str`），直接返回，避免后续操作报错。
- 步骤 1（删控制字符）：
  - `unicodedata.category(ch)` 返回字符的 Unicode 类别，例如 `"Cc"` 表示 Control character（控制字符）。
  - 用生成器表达式遍历每个字符 `ch`，保留三种合法字符 `\n \r \t`，或非 `Cc` 类别的字符。
  - `"".join(...)` 把保留的字符重新拼成字符串。
  - 这一步会删掉 `\x00`（空字符）、`\x07`（响铃）、`\x1b`（ESC）等不可见垃圾。
- 步骤 2（统一换行）：
  - `_RE_CRLF.sub("\n", text)`：把 `\r\n` 替换成 `\n`。
  - `_RE_CR.sub("\n", text)`：把单独的 `\r` 也替换成 `\n`。
  - 这步保证全文换行符统一。
- 步骤 3（全角→半角）：
  - 遍历 `_PUNCT_MAP`，对每个 `full → half` 做一次 `str.replace`。
  - `if full in text` 是个小优化：如果文本里没这个全角符号，就不调用 replace。
- 步骤 4（压扁多余空行）：
  - `_RE_MANY_NEWLINES.sub("\n\n", text)`：3 个或更多 `\n` 压成 2 个 `\n`（段落分隔）。
- 步骤 5（修单词中间断行）：
  - `_RE_MIDWORD_NL.sub("", text)`：把 `处理\n器` 中间的 `\n` 删掉（替换成空串 `""`），变成 `处理器`。
  - 注意：段落分隔是 `\n\n`，不满足"前后都是非空白"的条件，所以不会被误删。
- 步骤 6（带空格的断行）：
  - `处理 \n 器` → `处理 器`（替换成单个空格）。
- 步骤 7（行首行尾空格）：
  - 行尾：`续航。   \n` → `续航。\n`。
  - 行首：`\n   续航` → `\n续航`。
- 步骤 8（多个空格压一个）：
  - `支持   Wi-Fi` → `支持 Wi-Fi`。
- 步骤 9（最后扫尾）：
  - `text.split("\n")` 按换行拆成行列表。
  - `line.rstrip() for line in ...`：每行去掉右侧空格。
  - `"\n".join(...)` 重新拼回。
  - `text.strip()` 去掉整体首尾空白。

**清洗前后对比**（用我们的示例文本演示）：

清洗前（模拟从 PDF 抠出来时有断行问题）：

```text
# X-2025 智能音箱产品手册\r\n\r\n\r\n## 1. 产品概述\r\nX-2025 是一款支持语音交互的智能音箱,搭载四核 ARM A55 处理\n器,\r\n配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。\r\n
```

清洗后：

```text
# X-2025 智能音箱产品手册

## 1. 产品概述
X-2025 是一款支持语音交互的智能音箱,搭载四核 ARM A55 处理器,
配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。
```

#### 5.2.5 `clean_document`：清洗一个 Document

```python
def clean_document(doc: Document) -> Document:
    """Return a copy of ``doc`` whose text has been cleaned.

    Args:
        doc: The source :class:`Document`.

    Returns:
        A new :class:`Document` with cleaned text and identical metadata.
    """
    cleaned = clean_text(doc.text)
    return doc.model_copy(update={"text": cleaned})
```

**函数签名**：

- 参数：`doc: Document`，一个 `Document` 对象。
- 返回值：`Document`，新的 `Document`（原对象不变）。

**逐行解释**：

- `cleaned = clean_text(doc.text)`：把文档正文喂给 `clean_text`，得到清洗后的字符串。
- `doc.model_copy(update={"text": cleaned})`：Pydantic v2 的方法，复制 `doc`，但把 `text` 字段替换成 `cleaned`。
  - 为什么不直接 `doc.text = cleaned`？因为 `Document` 是 Pydantic 模型，默认是不可变的（`model_config` 可以配置），而且函数式风格更安全——原对象不被改，调试时能对比前后。

#### 5.2.6 `clean_documents`：批量清洗

```python
def clean_documents(docs: list[Document]) -> list[Document]:
    """Clean a batch of documents.

    Args:
        docs: List of :class:`Document` instances to clean.

    Returns:
        A new list of cleaned :class:`Document` instances (same order, same
        length).
    """
    return [clean_document(doc) for doc in docs]
```

**函数签名**：

- 参数：`docs: list[Document]`，一组 `Document`。
- 返回值：`list[Document]`，清洗后的新列表，顺序、长度都不变。

**逐行解释**：

- 列表推导式 `[clean_document(doc) for doc in docs]`：对每个 `doc` 调一次 `clean_document`，结果收集成一个新列表。
- 没有用 `map(clean_document, docs)` 是因为列表推导式在 Python 里可读性更好。

---

## 第 6 章 · 分块——把长文本切成小段

### 6.1 为什么要分块

假设你把整本《X-2025 智能音箱产品手册》当作一个整体塞进向量库。
用户问"续航多久？"，系统把整本手册的向量取出来，丢给大模型让它回答。
问题：

1. **找不准**：一整本文档的向量是"全文平均意思"，会稀释"续航 12 小时"这种具体信息。
2. **超 token 限制**：大模型上下文窗口有限，整本 1 万字塞不下。
3. **响应慢**：传的字数越多，模型推理越慢、越贵。

**打比方**：一本 500 页的书，你要找"续航时间"。
- 不分块 = 整本翻，从头到尾找，慢得要命；
- 分块 = 按章节拆开，直接翻"技术规格"章节，秒找到。

**分块大小的权衡**：

| 太大 | 太小 |
| --- | --- |
| 检索不精确（一个 chunk 里混杂多个主题） | 上下文不够（一个 chunk 只有一句话，模型答不出来） |
| 嵌入向量"平均化"，丢细节 | chunk 数量爆炸，存储和检索成本高 |
| 推理 token 浪费 | 相邻信息被切散 |

项目默认值 `chunk_size=512, overlap=64` 是个比较平衡的选择（详见 6.2）。

### 6.2 base.py 逐行精读

文件路径：`app/chunkers/base.py`。

#### 6.2.1 import 与 ChunkerConfig

```python
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.models.document import Chunk, Document, Metadata


@dataclass
class ChunkerConfig:
    """Tuning parameters shared by all chunker implementations.

    Attributes:
        chunk_size: Target size of a chunk. Interpreted as tokens by
            token-based chunkers and as characters by character-based ones.
        overlap: Number of tokens/characters of overlap between adjacent chunks.
        min_chunk_size: Minimum size below which a chunk is discarded/merged.
    """

    chunk_size: int = 512
    overlap: int = 64
    min_chunk_size: int = 50
```

逐行解释：

- `import hashlib`：内置哈希库，用来生成 chunk id。
- `from abc import ABC, abstractmethod`：导入抽象基类工具。`ABC` 让一个类"不能直接被实例化"，`abstractmethod` 标记"子类必须实现这个方法"。
- `from dataclasses import dataclass`：装饰器，自动生成 `__init__`、`__repr__` 等方法，写配置类很方便。
- `from typing import Any`：任意类型。
- `from app.models.document import Chunk, Document, Metadata`：导入项目数据模型。
  - `Chunk` 字段：`id, text, vector, metadata, score`
  - `Metadata` 字段：`source, page, sheet, tag, created_at, doc_id, chunk_index, bbox`

**`ChunkerConfig` 字段详解**：

| 字段 | 类型 | 默认值 | 含义 | 为什么这个值 |
| --- | --- | --- | --- | --- |
| `chunk_size` | `int` | `512` | 每个 chunk 的目标大小（token 或字符） | 512 token 大约 1500 字符，是中文短段落的典型长度，对 4k 上下文模型很友好 |
| `overlap` | `int` | `64` | 相邻 chunk 重叠的 token/字符数 | 64 token ≈ 一句话，重叠保证切散的语义能跨 chunk 命中 |
| `min_chunk_size` | `int` | `50` | 小于这个值的 chunk 被丢弃或合并 | 50 以下的 chunk 信息太少，存进向量库是噪音 |

#### 6.2.2 抽象基类 `Chunker`

```python
class Chunker(ABC):
    """Abstract base class for all chunkers."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialise the chunker with an optional configuration.

        Args:
            config: Chunker tuning parameters. Defaults to a fresh
                :class:`ChunkerConfig` when omitted.
        """
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a single document into chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances derived from ``doc``.
        """
        raise NotImplementedError
```

逐行解释：

- `class Chunker(ABC)`：`Chunker` 继承 `ABC`，意味着 `Chunker()` 直接实例化会报错——必须用子类。
- `def __init__(self, config: ChunkerConfig | None = None)`：
  - `config` 参数：可选，类型是 `ChunkerConfig` 或 `None`，默认 `None`。
  - `|` 是 Python 3.10+ 的"联合类型"语法，等价于 `Optional[ChunkerConfig]`。
- `self.config = config or ChunkerConfig()`：如果 `config` 是 `None`，就用默认的 `ChunkerConfig()`（即 `chunk_size=512, overlap=64, min_chunk_size=50`）。
- `@abstractmethod` 装饰 `chunk`：子类必须实现 `chunk` 方法，否则子类也不能被实例化。
- `raise NotImplementedError`：这只是占位，子类必须重写。

#### 6.2.3 批量入口 `chunk_documents`

```python
def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
    """Chunk a batch of documents with globally continuous ``chunk_index``.

    Args:
        docs: List of :class:`Document` instances to chunk.

    Returns:
        A flat list of :class:`Chunk` instances whose ``metadata.chunk_index``
        values form a continuous 0-based sequence across all input documents.
    """
    chunks: list[Chunk] = []
    global_idx = 0
    for doc in docs:
        doc_chunks = self.chunk(doc)
        for ch in doc_chunks:
            ch.metadata.chunk_index = global_idx
            global_idx += 1
        chunks.extend(doc_chunks)
    return chunks
```

**函数签名**：

- 参数：`docs: list[Document]`
- 返回值：`list[Chunk]`，所有文档的 chunk 拼成一个扁平列表。

**逐行解释**：

- `chunks: list[Chunk] = []`：用列表累积所有 chunk。
- `global_idx = 0`：**全局** chunk 索引计数器，从 0 开始。
- 外层 `for doc in docs:`：遍历每个文档。
- `doc_chunks = self.chunk(doc)`：调用子类实现的 `chunk` 方法，把这个文档切成 chunks。
- 内层 `for ch in doc_chunks:`：遍历这个文档的所有 chunk。
- `ch.metadata.chunk_index = global_idx`：**改写** chunk 自己的 `chunk_index`（子类只填了局部 idx，这里改成全局 idx，保证整个 batch 的 chunk_index 是连续的 0,1,2,...）。
- `global_idx += 1`：自增。
- `chunks.extend(doc_chunks)`：把这份 chunk 加到总列表里。

**为什么需要全局 `chunk_index`？**
打比方：你有 3 本手册，每本切成 5 段。如果只用"每本内的局部 idx"，那 3 本的 chunk 都是 0-4，下游想"取第 7 段"就找不到。改成全局 0-14 就唯一了。

#### 6.2.4 chunk id 生成 `_make_chunk_id`

```python
def _make_chunk_id(self, doc_id: str, idx: int) -> str:
    """Build a deterministic 16-char hex chunk id from a doc id and index.

    Args:
        doc_id: The parent document id.
        idx: The local chunk index within the document.

    Returns:
        The first 16 characters of the SHA-1 hex digest of
        ``"{doc_id}::chunk::{idx}"``.
    """
    raw = f"{doc_id}::chunk::{idx}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
```

**函数签名**：

- 参数：
  - `doc_id: str`：父文档 id。
  - `idx: int`：chunk 在文档内的局部索引。
- 返回值：`str`，16 个字符的十六进制字符串。

**逐行解释**：

- `raw = f"{doc_id}::chunk::{idx}".encode("utf-8")`：把字符串拼成 `xxx::chunk::0` 这种形式，再编码成 bytes（SHA-1 需要字节串）。
- `hashlib.sha1(raw).hexdigest()`：算 SHA-1 哈希，返回 40 位十六进制字符串。
- `[:16]`：截前 16 位作为 chunk id。

**为什么用 SHA-1 而不是 `f"{doc_id}_{idx}"`？**
- 想象 doc_id 是个 UUID（36 字符），如果用 `f"{doc_id}_{idx}"` 当主键，主键会很长（38 字符）。
- SHA-1 截 16 位后是定长 16 字符，更整齐；同时仍然**确定性**（同一个 doc_id+idx 总是产生同一个 chunk id），方便重复 upsert 时覆盖而不是新增。

#### 6.2.5 元数据与 chunk 构建

```python
def _build_metadata(self, doc: Document, idx: int, offset: int | None) -> Metadata:
    """Build chunk metadata inheriting document-level fields.

    Args:
        doc: The parent :class:`Document`.
        idx: Local chunk index within the document.
        offset: Optional character offset of the chunk within the document
            text; recorded in ``bbox.char_offset`` for traceability.

    Returns:
        A :class:`Metadata` instance populated with inherited fields.
    """
    bbox: dict[str, Any] | None = None
    if offset is not None:
        bbox = {"char_offset": offset}
    return Metadata(
        source=doc.metadata.source,
        page=doc.metadata.page,
        sheet=doc.metadata.sheet,
        tag=list(doc.metadata.tag),
        doc_id=doc.metadata.doc_id,
        chunk_index=idx,
        bbox=bbox,
    )

def _build_chunk(
    self, doc: Document, text: str, idx: int, offset: int | None = None
) -> Chunk:
    """Build a :class:`Chunk` with deterministic id and inherited metadata.

    Args:
        doc: The parent :class:`Document`.
        text: The chunk text.
        idx: Local chunk index within the document.
        offset: Optional character offset of the chunk within the document.

    Returns:
        A populated :class:`Chunk` instance.
    """
    return Chunk(
        id=self._make_chunk_id(doc.metadata.doc_id, idx),
        text=text,
        metadata=self._build_metadata(doc, idx, offset),
    )
```

**`_build_metadata` 解释**：

- `bbox`：可选，记录 chunk 在原文中的字符偏移（`char_offset`）。后续如果要做"高亮原文"功能，靠这个找位置。
- `Metadata(...)`：构造一个元数据对象，把文档级别的 `source / page / sheet / tag / doc_id` 继承下来，加上 chunk 自己的 `chunk_index` 和 `bbox`。

**`_build_chunk` 解释**：

- 把 `id`、`text`、`metadata` 三个字段组合成一个 `Chunk` 对象。
- 子类只需要调 `self._build_chunk(doc, text, idx, offset)` 就能产出一个完整的 chunk，不用重复写 id 生成和元数据继承逻辑。

### 6.3 fixed_token.py 逐行精读

文件路径：`app/chunkers/fixed_token.py`。

```python
"""Fixed-size token chunker using tiktoken.

Splits document text into overlapping windows measured in tokens (encoding
``cl100k_base``). Windows shorter than ``min_chunk_size`` tokens are skipped so
the final chunk is not a tiny fragment.
"""
from __future__ import annotations

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document


class FixedTokenChunker(Chunker):
    """Chunk text by token count using the ``cl100k_base`` tiktoken encoding."""

    ENCODING_NAME = "cl100k_base"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialise the chunker and load the tiktoken encoding.

        Args:
            config: Chunker tuning parameters.
        """
        super().__init__(config)
        import tiktoken

        self._enc = tiktoken.get_encoding(self.ENCODING_NAME)

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into overlapping token windows.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Empty or sub-``min_chunk_size``
            trailing windows are skipped.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        tokens = self._enc.encode(text)
        size = self.config.chunk_size
        overlap = self.config.overlap
        min_size = self.config.min_chunk_size
        step = max(1, size - overlap)

        chunks: list[Chunk] = []
        idx = 0
        i = 0
        while i < len(tokens):
            window = tokens[i : i + size]
            if len(window) < min_size:
                # Trailing fragment too small to keep; advancing cannot grow it.
                break
            chunk_text = self._enc.decode(window)
            if chunk_text.strip():
                chunks.append(self._build_chunk(doc, chunk_text, idx, offset=i))
                idx += 1
            i += step
        return chunks
```

#### 名词解释

- **tiktoken**：OpenAI 开源的 BPE 分词器库，用来把字符串切成 token id 列表。`pip install tiktoken`。
- **cl100k_base**：GPT-4 / GPT-3.5-turbo 用的 BPE 编码表，对中英文混合文本比较友好。
- **token**：模型眼里"一个语义单元"。一个汉字通常是 1-2 个 token，一个英文单词通常是 1 个 token。

#### 逐行解释

- `class FixedTokenChunker(Chunker)`：继承 `Chunker`。
- `ENCODING_NAME = "cl100k_base"`：类常量。
- `def __init__(self, config=None)`：
  - `super().__init__(config)`：先调父类初始化（设好 `self.config`）。
  - `import tiktoken`：**延迟导入**（在方法里 import 而不是文件顶部）。好处：没用到这个分块器时不强制安装 tiktoken。
  - `self._enc = tiktoken.get_encoding("cl100k_base")`：加载编码表，返回一个 `Encoding` 对象。
- `chunk` 方法：
  - `text = doc.text or ""`：取出正文，如果是 `None` 就用空串。
  - `if not text.strip(): return []`：空白或空文本直接返回空列表。
  - `tokens = self._enc.encode(text)`：把文本编码成 token id 列表，例如 `[123, 4567, 89, ...]`。
  - `size = self.config.chunk_size`：窗口大小（默认 512）。
  - `overlap = self.config.overlap`：重叠（默认 64）。
  - `min_size = self.config.min_chunk_size`：最小 chunk 大小（默认 50）。
  - `step = max(1, size - overlap)`：滑动窗口步长 = 512 - 64 = 448。
    - `max(1, ...)`：防止 overlap ≥ size 时 step 变成 0 或负数导致死循环。
  - `while i < len(tokens):`：滑动窗口主循环。
    - `window = tokens[i : i + size]`：取一个窗口的 token。
    - `if len(window) < min_size: break`：如果剩余 token 不足 50 个，直接结束（不保留尾部碎片）。
    - `chunk_text = self._enc.decode(window)`：把 token 解码回字符串。
    - `if chunk_text.strip():`：如果解码后不是纯空白，才保留。
    - `chunks.append(self._build_chunk(doc, chunk_text, idx, offset=i))`：用父类方法构建 chunk 并加入列表。`offset=i` 记录 token 偏移。
    - `idx += 1`：局部 chunk 索引 +1。
    - `i += step`：窗口向后滑动 448 个 token。

**overlap 滑动示意**（size=8, overlap=2, step=6）：

```
tokens:  [0 1 2 3 4 5 6 7 8 9 10 11 12 13 ...]
chunk0:   [0 1 2 3 4 5 6 7]
chunk1:           [6 7 8 9 10 11 12 13]
                          ↑ 重叠 6,7
```

#### 用示例文本演示

我们的示例文档（清洗后）大约 100 多个 token（中文为主），所以会被切成 1 个 chunk，正好等于全文（不超过 512 token）。`offset=0`，`idx=0`，`chunk_index=0`。

### 6.4 recursive_char.py 逐行精读（默认分块器）

文件路径：`app/chunkers/recursive_char.py`。这是项目的默认分块器。

```python
"""Recursive character chunker (LangChain-style).

Splits text by a prioritised list of separators, falling back to the next one
when a split is still larger than ``chunk_size``. The resulting pieces are
merged into chunks of approximately ``chunk_size`` characters with ``overlap``
characters of overlap between adjacent chunks.

The implementation is self-contained: it does not depend on LangChain.
"""
from __future__ import annotations

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document


class RecursiveCharChunker(Chunker):
    """Recursive character chunker with configurable separators."""

    SEPARATORS: list[str] = ["\n\n", "\n", "。", ".", " ", ""]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` recursively and merge into overlapping chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Each chunk text is at most
            ``chunk_size + overlap`` characters long.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        raw_splits = self._recursive_split(text, self.SEPARATORS)
        merged = self._merge_splits(raw_splits)
        chunks: list[Chunk] = []
        idx = 0
        offset = 0
        max_len = self.config.chunk_size + self.config.overlap
        for piece in merged:
            piece = piece.strip()
            if not piece:
                continue
            # Safety hard cut so we never exceed chunk_size + overlap.
            if len(piece) > max_len:
                piece = piece[:max_len]
            chunks.append(self._build_chunk(doc, piece, idx, offset))
            offset += len(piece)
            idx += 1
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split ``text`` using the separator priority list.

        Args:
            text: The text to split.
            separators: Ordered list of separators to try.

        Returns:
            A list of text fragments all no larger than ``chunk_size`` (the
            empty-string separator guarantees the recursion terminates by
            falling back to hard character slicing).
        """
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [text]
        if not separators:
            return [text]
        sep = separators[0]
        remaining = separators[1:]
        if sep == "":
            # Character-level fallback: hard slice into chunk_size pieces.
            size = self.config.chunk_size
            return [text[i : i + size] for i in range(0, len(text), size)]
        parts = text.split(sep)
        # Re-attach the separator to the start of every part except the first so
        # paragraph/sentence boundaries survive the merge step.
        rebuilt: list[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                rebuilt.append(part)
            else:
                rebuilt.append(sep + part)
        result: list[str] = []
        for part in rebuilt:
            if len(part) > self.config.chunk_size:
                result.extend(self._recursive_split(part, remaining))
            elif part:
                result.append(part)
        return result

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits into chunks of ~``chunk_size`` with overlap.

        Args:
            splits: Pre-split text fragments (may be smaller or larger than
                ``chunk_size``).

        Returns:
            A list of merged chunk strings.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        merged: list[str] = []
        current = ""
        for piece in splits:
            if not piece:
                continue
            candidate = current + piece if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            # Candidate overflows: flush the current buffer.
            if current:
                merged.append(current)
            if overlap > 0 and len(current) >= overlap:
                current = current[-overlap:] + piece
            else:
                current = piece
            # If the new current is still too large, hard-cut it.
            if len(current) > chunk_size:
                merged.append(current[:chunk_size])
                current = current[chunk_size:]
        if current:
            merged.append(current)
        return merged
```

#### 分隔符优先级解释

`SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]`

为什么这个顺序？

1. `"\n\n"`（段落分隔）：最优先在段落边界切，保留语义完整。
2. `"\n"`（换行）：段落不够切时，按行切。
3. `"。"`（中文句号）：行内太长时，按中文句子切。
4. `"."`（英文句号）：再按英文句子切。
5. `" "`（空格）：英文单词之间切。
6. `""`（空字符串）：最后兜底，硬切字符。

**直觉**：优先在大语义边界（段落 > 行 > 句子 > 词 > 字符）切，越往后切得越"暴力"。

#### `chunk` 方法逐行

- `text = doc.text or ""`：取正文。
- `if not text.strip(): return []`：空文本返回空列表。
- `raw_splits = self._recursive_split(text, self.SEPARATORS)`：递归切成小片段。
- `merged = self._merge_splits(raw_splits)`：把小片段合并成接近 `chunk_size` 的 chunk，并加 overlap。
- `max_len = chunk_size + overlap`：硬上限，避免 merge 后超出。
- `for piece in merged:`：
  - `piece = piece.strip()`：去掉首尾空白。
  - `if not piece: continue`：跳过空片段。
  - `if len(piece) > max_len: piece = piece[:max_len]`：保险起见，硬切。
  - `chunks.append(self._build_chunk(doc, piece, idx, offset))`：构建 chunk。
  - `offset += len(piece)`：累加字符偏移。
  - `idx += 1`。

#### `_recursive_split` 递归逻辑

- 递归出口 1：`text` 为空 → 返回 `[]`。
- 递归出口 2：`len(text) <= chunk_size` → 不用再切，直接返回 `[text]`。
- 递归出口 3：`separators` 用完 → 返回 `[text]`（其实走不到，因为最后一个是 `""`）。
- `sep = separators[0]`：取第一个分隔符。
- `remaining = separators[1:]`：剩下的分隔符列表，传给下一层递归。
- `if sep == "":` 兜底分支：
  - `[text[i : i + size] for i in range(0, len(text), size)]`：按 `chunk_size` 硬切。
- `parts = text.split(sep)`：用当前分隔符切。
- 重新拼接：除了第一段，每段前面加回 `sep`，保留分隔符信息。
- `for part in rebuilt:`：
  - 如果 `part` 还超过 `chunk_size`，用下一级分隔符递归切。
  - 否则加入 `result`。

#### `_merge_splits` 合并逻辑

- `current = ""`：当前缓冲区。
- `for piece in splits:`：
  - `candidate = current + piece`：试着把新片段拼到缓冲区。
  - `if len(candidate) <= chunk_size:`：没超 → 更新 `current`，继续。
  - 超了 → 把 `current` 推入 `merged`，然后开始新 chunk：
    - 如果 `overlap > 0` 且 `current` 够长：`current = current[-overlap:] + piece`，新 chunk 包含上一 chunk 的最后 `overlap` 个字符。
    - 否则 `current = piece`。
  - 如果新 `current` 还超 `chunk_size`：硬切。
- 收尾：`if current: merged.append(current)`。

#### 用示例文本演示

输入（清洗后）：

```text
# X-2025 智能音箱产品手册

## 1. 产品概述
X-2025 是一款支持语音交互的智能音箱,搭载四核 ARM A55 处理器,
配备 4GB 内存与 32GB 存储空间。支持 Wi-Fi 6 与蓝牙 5.0 双模连接。

## 2. 技术规格
续航时间 12 小时,充电时间 2 小时,扬声器功率 20W,重量 1.2kg。

## 3. 使用说明
首次使用时,请先充电至 80% 以上。长按电源键 3 秒开机。
```

由于全文只有约 200 字符，远小于 `chunk_size=512`，`_recursive_split` 第一次判断 `len(text) <= chunk_size` 就直接返回 `[text]`，最终只产出 **1 个 chunk**：

```python
Chunk(
    id="<sha1前16位>",
    text="# X-2025 智能音箱产品手册\n\n## 1. 产品概述\n...",
    metadata=Metadata(
        source="manual.md",
        doc_id="<doc_uuid>",
        chunk_index=0,           # 全局
        bbox={"char_offset": 0},
        ...
    )
)
```

如果你把 `chunk_size` 调小到 100，就会按 `\n\n` 切出 4 段：标题段、产品概述、技术规格、使用说明。

### 6.5 semantic.py 逐行精读

文件路径：`app/chunkers/semantic.py`。

```python
"""Semantic chunker based on sentence-embedding cosine similarity.

The document is split into sentences (Chinese- and English-aware), each
sentence is embedded with a lightweight sentence-transformers model, and
adjacent sentences whose cosine similarity drops below a threshold are placed
into separate chunks. Sentence accumulation also respects ``chunk_size``
characters so no chunk grows without bound.

The sentence-transformers model is loaded lazily on first use; if it is not
available the chunker degrades gracefully and returns the whole document as a
single chunk rather than raising.
"""
from __future__ import annotations

import re
from typing import Any

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Split after sentence-ending punctuation (Chinese and English). The regex
# keeps the terminator attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.。！？!?])\s*")


class SemanticChunker(Chunker):
    """Chunk text by semantic similarity between adjacent sentences."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.75

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Initialise the semantic chunker.

        Args:
            config: Chunker tuning parameters.
            model_name: Override for the sentence-transformers model name.
            threshold: Override for the cosine similarity split threshold.
        """
        super().__init__(config)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        # ``None`` = uninitialised, ``False`` = unavailable, an instance = ready.
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the sentence-transformers model.

        Returns:
            A ``SentenceTransformer`` instance, or ``None`` if the dependency is
            unavailable.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(
                    "sentence-transformers unavailable; semantic chunker degrading to single chunk",
                    model=self.model_name,
                    error=str(exc),
                )
                self._model = False
        return self._model if self._model is not False else None

    def _split_sentences(self, text: str) -> list[str]:
        """Split ``text`` into sentences using bilingual punctuation.

        Args:
            text: The text to split.

        Returns:
            A list of non-empty sentence strings.
        """
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into semantic chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. When the embedding model is
            unavailable the whole document is returned as a single chunk.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        model = self._ensure_model()
        if model is None:
            return [self._build_chunk(doc, text, 0, 0)]
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        embeddings = model.encode(sentences, normalize_embeddings=True)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        offset = 0
        idx = 0
        for i, sent in enumerate(sentences):
            would_exceed = (
                current
                and current_len + len(sent) > self.config.chunk_size
            )
            boundary = False
            if i + 1 < len(sentences):
                sim = float(embeddings[i] @ embeddings[i + 1])
                if sim < self.threshold and current_len >= self.config.min_chunk_size:
                    boundary = True
            if (would_exceed or boundary) and current:
                chunk_text = "".join(current)
                chunks.append(self._build_chunk(doc, chunk_text, idx, offset))
                offset += len(chunk_text)
                idx += 1
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)
        if current:
            chunk_text = "".join(current)
            chunks.append(self._build_chunk(doc, chunk_text, idx, offset))
        return chunks
```

#### 名词解释

- **sentence-transformers**：一个基于 PyTorch 的库，专门做"句子级嵌入"。`pip install sentence-transformers`。
- **all-MiniLM-L6-v2**：一个轻量级英文嵌入模型，只有 ~80MB，6 层 transformer，速度快。对中文效果一般，但作为分块的"相似度判断器"够用。
- **cosine 相似度**：两个向量的夹角余弦值，范围 [-1, 1]，越接近 1 越相似。
- **语义分块原理**：相邻两句意思相近 → 合在一起；意思跳变 → 在这里切开。

#### 逐行解释

- `_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.。！？!?])\s*")`：
  - `(?<=[.。！？!?])`：lookbehind，"前面是中英文句末标点"。
  - `\s*`：匹配 0 个或多个空白。
  - 效果：在句号后面切开，但保留句号本身在前一句。
- `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`：默认模型。
- `DEFAULT_THRESHOLD = 0.75`：相似度阈值。
  - 为什么 0.75？经验值：低于这个数意味着两句主题已经明显不同，应该切开。太高（如 0.9）会切得过细，太低（如 0.5）几乎不切。
- `__init__`：
  - `super().__init__(config)`：父类初始化。
  - `self.model_name = model_name or self.DEFAULT_MODEL`：允许传入自定义模型名。
  - `self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD`：注意 `if threshold is not None`，因为 `0.0` 也是合法阈值，不能直接 `or`。
  - `self._model: Any = None`：模型占位，懒加载。
- `_ensure_model`：懒加载逻辑。
  - `if self._model is None:`：第一次调用时进入。
  - `from sentence_transformers import SentenceTransformer`：延迟导入，避免没装库时整个文件 import 失败。
  - `self._model = SentenceTransformer(self.model_name)`：加载模型。
  - `except Exception as exc:`：捕获所有异常（网络、依赖、模型文件等）。
  - `self._model = False`：标记"加载失败"，下次不再尝试。
  - 返回 `None` 表示不可用。
- `_split_sentences`：用正则切句子。
- `chunk` 方法核心循环：
  - `sentences = self._split_sentences(text)`：分句。
  - `embeddings = model.encode(sentences, normalize_embeddings=True)`：
    - `normalize_embeddings=True`：归一化向量，让 `a @ b` 直接等于 cosine 相似度。
  - `current: list[str] = []`：当前 chunk 累积的句子。
  - `current_len = 0`：当前 chunk 字符数。
  - `for i, sent in enumerate(sentences):`：
    - `would_exceed`：如果加这句会让 chunk 超过 `chunk_size`，标记需要切。
    - `boundary`：算当前句和下一句的相似度 `sim = embeddings[i] @ embeddings[i+1]`，如果 `sim < threshold` 且当前 chunk 已经够长（≥ `min_chunk_size`），标记需要切。
    - 如果需要切：把当前累积的句子拼成 chunk，存起来，重新开始。
    - 否则：累积。

#### 用示例文本演示

分句结果：

```
1. # X-2025 智能音箱产品手册\n\n## 1. 产品概述\nX-2025 是一款支持语音交互的智能音箱,搭载四核 ARM A55 处理器,\n配备 4GB 内存与 32GB 存储空间。
2. 支持 Wi-Fi 6 与蓝牙 5.0 双模连接。
3. ## 2. 技术规格\n续航时间 12 小时,充电时间 2 小时,扬声器功率 20W,重量 1.2kg。
4. ## 3. 使用说明\n首次使用时,请先充电至 80% 以上。
5. 长按电源键 3 秒开机。
```

第 2 句和第 3 句主题从"连接方式"跳到"技术规格"，相似度可能低于 0.75 → 在这里切一刀。
最终可能产出 2-3 个语义连贯的 chunk。

### 6.6 structural.py 逐行精读

文件路径：`app/chunkers/structural.py`。

```python
"""Structural chunker that respects Markdown headings and tables.

The document is split into blocks delimited by Markdown headings
(``^#{1,6}\\s``). Consecutive Markdown table rows are kept as an atomic block
so tables are never split mid-row. Any block exceeding ``chunk_size * 2``
characters is recursively re-chunked with :class:`RecursiveCharChunker`.
"""
from __future__ import annotations

import re

from app.chunkers.base import Chunker, ChunkerConfig
from app.chunkers.recursive_char import RecursiveCharChunker
from app.models.document import Chunk, Document, Metadata

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")


class StructuralChunker(Chunker):
    """Chunk by Markdown structure (headings + atomic tables)."""

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into structural blocks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Oversized blocks are re-chunked
            with :class:`RecursiveCharChunker`.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        blocks = self._split_blocks(text)
        chunks: list[Chunk] = []
        idx = 0
        offset = 0
        max_block = self.config.chunk_size * 2
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if len(block) > max_block:
                # Recursive fallback for oversized blocks.
                sub_doc = self._make_sub_doc(doc, block)
                sub_chunks = RecursiveCharChunker(self.config).chunk(sub_doc)
                for sc in sub_chunks:
                    sc.metadata.chunk_index = idx
                    chunks.append(sc)
                    idx += 1
                offset += len(block)
            else:
                chunks.append(self._build_chunk(doc, block, idx, offset))
                offset += len(block)
                idx += 1
        return chunks

    def _split_blocks(self, text: str) -> list[str]:
        """Split ``text`` into heading-delimited blocks.

        Markdown table rows immediately following a non-table block are kept
        attached to that block; consecutive table rows form their own block.

        Args:
            text: The full document text.

        Returns:
            A list of block strings.
        """
        # Split on lines that start with a Markdown heading marker.
        positions: list[int] = [m.start() for m in _HEADING_RE.finditer(text)]
        if not positions:
            return self._split_tables(text)
        positions.append(len(text))
        blocks: list[str] = []
        for i in range(len(positions) - 1):
            segment = text[positions[i] : positions[i + 1]]
            blocks.extend(self._split_tables(segment))
        return blocks

    def _split_tables(self, segment: str) -> list[str]:
        """Separate Markdown table runs from prose within a segment.

        Args:
            segment: A text segment (typically one heading section).

        Returns:
            A list of blocks where each table run is kept intact.
        """
        lines = segment.split("\n")
        blocks: list[str] = []
        buf: list[str] = []
        in_table = False
        for line in lines:
            is_table = bool(_TABLE_ROW_RE.match(line))
            if is_table:
                if not in_table and buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                buf.append(line)
                in_table = True
            else:
                if in_table and buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                buf.append(line)
                in_table = False
        if buf:
            blocks.append("\n".join(buf).strip())
        return [b for b in blocks if b]

    def _make_sub_doc(self, doc: Document, block: str) -> Document:
        """Build a temporary Document for recursive sub-chunking.

        Args:
            doc: The parent :class:`Document`.
            block: The block text to wrap.

        Returns:
            A new :class:`Document` carrying the parent's metadata.
        """
        meta = Metadata(
            source=doc.metadata.source,
            page=doc.metadata.page,
            sheet=doc.metadata.sheet,
            tag=list(doc.metadata.tag),
            doc_id=doc.metadata.doc_id,
        )
        return Document(id=doc.id, text=block, metadata=meta)
```

#### 正则解释

- `_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)`：
  - `^`：行首。
  - `#{1,6}`：1 到 6 个 `#`（Markdown 标题级别）。
  - `\s`：一个空白字符（标题后必须跟空格）。
  - `re.MULTILINE`：让 `^` 匹配每一行的开头，而不是整个字符串的开头。
- `_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")`：
  - `^\|`：行首是 `|`（Markdown 表格行）。
  - `.*`：中间任意字符。
  - `\|\s*$`：行尾是 `|` 加可选空白。

#### `chunk` 方法逐行

- `blocks = self._split_blocks(text)`：按标题切成块。
- `max_block = self.config.chunk_size * 2`：阈值，块超过 `chunk_size` 的 2 倍才回退到递归分块。
  - 为什么 2 倍？给标题块一点宽容度，避免稍微超长就回退。
- `for block in blocks:`：
  - 如果 block 过大：用 `RecursiveCharChunker` 重新切（`_make_sub_doc` 把 block 包成临时 Document），子 chunk 的 `chunk_index` 用当前 `idx`。
  - 否则：直接作为一个 chunk。

#### `_split_blocks` 逻辑

- `positions = [m.start() for m in _HEADING_RE.finditer(text)]`：找出所有标题行的起始位置。
- `if not positions: return self._split_tables(text)`：没标题，直接按表格切。
- `positions.append(len(text))`：末尾位置作为最后一个段的终点。
- 循环：每两个相邻位置之间是一个"标题块"。
- 对每个 segment 调 `_split_tables`，把里面的表格分离出来。

#### `_split_tables` 表格整体保留

- 逐行扫描：
  - 如果是表格行：累积到 `buf`，标记 `in_table=True`。
  - 如果不是：如果之前在表格里，把表格整体作为一个 block 推出去。
- 效果：连续的表格行不会被切散。

#### 用示例文本演示

输入有 3 个 `## ` 标题，会被切成 3 个 block：

```
Block 0: "# X-2025 智能音箱产品手册\n\n" (顶部标题段)
Block 1: "## 1. 产品概述\nX-2025 是一款..."
Block 2: "## 2. 技术规格\n续航时间 12 小时..."
Block 3: "## 3. 使用说明\n首次使用时..."
```

每个 block 字符数远小于 `max_block=1024`，所以每个 block 直接成为一个 chunk，结构清晰，非常适合手册类文档。

### 6.7 factory.py 逐行精读

文件路径：`app/chunkers/factory.py`。

```python
"""Chunker factory dispatching on ``settings.chunker_type``."""
from __future__ import annotations

from typing import Any

from app.chunkers.base import Chunker, ChunkerConfig
from app.chunkers.fixed_token import FixedTokenChunker
from app.chunkers.recursive_char import RecursiveCharChunker
from app.chunkers.semantic import SemanticChunker
from app.chunkers.structural import StructuralChunker


def get_chunker(settings: Any) -> Chunker:
    """Build a :class:`Chunker` from a Settings-like object.

    Args:
        settings: An object exposing ``chunker_type``, ``chunk_size`` and
            ``chunk_overlap`` attributes (e.g. the project :class:`Settings` or
            a ``MagicMock`` in tests).

    Returns:
        A concrete :class:`Chunker` instance matching ``settings.chunker_type``.
        Unknown / missing types fall back to :class:`RecursiveCharChunker`.
    """
    config = ChunkerConfig(
        chunk_size=getattr(settings, "chunk_size", 512),
        overlap=getattr(settings, "chunk_overlap", 64),
    )
    ctype = getattr(settings, "chunker_type", "recursive")
    if ctype == "fixed":
        return FixedTokenChunker(config)
    if ctype == "semantic":
        return SemanticChunker(config)
    if ctype == "structural":
        return StructuralChunker(config)
    # Default to recursive for "recursive" and any unknown value.
    return RecursiveCharChunker(config)
```

#### 逐行解释

- `from typing import Any`：用 `Any` 是为了让函数能接受任何 settings 对象（不强制依赖具体的 `Settings` 类），方便测试时传 `MagicMock`。
- `get_chunker(settings)`：
  - `getattr(settings, "chunk_size", 512)`：从 settings 读 `chunk_size`，读不到就用默认 512。
  - `getattr(settings, "chunk_overlap", 64)`：读 overlap。
  - `getattr(settings, "chunker_type", "recursive")`：读分块器类型，默认 recursive。
  - 根据 `ctype` 字符串分发到具体实现。
  - 未知值兜底 → `RecursiveCharChunker`。

#### 四种分块器什么时候选哪种

| 场景 | 推荐分块器 | 理由 |
| --- | --- | --- |
| 通用文档（混合中英文、长短不一） | `recursive`（默认） | 兼顾语义边界和长度控制，无外部依赖 |
| 严格的 token 预算（如 OpenAI 4k 上下文） | `fixed` | 精确控制 token 数，避免超限 |
| 长篇连贯叙述（小说、论文） | `semantic` | 按语义切分，chunk 内主题集中 |
| Markdown 手册、技术文档（带标题和表格） | `structural` | 尊重文档结构，表格不破 |

### 6.8 四种分块器对比表

| 分块器 | 原理 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| `FixedTokenChunker` | 按 token 数滑动窗口 | 严格 token 预算 | 长度精确、可预测 | 可能在词/句中间切断，语义不完整 |
| `RecursiveCharChunker` | 按分隔符优先级递归切 | 通用混合文本 | 无依赖、保留语义边界 | 字符数 ≠ token 数，模型侧可能超限 |
| `SemanticChunker` | 句向量相似度切分 | 长篇叙述 | chunk 主题集中 | 依赖模型、速度慢、对中文一般 |
| `StructuralChunker` | Markdown 标题+表格 | 手册/技术文档 | 结构清晰、表格完整 | 仅对 Markdown 有效，纯文本无结构可依 |

---

## 第 7 章 · 嵌入——把文本变成数字向量

### 7.1 什么是嵌入（用大白话讲）

电脑不懂"语义"，它只懂数字。所以我们要把一段文本（比如 "续航时间 12 小时"）变成一串数字，比如：

```
[0.12, -0.34, 0.56, 0.78, -0.91, ..., 0.45]   # 一共 1024 个数字
```

这串数字就叫**向量**（vector），它代表了文本的"意思"。

**打比方**：就像给每个人发身份证号，号码本身没意义，但号码相近的人来自同一个地区。
向量也是这样：意思相近的文本，向量也相近。

**为什么有用？**
意思相近的文本，向量也相近，可以用数学方法比较。最常用的是 **cosine 相似度**：

```
cosine(A, B) = (A · B) / (|A| × |B|)
```

- `A · B`：两个向量的点积（对应位置相乘再求和）。
- `|A|`、`|B|`：向量的长度（平方和开根号）。
- 结果范围 [-1, 1]，越接近 1 越相似。

**手算例子**：
- A = [1, 0, 0]
- B = [0.9, 0.1, 0]
- 点积 = 1×0.9 + 0×0.1 + 0×0 = 0.9
- |A| = 1, |B| = √(0.81+0.01) ≈ 0.905
- cosine ≈ 0.9 / 0.905 ≈ 0.994 → 非常相似

如果向量被**归一化**（长度=1），那 `|A|×|B|` 就是 1，cosine 相似度就等于点积 `A·B`，计算更快。这就是项目里 `normalize_embeddings=True` 的原因。

### 7.2 base.py 逐行精读

文件路径：`app/embedders/base.py`。

```python
"""Abstract base class for embedding backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract embedding interface used across the kb-rag pipeline.

    Concrete implementations convert text into dense vector representations
    that can be persisted in a :class:`~app.stores.base.VectorStore` and used
    for similarity search during retrieval.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the vectors produced by this embedder."""
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text documents.

        Args:
            texts: List of raw text strings to embed.

        Returns:
            A list of embedding vectors, one per input text, ordered the same
            way as the input. Each vector has length :attr:`dim`.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Raw query text.

        Returns:
            A single embedding vector of length :attr:`dim`.
        """
        raise NotImplementedError
```

#### 解释

- `class Embedder(ABC)`：抽象基类，不能直接实例化。
- `@property @abstractmethod def dim(self) -> int:`：
  - `@property`：把方法变成属性，访问时 `embedder.dim` 不加括号。
  - 返回向量维度（如 1024）。
- `embed_texts(texts: list[str]) -> list[list[float]]`：
  - **批量**嵌入。
  - 输入：字符串列表，例如 `["续航时间 12 小时", "充电时间 2 小时"]`。
  - 输出：向量的列表，例如 `[[0.12, ...], [0.34, ...]]`。
  - **为什么批量？** 性能。模型一次处理多条比循环调用快得多（GPU 并行、API 也支持批量）。
- `embed_query(text: str) -> list[float]`：
  - **单条**查询嵌入。
  - 输入：一个字符串。
  - 输出：一个向量。
  - **为什么单独一个方法？** 有些模型对"查询"和"文档"用不同前缀（比如 BGE-large-zh 查询要加 "为这个句子生成表示以用于检索相关文章："）。即使 bge-m3 不需要前缀，保留这个方法让接口清晰、未来可扩展。

### 7.3 local_embedder.py 逐行精读

文件路径：`app/embedders/local_embedder.py`。

```python
"""Local sentence-transformers embedder (in-process)."""
from __future__ import annotations

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


class LocalEmbedder(Embedder):
    """Embedder backed by ``sentence-transformers.SentenceTransformer``.

    Suitable for self-hosted deployments where the model weights can be loaded
    into the process. Defaults to ``BAAI/bge-m3``.

    Note on bge-m3:
        Unlike some other BGE checkpoints (e.g. ``bge-large-zh``), ``bge-m3``
        does **not** require query/document instruction prefixes. Both indexed
        documents and user queries are encoded directly without any prefix
        transformation.

    Args:
        model_name: HuggingFace model id. Defaults to ``"BAAI/bge-m3"``.
        dim: Vector dimensionality (used for the abstract :attr:`dim` property
            and to cross-check the loaded model when possible).
        batch_size: Default mini-batch size passed to ``SentenceTransformer.encode``.
        device: Torch device, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dim: int = 1024,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        """Initialize and eagerly load the SentenceTransformer model."""
        self._model_name = model_name
        self._dim = int(dim)
        self._batch_size = int(batch_size)
        self._device = device
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001 - re-raise after logging
            logger.error("failed to load local embedder model", model=model_name, error=str(exc))
            raise

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents using L2-normalized vectors.

        Args:
            texts: List of raw text strings.

        Returns:
            List of normalized embedding vectors (one per input text).
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        bge-m3 requires no query prefix, so the query is encoded verbatim,
        identical to how documents are encoded.
        """
        return self.embed_texts([text])[0]
```

#### 名词解释

- **sentence-transformers**：基于 PyTorch 的句子嵌入库，封装了多种预训练模型。`pip install sentence-transformers`。
- **BAAI/bge-m3**：智源研究院（BAAI）开源的多语言嵌入模型。
  - **bge**：BAAI General Embedding。
  - **m3**：Multi-linguality, Multi-granularity, Multi-Functionality。
  - 支持 100+ 语言（中英文效果都很好），输出 1024 维向量。
  - 不需要查询/文档前缀，调用简单。
- **normalize_embeddings**：把向量除以它的长度，让长度=1。归一化后点积=cosine 相似度。

#### 逐行解释

- `__init__` 参数详解：
  - `model_name: str = "BAAI/bge-m3"`：默认用 bge-m3。为什么？多语言、效果强、1024 维兼顾性能和精度。
  - `dim: int = 1024`：bge-m3 输出维度。
  - `batch_size: int = 32`：每次 encode 32 条。为什么 32？CPU 上太大（如 256）会爆内存，太小（如 1）没并行优势。32 是经验平衡点。
  - `device: str = "cpu"`：默认 CPU。为什么？生产环境不一定有 GPU，CPU 可用性更高。有 GPU 的用户可以传 `"cuda"`。
- `self._model = SentenceTransformer(model_name, device=device)`：**立即加载**模型（不像 SemanticChunker 懒加载）。为什么这里要 eager？因为 LocalEmbedder 是核心组件，启动时加载失败应该立刻报错，而不是检索时才发现。
- `try/except`：捕获加载异常，记日志后**重新抛出**（`raise`），不让程序带病运行。
- `dim` 属性：返回 `self._dim`。
- `embed_texts`：
  - `if not texts: return []`：空输入返回空列表。
  - `self._model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)`：
    - `batch_size`：内部 mini-batch 大小。
    - `normalize_embeddings=True`：L2 归一化。
    - `convert_to_numpy=True`：返回 numpy 数组（比 torch tensor 省内存、易序列化）。
  - `[list(map(float, vec)) for vec in embeddings]`：把 numpy 数组转成 Python 原生 float 列表，方便 JSON 序列化存库。
- `embed_query`：复用 `embed_texts`，取第一个结果。`bge-m3` 不需要查询前缀，所以查询和文档走同一条路径。

### 7.4 ollama_embedder.py 逐行精读

文件路径：`app/embedders/ollama_embedder.py`。

```python
"""Ollama-hosted embedder using the ``/api/embeddings`` REST endpoint."""
from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


class OllamaEmbedder(Embedder):
    """Embedder that calls an Ollama server's ``/api/embeddings`` endpoint.

    Args:
        base_url: Base URL of the Ollama service, e.g. ``"http://ollama:11434"``.
        model: Ollama model name, defaults to ``"bge-m3"``.
        dim: Vector dimensionality returned by the model.
        timeout: HTTP request timeout in seconds.
        batch_size: Maximum number of concurrent in-flight requests when
            embedding multiple texts.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        dim: int = 1024,
        timeout: float = 30.0,
        batch_size: int = 8,
    ) -> None:
        """Initialize the HTTP client and configuration."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = int(dim)
        self._timeout = float(timeout)
        self._batch_size = int(batch_size)
        self._client = httpx.Client(timeout=self._timeout)

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> list[float]:
        """Call the Ollama embeddings endpoint for a single text."""
        url = f"{self._base_url}/api/embeddings"
        resp = self._client.post(url, json={"model": self._model, "prompt": prompt})
        resp.raise_for_status()
        payload = resp.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError("ollama embeddings response missing 'embedding' field")
        return [float(x) for x in embedding]

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed a batch of documents sequentially (with retry per call).

        Args:
            texts: List of raw text strings.
            batch_size: Optional override for the configured batch size. Kept
                for API compatibility; concurrent embedding is not used to
                avoid overwhelming the Ollama server.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        for text in texts:
            results.append(self._call_api(text))
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api(text)
```

#### 名词解释

- **Ollama**：本地运行的 LLM 服务（类似一个本地 OpenAI）。`ollama serve` 启动后监听 11434 端口，可以用 REST API 调用各种模型，包括嵌入模型。优点：数据不外传，适合企业内网部署。
- **httpx**：Python 的 HTTP 客户端库，支持同步和异步，API 类似 `requests` 但更现代。`pip install httpx`。
- **tenacity**：重试库，提供装饰器风格的_retry_逻辑。`pip install tenacity`。

#### 逐行解释

- `__init__` 参数：
  - `base_url: str`：Ollama 服务地址，例如 `"http://ollama:11434"`。
  - `model: str = "bge-m3"`：模型名，需要先 `ollama pull bge-m3`。
  - `dim: int = 1024`：维度。
  - `timeout: float = 30.0`：HTTP 超时 30 秒。
  - `batch_size: int = 8`：保留参数，但代码里其实没并发用（顺序调用）。
- `self._base_url = base_url.rstrip("/")`：去掉末尾 `/`，避免拼 URL 时出现 `//api`。
- `self._client = httpx.Client(timeout=self._timeout)`：创建一个**长连接** HTTP 客户端（连接池复用，比每次新建连接快）。
- `_call_api` 方法的 `@retry` 装饰器：
  - `retry_if_exception_type(httpx.HTTPError)`：只在 HTTP 错误时重试（网络抖动、5xx）。
  - `stop_after_attempt(3)`：最多重试 3 次。
  - `wait_exponential(multiplier=1, min=1, max=10)`：指数退避，第 1 次等 1s，第 2 次等 2s，第 3 次等 4s（封顶 10s）。
  - `reraise=True`：重试用完后重新抛出原异常。
- `_call_api` 方法体：
  - `url = f"{self._base_url}/api/embeddings"`：拼接口 URL。
  - `resp = self._client.post(url, json={"model": ..., "prompt": ...})`：POST 请求，body 是 JSON。
  - `resp.raise_for_status()`：4xx/5xx 直接抛异常。
  - `payload = resp.json()`：解析响应 JSON。
  - `embedding = payload.get("embedding")`：取出 embedding 字段。
  - `if not embedding: raise RuntimeError(...)`：响应里没有就报错。
  - `return [float(x) for x in embedding]`：转成 float 列表。
- `embed_texts`：循环调用 `_call_api`，顺序处理（不并发，避免压垮 Ollama 服务）。
- `embed_query`：单条调用。

#### 返回值解析

Ollama `/api/embeddings` 响应格式：

```json
{
  "embedding": [0.0123, -0.0456, ..., 0.0789]
}
```

代码把它转成 `list[float]` 返回。

### 7.5 api_embedder.py 逐行精读

文件路径：`app/embedders/api_embedder.py`。

```python
"""OpenAI-protocol API embedder (OpenAI / Zhipu BigModel)."""
from __future__ import annotations

from typing import Literal

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)

Provider = Literal["openai", "zhipu"]

_DEFAULTS: dict[str, dict[str, object]] = {
    "openai": {"model": "text-embedding-3-small", "dim": 1536, "base_url": "https://api.openai.com/v1"},
    "zhipu": {"model": "embedding-3", "dim": 1024, "base_url": "https://open.bigmodel.cn/api/paas/v4"},
}


class ApiEmbedder(Embedder):
    """Embedder that talks to an OpenAI-compatible embeddings endpoint.

    Both OpenAI and Zhipu BigModel expose the same ``/embeddings`` REST shape,
    so a single :class:`openai.OpenAI` client is reused; only the ``base_url``,
    ``api_key`` and default model differ.

    Args:
        provider: ``"openai"`` or ``"zhipu"``.
        api_key: API key for the chosen provider.
        base_url: Base URL override. Falls back to the provider default.
        model: Model name override. Falls back to the provider default.
        dim: Vector dimensionality override. Falls back to the provider default.
    """

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        """Initialize the OpenAI client for the chosen provider."""
        if provider not in _DEFAULTS:
            raise ValueError(f"unsupported provider: {provider!r}; expected one of {list(_DEFAULTS)}")
        defaults = _DEFAULTS[provider]
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url or str(defaults["base_url"])
        self._model = model or str(defaults["model"])
        self._dim = int(dim if dim is not None else defaults["dim"])  # type: ignore[arg-type]
        if not api_key:
            raise ValueError(f"api_key is required for provider {provider!r}")
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings endpoint with retry."""
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [[float(x) for x in item.embedding] for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents via the API.

        Args:
            texts: List of raw text strings.

        Returns:
            List of embedding vectors, ordered the same as the input.
        """
        if not texts:
            return []
        return self._create_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._create_embeddings([text])[0]
```

#### 为什么智谱和 OpenAI 共用同一个类

智谱 BigModel 的 `/embeddings` 接口**完全兼容 OpenAI 协议**（请求参数 `input` + `model`，响应 `data[i].embedding`），所以一个 `OpenAI` client 切换 `base_url` 和 `api_key` 就能复用。这种"协议兼容"在国内 API 厂商里很常见，方便用户切换。

#### 名词解释

- **openai** 库：OpenAI 官方 Python SDK。`pip install openai`。它不仅能调 OpenAI，也能调任何兼容 OpenAI 协议的服务（如智谱、DeepSeek、Moonshot 等）。
- **embedding-3**：智谱的嵌入模型，1024 维，对中文友好。
- **text-embedding-3-small**：OpenAI 的小型嵌入模型，1536 维，性价比高。

#### 逐行解释

- `Provider = Literal["openai", "zhipu"]`：用 `Literal` 限制 `provider` 只能是这两个字符串之一，IDE 会提示。
- `_DEFAULTS`：字典，存两个厂商的默认配置。
  - OpenAI：模型 `text-embedding-3-small`，1536 维。
  - 智谱：模型 `embedding-3`，1024 维。
- `__init__`：
  - `if provider not in _DEFAULTS: raise ValueError(...)`：校验 provider。
  - `defaults = _DEFAULTS[provider]`：取出该厂商默认配置。
  - `self._base_url = base_url or str(defaults["base_url"])`：允许用户覆盖 base_url（比如用代理）。
  - `self._model = model or str(defaults["model"])`：允许覆盖模型名。
  - `self._dim = int(dim if dim is not None else defaults["dim"])`：允许覆盖维度。
  - `if not api_key: raise ValueError(...)`：没 API key 直接报错。
  - `self._client = OpenAI(api_key=api_key, base_url=self._base_url)`：创建 OpenAI client，指向选定的 base_url。
- `_create_embeddings` 方法的 `@retry`：
  - `retry_if_exception_type(Exception)`：捕获所有异常（比 Ollama 版本宽，因为 API 调用可能遇到各种问题：限流、网络、5xx）。
  - 3 次重试，指数退避 1-10s。
- `_create_embeddings` 方法体：
  - `response = self._client.embeddings.create(input=texts, model=self._model)`：调 API，`input` 是文本列表。
  - `return [[float(x) for x in item.embedding] for item in response.data]`：从响应里取出每个 embedding，转成 float 列表。
- `embed_texts` / `embed_query`：复用 `_create_embeddings`。

#### 返回值解析

OpenAI 协议响应格式：

```json
{
  "data": [
    {"embedding": [0.012, -0.034, ...], "index": 0},
    {"embedding": [0.056, -0.078, ...], "index": 1}
  ]
}
```

代码把 `data[i].embedding` 提取出来，按顺序返回。

### 7.6 factory.py 逐行精读

文件路径：`app/embedders/factory.py`。

```python
"""Factory for selecting an embedder backend based on settings."""
from __future__ import annotations

from app.embedders.api_embedder import ApiEmbedder
from app.embedders.base import Embedder
from app.embedders.local_embedder import LocalEmbedder
from app.embedders.ollama_embedder import OllamaEmbedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


def get_embedder(settings) -> Embedder:
    """Build an :class:`Embedder` from the supplied :class:`Settings`.

    Dispatch rules:
        - ``"local"``: :class:`LocalEmbedder` using ``embedder_model`` /
          ``embedder_dim``.
        - ``"ollama"``: :class:`OllamaEmbedder` pointed at ``ollama_base_url``.
        - ``"api"``: :class:`ApiEmbedder`. Zhipu is preferred when
          ``zhipu_api_key`` is set; otherwise OpenAI is used with
          ``openai_api_key``.
        - Any other value raises :class:`ValueError`.

    Args:
        settings: Application :class:`Settings` instance.

    Returns:
        A concrete :class:`Embedder` instance.

    Raises:
        ValueError: If ``settings.embedder_provider`` is unknown or the
            selected API provider has no API key configured.
    """
    provider = settings.embedder_provider

    if provider == "local":
        logger.info("using local embedder", model=settings.embedder_model, dim=settings.embedder_dim)
        return LocalEmbedder(
            model_name=settings.embedder_model,
            dim=settings.embedder_dim,
        )

    if provider == "ollama":
        logger.info("using ollama embedder", base_url=settings.ollama_base_url, dim=settings.embedder_dim)
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            dim=settings.embedder_dim,
        )

    if provider == "api":
        if settings.zhipu_api_key:
            logger.info("using zhipu api embedder", dim=settings.embedder_dim)
            return ApiEmbedder(
                provider="zhipu",
                api_key=settings.zhipu_api_key,
                base_url=settings.zhipu_base_url,
                dim=settings.embedder_dim,
            )
        if settings.openai_api_key:
            logger.info("using openai api embedder", dim=settings.embedder_dim)
            return ApiEmbedder(
                provider="openai",
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                dim=settings.embedder_dim,
            )
        raise ValueError("api embedder selected but neither zhipu_api_key nor openai_api_key is set")

    raise ValueError(f"unknown embedder_provider: {provider!r}")
```

#### 逐行解释

- `provider = settings.embedder_provider`：从配置读 `embedder_provider` 字段（值为 `"local"` / `"ollama"` / `"api"`）。
- 三个 `if` 分支：
  - `"local"` → `LocalEmbedder`：用 `embedder_model` 和 `embedder_dim` 配置。
  - `"ollama"` → `OllamaEmbedder`：用 `ollama_base_url`。
  - `"api"` → 优先用智谱（`zhipu_api_key` 优先），其次 OpenAI（`openai_api_key`）。两者都没配置就报错。
- 其他值 → `ValueError`。

### 7.7 嵌入结果示例

假设我们用 `LocalEmbedder`（bge-m3，1024 维），把第一个 chunk（示例手册全文，约 200 字符）送进去：

```python
embedder = LocalEmbedder(model_name="BAAI/bge-m3", dim=1024)
vec = embedder.embed_query("续航时间 12 小时,充电时间 2 小时")
print(vec[:10])     # 只看前 10 个数字
print(len(vec))     # 1024
```

输出（示意，真实数值会不同）：

```text
[0.0234, -0.0178, 0.0412, -0.0089, 0.0334, -0.0256, 0.0198, -0.0421, 0.0112, -0.0067, ...]
1024
```

**这些数字代表什么？**

- 每个数字是模型从文本里提取的某个"语义特征"的强度。
- 第 1 个数 `0.0234` 可能代表"和时间有关"的程度（正数=相关）。
- 第 2 个数 `-0.0178` 可能代表"和地点有关"的程度（负数=不相关）。
- 我们不需要知道每个维度具体含义（这叫"不可解释性"），只要知道：**意思相近的文本，这串数字也相近**。

比如：

```python
v1 = embedder.embed_query("续航时间 12 小时")
v2 = embedder.embed_query("电池能用多久")
v3 = embedder.embed_query("今天天气不错")

# 归一化后点积 = cosine 相似度
sim12 = sum(a*b for a,b in zip(v1, v2))   # ≈ 0.85（意思相近）
sim13 = sum(a*b for a,b in zip(v1, v3))   # ≈ 0.20（意思无关）
```

这就是 RAG 检索的基础：用户问"电池能用多久"，能匹配到文档里的"续航时间 12 小时"。

---

## 第 8 章 · 向量存储——把向量存起来供检索

### 8.1 为什么不能用普通数据库存向量

**普通数据库**（MySQL、PostgreSQL 传统用法）只能**精确匹配**：

```sql
SELECT * FROM products WHERE name = '续航时间';
```

如果用户问"电池能用多久"，SQL 里 `name='电池能用多久'` 一条都查不到——因为表里存的是"续航时间"。

**向量数据库**能**模糊匹配**（语义相似）：

```python
store.search(query_vector=embed("电池能用多久"), top_n=5)
# 返回最相似的 5 个 chunk，即使字面完全不同
```

**打比方**：
- 普通数据库 = 精确查字典找"续航"；
- 向量库 = 描述"电池能用多久"也能找到"续航"相关内容。

#### HNSW 算法直觉解释

HNSW = **Hierarchical Navigable Small World**（分层可导航小世界图）。

想象你要在一栋 100 层的大楼里找一个人：
- 第 100 层：每层只有 10 个房间，但每个房间的人认识其他楼层的很多人（"快车"，跳跃大）。
- 第 50 层：每层 100 个房间，认识的人少一些。
- 第 1 层：每层 10000 个房间，认识的人最少（"慢车"，精确）。

找人的流程：
1. 先在第 100 层快速跳到大致区域；
2. 下到第 50 层，继续缩小范围；
3. 最后到第 1 层，精确找到目标。

HNSW 就是把向量建成这种"分层图"，检索时先走大步再走小步，速度比线性扫描快几个数量级。

### 8.2 base.py 逐行精读

文件路径：`app/stores/base.py`。

```python
"""Abstract base class for vector store backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.document import Chunk


class VectorStore(ABC):
    """Abstract vector store interface for the kb-rag pipeline.

    A vector store persists chunk embeddings together with their metadata and
    supports similarity search, deletion by document or chunk id, and basic
    counting.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks and their corresponding vectors.

        Args:
            chunks: List of :class:`~app.models.document.Chunk` objects. The
                chunk ``id``, ``text`` and ``metadata`` are persisted.
            vectors: Parallel list of embedding vectors. ``vectors[i]``
                corresponds to ``chunks[i]``.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Return the ``top_n`` most similar chunks for ``query_vector``.

        Args:
            query_vector: Embedding of the query.
            top_n: Maximum number of results to return.
            filters: Optional filter dictionary. Recognized keys:
                ``source`` (list[str] | str), ``tag`` (list[str] | str),
                ``doc_id`` (str), ``time_range`` (dict with ``gte`` / ``lte``
                ISO-8601 timestamps).

        Returns:
            List of :class:`~app.models.document.Chunk` objects with ``score``
            populated, sorted by descending similarity.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all chunks belonging to ``doc_id``."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete the chunks identified by ``chunk_ids``."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors stored in the collection."""
        raise NotImplementedError
```

#### 四个核心方法解释

1. **`upsert(chunks, vectors)`**：写入/更新。
   - 参数：
     - `chunks: list[Chunk]`：要写入的 chunk 列表。
     - `vectors: list[list[float]]`：对应的向量列表，`vectors[i]` 对应 `chunks[i]`。
   - 返回：`None`。
   - 行为：如果 chunk id 已存在则覆盖，不存在则新增（"upsert" = update + insert）。
2. **`search(query_vector, top_n=20, filters=None)`**：搜索。
   - 参数：
     - `query_vector: list[float]`：查询向量。
     - `top_n: int = 20`：返回前 20 条最相似的。
     - `filters: dict | None = None`：可选过滤条件，支持：
       - `source`：按来源过滤（字符串或列表）。
       - `tag`：按标签过滤。
       - `doc_id`：按文档 id 过滤。
       - `time_range`：按时间范围过滤（`{"gte": "...", "lte": "..."}`，ISO-8601 格式）。
   - 返回：`list[Chunk]`，按相似度从高到低排序，`chunk.score` 字段填充相似度。
3. **`delete_by_doc(doc_id)`**：按文档删除。
   - 参数：`doc_id: str`。
   - 返回：`None`。
   - 行为：删除该文档下的所有 chunk（用于文档更新或下架）。
4. **`delete_by_chunk_ids(chunk_ids)`**：按 chunk id 删除。
5. **`count()`**：返回当前集合里总共有多少条向量。

### 8.3 qdrant_store.py 逐行精读

文件路径：`app/stores/qdrant_store.py`。这是默认实现。

#### 8.3.1 import 与辅助函数

```python
"""Qdrant-backed vector store implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_OID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.models.document import Chunk, Metadata
from app.observability.logging import get_logger
from app.stores.base import VectorStore

logger = get_logger(__name__)


def _to_unix_ts(value: object) -> float:
    """Convert an ISO-8601 string or numeric value to a unix timestamp (float).

    Qdrant's :class:`Range` filter only accepts numeric values, so datetime
    filtering is performed against a numeric ``created_at`` payload field.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"cannot parse ISO timestamp: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    raise TypeError(f"unsupported time_range value type: {type(value).__name__}")
```

#### 名词解释

- **qdrant-client**：Qdrant 官方 Python SDK。`pip install qdrant-client`。
- **Qdrant**：开源向量数据库，Rust 写的，性能好、支持过滤、支持 HNSW 索引。可以用 Docker 一键启动。
- **uuid5(namespace, name)**：基于命名空间和名字生成的确定性 UUID。同样的输入永远生成同样的 UUID。

#### 逐行解释

- `from uuid import NAMESPACE_OID, uuid5`：导入 UUID 工具。
- `from qdrant_client import QdrantClient`：Qdrant 客户端。
- `from qdrant_client.http.models import ...`：导入 Qdrant 的数据模型类。
  - `Distance`：距离度量（Cosine、Euclid、Dot）。
  - `Filter`、`FieldCondition`：过滤条件。
  - `MatchAny`、`MatchValue`、`Range`：匹配方式。
  - `HnswConfigDiff`：HNSW 配置。
  - `PointStruct`：一个数据点（id + vector + payload）。
  - `VectorParams`：向量参数（维度 + 距离）。
- `_to_unix_ts(value)`：
  - 把 ISO-8601 字符串或数字转成 unix 时间戳。
  - 为什么需要？Qdrant 的 `Range` 过滤只接受数字，所以时间字段必须存成数字（unix 时间戳）才能范围过滤。
  - `dt.replace(tzinfo=timezone.utc)`：如果时间没带时区，默认当成 UTC。

#### 8.3.2 `__init__` 和 `_ensure_collection`

```python
class QdrantStore(VectorStore):
    """Vector store backed by `Qdrant <https://qdrant.tech>`_.

    Args:
        url: Qdrant REST endpoint, e.g. ``"http://qdrant:6333"``.
        collection_name: Name of the Qdrant collection to use.
        dim: Vector dimensionality. Used only when the collection is first
            created; existing collections keep their original dimensionality.
    """

    def __init__(self, url: str, collection_name: str, dim: int) -> None:
        """Initialize the client and ensure the collection exists."""
        self._url = url
        self._collection_name = collection_name
        self._dim = int(dim)
        self._client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection with HNSW + Cosine config if it is missing."""
        try:
            existing = self._client.get_collection(self._collection_name)
            # Collection exists; trust its configuration.
            logger.debug("qdrant collection exists", collection=self._collection_name, points=existing.points_count)
            return
        except Exception:
            # Collection does not exist (or other transient error); create it.
            logger.info("creating qdrant collection", collection=self._collection_name, dim=self._dim)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
```

#### 逐行解释

- `__init__` 参数：
  - `url: str`：Qdrant 服务地址，如 `"http://qdrant:6333"`。
  - `collection_name: str`：集合名（类似数据库的表名）。
  - `dim: int`：向量维度。
- `self._client = QdrantClient(url=url)`：创建客户端连接。
- `self._ensure_collection()`：确保集合存在。
- `_ensure_collection`：
  - `try: self._client.get_collection(...)`：先尝试获取集合信息。
  - 如果存在：记日志，直接返回（信任已有配置）。
  - 如果不存在（抛异常）：进入 `except`，记日志，然后创建集合。
  - `create_collection` 参数：
    - `vectors_config=VectorParams(size=1024, distance=Distance.COSINE)`：
      - `size`：向量维度。
      - `distance=Distance.COSINE`：用 cosine 相似度（适合归一化向量）。
    - `hnsw_config=HnswConfigDiff(m=16, ef_construct=100)`：
      - `m=16`：每个节点在图里的最大连接数。16 是默认值，平衡内存和召回率。越大召回越高但内存越大。
      - `ef_construct=100`：建索引时的探索深度。100 是默认值，越大建索引越慢但召回越好。

#### 8.3.3 uuid5 转换

```python
@staticmethod
def _point_id(chunk_id: str) -> str:
    """Convert a chunk id into a stable UUID string for Qdrant point storage."""
    return str(uuid5(NAMESPACE_OID, chunk_id))
```

**为什么用 uuid5？**
- Qdrant 的 point id 必须是 UUID 或整数。
- chunk id 是 16 位 hex（如 `"a3f2b1c9d8e7f6a5"`），不是 UUID 格式。
- `uuid5(NAMESPACE_OID, chunk_id)`：基于 `chunk_id` 生成确定性 UUID。
  - **确定性**：同一个 `chunk_id` 永远生成同一个 UUID。
  - **好处**：重复 upsert 时覆盖原点，而不是新增重复点。

#### 8.3.4 `upsert` 方法

```python
def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Upsert chunks and their vectors into the Qdrant collection."""
    if not chunks:
        return
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

    points: list[PointStruct] = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        meta = chunk.metadata
        payload = {
            "chunk_id": chunk.id,
            "source": meta.source,
            "page": meta.page,
            "sheet": meta.sheet,
            "tag": list(meta.tag) if meta.tag else [],
            # Store as numeric unix timestamp so Qdrant Range filters work.
            "created_at": meta.created_at.timestamp(),
            "created_at_iso": meta.created_at.isoformat(),
            "doc_id": meta.doc_id,
            "chunk_index": meta.chunk_index,
            "text": chunk.text,
        }
        if meta.bbox is not None:
            payload["bbox"] = meta.bbox
        points.append(
            PointStruct(id=self._point_id(chunk.id), vector=list(vec), payload=payload)
        )

    self._client.upsert(collection_name=self._collection_name, points=points)
    logger.debug("upserted points", count=len(points), collection=self._collection_name)
```

#### 逐行解释

- `if not chunks: return`：空列表直接返回。
- `if len(chunks) != len(vectors): raise ValueError(...)`：长度不一致报错（防止错位）。
- `for chunk, vec in zip(chunks, vectors, strict=True):`：`strict=True` 在 Python 3.10+ 保证长度一致（额外保险）。
- `payload` 字典：存到 Qdrant 的元数据，包括：
  - `chunk_id`：原始 chunk id（16 位 hex）。
  - `source`、`page`、`sheet`、`tag`：来源信息。
  - `created_at`：unix 时间戳（用于 Range 过滤）。
  - `created_at_iso`：ISO 字符串（人类可读）。
  - `doc_id`、`chunk_index`：定位信息。
  - `text`：原文（检索时直接返回，不用回查原库）。
  - `bbox`：可选，空间位置信息。
- `PointStruct(id=..., vector=..., payload=...)`：构造一个 Qdrant 点。
- `self._client.upsert(collection_name=..., points=points)`：批量写入。

#### 8.3.5 Filter 构建

```python
@staticmethod
def _build_filter(filters: dict | None) -> Filter | None:
    """Translate a filter dict into a Qdrant :class:`Filter` object."""
    if not filters:
        return None

    must: list[FieldCondition] = []

    sources = filters.get("source")
    if sources is not None:
        values = sources if isinstance(sources, list) else [sources]
        must.append(FieldCondition(key="source", match=MatchAny(any=[str(s) for s in values])))

    tags = filters.get("tag")
    if tags is not None:
        values = tags if isinstance(tags, list) else [tags]
        must.append(FieldCondition(key="tag", match=MatchAny(any=[str(t) for t in values])))

    doc_id = filters.get("doc_id")
    if doc_id is not None:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id))))

    time_range = filters.get("time_range")
    if time_range:
        range_kwargs: dict[str, float] = {}
        if time_range.get("gte") is not None:
            range_kwargs["gte"] = _to_unix_ts(time_range["gte"])
        if time_range.get("lte") is not None:
            range_kwargs["lte"] = _to_unix_ts(time_range["lte"])
        if range_kwargs:
            must.append(FieldCondition(key="created_at", range=Range(**range_kwargs)))

    if not must:
        return None
    return Filter(must=must)
```

#### 逐行解释

- `must: list[FieldCondition] = []`：累积"必须满足"的条件（AND 关系）。
- `source` 过滤：
  - `MatchAny(any=[...])`：匹配任一值（OR）。
  - 例：`source=["manual.md", "spec.pdf"]` → source 是这两个之一。
- `tag` 过滤：同 source，用 `MatchAny`。
- `doc_id` 过滤：
  - `MatchValue(value=...)`：精确等于。
  - 例：`doc_id="abc123"` → doc_id 必须等于 "abc123"。
- `time_range` 过滤：
  - `Range(gte=..., lte=...)`：范围过滤。
  - `gte` = greater than or equal（大于等于）。
  - `lte` = less than or equal（小于等于）。
  - 时间值用 `_to_unix_ts` 转成数字。
- `Filter(must=must)`：所有条件 AND 起来。

#### 8.3.6 `search` 方法

```python
def search(
    self,
    query_vector: list[float],
    top_n: int = 20,
    filters: dict | None = None,
) -> list[Chunk]:
    """Search the collection for the most similar chunks to ``query_vector``."""
    query_filter = self._build_filter(filters)
    hits = self._client.search(
        collection_name=self._collection_name,
        query_vector=list(query_vector),
        limit=top_n,
        query_filter=query_filter,
    )

    results: list[Chunk] = []
    for hit in hits:
        payload = hit.payload or {}
        created_at = self._parse_created_at(payload)

        metadata = Metadata(
            source=payload.get("source", ""),
            page=payload.get("page"),
            sheet=payload.get("sheet"),
            tag=list(payload.get("tag") or []),
            created_at=created_at,
            doc_id=payload.get("doc_id", ""),
            chunk_index=int(payload.get("chunk_index", 0) or 0),
            bbox=payload.get("bbox"),
        )
        results.append(
            Chunk(
                id=payload.get("chunk_id", str(hit.id)),
                text=payload.get("text", ""),
                metadata=metadata,
                score=float(hit.score) if hit.score is not None else None,
            )
        )
    return results
```

#### 逐行解释

- `query_filter = self._build_filter(filters)`：构建过滤器。
- `hits = self._client.search(...)`：调 Qdrant 搜索 API。
  - `query_vector`：查询向量。
  - `limit=top_n`：返回前 N 条。
  - `query_filter`：过滤条件。
- 遍历 `hits`：
  - `payload = hit.payload or {}`：取出元数据。
  - `created_at = self._parse_created_at(payload)`：解析时间。
  - 构造 `Metadata`：从 payload 还原所有字段。
  - 构造 `Chunk`：
    - `id`：优先用 payload 里的 `chunk_id`（原始 16 位 hex），没有就用 Qdrant 的 point id。
    - `text`：原文。
    - `metadata`：还原的元数据。
    - `score`：相似度分数（Qdrant 返回的）。

#### 8.3.7 `delete_by_doc` 和 `delete_by_chunk_ids`

```python
def delete_by_doc(self, doc_id: str) -> None:
    """Delete all points whose ``doc_id`` payload matches ``doc_id``."""
    self._client.delete(
        collection_name=self._collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=str(doc_id)))]
        ),
    )
    logger.info("deleted by doc_id", doc_id=doc_id, collection=self._collection_name)

def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
    """Delete the points identified by the given chunk ids."""
    if not chunk_ids:
        return
    point_ids = [self._point_id(cid) for cid in chunk_ids]
    self._client.delete(
        collection_name=self._collection_name,
        points_selector=qmodels.PointIdsList(points=point_ids),
    )
    logger.info("deleted by chunk_ids", count=len(point_ids), collection=self._collection_name)
```

#### 逐行解释

- `delete_by_doc`：
  - `points_selector=Filter(must=[...])`：按过滤条件删除（删所有 doc_id 匹配的点）。
- `delete_by_chunk_ids`：
  - `point_ids = [self._point_id(cid) for cid in chunk_ids]`：把 chunk id 转成 Qdrant 的 UUID。
  - `points_selector=qmodels.PointIdsList(points=point_ids)`：按 point id 列表删除。

#### 8.3.8 `count`

```python
def count(self) -> int:
    """Return the number of points in the collection."""
    result = self._client.count(collection_name=self._collection_name, exact=True)
    return int(result.count)
```

- `exact=True`：精确计数（不是近似）。
- `result.count`：返回的点数。

### 8.4 chroma_store.py 逐行精读

文件路径：`app/stores/chroma_store.py`。ChromaDB 是另一个开源向量库，主打"零配置"（嵌入式，不需要单独起服务）。

```python
"""ChromaDB-backed vector store implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.document import Chunk, Metadata
from app.observability.logging import get_logger
from app.stores.base import VectorStore

logger = get_logger(__name__)


class ChromaStore(VectorStore):
    """Vector store backed by ``chromadb.PersistentClient``.

    Args:
        path: Filesystem path where Chroma persists its data.
        collection_name: Name of the Chroma collection to use.
        dim: Vector dimensionality (kept for interface symmetry; Chroma infers
            dimensionality from the first upsert).
    """

    def __init__(self, path: str, collection_name: str, dim: int) -> None:
        """Initialize the persistent client and get/create the collection."""
        self._path = path
        self._collection_name = collection_name
        self._dim = int(dim)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _build_where(filters: dict | None) -> dict | None:
        """Translate a filter dict into a Chroma ``where`` clause (best-effort).

        Chroma only supports simple equality on metadata fields. Compound
        filters fall back to the first supported key encountered.
        """
        if not filters:
            return None

        doc_id = filters.get("doc_id")
        if doc_id is not None:
            return {"doc_id": str(doc_id)}

        source = filters.get("source")
        if isinstance(source, str):
            return {"source": source}

        tag = filters.get("tag")
        if isinstance(tag, str):
            return {"tag": tag}

        return None

    @staticmethod
    def _metadata_to_chroma(meta: Metadata) -> dict[str, Any]:
        """Convert a :class:`Metadata` into a JSON-serializable dict for Chroma."""
        out: dict[str, Any] = {
            "chunk_id": meta.doc_id,  # Chroma uses its own id; this is auxiliary
            "source": meta.source,
            "doc_id": meta.doc_id,
            "chunk_index": int(meta.chunk_index),
            "created_at": meta.created_at.isoformat(),
        }
        if meta.page is not None:
            out["page"] = int(meta.page)
        if meta.sheet is not None:
            out["sheet"] = str(meta.sheet)
        if meta.tag:
            out["tag"] = ",".join(meta.tag)
        return out

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert chunks and their vectors into the Chroma collection."""
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")

        ids = [chunk.id for chunk in chunks]
        embeddings = [list(map(float, v)) for v in vectors]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._metadata_to_chroma(chunk.metadata) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("upserted chunks", count=len(ids), collection=self._collection_name)

    def search(
        self,
        query_vector: list[float],
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Search the Chroma collection for the most similar chunks."""
        where = self._build_where(filters)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, query_vector))],
            "n_results": top_n,
        }
        if where is not None:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        results: list[Chunk] = []
        ids_batch = (raw.get("ids") or [[]])[0]
        docs_batch = (raw.get("documents") or [[]])[0]
        metas_batch = (raw.get("metadatas") or [[]])[0]
        dists_batch = (raw.get("distances") or [[]])[0]

        for cid, doc, meta, dist in zip(ids_batch, docs_batch, metas_batch, dists_batch, strict=False):
            meta = meta or {}
            created_at_value = meta.get("created_at")
            try:
                created_at = (
                    datetime.fromisoformat(created_at_value)
                    if isinstance(created_at_value, str)
                    else datetime.utcnow()
                )
            except (TypeError, ValueError):
                created_at = datetime.utcnow()

            tag_value = meta.get("tag")
            tag_list = tag_value.split(",") if isinstance(tag_value, str) and tag_value else []

            metadata = Metadata(
                source=meta.get("source", ""),
                page=meta.get("page"),
                sheet=meta.get("sheet"),
                tag=tag_list,
                created_at=created_at,
                doc_id=meta.get("doc_id", ""),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
            )
            try:
                score = float(dist)
            except (TypeError, ValueError):
                score = 0.0
            results.append(Chunk(id=str(cid), text=doc or "", metadata=metadata, score=score))
        return results

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all chunks whose ``doc_id`` metadata matches ``doc_id``."""
        self._collection.delete(where={"doc_id": str(doc_id)})
        logger.info("deleted by doc_id", doc_id=doc_id, collection=self._collection_name)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete the chunks identified by the given chunk ids."""
        if not chunk_ids:
            return
        self._collection.delete(ids=list(chunk_ids))
        logger.info("deleted by chunk_ids", count=len(chunk_ids), collection=self._collection_name)

    def count(self) -> int:
        """Return the number of items in the collection."""
        return int(self._collection.count())
```

#### 逐行解释（重点差异）

- `__init__`：
  - `chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))`：
    - `PersistentClient`：嵌入式模式，数据存到本地文件夹（不像 Qdrant 要单独起服务）。
    - `anonymized_telemetry=False`：关闭遥测（隐私）。
  - `self._collection = self._client.get_or_create_collection(name=collection_name)`：集合不存在就创建，存在就获取。**不需要预先指定维度**（Chroma 从第一次 upsert 推断）。
- `_build_where`：
  - Chroma 的 where 子句比 Qdrant 简单，只支持等值匹配。
  - 优先级：`doc_id` > `source` > `tag`，只取第一个能用的（best-effort）。
- `_metadata_to_chroma`：
  - Chroma 的 metadata 必须是基本类型（str/int/float/bool），不能存 list/dict。
  - 所以 `tag`（list）被拼成逗号分隔字符串：`"tag1,tag2"`。
  - `bbox`（dict）直接丢弃（Chroma 不支持）。
- `upsert`：
  - `ids`：直接用 chunk.id（不需要转 UUID，Chroma 接受任意字符串 id）。
  - `embeddings`、`documents`、`metadatas`：三个并行列表。
- `search`：
  - `query_embeddings`：注意是复数，可以传多个查询向量。
  - `n_results`：每个查询返回前 N 条。
  - 返回值是嵌套列表：`raw["ids"][0]` 是第一个查询的结果 id 列表。
  - `distances`：Chroma 返回的是"距离"（越小越相似），不是相似度（越大越相似）。代码直接存进 `score`，语义上和 Qdrant 相反，但接口统一了。
- `delete_by_doc`：`where={"doc_id": ...}` 按条件删。
- `delete_by_chunk_ids`：`ids=[...]` 按 id 删。
- `count`：`self._collection.count()` 直接返回整数。

#### 8.4.1 与 Qdrant 的差异对比表

| 特性 | QdrantStore | ChromaStore |
| --- | --- | --- |
| 部署模式 | 独立服务（Docker） | 嵌入式（本地文件） |
| 集合维度 | 创建时指定 | 自动推断 |
| point id | UUID（uuid5 转换） | 任意字符串 |
| metadata 类型 | 支持任意 JSON（含 list/dict） | 只支持基本类型 |
| 过滤能力 | 强（MatchAny/MatchValue/Range/AND） | 弱（等值匹配） |
| 时间范围过滤 | 支持（Range + 数字时间戳） | 不支持 |
| 距离度量 | Cosine（可配 Euclid/Dot） | 默认 L2 |
| 性能 | 高（Rust + HNSW 优化） | 中等（Python + SQLite 后端） |
| 适用场景 | 生产环境、大规模 | 开发/小规模、快速原型 |

### 8.5 factory.py 逐行精读

文件路径：`app/stores/factory.py`。

```python
"""Factory for selecting a vector store backend based on settings."""
from __future__ import annotations

from app.observability.logging import get_logger
from app.stores.base import VectorStore
from app.stores.chroma_store import ChromaStore
from app.stores.qdrant_store import QdrantStore

logger = get_logger(__name__)


def get_vector_store(settings) -> VectorStore:
    """Build a :class:`VectorStore` from the supplied :class:`Settings`.

    Dispatch rules:
        - ``"qdrant"`` (default): :class:`QdrantStore` pointed at
          ``qdrant_url`` using collection ``qdrant_collection``.
        - ``"chroma"``: :class:`ChromaStore` rooted at ``chroma_path``,
          reusing ``qdrant_collection`` as the collection name.

    Args:
        settings: Application :class:`Settings` instance.

    Returns:
        A concrete :class:`VectorStore` instance.
    """
    backend = (settings.vector_store or "qdrant").lower()

    if backend == "chroma":
        logger.info(
            "using chroma vector store",
            path=settings.chroma_path,
            collection=settings.qdrant_collection,
        )
        return ChromaStore(
            path=settings.chroma_path,
            collection_name=settings.qdrant_collection,
            dim=settings.embedder_dim,
        )

    if backend == "qdrant":
        logger.info(
            "using qdrant vector store",
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )
        return QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            dim=settings.embedder_dim,
        )

    logger.warning("unknown vector_store, falling back to qdrant", value=backend)
    return QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        dim=settings.embedder_dim,
    )
```

#### 逐行解释

- `backend = (settings.vector_store or "qdrant").lower()`：
  - 读 `settings.vector_store`，如果是 `None` 或空串就用 `"qdrant"`。
  - `.lower()`：统一成小写，避免 `"Qdrant"` / `"QDRANT"` 不匹配。
- 两个 `if` 分支：
  - `"chroma"` → `ChromaStore(path=chroma_path, ...)`：用本地文件存储。
  - `"qdrant"` → `QdrantStore(url=qdrant_url, ...)`：连远程 Qdrant 服务。
- 兜底：未知值 → 记 warning，回退到 Qdrant。
- **注意**：Chroma 复用 `qdrant_collection` 作为集合名（避免引入新配置项）。

### 8.6 存储结果示例

假设我们把示例手册的第一个 chunk 写入 Qdrant，一个 point 的完整结构如下：

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vector": [0.0234, -0.0178, 0.0412, -0.0089, 0.0334, ...],
  "payload": {
    "chunk_id": "a3f2b1c9d8e7f6a5",
    "source": "manual.md",
    "page": null,
    "sheet": null,
    "tag": ["产品手册", "音箱"],
    "created_at": 1722304800.0,
    "created_at_iso": "2026-07-29T10:00:00+00:00",
    "doc_id": "doc-001-uuid",
    "chunk_index": 0,
    "text": "# X-2025 智能音箱产品手册\n\n## 1. 产品概述\nX-2025 是一款支持语音交互的智能音箱...",
    "bbox": {"char_offset": 0}
  }
}
```

**字段含义**：

- `id`：Qdrant 内部 point id（UUID 格式，由 `uuid5(chunk_id)` 生成）。
- `vector`：1024 维浮点数向量（bge-m3 输出）。
- `payload`：
  - `chunk_id`：原始 16 位 hex chunk id（业务用）。
  - `source`：来源文件。
  - `page`、`sheet`：分页/分表信息（手册没有，所以 null）。
  - `tag`：业务标签。
  - `created_at`：unix 时间戳（用于 Range 过滤）。
  - `created_at_iso`：ISO 字符串（人类可读）。
  - `doc_id`：父文档 id。
  - `chunk_index`：chunk 在文档内的序号。
  - `text`：原文（检索时直接返回）。
  - `bbox`：字符偏移（用于高亮）。

**检索时**：用户问"电池能用多久" → 嵌入成向量 → Qdrant 用 HNSW 找最相似的 point → 返回 payload 里的 `text` 字段给大模型生成答案。

---

## 全篇总结

到这里，第二部分教程结束。我们跟着一份《X-2025 智能音箱产品手册》走完了 RAG 的四个核心阶段：

1. **清洗**（第 5 章）：把 PDF 抠出来的脏文本洗干净——去控制字符、统一换行、全角转半角、压扁空行、修单词断行。
2. **分块**（第 6 章）：把长文本切成小段——四种分块器各有侧重，默认用递归字符分块器。
3. **嵌入**（第 7 章）：把文本变成 1024 维数字向量——三种后端（本地、Ollama、API），默认用本地 bge-m3。
4. **存储**（第 8 章）：把向量存进向量库——两种后端（Qdrant、Chroma），默认用 Qdrant。

下一步（第三部分）会讲检索、重排、生成阶段，把"问问题 → 找答案 → 生成回复"的完整链路跑通。

---

**文档统计**：

- 章节：第 5、6、7、8 章（共 4 章，含 5.1-5.2、6.1-6.8、7.1-7.7、8.1-8.6 共 25 个小节）
- 涉及代码文件：16 个（cleaner、base/fixed_token/recursive_char/semantic/structural/factory 分块器、base/local/ollama/api/factory 嵌入器、base/qdrant/chroma/factory 存储）
- 总字数：约 1.2 万字
