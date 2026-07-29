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

# --------------------------------------------------------------------------- #
# Page config (must run before any other Streamlit call).                     #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="kb-rag 知识库",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Constants & configuration.                                                 #
# --------------------------------------------------------------------------- #
DEFAULT_API_URL = "http://localhost:8000"
ACCEPTED_FILE_TYPES = ["pdf", "docx", "xlsx", "md", "txt", "html"]
MAX_HISTORY = 10
API_TIMEOUT = 120  # seconds; bounded so st.spinner never blocks forever.


# --------------------------------------------------------------------------- #
# Local data structures.                                                      #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Helper functions.                                                           #
# --------------------------------------------------------------------------- #
def _api_call(
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float = API_TIMEOUT,
    **kwargs: Any,
) -> tuple[int, Any]:
    """Call the kb-rag API and return ``(status_code, data)``.

    Network / timeout errors are caught and returned as
    ``(0, {"error": "<message>"})`` so callers can render a friendly
    ``st.error`` instead of crashing.

    Parameters
    ----------
    method:
        HTTP verb (``GET``, ``POST``, ``DELETE`` …).
    base_url:
        API origin, e.g. ``http://localhost:8000``.
    path:
        Path appended to ``base_url``, e.g. ``/api/v1/documents``.
    timeout:
        Per-request timeout in seconds.
    **kwargs:
        Forwarded to :func:`requests.request` (``json=``, ``files=``, …).
    """
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


def _format_filesize(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


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


# --------------------------------------------------------------------------- #
# Sidebar.                                                                    #
# --------------------------------------------------------------------------- #
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

    # ---- File upload & ingest ----
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

    # ---- Document list ----
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
                if st.button(
                    "🗑️",
                    key=f"del_{doc.doc_id}",
                    help=f"删除文档 {doc.doc_id}",
                ):
                    _delete_doc(base_url, doc.doc_id)
                    st.session_state["docs"] = _load_documents(base_url)
                    st.rerun()

    return base_url


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


def _delete_doc(base_url: str, doc_id: str) -> None:
    """Delete a document via ``DELETE /api/v1/documents/{doc_id}``."""
    status, data = _api_call("DELETE", base_url, f"/api/v1/documents/{doc_id}")
    if status == 204:
        st.sidebar.success(f"已删除 `{doc_id[:8]}…`")
    else:
        msg = data.get("error") if isinstance(data, dict) else str(data)
        st.sidebar.error(f"删除失败：{msg}")


# --------------------------------------------------------------------------- #
# Main Q&A area.                                                              #
# --------------------------------------------------------------------------- #
def _render_main(base_url: str) -> None:
    """Render the main Q&A panel: input, filters, answer, references, history."""
    st.title("💬 知识库问答")
    st.caption(f"后端 API：`{base_url}`")

    # ---- Optional filters ----
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

    # ---- Question input ----
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

    # ---- Current answer area ----
    _render_current_answer()

    # ---- History ----
    _render_history()


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

    # Archive the previous "current" result into history, then set the new one.
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


def _render_current_answer() -> None:
    """Render the most recent query result (the live answer area)."""
    current = st.session_state.get("current_result")
    if not current:
        st.info("👆 输入问题并点击「提问」开始问答。")
        return

    st.subheader("📝 当前回答")
    _render_qa_block(current, expanded=True)


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
                    # 用 info 框高亮 snippet，醒目易读。
                    st.info(snippet)
                else:
                    st.caption("_(无 snippet)_")

    # Metadata footer.
    meta_parts = [f"trace_id: `{item.get('trace_id', '-')}`"]
    rl = item.get("retrieval_latency")
    gl = item.get("generation_latency")
    if rl is not None:
        try:
            meta_parts.append(f"retrieval: {float(rl) * 1000:.0f}ms")
        except (TypeError, ValueError):
            meta_parts.append(f"retrieval: {rl}")
    if gl is not None:
        try:
            meta_parts.append(f"generation: {float(gl) * 1000:.0f}ms")
        except (TypeError, ValueError):
            meta_parts.append(f"generation: {gl}")
    st.caption(" · ".join(meta_parts))


def _render_history() -> None:
    """Render the archived Q&A history (collapsed)."""
    st.subheader("🕘 历史会话")
    history: list[dict[str, Any]] = st.session_state.get("history", []) or []
    if not history:
        st.caption("暂无历史记录")
        return

    if st.button("🗑️ 清空历史", key="clear_history"):
        st.session_state["history"] = []
        st.rerun()

    st.caption(f"共 {len(history)} 条（最多保留 {MAX_HISTORY} 条）")
    for idx, item in enumerate(history, start=1):
        q_preview = (item.get("question", "") or "")[:60]
        with st.expander(f"Q{idx}: {q_preview}", expanded=False):
            _render_qa_block(item, expanded=False)


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the full Streamlit application."""
    base_url = _render_sidebar()
    _render_main(base_url)


if __name__ == "__main__":
    main()
