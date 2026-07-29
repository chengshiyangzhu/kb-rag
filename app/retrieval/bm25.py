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
