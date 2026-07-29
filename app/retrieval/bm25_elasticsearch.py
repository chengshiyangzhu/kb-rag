"""Elasticsearch-backed BM25 retriever placeholder.

In production deployments, Elasticsearch is a natural fit for sparse retrieval
at scale. This module is intentionally a stub: the interface mirrors
:class:`app.retrieval.bm25.BM25Retriever`, but the implementation is deferred.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models.document import Chunk


class ElasticsearchBM25Retriever:
    """Placeholder for an Elasticsearch BM25 retriever.

    The interface is intentionally identical to
    :class:`app.retrieval.bm25.BM25Retriever` so callers can swap backends
    via configuration. All operations raise ``NotImplementedError`` until a
    real implementation is provided.

    Args:
        index_path: Reserved for API parity (unused).
        es_url: Elasticsearch URL.
        index_name: Name of the ES index to query.
    """

    def __init__(
        self,
        index_path: Path | str | None = None,
        es_url: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """Construct the (unimplemented) retriever."""
        self.index_path = index_path
        self.es_url = es_url
        self.index_name = index_name

    def add(self, chunks: list[Chunk]) -> None:
        """Not implemented."""
        raise NotImplementedError("ElasticsearchBM25Retriever is a placeholder")

    def search(
        self,
        query: str,
        top_n: int = 20,
        filters: dict | None = None,
    ) -> list[Chunk]:
        """Not implemented."""
        raise NotImplementedError("ElasticsearchBM25Retriever is a placeholder")

    def remove_by_doc(self, doc_id: str) -> None:
        """Not implemented."""
        raise NotImplementedError("ElasticsearchBM25Retriever is a placeholder")

    def persist(self) -> None:
        """Not implemented."""
        raise NotImplementedError("ElasticsearchBM25Retriever is a placeholder")

    def count(self) -> int:
        """Not implemented."""
        raise NotImplementedError("ElasticsearchBM25Retriever is a placeholder")
