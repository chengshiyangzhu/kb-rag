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
