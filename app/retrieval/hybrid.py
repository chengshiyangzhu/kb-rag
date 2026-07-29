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
