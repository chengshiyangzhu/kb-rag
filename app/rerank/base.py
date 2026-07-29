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
