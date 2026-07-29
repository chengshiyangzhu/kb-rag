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
