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
