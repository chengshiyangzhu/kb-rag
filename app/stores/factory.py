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
