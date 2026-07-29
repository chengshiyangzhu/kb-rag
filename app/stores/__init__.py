# app.stores package
"""Vector store backends for the kb-rag pipeline.

Exposes the :class:`VectorStore` ABC, two concrete implementations
(:class:`QdrantStore`, :class:`ChromaStore`) and the :func:`get_vector_store`
factory.
"""
from __future__ import annotations

from app.stores.base import VectorStore
from app.stores.chroma_store import ChromaStore
from app.stores.factory import get_vector_store
from app.stores.qdrant_store import QdrantStore

__all__ = [
    "VectorStore",
    "QdrantStore",
    "ChromaStore",
    "get_vector_store",
]
