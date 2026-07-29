# app.embedders package
"""Embedding backends for the kb-rag pipeline.

Exposes the :class:`Embedder` ABC, three concrete implementations
(:class:`LocalEmbedder`, :class:`OllamaEmbedder`, :class:`ApiEmbedder`) and the
:func:`get_embedder` factory.
"""
from __future__ import annotations

from app.embedders.api_embedder import ApiEmbedder
from app.embedders.base import Embedder
from app.embedders.factory import get_embedder
from app.embedders.local_embedder import LocalEmbedder
from app.embedders.ollama_embedder import OllamaEmbedder

__all__ = [
    "Embedder",
    "LocalEmbedder",
    "OllamaEmbedder",
    "ApiEmbedder",
    "get_embedder",
]
