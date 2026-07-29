"""Chunker factory dispatching on ``settings.chunker_type``."""
from __future__ import annotations

from typing import Any

from app.chunkers.base import Chunker, ChunkerConfig
from app.chunkers.fixed_token import FixedTokenChunker
from app.chunkers.recursive_char import RecursiveCharChunker
from app.chunkers.semantic import SemanticChunker
from app.chunkers.structural import StructuralChunker


def get_chunker(settings: Any) -> Chunker:
    """Build a :class:`Chunker` from a Settings-like object.

    Args:
        settings: An object exposing ``chunker_type``, ``chunk_size`` and
            ``chunk_overlap`` attributes (e.g. the project :class:`Settings` or
            a ``MagicMock`` in tests).

    Returns:
        A concrete :class:`Chunker` instance matching ``settings.chunker_type``.
        Unknown / missing types fall back to :class:`RecursiveCharChunker`.
    """
    config = ChunkerConfig(
        chunk_size=getattr(settings, "chunk_size", 512),
        overlap=getattr(settings, "chunk_overlap", 64),
    )
    ctype = getattr(settings, "chunker_type", "recursive")
    if ctype == "fixed":
        return FixedTokenChunker(config)
    if ctype == "semantic":
        return SemanticChunker(config)
    if ctype == "structural":
        return StructuralChunker(config)
    # Default to recursive for "recursive" and any unknown value.
    return RecursiveCharChunker(config)
