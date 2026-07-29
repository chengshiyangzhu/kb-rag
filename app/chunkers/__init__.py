"""app.chunkers package.

Re-exports the :class:`Chunker` ABC, :class:`ChunkerConfig`, every concrete
chunker implementation, and the :func:`get_chunker` factory.
"""
from __future__ import annotations

from app.chunkers.base import Chunker, ChunkerConfig
from app.chunkers.factory import get_chunker
from app.chunkers.fixed_token import FixedTokenChunker
from app.chunkers.recursive_char import RecursiveCharChunker
from app.chunkers.semantic import SemanticChunker
from app.chunkers.structural import StructuralChunker

__all__ = [
    "Chunker",
    "ChunkerConfig",
    "FixedTokenChunker",
    "RecursiveCharChunker",
    "SemanticChunker",
    "StructuralChunker",
    "get_chunker",
]
