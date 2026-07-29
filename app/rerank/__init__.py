"""Rerank layer for kb-rag.

Exposes the abstract :class:`Reranker`, the BGE implementation and a factory.
"""
from __future__ import annotations

from app.rerank.base import Reranker
from app.rerank.bge_reranker import BgeReranker
from app.rerank.factory import get_reranker

__all__ = ["Reranker", "BgeReranker", "get_reranker"]
