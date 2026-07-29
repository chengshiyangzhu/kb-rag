"""Factory for constructing rerankers from settings."""
from __future__ import annotations

from typing import Any

from app.rerank.base import Reranker
from app.rerank.bge_reranker import BgeReranker


def get_reranker(settings: Any) -> Reranker:
    """Build a :class:`Reranker` from the application settings.

    Currently always returns a :class:`BgeReranker` configured with
    ``settings.rerank_model``.

    Args:
        settings: Application settings (duck-typed; must expose
            ``rerank_model``).

    Returns:
        A :class:`Reranker` instance.
    """
    model_name = getattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")
    return BgeReranker(model_name=model_name)
