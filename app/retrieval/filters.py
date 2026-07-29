"""Filter construction helpers for retrieval.

The ``build_filters`` helper produces a normalized filter dictionary understood
by vector store ``search`` implementations. Filters are intentionally permissive:
``None`` values are dropped so callers can pass partial arguments.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def build_filters(
    source: str | None = None,
    tag: str | None = None,
    doc_id: str | None = None,
    time_range: tuple[datetime, datetime] | None = None,
) -> dict[str, Any]:
    """Build a normalized metadata filter dictionary.

    Args:
        source: Exact-match source path/URI.
        tag: Tag that must appear in the chunk's ``metadata.tag`` list.
        doc_id: Exact-match parent document id.
        time_range: ``(start, end)`` inclusive ``created_at`` window.

    Returns:
        A dict with only the keys for which a non-``None`` value was supplied.
        Returns an empty dict when no arguments are provided.
    """
    filters: dict[str, Any] = {}
    if source is not None:
        filters["source"] = source
    if tag is not None:
        filters["tag"] = tag
    if doc_id is not None:
        filters["doc_id"] = doc_id
    if time_range is not None:
        start, end = time_range
        filters["time_range"] = (start, end)
    return filters
