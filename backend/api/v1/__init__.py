"""Aggregate v1 API router for the kb-rag API.

The :data:`router` aggregates the ``/ingest``, ``/query`` and ``/documents``
sub-routers.  It is mounted under the ``/api/v1`` prefix in
:mod:`backend.main`.  The ``/health`` endpoint lives at the root level and is
imported directly in :mod:`backend.main`.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1 import documents, ingest, query

router = APIRouter()
router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
router.include_router(query.router, prefix="/query", tags=["query"])
router.include_router(documents.router, prefix="/documents", tags=["documents"])

__all__ = ["router"]
