"""backend package for the kb-rag API.

Exposes shared FastAPI dependency accessors that wrap :class:`app.pipeline.Container`.
Defining them here (rather than in :mod:`backend.main`) breaks what would
otherwise be a circular import: route modules need the accessors at module
load time, but :mod:`backend.main` loads the routers after defining them.
"""
from __future__ import annotations

from app.pipeline import Container, IngestPipeline, QueryPipeline
from app.stores import VectorStore

__all__ = [
    "get_ingest_pipeline_dep",
    "get_query_pipeline_dep",
    "get_vector_store_dep",
]


def get_ingest_pipeline_dep() -> IngestPipeline:
    """Return the shared :class:`IngestPipeline` singleton.

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_ingest_pipeline()


def get_query_pipeline_dep() -> QueryPipeline:
    """Return the shared :class:`QueryPipeline` singleton.

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_query_pipeline()


def get_vector_store_dep() -> VectorStore:
    """Return the shared :class:`VectorStore` (sourced from the ingest pipeline).

    Overridable via ``app.dependency_overrides`` in tests.
    """
    return Container.get_ingest_pipeline().vector_store
