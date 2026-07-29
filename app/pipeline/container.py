"""Dependency-injection container caching shared pipeline instances.

The :class:`Container` provides singleton access to the application settings
and the two top-level pipelines (:class:`IngestPipeline` and
:class:`QueryPipeline`).  Using :func:`functools.lru_cache` on the accessor
methods ensures that repeated calls return the *same* instance, avoiding
re-loading of ML models and re-initialisation of stores on every request.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings

from app.pipeline.ingest_pipeline import IngestPipeline
from app.pipeline.query_pipeline import QueryPipeline


class Container:
    """Singleton container caching settings and pipeline instances.

    All accessors are class methods backed by :func:`functools.lru_cache` so
    that the first call constructs the instance and every subsequent call
    returns the cached singleton.

    Typical usage::

        pipeline = Container.get_query_pipeline()
        result = pipeline.query("What is RAG?")
    """

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls):
        """Return the cached :class:`Settings` singleton.

        Returns:
            The application :class:`Settings` instance (loaded once).
        """
        return get_settings()

    @classmethod
    @lru_cache(maxsize=1)
    def get_ingest_pipeline(cls) -> IngestPipeline:
        """Return the cached :class:`IngestPipeline` singleton.

        Returns:
            A shared :class:`IngestPipeline` instance (constructed once).
        """
        return IngestPipeline(cls.get_settings())

    @classmethod
    @lru_cache(maxsize=1)
    def get_query_pipeline(cls) -> QueryPipeline:
        """Return the cached :class:`QueryPipeline` singleton.

        Returns:
            A shared :class:`QueryPipeline` instance (constructed once).
        """
        return QueryPipeline(cls.get_settings())

    @classmethod
    def reset(cls) -> None:
        """Clear all cached singletons.

        Primarily useful in tests where a fresh pipeline is needed.
        """
        cls.get_settings.cache_clear()
        cls.get_ingest_pipeline.cache_clear()
        cls.get_query_pipeline.cache_clear()
