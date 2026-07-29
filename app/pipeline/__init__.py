"""RAG pipeline orchestration package.

Exposes the ingestion and query pipelines, their result models, the citation
reference model, and the dependency-injection container.
"""
from __future__ import annotations

from app.pipeline.container import Container
from app.pipeline.ingest_pipeline import IngestPipeline, IngestResult
from app.pipeline.query_pipeline import QueryPipeline, QueryResult, Reference

__all__ = [
    "Container",
    "IngestPipeline",
    "IngestResult",
    "QueryPipeline",
    "QueryResult",
    "Reference",
]
