"""Pydantic request/response models for the kb-rag API (Stage 9).

These models are used as ``response_model`` and request body types for the
REST endpoints exposed under :mod:`backend.api.v1`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Response model for ``POST /api/v1/ingest``.

    Attributes:
        doc_id: Identifier of the ingested document.
        num_chunks: Number of chunks produced and stored.
        file_type: File extension (without dot) of the source file.
        trace_id: Trace identifier correlating logs/metrics across the request.
        errors: Non-fatal errors encountered during ingestion.
    """

    doc_id: str
    num_chunks: int
    file_type: str
    trace_id: str
    errors: list[str] = Field(default_factory=list)


class QueryFilters(BaseModel):
    """Optional metadata filters for the query endpoint.

    Attributes:
        source: Filter by source file path/name (string or list of strings).
        tag: Filter by tag (string or list of strings).
        doc_id: Filter by document identifier.
    """

    source: str | list[str] | None = None
    tag: str | list[str] | None = None
    doc_id: str | None = None


class QueryRequest(BaseModel):
    """Request body for ``POST /api/v1/query``.

    Attributes:
        question: Natural-language question.
        filters: Optional metadata filters forwarded to the retriever.
        top_n: Override for the number of candidates to retrieve.
    """

    question: str
    filters: QueryFilters | None = None
    top_n: int | None = None


class ReferenceOut(BaseModel):
    """A citation reference mapped back to a source chunk.

    Attributes:
        chunk_id: Identifier of the source chunk.
        source: Original file path or URI of the chunk.
        page: 1-indexed page number (if applicable).
        score: Relevance score from the retrieval/rerank stage.
        snippet: Truncated preview of the chunk text.
    """

    chunk_id: str
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str = ""


class QueryResponse(BaseModel):
    """Response model for ``POST /api/v1/query``.

    Attributes:
        answer: Generated natural-language answer (or the no-result fallback).
        references: Citation references mapped to source chunks.
        trace_id: Trace identifier correlating logs/metrics across the request.
        no_result: ``True`` when retrieval yielded no confident result.
        retrieval_latency: Retrieval stage latency in seconds.
        generation_latency: Generation stage latency in seconds.
    """

    answer: str
    references: list[ReferenceOut] = Field(default_factory=list)
    trace_id: str
    no_result: bool = False
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0


class DocumentOut(BaseModel):
    """Response model for an ingested document.

    Attributes:
        doc_id: Identifier of the document.
        file_type: File extension (without dot) of the source file.
        num_chunks: Number of chunks stored for the document.
        ingested_at: ISO-8601 timestamp of ingestion.
    """

    doc_id: str
    file_type: str
    num_chunks: int
    ingested_at: str


class HealthResponse(BaseModel):
    """Response model for ``GET /health``.

    Attributes:
        status: Liveness status (``"ok"`` or error).
        vector_store: Vector store backend name (``"qdrant"`` or ``"chroma"``).
        chunks: Total number of vectors stored in the collection.
        version: API version string.
    """

    status: str
    vector_store: str
    chunks: int
    version: str


class ErrorResponse(BaseModel):
    """Standard error response payload.

    Attributes:
        detail: Human-readable error description.
        trace_id: Trace identifier for correlation (when available).
    """

    detail: str
    trace_id: str | None = None
