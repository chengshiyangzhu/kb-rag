"""Prometheus metrics for the kb-rag pipeline.

Exposes counters and histograms covering query volume, retrieval/generation
latency, no-result occurrences and ingestion throughput. Use the ``record_*``
helpers rather than touching the instruments directly so labels stay consistent.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---- Instruments ----

rag_query_total = Counter(
    "rag_query_total",
    "Total number of RAG queries processed.",
    ["status"],
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of the retrieval stage in seconds.",
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Latency of the generation stage in seconds.",
)

rag_no_result_total = Counter(
    "rag_no_result_total",
    "Number of queries that yielded no relevant context.",
)

rag_ingest_total = Counter(
    "rag_ingest_total",
    "Number of documents ingested.",
    ["file_type"],
)


# ---- Recording helpers ----


def record_query(status: str) -> None:
    """Increment :data:`rag_query_total` for the given status.

    Args:
        status: Outcome label, e.g. ``"ok"``, ``"error"``.
    """
    rag_query_total.labels(status=status).inc()


def record_retrieval_latency(seconds: float) -> None:
    """Observe a retrieval latency in seconds."""
    rag_retrieval_latency_seconds.observe(seconds)


def record_generation_latency(seconds: float) -> None:
    """Observe a generation latency in seconds."""
    rag_generation_latency_seconds.observe(seconds)


def record_no_result() -> None:
    """Increment the no-result counter."""
    rag_no_result_total.inc()


def record_ingest(file_type: str) -> None:
    """Increment the ingestion counter for a given file type.

    Args:
        file_type: File extension or type label, e.g. ``"pdf"``, ``"docx"``.
    """
    rag_ingest_total.labels(file_type=file_type).inc()
