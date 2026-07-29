"""Retrieval layer for kb-rag.

Exposes dense (vector), sparse (BM25), hybrid retrieval and RRF fusion.
"""
from __future__ import annotations

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_elasticsearch import ElasticsearchBM25Retriever
from app.retrieval.dense import VectorRetriever
from app.retrieval.filters import build_filters
from app.retrieval.fusion import RRFFusion
from app.retrieval.hybrid import HybridRetriever

__all__ = [
    "BM25Retriever",
    "ElasticsearchBM25Retriever",
    "VectorRetriever",
    "build_filters",
    "RRFFusion",
    "HybridRetriever",
]
