"""Unit tests for the RAG pipeline orchestration layer (Stage 8).

These tests exercise the :class:`IngestPipeline` and :class:`QueryPipeline`
end-to-end with mocked heavy components (embedder, reranker, generator) and a
real Chroma vector store to verify the orchestration logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.generation import GenerationResult
from app.models.document import Chunk, Metadata
from app.pipeline import (
    Container,
    IngestPipeline,
    IngestResult,
    QueryPipeline,
    QueryResult,
    Reference,
)

# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Embedder returning fixed-length constant vectors for testing."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one constant vector per input text."""
        return [[0.1] * self._dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a single constant vector."""
        return [0.1] * self._dim


class FakeReranker:
    """Reranker that returns input chunks with a fixed high score."""

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Return up to ``top_k`` candidates with ``score`` set to 0.9."""
        result = candidates[:top_k]
        for chunk in result:
            chunk.score = 0.9
        return result


class FakeGenerator:
    """Generator returning a fixed answer with a ``[1]`` citation."""

    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Return a :class:`GenerationResult` referencing the first context."""
        chunk_id = contexts[0].id if contexts else ""
        return GenerationResult(
            answer="答案[1]",
            citations=[1],
            used_chunk_ids=[chunk_id],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "test-chunk-1",
    text: str = "This is the retrieved context for the query.",
    doc_id: str = "doc-1",
    source: str = "test.txt",
) -> Chunk:
    """Build a minimal :class:`Chunk` for tests."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=Metadata(
            source=source,
            doc_id=doc_id,
            chunk_index=0,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_pipeline_with_chroma(tmp_path, mock_settings, monkeypatch):  # type: ignore[no-untyped-def]
    """IngestPipeline should ingest a txt file into a real Chroma store.

    Uses a real ChromaStore (via ``get_vector_store``) with a mock embedder
    returning fixed 8-dim vectors.  Asserts that at least one chunk is
    produced and the store count increases.
    """
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text(
        "This is a test document for the RAG pipeline ingestion. "
        "It contains enough text to produce at least one chunk "
        "when processed by the recursive character chunker.",
        encoding="utf-8",
    )

    # Match embedder_dim to FakeEmbedder's dimension.
    mock_settings.embedder_dim = 8

    fake_embedder = FakeEmbedder(dim=8)
    monkeypatch.setattr(
        "app.pipeline.ingest_pipeline.get_embedder",
        lambda settings: fake_embedder,
    )

    pipeline = IngestPipeline(mock_settings)
    result = pipeline.ingest_file(txt_path)

    assert isinstance(result, IngestResult)
    assert result.num_chunks >= 1
    assert result.file_type == "txt"
    assert result.doc_id
    assert result.trace_id
    assert pipeline.vector_store.count() >= 1


def test_query_pipeline_no_result(mock_settings) -> None:  # type: ignore[no-untyped-def]
    """QueryPipeline should return ``no_result`` when retrieval is empty.

    The hybrid retriever is mocked to return an empty list; the pipeline
    should short-circuit and return the canonical no-result answer.
    """
    pipeline = QueryPipeline(mock_settings)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    pipeline._hybrid_retriever = mock_retriever

    result = pipeline.query("What is RAG?")

    assert isinstance(result, QueryResult)
    assert result.no_result is True
    assert "未在知识库" in result.answer
    assert result.references == []
    mock_retriever.retrieve.assert_called_once()


def test_query_pipeline_with_mock(mock_settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """QueryPipeline should parse citations and map references on success.

    The hybrid retriever is mocked to return a single chunk; FakeReranker
    sets a high score (0.9) so the guardrail passes; FakeGenerator returns
    an answer containing ``[1]``.  The pipeline should map citation ``[1]``
    back to the source chunk.
    """
    chunk = _make_chunk()

    pipeline = QueryPipeline(mock_settings)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [chunk]
    pipeline._hybrid_retriever = mock_retriever

    monkeypatch.setattr(
        "app.pipeline.query_pipeline.get_reranker",
        lambda settings: FakeReranker(),
    )
    monkeypatch.setattr(
        "app.pipeline.query_pipeline.get_generator",
        lambda settings: FakeGenerator(),
    )

    result = pipeline.query("What is RAG?")

    assert isinstance(result, QueryResult)
    assert result.no_result is False
    assert result.answer == "答案[1]"
    assert len(result.references) == 1

    ref = result.references[0]
    assert isinstance(ref, Reference)
    assert ref.chunk_id == chunk.id
    assert ref.source == chunk.metadata.source
    assert ref.score == 0.9
    assert ref.snippet
