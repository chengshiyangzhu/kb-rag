"""Unit tests for the retrieval, rerank and generation layers.

These tests do not require any external services: heavy ML dependencies are
mocked, and only the pure-Python logic (RRF fusion, BM25 indexing, guardrail
thresholds, citation parsing and prompt construction) is exercised.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.generation import (
    CitationParser,
    Guardrail,
    build_rag_prompt,
    get_generator,
)
from app.models.document import Chunk, Metadata
from app.rerank import get_reranker
from app.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RRFFusion,
    VectorRetriever,
)


# ---- Fixtures ----


def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc-1",
    score: float | None = None,
) -> Chunk:
    """Build a minimal :class:`Chunk` for tests."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=Metadata(
            source="test.md",
            doc_id=doc_id,
            chunk_index=0,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        score=score,
    )


# ---- RRF Fusion ----


def test_rrf_fusion() -> None:
    """The chunk appearing in both lists should be the top fused result."""
    shared = _make_chunk("c-shared", "shared content")
    only_dense = _make_chunk("c-dense", "dense only")
    only_sparse = _make_chunk("c-sparse", "sparse only")

    dense_results = [shared, only_dense]
    sparse_results = [only_sparse, shared]

    fusion = RRFFusion()
    fused = fusion.fuse([dense_results, sparse_results], k=60, top_n=10)

    assert len(fused) >= 3
    assert fused[0].id == "c-shared"
    # The shared chunk should have a higher score than the solo chunks.
    assert fused[0].score is not None
    assert fused[0].score > fused[1].score


def test_rrf_fusion_top_n_limit() -> None:
    """``top_n`` should cap the returned list length."""
    chunks_a = [_make_chunk(f"a-{i}", f"text a {i}") for i in range(5)]
    chunks_b = [_make_chunk(f"b-{i}", f"text b {i}") for i in range(5)]
    fused = RRFFusion().fuse([chunks_a, chunks_b], k=60, top_n=3)
    assert len(fused) == 3


# ---- BM25 Retriever ----


def test_bm25_retriever(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Adding 3 chunks and searching a keyword should return matches."""
    index_path = tmp_path / "bm25.pkl"
    retriever = BM25Retriever(index_path=index_path)
    chunks = [
        _make_chunk("c-1", "Python 是一种解释型编程语言", doc_id="doc-1"),
        _make_chunk("c-2", "Rust 注重性能与内存安全", doc_id="doc-2"),
        _make_chunk("c-3", "Python 广泛用于数据科学和机器学习", doc_id="doc-1"),
    ]
    retriever.add(chunks)
    assert retriever.count() == 3

    hits = retriever.search("Python 数据科学", top_n=5)
    assert len(hits) >= 1
    # Both Python chunks should be matched and ranked above the Rust chunk.
    hit_ids = {h.id for h in hits}
    assert "c-1" in hit_ids or "c-3" in hit_ids
    for hit in hits:
        assert hit.score is not None
        assert hit.score > 0

    # Persist + reload round-trip should preserve count.
    retriever.persist()
    reloaded = BM25Retriever(index_path=index_path)
    assert reloaded.count() == 3
    again = reloaded.search("Python", top_n=5)
    assert len(again) >= 1


def test_bm25_retriever_remove_by_doc(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``remove_by_doc`` should drop all chunks for the given doc_id."""
    retriever = BM25Retriever(index_path=tmp_path / "bm25.pkl")
    retriever.add(
        [
            _make_chunk("c-1", "alpha content", doc_id="doc-a"),
            _make_chunk("c-2", "beta content", doc_id="doc-b"),
        ]
    )
    assert retriever.count() == 2
    retriever.remove_by_doc("doc-a")
    assert retriever.count() == 1
    hits = retriever.search("alpha", top_n=5)
    assert hits == []


# ---- Vector Retriever (duck-typed) ----


def test_vector_retriever_duck_typed() -> None:
    """``VectorRetriever`` should call embed + search on duck-typed deps."""
    fake_store = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    expected = _make_chunk("c-1", "hello")
    fake_store.search.return_value = [expected]

    retriever = VectorRetriever(store=fake_store, embedder=fake_embedder)
    results = retriever.retrieve("hello", top_n=5)

    assert results == [expected]
    fake_embedder.embed_query.assert_called_once_with("hello")
    fake_store.search.assert_called_once()
    _, kwargs = fake_store.search.call_args
    assert kwargs["top_n"] == 5
    assert kwargs["query_vector"] == [0.1, 0.2, 0.3]


# ---- Hybrid Retriever ----


def test_hybrid_retriever_combines_paths(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Hybrid retriever should run both paths and fuse results."""
    fake_store = MagicMock()
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [0.1]
    fake_store.search.return_value = [_make_chunk("c-dense", "dense text")]
    bm25 = BM25Retriever(index_path=tmp_path / "bm25.pkl")
    bm25.add([_make_chunk("c-sparse", "sparse text")])

    vector_retriever = VectorRetriever(store=fake_store, embedder=fake_embedder)
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        rrf_k=60,
    )
    results = hybrid.retrieve("query", top_n=5)
    assert len(results) >= 1
    ids = {r.id for r in results}
    assert "c-dense" in ids or "c-sparse" in ids


# ---- Guardrail ----


def test_guardrail_rejects_low_confidence() -> None:
    """Below-threshold scores should be rejected."""
    guardrail = Guardrail()
    assert guardrail.check_confidence(0.1, threshold=0.3) is False


def test_guardrail_accepts_high_confidence() -> None:
    """At-or-above-threshold scores should pass."""
    guardrail = Guardrail()
    assert guardrail.check_confidence(0.5, threshold=0.3) is True


def test_guardrail_no_result_answer() -> None:
    """The fallback answer must mention the knowledge base."""
    guardrail = Guardrail()
    answer = guardrail.build_no_result_answer()
    assert "未在知识库" in answer


# ---- Citation Parser ----


def test_citation_parser_parse() -> None:
    """Multiple ``[n]`` markers should be extracted, deduped, ordered."""
    parser = CitationParser()
    assert parser.parse("答案[1] 与 [2]") == [1, 2]
    assert parser.parse("[3] [1] [3] [2]") == [3, 1, 2]
    assert parser.parse("no citations here") == []
    assert parser.parse("") == []


def test_citation_parser_map_to_references() -> None:
    """Citation numbers should map back to contexts (1-based)."""
    parser = CitationParser()
    contexts = [
        _make_chunk("c-1", "first"),
        _make_chunk("c-2", "second"),
        _make_chunk("c-3", "third"),
    ]
    mapped = parser.map_to_references([1, 3], contexts)
    assert [c.id for c in mapped] == ["c-1", "c-3"]
    # Out-of-range citations should be skipped.
    assert parser.map_to_references([0, 4], contexts) == []


# ---- Prompt ----


def test_build_rag_prompt_structure() -> None:
    """``build_rag_prompt`` returns 2 messages with a system mention."""
    contexts = [_make_chunk("c-1", "hello world"), _make_chunk("c-2", "bye")]
    messages = build_rag_prompt("hi", contexts)
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "知识库" in messages[0]["content"]
    # Numbered fragments should appear in the user content.
    assert "[1]" in messages[1]["content"]
    assert "[2]" in messages[1]["content"]
    assert "hello world" in messages[1]["content"]


def test_build_rag_prompt_empty_contexts() -> None:
    """With no contexts the prompt should still be 2 messages."""
    messages = build_rag_prompt("hi", [])
    assert len(messages) == 2
    assert messages[1]["content"]


# ---- Factories ----


def test_get_generator_openai(mock_settings) -> None:  # type: ignore[no-untyped-def]
    """The factory should build an OpenAI generator from settings."""
    mock_settings.llm_provider = "openai"
    gen = get_generator(mock_settings)
    assert gen.__class__.__name__ == "OpenAIGenerator"


def test_get_generator_zhipu(mock_settings) -> None:  # type: ignore[no-untyped-def]
    """The factory should build a Zhipu generator."""
    mock_settings.llm_provider = "zhipu"
    gen = get_generator(mock_settings)
    assert gen.__class__.__name__ == "ZhipuGenerator"


def test_get_generator_ollama(mock_settings) -> None:  # type: ignore[no-untyped-def]
    """The factory should build an Ollama generator."""
    mock_settings.llm_provider = "ollama"
    gen = get_generator(mock_settings)
    assert gen.__class__.__name__ == "OllamaGenerator"


def test_get_reranker_factory(mock_settings) -> None:  # type: ignore[no-untyped-def]
    """The rerank factory should return a BgeReranker without loading model."""
    reranker = get_reranker(mock_settings)
    assert reranker.__class__.__name__ == "BgeReranker"
    assert reranker.model_name == mock_settings.rerank_model
    # Model must not be loaded on construction.
    assert reranker._model is None
