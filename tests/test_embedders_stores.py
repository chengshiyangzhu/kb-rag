"""Tests for the embedder and vector-store backends (Stage 3 + Stage 4)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.embedders import ApiEmbedder, Embedder, get_embedder
from app.models.document import Chunk, Metadata
from app.stores import ChromaStore, QdrantStore, VectorStore, get_vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, doc_id: str = "doc-1", chunk_id: str | None = None) -> Chunk:
    """Build a small :class:`Chunk` for tests."""
    return Chunk(
        id=chunk_id or f"chunk-{doc_id}-{text[:4]}",
        text=text,
        metadata=Metadata(
            source="test.md",
            doc_id=doc_id,
            chunk_index=0,
            created_at=datetime.now(timezone.utc),
        ),
    )


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


class _MockEmbeddingData:
    """Mimics ``openai.types.CreateEmbeddingResponse.data[i]``."""

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    """Mimics ``openai.types.CreateEmbeddingResponse``."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_MockEmbeddingData(v) for v in vectors]


def test_api_embedder_interface() -> None:
    """ApiEmbedder should return vectors of the configured dimension and pass
    the correct model/input to the underlying OpenAI client."""
    dim = 8
    fake_vectors = [[float(i) for i in range(dim)], [float(i + 1) for i in range(dim)]]
    mock_create = MagicMock(return_value=_MockEmbeddingResponse(fake_vectors))
    mock_embeddings = MagicMock(create=mock_create)
    mock_client = MagicMock(embeddings=mock_embeddings)

    with patch("app.embedders.api_embedder.OpenAI", return_value=mock_client):
        embedder = ApiEmbedder(
            provider="openai",
            api_key="fake-key",
            model="text-embedding-3-small",
            dim=dim,
        )

    # Interface conformance
    assert isinstance(embedder, Embedder)
    assert embedder.dim == dim

    texts = ["hello world", "second doc"]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == dim
    assert vectors == fake_vectors

    # The mock should have been called with the right input + model
    mock_create.assert_called_once_with(input=texts, model="text-embedding-3-small")

    # embed_query goes through the same path and returns a single vector
    mock_create.return_value = _MockEmbeddingResponse([fake_vectors[0]])
    single = embedder.embed_query("query")
    assert len(single) == dim


def test_get_embedder_factory_api_zhipu() -> None:
    """When zhipu_api_key is set, the factory should choose the zhipu provider."""
    settings = MagicMock()
    settings.embedder_provider = "api"
    settings.embedder_model = "BAAI/bge-m3"
    settings.embedder_dim = 1024
    settings.ollama_base_url = "http://ollama:11434"
    settings.openai_api_key = "openai-key"
    settings.openai_base_url = "https://api.openai.com/v1"
    settings.zhipu_api_key = "zhipu-key"
    settings.zhipu_base_url = "https://open.bigmodel.cn/api/paas/v4"

    with patch("app.embedders.api_embedder.OpenAI", return_value=MagicMock()):
        embedder = get_embedder(settings)

    assert isinstance(embedder, ApiEmbedder)
    assert embedder.dim == 1024


# ---------------------------------------------------------------------------
# Chroma store tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def chroma_store(tmp_path: Any) -> ChromaStore:
    """Build a fresh ChromaStore rooted at a tmp directory."""
    return ChromaStore(
        path=str(tmp_path / "chroma"),
        collection_name="kb_rag_test",
        dim=4,
    )


def test_chroma_store_roundtrip(chroma_store: ChromaStore) -> None:
    """End-to-end Chroma roundtrip: upsert, search, count, delete."""
    dim = 4
    chunks = [
        _make_chunk("alpha chunk", doc_id="doc-A", chunk_id="c-alpha"),
        _make_chunk("beta chunk", doc_id="doc-A", chunk_id="c-beta"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    chroma_store.upsert(chunks, vectors)
    assert chroma_store.count() == 2

    results = chroma_store.search([1.0, 0.0, 0.0, 0.0], top_n=2)
    assert len(results) >= 1
    assert all(isinstance(c, Chunk) for c in results)
    # Top hit should match the alpha chunk
    assert results[0].id == "c-alpha"

    # Filter by doc_id
    filtered = chroma_store.search([1.0, 0.0, 0.0, 0.0], top_n=5, filters={"doc_id": "doc-A"})
    assert len(filtered) == 2

    # Delete by doc and verify count drops to 0
    chroma_store.delete_by_doc("doc-A")
    assert chroma_store.count() == 0

    # Delete by chunk_ids is a no-op on an empty collection
    chroma_store.delete_by_chunk_ids(["c-alpha", "c-beta"])
    assert chroma_store.count() == 0


def test_get_vector_store_factory_chroma(tmp_path: Any) -> None:
    """The factory should return a ChromaStore when ``vector_store == 'chroma'``."""
    settings = MagicMock()
    settings.vector_store = "chroma"
    settings.chroma_path = str(tmp_path / "chroma_factory")
    settings.qdrant_collection = "kb_rag_test"
    settings.qdrant_url = "http://qdrant:6333"
    settings.embedder_dim = 4

    store = get_vector_store(settings)
    assert isinstance(store, ChromaStore)
    assert isinstance(store, VectorStore)


# ---------------------------------------------------------------------------
# Qdrant store tests (mocked client)
# ---------------------------------------------------------------------------


def test_qdrant_store_interface() -> None:
    """QdrantStore should translate upsert/search into the right client calls."""
    dim = 4
    mock_client = MagicMock()
    # _ensure_collection calls get_collection (raises) then create_collection
    mock_client.get_collection.side_effect = RuntimeError("not found")

    with patch("app.stores.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(url="http://qdrant:6333", collection_name="kb_rag_test", dim=dim)

    # Collection was created with the right vector config
    mock_client.get_collection.assert_called_once_with("kb_rag_test")
    mock_client.create_collection.assert_called_once()
    create_kwargs = mock_client.create_collection.call_args.kwargs
    assert create_kwargs["collection_name"] == "kb_rag_test"
    vectors_config = create_kwargs["vectors_config"]
    assert vectors_config.size == dim
    assert vectors_config.distance is not None  # Distance.COSINE

    # Upsert
    chunks = [
        _make_chunk("hello", doc_id="doc-1", chunk_id="chunk-1"),
        _make_chunk("world", doc_id="doc-1", chunk_id="chunk-2"),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.upsert(chunks, vectors)

    mock_client.upsert.assert_called_once()
    upsert_kwargs = mock_client.upsert.call_args.kwargs
    assert upsert_kwargs["collection_name"] == "kb_rag_test"
    points = upsert_kwargs["points"]
    assert len(points) == 2
    assert all(p.vector == list(v) for p, v in zip(points, vectors, strict=True))
    # payload must include the required keys
    payload = points[0].payload
    for key in ("chunk_id", "source", "page", "sheet", "tag", "created_at", "doc_id", "text"):
        assert key in payload, f"missing payload key: {key}"

    # Search
    mock_client.search.return_value = []
    store.search([1.0, 0.0, 0.0, 0.0], top_n=5)
    mock_client.search.assert_called_once()
    search_kwargs = mock_client.search.call_args.kwargs
    assert search_kwargs["collection_name"] == "kb_rag_test"
    assert search_kwargs["limit"] == 5
    assert search_kwargs["query_vector"] == [1.0, 0.0, 0.0, 0.0]

    # count
    mock_client.count.return_value = MagicMock(count=7)
    assert store.count() == 7

    # delete_by_doc uses a Filter selector
    store.delete_by_doc("doc-1")
    delete_kwargs = mock_client.delete.call_args.kwargs
    assert delete_kwargs["collection_name"] == "kb_rag_test"
    selector = delete_kwargs["points_selector"]
    # Filter must target the doc_id field
    assert any(fc.key == "doc_id" for fc in selector.must)

    # delete_by_chunk_ids uses a PointIdsList selector
    mock_client.reset_mock()
    store.delete_by_chunk_ids(["chunk-1", "chunk-2"])
    assert mock_client.delete.called
    selector = mock_client.delete.call_args.kwargs["points_selector"]
    assert hasattr(selector, "points")
    assert len(selector.points) == 2


def test_qdrant_search_with_filters() -> None:
    """The Filter builder should translate source/tag/doc_id/time_range correctly."""
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = RuntimeError("not found")

    with patch("app.stores.qdrant_store.QdrantClient", return_value=mock_client):
        store = QdrantStore(url="http://qdrant:6333", collection_name="kb_rag_test", dim=4)

    mock_client.search.return_value = []
    store.search(
        [0.1, 0.2, 0.3, 0.4],
        top_n=10,
        filters={
            "source": ["a.md", "b.md"],
            "tag": "important",
            "doc_id": "doc-1",
            "time_range": {"gte": "2024-01-01T00:00:00+00:00", "lte": "2024-12-31T23:59:59+00:00"},
        },
    )
    search_kwargs = mock_client.search.call_args.kwargs
    flt = search_kwargs["query_filter"]
    keys = {fc.key for fc in flt.must}
    assert {"source", "tag", "doc_id", "created_at"}.issubset(keys)
