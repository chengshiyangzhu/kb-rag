"""API tests for the FastAPI backend (Stage 9).

Uses :class:`fastapi.testclient.TestClient` with mocked pipelines and a
temporary working directory so no real model loading or network access is
required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.pipeline import IngestResult, QueryResult, Reference
from backend import (
    get_ingest_pipeline_dep,
    get_query_pipeline_dep,
    get_vector_store_dep,
)
from backend.api.v1.documents import DocRegistry
from backend.main import app


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient wrapping the kb-rag FastAPI app."""
    return TestClient(app)


@pytest.fixture()
def mock_vector_store() -> MagicMock:
    """Return a mock VectorStore whose ``count()`` returns 42."""
    store = MagicMock()
    store.count.return_value = 42
    return store


@pytest.fixture()
def mock_ingest_pipeline() -> MagicMock:
    """Return a mock IngestPipeline returning a canned IngestResult."""
    pipeline = MagicMock()
    pipeline.ingest_file.return_value = IngestResult(
        doc_id="doc-123",
        num_chunks=5,
        file_type="txt",
        trace_id="trace-abc",
        errors=[],
    )
    pipeline.delete_document.return_value = None
    return pipeline


@pytest.fixture()
def mock_query_pipeline() -> MagicMock:
    """Return a mock QueryPipeline returning a canned QueryResult."""
    pipeline = MagicMock()
    pipeline.query.return_value = QueryResult(
        answer="The answer is 42.",
        references=[
            Reference(
                chunk_id="c1",
                source="doc.txt",
                page=1,
                score=0.95,
                snippet="hello world",
            )
        ],
        trace_id="trace-query",
        no_result=False,
        retrieval_latency=0.05,
        generation_latency=0.1,
    )
    return pipeline


@pytest.fixture(autouse=True)
def _override_deps(
    mock_vector_store: MagicMock,
    mock_ingest_pipeline: MagicMock,
    mock_query_pipeline: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override FastAPI deps with mocks and chdir into a temp directory.

    The chdir ensures ``data/raw`` and ``data/processed`` (used by the ingest
    endpoint and the doc registry) are created inside ``tmp_path`` rather than
    polluting the repository working tree.
    """
    monkeypatch.chdir(tmp_path)
    app.dependency_overrides[get_ingest_pipeline_dep] = lambda: mock_ingest_pipeline
    app.dependency_overrides[get_query_pipeline_dep] = lambda: mock_query_pipeline
    app.dependency_overrides[get_vector_store_dep] = lambda: mock_vector_store
    yield
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    """GET /health returns 200 with status='ok' and chunk count."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunks"] == 42
    assert data["version"] == "0.1.0"


def test_docs_accessible(client: TestClient) -> None:
    """GET /docs returns 200 (Swagger UI is reachable)."""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_ingest_txt(client: TestClient) -> None:
    """POST /api/v1/ingest with a .txt file returns 201 with the result."""
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["doc_id"] == "doc-123"
    assert data["num_chunks"] == 5
    assert data["file_type"] == "txt"
    assert data["trace_id"] == "trace-abc"
    assert data["errors"] == []


def test_query(client: TestClient) -> None:
    """POST /api/v1/query returns the generated answer and references."""
    resp = client.post(
        "/api/v1/query",
        json={"question": "what is the answer?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The answer is 42."
    assert data["trace_id"] == "trace-query"
    assert data["no_result"] is False
    assert len(data["references"]) == 1
    assert data["references"][0]["chunk_id"] == "c1"
    assert data["references"][0]["source"] == "doc.txt"


def test_ingest_bad_extension(client: TestClient) -> None:
    """POST /api/v1/ingest with a .exe file returns 400."""
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_documents_list_and_delete(client: TestClient) -> None:
    """GET /api/v1/documents lists docs; DELETE removes one and returns 204."""
    # Seed the registry (uses the default path under tmp_path due to chdir).
    registry = DocRegistry()
    registry.add(doc_id="d1", file_type="txt", num_chunks=3)
    registry.add(doc_id="d2", file_type="md", num_chunks=2)

    # List
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) == 2
    assert {d["doc_id"] for d in docs} == {"d1", "d2"}

    # Delete d1
    resp = client.delete("/api/v1/documents/d1")
    assert resp.status_code == 204

    # List again: only d2 remains
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "d2"
