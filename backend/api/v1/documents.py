"""GET/DELETE ``/api/v1/documents`` endpoints with a JSON-backed doc registry.

The :class:`DocRegistry` persists a list of ingested document metadata to
``data/processed/doc_registry.json`` so the API can list and delete documents
without modifying :mod:`app.stores`.  All file operations are guarded by a
class-level :class:`threading.Lock` to remain safe under concurrent requests.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.observability.logging import get_logger
from app.pipeline import IngestPipeline
from backend import get_ingest_pipeline_dep
from backend.schemas import DocumentOut

logger = get_logger(__name__)

router = APIRouter()

REGISTRY_PATH = Path("data/processed/doc_registry.json")


class DocRegistry:
    """Thread-safe JSON-backed document registry.

    Stores a list of ingested document metadata at the configured path.  All
    read/write operations are serialized via a class-level lock so concurrent
    requests do not corrupt the file.

    Args:
        path: Optional override for the registry file location (used in tests).
    """

    _lock: threading.Lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the registry with an optional path override.

        Args:
            path: Path to the registry JSON file.  Defaults to
                :data:`REGISTRY_PATH`.
        """
        self.path = path or REGISTRY_PATH

    # ------------------------------------------------------------------
    # Internal I/O (callers must already hold the lock)
    # ------------------------------------------------------------------
    def _read(self) -> list[dict[str, Any]]:
        """Read and return the registry list (empty if missing/corrupt)."""
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            logger.warning("registry.read.failed", error=str(exc), path=str(self.path))
        return []

    def _write(self, docs: list[dict[str, Any]]) -> None:
        """Write the registry list to disk, creating parent dirs as needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(docs, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    def list_all(self) -> list[dict[str, Any]]:
        """Return a copy of all registered documents."""
        with self._lock:
            return self._read()

    def add(self, doc_id: str, file_type: str, num_chunks: int) -> None:
        """Register a newly ingested document.

        Args:
            doc_id: Identifier of the document.
            file_type: File extension (without dot) of the source file.
            num_chunks: Number of chunks produced for the document.
        """
        with self._lock:
            docs = self._read()
            docs.append(
                {
                    "doc_id": doc_id,
                    "file_type": file_type,
                    "num_chunks": num_chunks,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._write(docs)

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the registry.

        Args:
            doc_id: Identifier of the document to remove.

        Returns:
            ``True`` if the document was found and removed, ``False`` otherwise.
        """
        with self._lock:
            docs = self._read()
            new_docs = [d for d in docs if d.get("doc_id") != doc_id]
            removed = len(new_docs) < len(docs)
            if removed:
                self._write(new_docs)
            return removed


def get_doc_registry() -> DocRegistry:
    """Return a :class:`DocRegistry` instance (overridable in tests)."""
    return DocRegistry()


@router.get(
    "",
    response_model=list[DocumentOut],
    summary="List ingested documents",
)
def list_documents(
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> list[DocumentOut]:
    """Return the list of documents recorded in the doc registry."""
    docs = registry.list_all()
    return [
        DocumentOut(
            doc_id=d["doc_id"],
            file_type=d.get("file_type", ""),
            num_chunks=d.get("num_chunks", 0),
            ingested_at=d.get("ingested_at", ""),
        )
        for d in docs
    ]


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an ingested document",
)
def delete_document(
    doc_id: str,
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> None:
    """Delete a document and all its chunks from every store and the registry.

    Args:
        doc_id: Identifier of the document to remove.
    """
    pipeline.delete_document(doc_id)
    registry.remove(doc_id)
