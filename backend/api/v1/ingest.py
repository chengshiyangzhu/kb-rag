"""``POST /api/v1/ingest`` endpoint for uploading and ingesting documents."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.observability.logging import get_logger
from app.pipeline import IngestPipeline
from backend import get_ingest_pipeline_dep
from backend.api.v1.documents import DocRegistry, get_doc_registry
from backend.schemas import IngestResponse

logger = get_logger(__name__)

router = APIRouter()

# Whitelist of supported file extensions (lowercase, no leading dot).
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {"pdf", "docx", "xlsx", "xls", "md", "html", "htm", "txt"}
)

# Maximum accepted upload size (50 MB).
MAX_FILE_SIZE: int = 50 * 1024 * 1024


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document file",
)
def ingest_file(
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    pipeline: Annotated[IngestPipeline, Depends(get_ingest_pipeline_dep)],
    registry: Annotated[DocRegistry, Depends(get_doc_registry)],
) -> IngestResponse:
    """Ingest a single document file into the RAG knowledge base.

    The uploaded file is saved to ``data/raw/{uuid}_{filename}`` and then
    processed end-to-end by the ingest pipeline.  Supported types: pdf, docx,
    xlsx, xls, md, html, htm, txt.  Maximum file size: 50 MB.  Unknown
    extensions are rejected with HTTP 400.
    """
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type: .{extension}",
        )

    # Read content and enforce size limit.
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large: {len(content)} bytes (max {MAX_FILE_SIZE})",
        )

    # Persist to data/raw/{uuid}_{filename}
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = raw_dir / saved_name
    saved_path.write_bytes(content)
    logger.info("ingest.saved", path=str(saved_path), size=len(content))

    # Run the ingest pipeline.
    result = pipeline.ingest_file(saved_path)

    # Record in the doc registry.
    registry.add(
        doc_id=result.doc_id,
        file_type=result.file_type,
        num_chunks=result.num_chunks,
    )

    return IngestResponse(
        doc_id=result.doc_id,
        num_chunks=result.num_chunks,
        file_type=result.file_type,
        trace_id=result.trace_id,
        errors=list(result.errors),
    )
