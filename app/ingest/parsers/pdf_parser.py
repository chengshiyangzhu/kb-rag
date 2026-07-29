"""PDF parser using pypdf with a RapidOCR fallback for scanned pages.

For each page the parser first attempts direct text extraction with
``pypdf.PdfReader``. When the extracted text is empty or very short the page is
rendered to an image with ``pdf2image`` and passed through ``rapidocr_onnxruntime``.

Both OCR-only dependencies are imported lazily so the module can be imported on
machines where they (or their native backends such as poppler) are unavailable;
in that case the parser logs a warning and returns an empty string for the page
instead of raising.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Module-level OCR singleton. ``None`` = not initialised, ``False`` = unavailable,
# an instance = ready to use.
_OCR_INSTANCE: Any = None


def _get_ocr() -> Any:
    """Return a lazily-initialised :class:`RapidOCR` instance or ``None``.

    The import is performed on first use so the parser module itself can be
    imported without ``rapidocr_onnxruntime`` installed. If the dependency (or
    its ONNX runtime) is missing, a warning is logged and subsequent calls
    short-circuit to ``None``.

    Returns:
        A ``RapidOCR`` instance, or ``None`` if OCR is unavailable.
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _OCR_INSTANCE = RapidOCR()
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning(
                "rapidocr_onnxruntime not available; OCR fallback disabled",
                error=str(exc),
            )
            _OCR_INSTANCE = False
    return _OCR_INSTANCE if _OCR_INSTANCE is not False else None


class PdfParser(Parser):
    """Parse a ``.pdf`` file into one :class:`Document` per page.

    Pages whose direct text extraction yields fewer than 10 characters are
    forwarded to the OCR fallback. When OCR is unavailable the page text is
    left empty rather than raising.
    """

    MIN_TEXT_LEN = 10

    def parse(self, file_path: Path) -> list[Document]:
        """Extract text from each PDF page.

        Args:
            file_path: Path to the ``.pdf`` file.

        Returns:
            A list of :class:`Document` instances, one per page (1-indexed).
        """
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        file_doc_id = _make_doc_id(file_path, None)
        docs: list[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < self.MIN_TEXT_LEN:
                ocr_text = self._ocr_page(file_path, page_idx)
                if ocr_text:
                    text = ocr_text
            metadata = Metadata(
                source=file_path.name,
                page=page_idx,
                doc_id=file_doc_id,
            )
            docs.append(
                Document(
                    id=_make_doc_id(file_path, page_idx),
                    text=text,
                    metadata=metadata,
                )
            )
        return docs

    def _ocr_page(self, file_path: Path, page_idx: int) -> str:
        """Render a single page to an image and run OCR over it.

        Args:
            file_path: Path to the source PDF.
            page_idx: 1-indexed page number to render.

        Returns:
            The OCR-extracted text, or an empty string on any failure.
        """
        ocr = _get_ocr()
        if ocr is None:
            return ""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(file_path),
                first_page=page_idx,
                last_page=page_idx,
                dpi=200,
            )
        except Exception as exc:  # pragma: no cover - depends on poppler
            logger.warning(
                "pdf2image rendering failed; skipping OCR for page",
                page=page_idx,
                error=str(exc),
            )
            return ""
        texts: list[str] = []
        for img in images:
            try:
                result, _elapse = ocr(img)
            except Exception as exc:  # pragma: no cover - depends on OCR runtime
                logger.warning(
                    "OCR inference failed for page",
                    page=page_idx,
                    error=str(exc),
                )
                continue
            if result:
                texts.append("\n".join(line[1] for line in result))
        return "\n".join(texts).strip()
