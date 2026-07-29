"""Parser abstract base class and shared utilities.

All concrete parsers implement the :class:`Parser` ABC. The shared helpers
``_make_doc_id`` and ``_now_utc`` provide deterministic identifiers and
timestamps used across the ingestion stage.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from app.models.document import Document


class Parser(ABC):
    """Abstract base class for all document parsers.

    A parser converts a single file on disk into one or more
    :class:`~app.models.document.Document` instances. Subclasses must implement
    :meth:`parse`.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> list[Document]:
        """Parse a file into a list of Documents.

        Args:
            file_path: Path to the file to parse.

        Returns:
            A list of :class:`Document` instances extracted from the file.
        """
        raise NotImplementedError


def _make_doc_id(file_path: Path, page_idx: int | None = None) -> str:
    """Build a deterministic 12-char hex identifier from a file and page index.

    The identifier is derived from the SHA-1 hash of the file's binary content
    combined with the optional page index. When ``page_idx`` is ``None`` the
    returned id is file-level (shared by every page/sheet of the file); when
    provided, the id becomes unique to that page/sheet.

    Args:
        file_path: Path to the source file.
        page_idx: Optional page/sheet index used to disambiguate sub-documents.

    Returns:
        The first 12 characters of the SHA-1 hex digest.
    """
    hasher = hashlib.sha1()
    try:
        with file_path.open("rb") as fh:
            hasher.update(fh.read())
    except OSError:
        # Fall back to hashing the path string if the file cannot be read.
        hasher.update(str(file_path).encode("utf-8"))
    if page_idx is not None:
        hasher.update(f":p{page_idx}".encode("utf-8"))
    return hasher.hexdigest()[:12]


def _now_utc() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        A timezone-aware :class:`~datetime.datetime` in UTC.
    """
    return datetime.now(timezone.utc)
