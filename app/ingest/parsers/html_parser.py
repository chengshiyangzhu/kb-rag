"""HTML parser using BeautifulSoup."""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


class HtmlParser(Parser):
    """Parse an ``.html``/``.htm`` file into a single :class:`Document`.

    The parser strips ``<script>`` and ``<style>`` elements and extracts the
    visible text of the ``<body>`` (falling back to the whole document when no
    ``<body>`` tag is present).
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Extract visible body text from the HTML file.

        Args:
            file_path: Path to the HTML file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        container = soup.body if soup.body is not None else soup
        # Remove non-content elements before extracting text.
        for tag in container(["script", "style"]):
            tag.decompose()
        text = container.get_text(separator="\n").strip()
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
