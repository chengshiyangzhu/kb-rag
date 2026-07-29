"""Parser factory dispatching on file extension."""
from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.base import Parser
from app.ingest.parsers.docx_parser import DocxParser
from app.ingest.parsers.html_parser import HtmlParser
from app.ingest.parsers.markdown_parser import MarkdownParser
from app.ingest.parsers.pdf_parser import PdfParser
from app.ingest.parsers.txt_parser import TxtParser
from app.ingest.parsers.xlsx_parser import XlsxParser

#: Mapping of supported file extensions (lower-case, without dot) to parsers.
_PARSER_REGISTRY: dict[str, type[Parser]] = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "xlsx": XlsxParser,
    "xls": XlsxParser,
    "md": MarkdownParser,
    "markdown": MarkdownParser,
    "html": HtmlParser,
    "htm": HtmlParser,
    "txt": TxtParser,
}


def get_parser(file_path: Path) -> Parser:
    """Return a :class:`Parser` instance appropriate for the file extension.

    Args:
        file_path: Path to the file whose extension determines the parser.

    Returns:
        A concrete :class:`Parser` instance.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.suffix.lower().lstrip(".")
    parser_cls = _PARSER_REGISTRY.get(ext)
    if parser_cls is None:
        raise ValueError(
            f"Unsupported file extension: '{file_path.suffix}' for {file_path}"
        )
    return parser_cls()
