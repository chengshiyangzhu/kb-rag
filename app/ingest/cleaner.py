"""Text cleaning utilities for the ingestion stage.

The cleaner normalises text extracted from heterogeneous file formats so
downstream chunkers and embedders operate on consistent input. It handles:

* removal of control characters,
* normalisation of full-width (CJK) punctuation to half-width equivalents,
* collapsing excessive blank lines into paragraph breaks,
* removing spurious in-word line breaks while preserving paragraph boundaries,
* collapsing runs of spaces into a single space.
"""
from __future__ import annotations

import re
import unicodedata

from app.models.document import Document

#: Mapping of common full-width (CJK) punctuation to their half-width
#: equivalents. This is intentionally a basic, explicit mapping rather than a
#: full-width-to-ASCII fold so semantic symbols (e.g. ``……``) remain readable.
_PUNCT_MAP: dict[str, str] = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "「": "'",
    "」": "'",
    "『": '"',
    "』": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "、": ",",
    "～": "~",
}

# Pre-compiled regexes used by :func:`clean_text`.
_RE_CRLF = re.compile(r"\r\n")
_RE_CR = re.compile(r"\r")
_RE_MANY_NEWLINES = re.compile(r"\n{3,}")
# A single \n flanked by non-space characters is a broken in-word line break.
_RE_MIDWORD_NL = re.compile(r"(?<=\S)\n(?=\S)")
# A single \n with surrounding spaces should collapse to a single space.
_RE_NL_WITH_SPACES = re.compile(r"(?<=\S)\n[ ]+(?=\S)")
# Trailing spaces before a newline.
_RE_SPACES_BEFORE_NL = re.compile(r"[ ]+\n")
# Leading spaces after a newline.
_RE_NL_THEN_SPACES = re.compile(r"\n[ ]+")
# Runs of two or more spaces.
_RE_MANY_SPACES = re.compile(r"[ ]{2,}")


def clean_text(text: str) -> str:
    """Normalise whitespace and punctuation in a text string.

    Args:
        text: Raw text potentially containing control characters, full-width
            punctuation, excessive blank lines, broken in-word line breaks, and
            runs of spaces.

    Returns:
        Cleaned text with paragraph breaks (``\\n\\n``) preserved, in-word line
        breaks removed, and runs of spaces collapsed to a single space.
    """
    if not text:
        return text

    # 1. Strip control characters (category Cc) except \n, \r, \t.
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\r", "\t") or unicodedata.category(ch) != "Cc"
    )

    # 2. Normalise line endings to \n.
    text = _RE_CRLF.sub("\n", text)
    text = _RE_CR.sub("\n", text)

    # 3. Normalise full-width punctuation to half-width.
    for full, half in _PUNCT_MAP.items():
        if full in text:
            text = text.replace(full, half)

    # 4. Collapse 3+ consecutive newlines into a paragraph break.
    text = _RE_MANY_NEWLINES.sub("\n\n", text)

    # 5. Remove in-word single newlines (keep \n\n paragraph breaks intact).
    text = _RE_MIDWORD_NL.sub("", text)

    # 6. Replace newlines surrounded by spaces with a single space.
    text = _RE_NL_WITH_SPACES.sub(" ", text)

    # 7. Trim spaces around newlines.
    text = _RE_SPACES_BEFORE_NL.sub("\n", text)
    text = _RE_NL_THEN_SPACES.sub("\n", text)

    # 8. Collapse runs of spaces into one.
    text = _RE_MANY_SPACES.sub(" ", text)

    # 9. Strip trailing whitespace on each line and overall.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def clean_document(doc: Document) -> Document:
    """Return a copy of ``doc`` whose text has been cleaned.

    Args:
        doc: The source :class:`Document`.

    Returns:
        A new :class:`Document` with cleaned text and identical metadata.
    """
    cleaned = clean_text(doc.text)
    return doc.model_copy(update={"text": cleaned})


def clean_documents(docs: list[Document]) -> list[Document]:
    """Clean a batch of documents.

    Args:
        docs: List of :class:`Document` instances to clean.

    Returns:
        A new list of cleaned :class:`Document` instances (same order, same
        length).
    """
    return [clean_document(doc) for doc in docs]
