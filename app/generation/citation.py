"""Citation parsing and mapping utilities.

The :class:`CitationParser` extracts ``[n]`` citation markers from the LLM
answer and maps them back to the :class:`Chunk` instances that were supplied
to the prompt (1-based indexing, matching :func:`build_rag_prompt`).
"""
from __future__ import annotations

import re

from app.models.document import Chunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationParser:
    """Parse ``[n]`` citation markers and map them to source chunks."""

    def parse(self, answer: str) -> list[int]:
        """Extract unique citation numbers from ``answer``, preserving order.

        Args:
            answer: LLM-generated answer potentially containing ``[n]``
                citation markers.

        Returns:
            A deduplicated, order-preserving list of 1-based citation ints.
        """
        if not answer:
            return []
        seen: set[int] = set()
        result: list[int] = []
        for match in _CITATION_RE.finditer(answer):
            num = int(match.group(1))
            if num in seen:
                continue
            seen.add(num)
            result.append(num)
        return result

    def map_to_references(
        self,
        citations: list[int],
        contexts: list[Chunk],
    ) -> list[Chunk]:
        """Map 1-based citation numbers back to ``contexts``.

        Args:
            citations: Citation numbers (1-based) extracted from the answer.
            contexts: The chunks that were passed to the prompt, in the same
                order (so citation ``n`` maps to ``contexts[n-1]``).

        Returns:
            The subset of ``contexts`` referenced by ``citations``, in
            citation order. Out-of-range citations are skipped.
        """
        result: list[Chunk] = []
        for num in citations:
            if num <= 0 or num > len(contexts):
                continue
            result.append(contexts[num - 1])
        return result
