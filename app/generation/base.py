"""Abstract generator interface and shared result model."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.models.document import Chunk


class GenerationResult(BaseModel):
    """Output of a generation call.

    Attributes:
        answer: Generated natural-language answer.
        citations: 1-based citation numbers extracted from the answer.
        used_chunk_ids: Ids of the chunks that contributed to the answer.
    """

    answer: str
    citations: list[int] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)


class Generator(ABC):
    """Abstract base class for LLM-backed answer generators."""

    @abstractmethod
    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Generate an answer grounded in ``contexts`` for ``query``.

        Args:
            query: The user query.
            contexts: Reranked chunks to ground the answer.

        Returns:
            A :class:`GenerationResult` with the answer, extracted citations
            and the list of used chunk ids.
        """
        raise NotImplementedError
