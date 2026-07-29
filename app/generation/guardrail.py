"""Hallucination guardrail for the generation stage.

The :class:`Guardrail` decides whether the top retrieval score is high
enough to trust the LLM with answering, and provides a fixed fallback
answer when retrieval yields nothing relevant.
"""
from __future__ import annotations

from app.observability.logging import get_logger
from app.observability.metrics import record_no_result

logger = get_logger(__name__)

NO_RESULT_ANSWER = "未在知识库中找到相关内容。"


class Guardrail:
    """Confidence-based guardrail for the RAG generation stage."""

    def check_confidence(self, top_score: float, threshold: float = 0.3) -> bool:
        """Return ``True`` if the top retrieval score clears ``threshold``.

        Args:
            top_score: Best score from the retrieval/rerank stage.
            threshold: Minimum score required to attempt generation.

        Returns:
            ``True`` when ``top_score >= threshold``; ``False`` (and a
            no-result counter increment) otherwise.
        """
        if top_score is None or top_score < threshold:
            logger.info(
                "guardrail.reject",
                top_score=top_score,
                threshold=threshold,
            )
            record_no_result()
            return False
        return True

    def build_no_result_answer(self) -> str:
        """Return the canonical answer used when retrieval fails."""
        return NO_RESULT_ANSWER
