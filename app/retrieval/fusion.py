"""Reciprocal Rank Fusion (RRF) for combining multiple result lists.

RRF assigns each candidate a score of ``1 / (k + rank)`` from every list it
appears in, then sums the contributions. It is a robust, parameter-light way
to fuse dense and sparse retrievers.
"""
from __future__ import annotations

from app.models.document import Chunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class RRFFusion:
    """Fuse multiple ranked chunk lists via Reciprocal Rank Fusion."""

    def fuse(
        self,
        result_lists: list[list[Chunk]],
        k: int = 60,
        top_n: int = 20,
    ) -> list[Chunk]:
        """Fuse ranked lists with RRF and return the top-N chunks.

        Args:
            result_lists: A list of ranked chunk lists. Within each list,
                rank 0 is the most relevant item.
            k: RRF smoothing constant (default 60, per the original paper).
            top_n: Maximum number of fused chunks to return.

        Returns:
            A list of :class:`Chunk` sorted by summed RRF score descending,
            with ``score`` set to the fused score. Chunks sharing the same
            ``id`` are merged across lists.
        """
        scores: dict[str, float] = {}
        # Keep one representative Chunk per id (the first occurrence).
        representatives: dict[str, Chunk] = {}

        for result_list in result_lists:
            for rank, chunk in enumerate(result_list):
                cid = chunk.id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in representatives:
                    representatives[cid] = chunk

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        results: list[Chunk] = []
        for cid in ranked_ids[:top_n]:
            chunk = representatives[cid].model_copy(deep=True)
            chunk.score = scores[cid]
            results.append(chunk)
        logger.info(
            "rrf.fuse.done",
            input_lists=len(result_lists),
            output=len(results),
            top_score=results[0].score if results else 0.0,
        )
        return results
