"""BGE-based cross-encoder reranker.

Prefers :class:`FlagEmbedding.FlagReranker` and transparently falls back to
:class:`sentence_transformers.CrossEncoder` when FlagEmbedding is unavailable.
The model is loaded lazily on the first ``rerank`` call so the module can be
imported in environments without the heavy ML dependencies.
"""
from __future__ import annotations

import time
from typing import Any

from app.models.document import Chunk
from app.observability.logging import get_logger
from app.rerank.base import Reranker

logger = get_logger(__name__)


class BgeReranker(Reranker):
    """Cross-encoder reranker backed by BGE-reranker-v2-m3.

    Args:
        model_name: Hugging Face model id (default
            ``"BAAI/bge-reranker-v2-m3"``).
        use_fp16: Forwarded to ``FlagReranker`` to enable half-precision
            inference. Ignored when falling back to CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = False,
    ) -> None:
        """Initialize the reranker without loading the model yet."""
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model: Any | None = None
        self._backend: str | None = None

    # ---- Model loading ----

    def _load_model(self) -> None:
        """Lazily load the reranker model.

        Tries FlagEmbedding first, then falls back to sentence-transformers.
        Raises ``ImportError`` if neither is available.
        """
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
            self._backend = "flag_embedding"
            logger.info("reranker.load.flag_embedding", model=self.model_name)
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "reranker.flag_embedding.unavailable",
                error=str(exc),
            )
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self.model_name)
            self._backend = "sentence_transformers"
            logger.info("reranker.load.sentence_transformers", model=self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.error("reranker.load.failed", error=str(exc))
            raise ImportError(
                "Neither FlagEmbedding nor sentence-transformers is available; "
                "install one of them to use BgeReranker."
            ) from exc

    # ---- Public API ----

    def rerank(
        self,
        query: str,
        candidates: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Score each candidate against the query and return the top-K.

        Args:
            query: The user query.
            candidates: Chunks to re-score.
            top_k: Maximum number of chunks to return.

        Returns:
            Top-K chunks sorted by rerank score descending. The returned
            :class:`Chunk` instances have ``score`` set to the rerank score.
        """
        if not candidates:
            return []
        self._load_model()
        start = time.perf_counter()
        pairs = [(query, c.text) for c in candidates]
        scores = self._compute_scores(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=False),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        results: list[Chunk] = []
        for chunk, score in ranked[:top_k]:
            new_chunk = chunk.model_copy(deep=True)
            new_chunk.score = float(score)
            results.append(new_chunk)
        logger.info(
            "rerank.done",
            candidates=len(candidates),
            top_k=top_k,
            backend=self._backend,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )
        return results

    # ---- Internal ----

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute rerank scores for query/text pairs.

        Args:
            pairs: List of ``(query, text)`` tuples.

        Returns:
            A list of float scores, one per pair. FlagReranker applies
            sigmoid internally; CrossEncoder outputs are passed through a
            sigmoid to normalize them to ``[0, 1]``.
        """
        assert self._model is not None and self._backend is not None
        if self._backend == "flag_embedding":
            # FlagReranker.compute_score already applies sigmoid.
            raw = self._model.compute_score(pairs, normalize=True)
            if isinstance(raw, float):
                return [float(raw)]
            return [float(s) for s in raw]
        # CrossEncoder returns logits; apply sigmoid for normalization.
        raw = self._model.predict(pairs)
        import numpy as np  # local import keeps module import light

        probs = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
        return [float(p) for p in probs]
