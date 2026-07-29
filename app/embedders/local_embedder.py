"""Local sentence-transformers embedder (in-process)."""
from __future__ import annotations

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


class LocalEmbedder(Embedder):
    """Embedder backed by ``sentence-transformers.SentenceTransformer``.

    Suitable for self-hosted deployments where the model weights can be loaded
    into the process. Defaults to ``BAAI/bge-m3``.

    Note on bge-m3:
        Unlike some other BGE checkpoints (e.g. ``bge-large-zh``), ``bge-m3``
        does **not** require query/document instruction prefixes. Both indexed
        documents and user queries are encoded directly without any prefix
        transformation.

    Args:
        model_name: HuggingFace model id. Defaults to ``"BAAI/bge-m3"``.
        dim: Vector dimensionality (used for the abstract :attr:`dim` property
            and to cross-check the loaded model when possible).
        batch_size: Default mini-batch size passed to ``SentenceTransformer.encode``.
        device: Torch device, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dim: int = 1024,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        """Initialize and eagerly load the SentenceTransformer model."""
        self._model_name = model_name
        self._dim = int(dim)
        self._batch_size = int(batch_size)
        self._device = device
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001 - re-raise after logging
            logger.error("failed to load local embedder model", model=model_name, error=str(exc))
            raise

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents using L2-normalized vectors.

        Args:
            texts: List of raw text strings.

        Returns:
            List of normalized embedding vectors (one per input text).
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        bge-m3 requires no query prefix, so the query is encoded verbatim,
        identical to how documents are encoded.
        """
        return self.embed_texts([text])[0]
