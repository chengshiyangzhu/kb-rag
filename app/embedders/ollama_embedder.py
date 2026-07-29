"""Ollama-hosted embedder using the ``/api/embeddings`` REST endpoint."""
from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


class OllamaEmbedder(Embedder):
    """Embedder that calls an Ollama server's ``/api/embeddings`` endpoint.

    Args:
        base_url: Base URL of the Ollama service, e.g. ``"http://ollama:11434"``.
        model: Ollama model name, defaults to ``"bge-m3"``.
        dim: Vector dimensionality returned by the model.
        timeout: HTTP request timeout in seconds.
        batch_size: Maximum number of concurrent in-flight requests when
            embedding multiple texts.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        dim: int = 1024,
        timeout: float = 30.0,
        batch_size: int = 8,
    ) -> None:
        """Initialize the HTTP client and configuration."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = int(dim)
        self._timeout = float(timeout)
        self._batch_size = int(batch_size)
        self._client = httpx.Client(timeout=self._timeout)

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> list[float]:
        """Call the Ollama embeddings endpoint for a single text."""
        url = f"{self._base_url}/api/embeddings"
        resp = self._client.post(url, json={"model": self._model, "prompt": prompt})
        resp.raise_for_status()
        payload = resp.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError("ollama embeddings response missing 'embedding' field")
        return [float(x) for x in embedding]

    def embed_texts(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed a batch of documents sequentially (with retry per call).

        Args:
            texts: List of raw text strings.
            batch_size: Optional override for the configured batch size. Kept
                for API compatibility; concurrent embedding is not used to
                avoid overwhelming the Ollama server.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        results: list[list[float]] = []
        for text in texts:
            results.append(self._call_api(text))
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api(text)
