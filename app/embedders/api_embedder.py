"""OpenAI-protocol API embedder (OpenAI / Zhipu BigModel)."""
from __future__ import annotations

from typing import Literal

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.embedders.base import Embedder
from app.observability.logging import get_logger

logger = get_logger(__name__)

Provider = Literal["openai", "zhipu"]

_DEFAULTS: dict[str, dict[str, object]] = {
    "openai": {"model": "text-embedding-3-small", "dim": 1536, "base_url": "https://api.openai.com/v1"},
    "zhipu": {"model": "embedding-3", "dim": 1024, "base_url": "https://open.bigmodel.cn/api/paas/v4"},
}


class ApiEmbedder(Embedder):
    """Embedder that talks to an OpenAI-compatible embeddings endpoint.

    Both OpenAI and Zhipu BigModel expose the same ``/embeddings`` REST shape,
    so a single :class:`openai.OpenAI` client is reused; only the ``base_url``,
    ``api_key`` and default model differ.

    Args:
        provider: ``"openai"`` or ``"zhipu"``.
        api_key: API key for the chosen provider.
        base_url: Base URL override. Falls back to the provider default.
        model: Model name override. Falls back to the provider default.
        dim: Vector dimensionality override. Falls back to the provider default.
    """

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        """Initialize the OpenAI client for the chosen provider."""
        if provider not in _DEFAULTS:
            raise ValueError(f"unsupported provider: {provider!r}; expected one of {list(_DEFAULTS)}")
        defaults = _DEFAULTS[provider]
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url or str(defaults["base_url"])
        self._model = model or str(defaults["model"])
        self._dim = int(dim if dim is not None else defaults["dim"])  # type: ignore[arg-type]
        if not api_key:
            raise ValueError(f"api_key is required for provider {provider!r}")
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)

    @property
    def dim(self) -> int:
        """Return the configured vector dimensionality."""
        return self._dim

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings endpoint with retry."""
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [[float(x) for x in item.embedding] for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents via the API.

        Args:
            texts: List of raw text strings.

        Returns:
            List of embedding vectors, ordered the same as the input.
        """
        if not texts:
            return []
        return self._create_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._create_embeddings([text])[0]
