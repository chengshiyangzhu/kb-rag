"""Factory for selecting an embedder backend based on settings."""
from __future__ import annotations

from app.embedders.api_embedder import ApiEmbedder
from app.embedders.base import Embedder
from app.embedders.local_embedder import LocalEmbedder
from app.embedders.ollama_embedder import OllamaEmbedder
from app.observability.logging import get_logger

logger = get_logger(__name__)


def get_embedder(settings) -> Embedder:
    """Build an :class:`Embedder` from the supplied :class:`Settings`.

    Dispatch rules:
        - ``"local"``: :class:`LocalEmbedder` using ``embedder_model`` /
          ``embedder_dim``.
        - ``"ollama"``: :class:`OllamaEmbedder` pointed at ``ollama_base_url``.
        - ``"api"``: :class:`ApiEmbedder`. Zhipu is preferred when
          ``zhipu_api_key`` is set; otherwise OpenAI is used with
          ``openai_api_key``.
        - Any other value raises :class:`ValueError`.

    Args:
        settings: Application :class:`Settings` instance.

    Returns:
        A concrete :class:`Embedder` instance.

    Raises:
        ValueError: If ``settings.embedder_provider`` is unknown or the
            selected API provider has no API key configured.
    """
    provider = settings.embedder_provider

    if provider == "local":
        logger.info("using local embedder", model=settings.embedder_model, dim=settings.embedder_dim)
        return LocalEmbedder(
            model_name=settings.embedder_model,
            dim=settings.embedder_dim,
        )

    if provider == "ollama":
        logger.info("using ollama embedder", base_url=settings.ollama_base_url, dim=settings.embedder_dim)
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            dim=settings.embedder_dim,
        )

    if provider == "api":
        if settings.zhipu_api_key:
            logger.info("using zhipu api embedder", dim=settings.embedder_dim)
            return ApiEmbedder(
                provider="zhipu",
                api_key=settings.zhipu_api_key,
                base_url=settings.zhipu_base_url,
                dim=settings.embedder_dim,
            )
        if settings.openai_api_key:
            logger.info("using openai api embedder", dim=settings.embedder_dim)
            return ApiEmbedder(
                provider="openai",
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                dim=settings.embedder_dim,
            )
        raise ValueError("api embedder selected but neither zhipu_api_key nor openai_api_key is set")

    raise ValueError(f"unknown embedder_provider: {provider!r}")
