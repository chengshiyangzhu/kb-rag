"""Factory for constructing generators from settings."""
from __future__ import annotations

from typing import Any

from app.generation.base import Generator
from app.generation.llm import OllamaGenerator, OpenAIGenerator, ZhipuGenerator


def get_generator(settings: Any) -> Generator:
    """Build a :class:`Generator` based on ``settings.llm_provider``.

    Args:
        settings: Application settings (duck-typed). Must expose
            ``llm_provider``, ``llm_model`` and the credentials/URLs for the
            requested provider.

    Returns:
        One of :class:`OpenAIGenerator`, :class:`ZhipuGenerator`,
        :class:`OllamaGenerator`.

    Raises:
        ValueError: If ``llm_provider`` is not one of the supported values.
    """
    provider = getattr(settings, "llm_provider", "openai").lower()
    model = getattr(settings, "llm_model", "gpt-4o-mini")
    temperature = getattr(settings, "llm_temperature", 0.0)
    max_tokens = getattr(settings, "llm_max_tokens", 2048)
    top_p = getattr(settings, "llm_top_p", 1.0)
    if provider == "openai":
        return OpenAIGenerator(
            api_key=getattr(settings, "openai_api_key", ""),
            base_url=getattr(settings, "openai_base_url", None),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    if provider == "zhipu":
        return ZhipuGenerator(
            api_key=getattr(settings, "zhipu_api_key", ""),
            base_url=getattr(settings, "zhipu_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    if provider == "ollama":
        return OllamaGenerator(
            base_url=getattr(settings, "ollama_base_url", "http://ollama:11434"),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
    raise ValueError(f"Unsupported llm_provider: {provider!r}")
