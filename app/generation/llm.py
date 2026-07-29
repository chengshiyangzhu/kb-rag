"""LLM-backed generators for the RAG pipeline.

Three concrete implementations of :class:`app.generation.base.Generator`:

* :class:`OpenAIGenerator`  - OpenAI-compatible Chat Completions API.
* :class:`ZhipuGenerator`   - Zhipu (BigModel) via the OpenAI client.
* :class:`OllamaGenerator`  - Local Ollama ``/api/chat`` REST endpoint.

All generators wrap their network calls in tenacity retries (3 attempts).
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.generation.base import GenerationResult, Generator
from app.generation.citation import CitationParser
from app.generation.prompts import build_rag_prompt
from app.models.document import Chunk
from app.observability.logging import get_logger
from app.observability.metrics import record_generation_latency

logger = get_logger(__name__)

# Shared retry decorator: 3 attempts, exponential backoff, retry on network
# errors and transient HTTP errors.
_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (
            httpx.HTTPError,
            httpx.TimeoutException,
            httpx.NetworkError,
            RuntimeError,
        )
    ),
    reraise=True,
)


def _extract_answer(response_obj: Any) -> str:
    """Pull the assistant text out of an OpenAI-style chat response.

    Args:
        response_obj: Response object from ``OpenAI.chat.completions.create``.

    Returns:
        The first choice's message content as a string.
    """
    try:
        return response_obj.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("llm.parse.failed", error=str(exc))
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc


class OpenAIGenerator(Generator):
    """Generator backed by the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.
        base_url: Optional override (e.g. for Azure or compatible proxies).
        model: Model id (e.g. ``"gpt-4o-mini"``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        """Initialize the generator (does not create the client yet).

        Args:
            api_key: OpenAI API key.
            base_url: Optional override (e.g. for Azure or compatible proxies).
            model: Model id (e.g. ``"gpt-4o-mini"``).
            temperature: 0.0 = deterministic (RAG recommended), 1.0 = creative.
            max_tokens: Maximum tokens in the generated response.
            top_p: Nucleus sampling threshold; 1.0 = disabled (use temperature).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._client: Any | None = None
        self._citation_parser = CitationParser()

    def _get_client(self) -> Any:
        """Lazily construct the OpenAI client."""
        if self._client is None:
            from openai import OpenAI  # type: ignore[import-untyped]

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Generate an answer with retries, then extract citations."""
        messages = build_rag_prompt(query, contexts)
        start = time.perf_counter()
        try:
            answer = self._call_with_retry(messages)
            citations = self._citation_parser.parse(answer)
            used = self._citation_parser.map_to_references(citations, contexts)
            used_ids = [c.id for c in used]
            logger.info(
                "openai.generate.done",
                model=self.model,
                citations=citations,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return GenerationResult(
                answer=answer,
                citations=citations,
                used_chunk_ids=used_ids,
            )
        finally:
            record_generation_latency(time.perf_counter() - start)

    @_RETRY_DECORATOR
    def _call_with_retry(self, messages: list[dict]) -> str:
        """Send the chat completion request, retrying on transient errors."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,    # ← 控制随机性，0=确定性输出
            max_tokens=self.max_tokens,      # ← 回答最大长度
            top_p=self.top_p,                # ← 核采样阈值
        )
        return _extract_answer(response)


class ZhipuGenerator(Generator):
    """Generator backed by Zhipu (BigModel), via the OpenAI-compatible client.

    Args:
        api_key: Zhipu API key.
        base_url: Zhipu API base URL.
        model: Zhipu model id (e.g. ``"glm-4"``).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: Zhipu API key.
            base_url: Zhipu API base URL.
            model: Zhipu model id (e.g. ``"glm-4"``).
            temperature: 0.0 = deterministic (RAG recommended), 1.0 = creative.
            max_tokens: Maximum tokens in the generated response.
            top_p: Nucleus sampling threshold; 1.0 = disabled.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._client: Any | None = None
        self._citation_parser = CitationParser()

    def _get_client(self) -> Any:
        """Lazily construct the OpenAI client pointed at Zhipu's base URL."""
        if self._client is None:
            from openai import OpenAI  # type: ignore[import-untyped]

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Generate an answer via Zhipu, then extract citations."""
        messages = build_rag_prompt(query, contexts)
        start = time.perf_counter()
        try:
            answer = self._call_with_retry(messages)
            citations = self._citation_parser.parse(answer)
            used = self._citation_parser.map_to_references(citations, contexts)
            used_ids = [c.id for c in used]
            logger.info(
                "zhipu.generate.done",
                model=self.model,
                citations=citations,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return GenerationResult(
                answer=answer,
                citations=citations,
                used_chunk_ids=used_ids,
            )
        finally:
            record_generation_latency(time.perf_counter() - start)

    @_RETRY_DECORATOR
    def _call_with_retry(self, messages: list[dict]) -> str:
        """Send the chat completion request to Zhipu, with retries."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return _extract_answer(response)


class OllamaGenerator(Generator):
    """Generator backed by a local Ollama ``/api/chat`` REST endpoint.

    Args:
        base_url: Ollama base URL (e.g. ``"http://ollama:11434"``).
        model: Ollama model name (e.g. ``"qwen2.5"``).
    """

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "qwen2.5",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> None:
        """Initialize the generator.

        Args:
            base_url: Ollama base URL (e.g. ``"http://ollama:11434"``).
            model: Ollama model name (e.g. ``"qwen2.5"``).
            temperature: 0.0 = deterministic (RAG recommended), 1.0 = creative.
            max_tokens: Maximum tokens in the generated response.
            top_p: Nucleus sampling threshold; 1.0 = disabled.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._citation_parser = CitationParser()

    def generate(self, query: str, contexts: list[Chunk]) -> GenerationResult:
        """Generate an answer via Ollama, then extract citations."""
        messages = build_rag_prompt(query, contexts)
        start = time.perf_counter()
        try:
            answer = self._call_with_retry(messages)
            citations = self._citation_parser.parse(answer)
            used = self._citation_parser.map_to_references(citations, contexts)
            used_ids = [c.id for c in used]
            logger.info(
                "ollama.generate.done",
                model=self.model,
                citations=citations,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )
            return GenerationResult(
                answer=answer,
                citations=citations,
                used_chunk_ids=used_ids,
            )
        finally:
            record_generation_latency(time.perf_counter() - start)

    @_RETRY_DECORATOR
    def _call_with_retry(self, messages: list[dict]) -> str:
        """POST messages to Ollama ``/api/chat`` and assemble the response."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,   # ← Ollama 用 options 包裹生成参数
                "num_predict": self.max_tokens,    # ← Ollama 里 max_tokens 叫 num_predict
                "top_p": self.top_p,
            },
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        try:
            return data.get("message", {}).get("content", "") or ""
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc
