"""Generation layer for kb-rag.

Exposes the abstract :class:`Generator` and :class:`GenerationResult`, the
three concrete LLM implementations, a factory, the prompt builder, the
hallucination guardrail and the citation parser.
"""
from __future__ import annotations

from app.generation.base import GenerationResult, Generator
from app.generation.citation import CitationParser
from app.generation.factory import get_generator
from app.generation.guardrail import Guardrail
from app.generation.llm import OllamaGenerator, OpenAIGenerator, ZhipuGenerator
from app.generation.prompts import build_rag_prompt

__all__ = [
    "GenerationResult",
    "Generator",
    "CitationParser",
    "get_generator",
    "Guardrail",
    "OllamaGenerator",
    "OpenAIGenerator",
    "ZhipuGenerator",
    "build_rag_prompt",
]
