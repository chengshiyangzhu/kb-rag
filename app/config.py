"""Application settings loaded from environment variables and .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the kb-rag platform.

    Values are read from environment variables first, then from a ``.env`` file
    located at the project root. Non-sensitive defaults mirror ``config.yaml``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application ----
    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- Embedder ----
    embedder_provider: Literal["local", "ollama", "api"] = "local"
    embedder_model: str = "BAAI/bge-m3"
    embedder_dim: int = 1024

    # ---- Ollama ----
    ollama_base_url: str = "http://ollama:11434"

    # ---- OpenAI ----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # ---- Zhipu (BigModel) ----
    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # ---- LLM ----
    llm_provider: Literal["openai", "zhipu", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0       # RAG 需要确定性输出，0 = 每次回答相同
    llm_max_tokens: int = 2048         # 回答最大长度，2048 token 约 1500 字中文
    llm_top_p: float = 1.0             # 核采样，1.0 = 不限制（用 temperature 控制即可）

    # ---- Vector store ----
    vector_store: Literal["qdrant", "chroma"] = "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "kb_rag"
    chroma_path: str = "./data/chroma"

    # ---- Chunker ----
    chunker_type: Literal["recursive", "fixed", "semantic", "structural"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- Retrieval / Rerank ----
    retrieve_top_n: int = 20
    rerank_top_k: int = 5
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_threshold: float = 0.3

    # ---- Fusion ----
    rrf_k: int = 60

    # ---- BM25 ----
    bm25_index_path: str = "./data/bm25.pkl"

    # ---- Observability ports ----
    prometheus_port: int = 9090
    grafana_port: int = 3000
    api_port: int = 8000
    ui_port: int = 8501

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v

    @field_validator("embedder_dim")
    @classmethod
    def _dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedder_dim must be positive")
        return v

    @field_validator("retrieve_top_n", "rerank_top_k", "rrf_k")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def _temperature_range(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return v

    @field_validator("llm_top_p")
    @classmethod
    def _top_p_range(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError("llm_top_p must be between 0.0 and 1.0 (exclusive 0)")
        return v

    @field_validator("llm_max_tokens")
    @classmethod
    def _max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("llm_max_tokens must be positive")
        return v

    @field_validator("log_level")
    @classmethod
    def _normalize_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()
