from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Provider selection ─────────────────────────────────────────────────────
    llm_provider: Literal["azure_openai", "bedrock"] = Field(
        default="azure_openai",
        description="Active LLM/embedding provider",
    )

    # ── Azure OpenAI ───────────────────────────────────────────────────────────
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version",
    )
    azure_chat_deployment: str = Field(
        default="gpt-4o",
        description="Deployment name for the chat model",
    )
    azure_embedding_deployment: str = Field(
        default="text-embedding-3-large",
        description="Deployment name for the embedding model",
    )

    # ── AWS Bedrock ────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: str = Field(default="", description="AWS access key ID")
    aws_secret_access_key: str = Field(default="", description="AWS secret access key")
    aws_session_token: str = Field(default="", description="AWS session token (optional)")
    bedrock_chat_model: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Bedrock model ID for chat completions",
    )
    bedrock_embedding_model: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock model ID for embeddings",
    )

    # ── Milvus ─────────────────────────────────────────────────────────────────
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    milvus_collection_name: str = Field(
        default="document_chunks",
        description="Target Milvus collection name",
    )
    milvus_embedding_dim: int = Field(
        default=1536,
        description="Vector dimension (must match embedding model output)",
    )

    # ── Chunking defaults ──────────────────────────────────────────────────────
    default_chunk_size: int = Field(default=1000, ge=100, le=8000)
    default_chunk_overlap: int = Field(default=200, ge=0, le=2000)
    semantic_breakpoint_threshold: float = Field(
        default=85.0,
        ge=50.0,
        le=99.0,
        description="Percentile for semantic topic-shift detection",
    )

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str = Field(default="", description="Bearer / header API key")
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )
