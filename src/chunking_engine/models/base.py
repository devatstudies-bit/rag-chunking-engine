from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from chunking_engine.config import Settings, get_settings


class LLMProvider(ABC):
    """Abstract interface for all LLM/embedding providers.

    Implementations must supply a chat model and an embedding model.
    Switching providers is a single .env change — no code changes required.
    """

    @abstractmethod
    def get_chat_model(self, temperature: float = 0.0, **kwargs: object) -> BaseChatModel:
        """Return a configured chat completion model."""

    @abstractmethod
    def get_embeddings(self, **kwargs: object) -> Embeddings:
        """Return a configured embedding model."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""


class ProviderFactory:
    """Resolves the active LLMProvider from settings."""

    @staticmethod
    def create(settings: Settings | None = None) -> LLMProvider:
        s = settings or get_settings()
        if s.llm_provider == "bedrock":
            from chunking_engine.models.bedrock import BedrockProvider
            return BedrockProvider(s)
        from chunking_engine.models.azure_openai import AzureOpenAIProvider
        return AzureOpenAIProvider(s)
