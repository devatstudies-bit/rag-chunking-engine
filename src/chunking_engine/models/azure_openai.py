from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from chunking_engine.config import Settings
from chunking_engine.models.base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """LLM + embedding provider backed by Azure OpenAI Service."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    def get_chat_model(self, temperature: float = 0.0, **kwargs: object) -> BaseChatModel:
        s = self._settings
        return AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,  # type: ignore[arg-type]
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_chat_deployment,
            temperature=temperature,
            **kwargs,
        )

    def get_embeddings(self, **kwargs: object) -> Embeddings:
        s = self._settings
        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,  # type: ignore[arg-type]
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_embedding_deployment,
            **kwargs,
        )
