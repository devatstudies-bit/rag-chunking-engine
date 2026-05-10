from __future__ import annotations

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from chunking_engine.config import Settings
from chunking_engine.models.base import LLMProvider


class BedrockProvider(LLMProvider):
    """LLM + embedding provider backed by AWS Bedrock."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _boto_session(self) -> boto3.Session:
        s = self._settings
        kwargs: dict[str, str] = {"region_name": s.aws_region}
        if s.aws_access_key_id:
            kwargs["aws_access_key_id"] = s.aws_access_key_id
        if s.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = s.aws_secret_access_key
        if s.aws_session_token:
            kwargs["aws_session_token"] = s.aws_session_token
        return boto3.Session(**kwargs)

    def get_chat_model(self, temperature: float = 0.0, **kwargs: object) -> BaseChatModel:
        s = self._settings
        return ChatBedrock(
            model_id=s.bedrock_chat_model,
            client=self._boto_session().client("bedrock-runtime"),
            model_kwargs={"temperature": temperature},
            **kwargs,
        )

    def get_embeddings(self, **kwargs: object) -> Embeddings:
        s = self._settings
        return BedrockEmbeddings(
            model_id=s.bedrock_embedding_model,
            client=self._boto_session().client("bedrock-runtime"),
            **kwargs,
        )
