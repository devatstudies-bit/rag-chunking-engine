from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document


class DocumentType(str, Enum):
    """Canonical document-type labels used by the strategy registry."""

    STRUCTURED_DOCUMENT = "structured_document"   # known sections/schema
    GENERAL_TEXT = "general_text"                  # prose without fixed schema
    TRANSCRIPT = "transcript"                      # conversational / interview
    SOURCE_CODE = "source_code"                    # any programming language
    TABULAR_DATA = "tabular_data"                  # CSV / spreadsheet rows
    TECHNICAL_DOC = "technical_doc"               # dense cross-referencing prose
    HETEROGENEOUS = "heterogeneous"                # mixed: tables + code + prose
    UNKNOWN = "unknown"                            # fallback → general_text


class ChunkingStrategy(str, Enum):
    """Enumerated strategy identifiers."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE_CHARACTER = "recursive_character"
    DOCUMENT_AWARE = "document_aware"
    SEMANTIC = "semantic"
    CODE_AWARE = "code_aware"
    ROW_AWARE = "row_aware"
    SLIDING_WINDOW = "sliding_window"
    AGENTIC = "agentic"


@dataclass
class ChunkingConfig:
    """Unified configuration object passed to every chunker."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    language: str = "generic"                      # for code-aware strategy
    section_patterns: list[str] = field(default_factory=list)
    semantic_threshold: float = 85.0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    strategy: ChunkingStrategy

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split *text* into LangChain Documents with populated metadata."""

    def _base_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {"strategy": self.strategy.value}
        if extra:
            meta.update(extra)
        return meta

    @classmethod
    def description(cls) -> str:
        return cls.__doc__ or ""
