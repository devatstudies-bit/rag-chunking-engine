"""Strategy registry — maps document types to optimal chunking strategies."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from chunking_engine.chunkers import (
    AgenticChunker,
    BaseChunker,
    ChunkingConfig,
    ChunkingStrategy,
    CodeAwareChunker,
    DocumentAwareChunker,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    RowAwareChunker,
    SemanticChunker,
    SlidingWindowChunker,
)
from chunking_engine.chunkers.base import DocumentType

# ── Golden rule mapping ────────────────────────────────────────────────────────
# Each document type maps to the strategy that best preserves its semantics.
_TYPE_TO_STRATEGY: dict[DocumentType, ChunkingStrategy] = {
    DocumentType.STRUCTURED_DOCUMENT: ChunkingStrategy.DOCUMENT_AWARE,
    DocumentType.GENERAL_TEXT:        ChunkingStrategy.RECURSIVE_CHARACTER,
    DocumentType.TRANSCRIPT:          ChunkingStrategy.SEMANTIC,
    DocumentType.SOURCE_CODE:         ChunkingStrategy.CODE_AWARE,
    DocumentType.TABULAR_DATA:        ChunkingStrategy.ROW_AWARE,
    DocumentType.TECHNICAL_DOC:       ChunkingStrategy.SLIDING_WINDOW,
    DocumentType.HETEROGENEOUS:       ChunkingStrategy.AGENTIC,
    DocumentType.UNKNOWN:             ChunkingStrategy.RECURSIVE_CHARACTER,
}

# ── Detection heuristics ──────────────────────────────────────────────────────
# Each entry: (DocumentType, score_function(text, meta) → float 0–1)
_HEURISTICS: list[tuple[DocumentType, Any]] = [
    (
        DocumentType.TABULAR_DATA,
        lambda t, m: 1.0 if (
            m.get("file_extension") in {".csv", ".tsv"}
            or _csv_confidence(t) > 0.7
        ) else 0.0,
    ),
    (
        DocumentType.SOURCE_CODE,
        lambda t, m: 1.0 if (
            m.get("file_extension") in {".py", ".js", ".ts", ".java", ".go",
                                         ".rs", ".cpp", ".c", ".rb", ".kt"}
            or _code_confidence(t) > 0.6
        ) else 0.0,
    ),
    (
        DocumentType.STRUCTURED_DOCUMENT,
        lambda t, m: _section_header_confidence(t),
    ),
    (
        DocumentType.TRANSCRIPT,
        lambda t, m: _transcript_confidence(t),
    ),
    (
        DocumentType.TECHNICAL_DOC,
        lambda t, m: _technical_doc_confidence(t),
    ),
]


class StrategyRegistry:
    """Static utility: document-type detection and chunker instantiation."""

    @staticmethod
    def detect_document_type(
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = metadata or {}
        # Explicit override from caller always wins.
        if explicit := meta.get("doc_type"):
            return explicit

        scores: dict[DocumentType, float] = {}
        for doc_type, score_fn in _HEURISTICS:
            try:
                s = float(score_fn(text, meta))
            except Exception:
                s = 0.0
            scores[doc_type] = s

        best = max(scores, key=lambda k: scores[k])
        return best.value if scores[best] > 0.3 else DocumentType.GENERAL_TEXT.value

    @staticmethod
    def select_strategy(doc_type: DocumentType | str) -> ChunkingStrategy:
        if isinstance(doc_type, str):
            try:
                doc_type = DocumentType(doc_type)
            except ValueError:
                doc_type = DocumentType.UNKNOWN
        return _TYPE_TO_STRATEGY.get(doc_type, ChunkingStrategy.RECURSIVE_CHARACTER)

    @staticmethod
    def build_chunker(
        strategy: ChunkingStrategy,
        config: ChunkingConfig | None = None,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
    ) -> BaseChunker:
        cfg = config or ChunkingConfig()
        match strategy:
            case ChunkingStrategy.FIXED_SIZE:
                return FixedSizeChunker(cfg)
            case ChunkingStrategy.RECURSIVE_CHARACTER:
                return RecursiveCharacterChunker(cfg)
            case ChunkingStrategy.DOCUMENT_AWARE:
                return DocumentAwareChunker(cfg)
            case ChunkingStrategy.SEMANTIC:
                if embeddings is None:
                    raise ValueError("SemanticChunker requires an embeddings model")
                return SemanticChunker(embeddings=embeddings, config=cfg)
            case ChunkingStrategy.CODE_AWARE:
                return CodeAwareChunker(cfg)
            case ChunkingStrategy.ROW_AWARE:
                return RowAwareChunker(cfg)
            case ChunkingStrategy.SLIDING_WINDOW:
                return SlidingWindowChunker(cfg)
            case ChunkingStrategy.AGENTIC:
                if llm is None:
                    raise ValueError("AgenticChunker requires an LLM")
                return AgenticChunker(llm=llm, config=cfg)
            case _:
                return RecursiveCharacterChunker(cfg)

    @staticmethod
    def list_strategies() -> list[dict[str, str]]:
        return [
            {
                "strategy": s.value,
                "document_type": dt.value,
                "description": StrategyRegistry.build_chunker(s).description(),
            }
            for dt, s in _TYPE_TO_STRATEGY.items()
        ]


# ── Heuristic helpers ─────────────────────────────────────────────────────────

def _csv_confidence(text: str) -> float:
    lines = text.strip().splitlines()[:20]
    if len(lines) < 2:
        return 0.0
    comma_counts = [line.count(",") for line in lines]
    avg = sum(comma_counts) / len(comma_counts)
    return min(avg / 5, 1.0)


def _code_confidence(text: str) -> float:
    indicators = [
        r"\bdef\s+\w+\s*\(",
        r"\bfunction\s+\w+\s*\(",
        r"\bclass\s+\w+[\s:{]",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"^\s*//",
        r"^\s*#\s",
        r"\{[\s\S]*?\}",
    ]
    hits = sum(1 for p in indicators if re.search(p, text, re.MULTILINE))
    return min(hits / 4, 1.0)


def _section_header_confidence(text: str) -> float:
    header_pattern = r"^(#{1,6}\s+\w|\b(?:Overview|Background|Description|Details|Summary|Analysis|Findings|Recommendations|Notes|References|Introduction|Conclusion|Results)\b)"
    matches = len(re.findall(header_pattern, text, re.MULTILINE | re.IGNORECASE))
    return min(matches / 3, 1.0)


def _transcript_confidence(text: str) -> float:
    indicators = [
        r"\b\d{1,2}:\d{2}(:\d{2})?\b",          # timestamps
        r"^[A-Z][a-z]+\s[A-Z][a-z]+\s*:",        # Speaker Name:
        r"\b(?:said|asked|replied|mentioned)\b",
        r"\b(?:yeah|um|uh|okay|so basically)\b",
    ]
    hits = sum(1 for p in indicators if re.search(p, text, re.MULTILINE | re.IGNORECASE))
    return min(hits / 2, 1.0)


def _technical_doc_confidence(text: str) -> float:
    indicators = [
        r"(?:see also|refer to|as described in|cf\.)\s+(?:section|chapter|figure|table)",
        r"\b(?:algorithm|implementation|specification|interface|protocol)\b",
        r"[A-Z]{2,}\s*\(\w+\)",                  # abbreviation patterns
        r"\b\d+\.\d+\.\d+\b",                     # version numbers
    ]
    hits = sum(1 for p in indicators if re.search(p, text, re.IGNORECASE))
    return min(hits / 2, 1.0)
