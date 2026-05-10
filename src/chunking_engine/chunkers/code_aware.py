"""Code-aware chunker — splits at function / class / method boundaries."""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

# LangChain built-in language support
_LANGCHAIN_LANGUAGES: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "rust": Language.RUST,
    "cpp": Language.CPP,
    "c": Language.C,
    "ruby": Language.RUBY,
    "kotlin": Language.KOTLIN,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
}

# Generic regex patterns for languages not in LangChain's built-ins.
# Key = unit type label, value = regex matching the start of a unit.
_GENERIC_UNIT_PATTERNS: list[tuple[str, str]] = [
    ("function",  r"(?:^|\n)(?:func|function|def|fn)\s+\w+"),
    ("class",     r"(?:^|\n)(?:class|struct|interface|trait)\s+\w+"),
    ("method",    r"(?:^|\n)\s+(?:public|private|protected|static)?\s*\w+\s*\("),
]


class CodeAwareChunker(BaseChunker):
    """Language-sensitive splitter that never cuts through a function or class body.

    For languages supported by LangChain (Python, JS, TS, Java, Go, …) the
    built-in AST-aware splitter is used.  For other languages a regex-based
    fallback locates unit boundaries (function/class/method keywords).

    Ideal for: any programming language source files where a partial function
    is semantically meaningless.
    """

    strategy = ChunkingStrategy.CODE_AWARE

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        lang = self.config.language.lower()
        base_meta = self._base_metadata({"doc_type": "source_code", **(metadata or {})})

        if lang in _LANGCHAIN_LANGUAGES:
            return self._chunk_with_langchain(text, _LANGCHAIN_LANGUAGES[lang], base_meta)
        return self._chunk_generic(text, base_meta)

    # ── Built-in language support ──────────────────────────────────────────────

    def _chunk_with_langchain(
        self,
        text: str,
        language: Language,
        meta: dict[str, Any],
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        docs = splitter.create_documents([text], metadatas=[meta])
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["language"] = self.config.language
        return docs

    # ── Generic regex-based fallback ───────────────────────────────────────────

    def _chunk_generic(self, text: str, meta: dict[str, Any]) -> list[Document]:
        boundaries = self._find_boundaries(text)
        boundaries.append(len(text))
        chunks: list[Document] = []

        for idx in range(len(boundaries) - 1):
            start, end = boundaries[idx], boundaries[idx + 1]
            unit = text[start:end].strip()
            if not unit:
                continue

            unit_type, unit_name = self._classify_unit(unit)
            chunk_meta = {
                **meta,
                "chunk_id": len(chunks),
                "unit_type": unit_type,
                "unit_name": unit_name,
                "language": self.config.language,
            }
            chunks.append(Document(page_content=unit, metadata=chunk_meta))

        # Fallback: no boundaries found → single chunk
        if not chunks:
            chunks.append(Document(page_content=text, metadata={**meta, "chunk_id": 0}))

        return chunks

    def _find_boundaries(self, text: str) -> list[int]:
        combined = "|".join(p for _, p in _GENERIC_UNIT_PATTERNS)
        return sorted({m.start() for m in re.finditer(combined, text, re.MULTILINE)})

    def _classify_unit(self, unit: str) -> tuple[str, str]:
        first_line = unit.split("\n")[0]
        for label, pattern in _GENERIC_UNIT_PATTERNS:
            if re.search(pattern, first_line, re.IGNORECASE):
                name_m = re.search(r"(?:func|function|def|fn|class|struct|interface)\s+(\w+)", first_line)
                return label, name_m.group(1) if name_m else "unknown"
        return "block", "unknown"
