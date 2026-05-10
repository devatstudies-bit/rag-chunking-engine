"""Document-aware chunker — splits on explicit structural section boundaries."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

# Default section headers for generic structured documents.
# Override via ChunkingConfig.section_patterns for domain-specific schemas.
DEFAULT_SECTION_PATTERNS: list[str] = [
    "Overview",
    "Background",
    "Description",
    "Details",
    "Summary",
    "Analysis",
    "Findings",
    "Recommendations",
    "Notes",
    "References",
    "Appendix",
    "Conclusion",
    "Introduction",
    "Methodology",
    "Results",
]


class DocumentAwareChunker(BaseChunker):
    """Exploits a document's own section structure as chunk boundaries.

    Each chunk maps to exactly one logical section of the source document.
    The document header (content before the first named section) is prepended
    to every chunk so that retrieval always has the document identity context.

    Supports any domain via configurable section_patterns.
    """

    strategy = ChunkingStrategy.DOCUMENT_AWARE

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        super().__init__(config)
        self._sections = (
            self.config.section_patterns if self.config.section_patterns
            else DEFAULT_SECTION_PATTERNS
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        header = self._extract_header(text)
        chunks: list[Document] = []

        for section in self._sections:
            content = self._extract_section(text, section)
            if not content:
                continue

            # Prepend document header so every chunk is self-contained.
            chunk_text = f"{header}\n\n{section}:\n{content}" if header else f"{section}:\n{content}"

            meta = self._base_metadata({
                "doc_type": "structured_document",
                "section": section,
                "chunk_id": len(chunks),
                **(metadata or {}),
            })
            chunks.append(Document(page_content=chunk_text.strip(), metadata=meta))

        # Fallback: if no known sections found, treat whole doc as one chunk.
        if not chunks:
            meta = self._base_metadata({
                "doc_type": "structured_document",
                "section": "full_document",
                "chunk_id": 0,
                **(metadata or {}),
            })
            chunks.append(Document(page_content=text.strip(), metadata=meta))

        return chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_header(self, text: str) -> str:
        """Return everything before the first recognised section header."""
        all_sections_re = "|".join(re.escape(s) for s in self._sections)
        m = re.search(
            rf"^(.*?)(?={all_sections_re})",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def _extract_section(self, text: str, section: str) -> str:
        """Return the body of *section*, stopping at the next known section."""
        all_sections_re = "|".join(re.escape(s) for s in self._sections)
        pattern = rf"{re.escape(section)}\s*[:\n](.*?)(?={all_sections_re}|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
