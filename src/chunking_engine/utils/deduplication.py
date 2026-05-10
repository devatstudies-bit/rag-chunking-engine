"""Deduplication utilities for post-retrieval and post-chunking pipelines."""

from __future__ import annotations

from langchain_core.documents import Document


def deduplicate_by_content(docs: list[Document], threshold: float = 0.85) -> list[Document]:
    """Remove near-duplicate Documents using Jaccard token-overlap similarity.

    Used after sliding-window chunking or when multiple overlapping results
    are returned by ANN search for closely related queries.
    """
    seen: list[Document] = []
    for doc in docs:
        tokens = set(doc.page_content.lower().split())
        is_dup = any(
            _jaccard(tokens, set(s.page_content.lower().split())) > threshold
            for s in seen
        )
        if not is_dup:
            seen.append(doc)
    return seen


def deduplicate_by_source(docs: list[Document]) -> list[Document]:
    """Keep the highest-scored chunk per unique (source, section) pair.

    Useful after retrieval to avoid returning multiple chunks from the same
    document section to the LLM context window.
    """
    best: dict[tuple[str, str], Document] = {}
    for doc in docs:
        key = (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("section", "")),
        )
        if key not in best:
            best[key] = doc
    return list(best.values())


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
