# Strategy 2 — Recursive Character

> **Best for: general prose, mixed-content documentation, any text without machine-readable structure.**

---

## How It Works

Tries to split on a hierarchy of separators in descending order of "naturalness":

```
1st choice: \n\n   (paragraph break)
2nd choice: \n     (line break)
3rd choice: ". "   (sentence end)
4th choice: " "    (word boundary)
Last resort: ""    (character — only if no other separator fits)
```

The splitter only descends to a finer separator if the chunk produced by the coarser one is still too large. This means chunks are always split at the most natural boundary available.

---

## The Newspaper Analogy

A librarian cutting articles out of a newspaper:
- First tries to keep each article whole.
- If an article is too long for the folder, cuts at paragraph breaks.
- If a paragraph is still too long, cuts at sentence ends.
- Never cuts mid-word unless there is truly no other option.

---

## Advantage over Fixed Size

| Property | Fixed Size | Recursive Character |
|---|---|---|
| Sentence integrity | ❌ Splits mid-sentence | ✅ Sentences stay whole |
| Paragraph integrity | ❌ Splits mid-paragraph | ✅ Paragraphs stay whole where possible |
| Cross-chunk coherence | ❌ Arbitrary | ✅ Natural topic grouping |

---

## Limitation

Does not understand document structure (headers, sections, XML tags, code blocks). It treats all text equally — a section header and a footnote are treated identically. For documents with known structure, use Document-Aware chunking instead.

---

## Best Fit Document Types

- General-purpose prose
- Mixed-content documents (narrative + lists + quotes)
- Documentation where sections are separated by blank lines but have no machine-readable markers
- Default fallback when no better strategy applies

---

## Code Reference

```python
from chunking_engine.chunkers import RecursiveCharacterChunker, ChunkingConfig

config = ChunkingConfig(chunk_size=1000, chunk_overlap=150)
chunker = RecursiveCharacterChunker(config)
chunks = chunker.chunk(document_text, {"source": "doc-001"})
```

---

## Configuration Parameters

| Parameter | Default | Notes |
|---|---|---|
| `chunk_size` | 1000 | Target character count per chunk |
| `chunk_overlap` | 200 | Characters of overlap between adjacent chunks |

---

## Metadata Fields

| Field | Value |
|---|---|
| `strategy` | `recursive_character` |
| `doc_type` | `general_text` |
| `chunk_id` | Sequential integer |

---

## Interview Line

> *"Recursive character chunking is my general-purpose fallback. It tries paragraph breaks first, then sentences, then word boundaries — always the most natural split available. It preserves sentence coherence where fixed-size chunking would destroy it, without requiring any knowledge of the document's internal structure."*
