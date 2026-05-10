# Strategy 3 — Document-Aware

> **Best for: structured documents with a known, fixed section schema.**
> This is the highest-precision strategy when document structure is predictable.

---

## How It Works

Reads the document's own section structure — headers, named sections — and splits exactly at those boundaries. Each chunk maps to **exactly one logical section** of the source document.

The document header (all content before the first named section) is **prepended to every chunk** so that retrieval always returns context with the document's identity attached.

```
Input document:
  ┌─────────────────────────────────────────────────┐
  │ Advisory ADV-2024-0187 — PostgreSQL Performance │  ← Header
  │                                                 │
  │ Overview:                                       │
  │   A performance regression has been identified… │  ← Section 1
  │                                                 │
  │ Findings:                                       │
  │   Root cause confirmed via kernel profiling…    │  ← Section 2
  │                                                 │
  │ Recommendations:                                │
  │   1. Upgrade to PostgreSQL 15.4…                │  ← Section 3
  └─────────────────────────────────────────────────┘

Output chunks:
  Chunk 1: [Header] + [Overview content]
  Chunk 2: [Header] + [Findings content]
  Chunk 3: [Header] + [Recommendations content]
```

---

## The Librarian Analogy

A librarian who reads the table of contents first, then cuts the book at chapter boundaries. Each chunk **is** a complete chapter — no chapter is ever split in half, and every chapter has the book title on its cover.

---

## Why the Header Is Repeated in Every Chunk

Without the header, a retrieved chunk saying *"Apply the patch before Friday"* gives no context. With the header, the same chunk says *"Advisory ADV-2024-0187 — PostgreSQL Performance: Apply the patch before Friday."* The LLM always knows which document it came from.

---

## The Power of Section Metadata for Filtering

The section name is stored in chunk metadata. This enables **metadata-filtered retrieval** at query time:

```python
# Retrieve only Recommendation chunks — highest precision for "how do I fix X?" queries
results = milvus.search(query_vec, filter_expr='section == "Recommendations"')

# Retrieve only Findings chunks for root-cause analysis queries
results = milvus.search(query_vec, filter_expr='section == "Findings"')
```

This is dramatically more precise than pure vector similarity.

---

## Configurable Section Patterns

The default section list covers common document schemas. Override for any domain:

```python
from chunking_engine.chunkers import DocumentAwareChunker, ChunkingConfig

# Healthcare records
config = ChunkingConfig(section_patterns=[
    "Patient History", "Diagnosis", "Treatment Plan", "Medications", "Follow-up"
])

# Legal contracts
config = ChunkingConfig(section_patterns=[
    "Parties", "Definitions", "Obligations", "Termination", "Governing Law"
])

# Incident reports
config = ChunkingConfig(section_patterns=[
    "Overview", "Background", "Findings", "Recommendations", "References"
])

chunker = DocumentAwareChunker(config)
```

---

## Default Section Patterns

```
Overview, Background, Description, Details, Summary, Analysis,
Findings, Recommendations, Notes, References, Appendix,
Conclusion, Introduction, Methodology, Results
```

---

## Fallback Behaviour

If no known section headers are found in the document, the entire document becomes a single chunk with `section = "full_document"`. This prevents empty output for unseen document formats.

---

## Code Reference

```python
from chunking_engine.chunkers import DocumentAwareChunker, ChunkingConfig

chunker = DocumentAwareChunker()  # uses default section patterns
chunks = chunker.chunk(document_text, {"source": "advisory-0187"})

# Each chunk has:
# chunk.metadata["section"]  → "Findings"
# chunk.metadata["doc_type"] → "structured_document"
# chunk.metadata["source"]   → "advisory-0187"
```

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `document_aware` | |
| `doc_type` | `structured_document` | |
| `section` | e.g. `"Findings"` | Enables metadata filtering |
| `chunk_id` | Sequential integer | |
| `source` | Caller-provided | Document identifier |

---

## Interview Line

> *"For structured documents I parse the document's own section headers directly. Each section becomes one chunk, and I always prepend the document header to every chunk so retrieval always knows which document the section belongs to. The section name is stored in metadata, which enables server-side filtering at query time — for example, retrieving only Recommendation sections when the user asks how to fix something."*
