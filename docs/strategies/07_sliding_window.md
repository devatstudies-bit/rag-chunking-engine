# Strategy 7 — Sliding Window

> **Best for: dense technical documentation, API references, manuals with heavy cross-references between sections.**

---

## How It Works

Fixed-size chunks with configurable **overlap** between adjacent windows. Each chunk shares some content with its left and right neighbours.

```
chunk_size=1000, chunk_overlap=200:

Chunk 0: characters   0 – 1000
Chunk 1: characters 800 – 1800   ← 200-char overlap with Chunk 0
Chunk 2: characters 1600 – 2600  ← 200-char overlap with Chunk 1
Chunk 3: characters 2400 – 3400  ← 200-char overlap with Chunk 2
```

The overlap window is computed using the same separator hierarchy as Recursive Character chunking (`\n\n` → `\n` → `.` → space), so the boundary of the overlap still falls at a natural text position.

---

## The Magnifying Glass Analogy

Reading a long document by sliding a magnifying glass across the page, with each position overlapping the previous one by a third. Concepts that appear at the edge of one "view" always appear fully within the next view — nothing is lost at the junction.

---

## The Problem Overlap Solves

```
Without overlap (chunk_size=1000):

  Chunk 2 ends with:    "… The connection pool timeout is configur"
  Chunk 3 starts with:  "ed via the MAX_POOL_SIZE environment variable."

→ Retrieval for "connection pool configuration" might return Chunk 2 (missing the value)
  or Chunk 3 (missing the concept name), but not both — and neither is complete.

With overlap=200:

  Chunk 2 ends with:    "… The connection pool timeout is configured via the MAX_POOL_SIZE environment variable."
  Chunk 3 starts with:  "via the MAX_POOL_SIZE environment variable. Set it to the number of…"

→ The complete concept appears in both Chunk 2 and Chunk 3.
  At least one retrieved chunk is always self-contained.
```

---

## The 20% Rule of Thumb

A chunk_overlap of approximately 20% of chunk_size is the standard starting point:

| chunk_size | chunk_overlap (20%) | Note |
|---|---|---|
| 500 | 100 | Good for dense reference material |
| 1000 | 200 | Standard configuration |
| 2000 | 400 | For long-form technical prose |

Increase overlap if your documents have many cross-references. Decrease if token costs are a concern.

---

## Deduplication

Overlap means the same content appears in multiple chunks. If two overlapping chunks are both retrieved for the same query and passed to the LLM, the model sees redundant context, which wastes tokens.

The `SlidingWindowChunker.deduplicate()` helper removes near-duplicate chunks using Jaccard similarity:

```python
from chunking_engine.chunkers import SlidingWindowChunker

chunks = chunker.chunk(text)
# After retrieval, before passing to LLM:
retrieved = milvus.search(query_vec, top_k=10)
deduped = SlidingWindowChunker.deduplicate(retrieved_docs, threshold=0.85)
# Now pass deduped to the LLM — no wasted tokens
```

Jaccard similarity is O(n²) in the number of retrieved results. For `top_k ≤ 20` this is negligible. For larger result sets, consider MinHash LSH for approximate deduplication.

---

## Comparison with Recursive Character

| Property | Recursive Character | Sliding Window |
|---|---|---|
| Overlap | Configurable (both support it) | Same |
| Cross-reference handling | ❌ Content at boundaries may be lost | ✅ Overlap ensures boundary concepts appear in full |
| Token cost | Lower (no duplication) | Higher (overlap introduces duplication) |
| Best for | General text | Dense cross-referencing docs |

---

## Code Reference

```python
from chunking_engine.chunkers import SlidingWindowChunker, ChunkingConfig

config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
chunker = SlidingWindowChunker(config)
chunks = chunker.chunk(technical_doc, {"source": "api-reference.txt"})

# Deduplicate after retrieval
deduped_results = SlidingWindowChunker.deduplicate(retrieved_docs, threshold=0.85)
```

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `sliding_window` | |
| `doc_type` | `technical_doc` | |
| `chunk_id` | Sequential integer | |
| `has_overlap` | `True` / `False` | `False` only for the first chunk |

---

## Interview Line

> *"For dense technical documentation I use a sliding window with 20% overlap. The overlap ensures that concepts spanning a chunk boundary always appear complete in at least one chunk — without it, a concept defined at the end of one chunk and referenced at the start of the next would be split, and retrieval would return an incomplete answer. After retrieval, I deduplicate near-identical overlapping chunks using Jaccard similarity before sending them to the LLM to avoid wasting context tokens."*
