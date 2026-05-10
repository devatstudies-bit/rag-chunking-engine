# Strategy 1 — Fixed Size

> **Status: BENCHMARK BASELINE ONLY — never use in production.**

---

## How It Works

Splits text every N characters regardless of content. No awareness of sentences, paragraphs, topics, or document structure.

```
Input:  "The function calculates the total revenue for all active orders."
N=30 →  Chunk 1: "The function calculates the to"
        Chunk 2: "tal revenue for all active ord"   ← word split mid-token
        Chunk 3: "ers."
```

---

## The Newspaper Analogy

Imagine cutting a newspaper with a ruler set to 30 cm, ignoring headlines, article boundaries, and columns. You will always cut through words, sentences, and stories. The result is fragments that have no meaning on their own.

---

## Why It Fails in Production

| Problem | Impact |
|---|---|
| Splits mid-sentence | LLM receives half a thought — retrieval accuracy drops dramatically |
| Separates header from body | A section heading lands in Chunk N, its content in Chunk N+1 — retrieval returns orphaned content |
| No topic coherence | Adjacent chunks share no thematic relationship |
| Overlap is a bandage, not a fix | A 50-char overlap on a 500-char chunk barely helps when the split was mid-word |

---

## When To Use It

| Scenario | Acceptable? |
|---|---|
| 30-minute prototype to verify pipeline wiring | ✅ Yes |
| Benchmarking — measuring how much better other strategies are | ✅ Yes |
| Any production workload | ❌ Never |

---

## Code Reference

```python
from langchain_text_splitters import CharacterTextSplitter
from chunking_engine.chunkers import FixedSizeChunker, ChunkingConfig

config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
chunker = FixedSizeChunker(config)
chunks = chunker.chunk(document_text, {"source": "doc-001"})
```

---

## Configuration Parameters

| Parameter | Default | Notes |
|---|---|---|
| `chunk_size` | 1000 | Characters per chunk |
| `chunk_overlap` | 200 | Characters shared between adjacent chunks |

---

## Interview Line

> *"Fixed-size chunking is the naive baseline — I use it only to benchmark against. In production it fails because it splits mid-sentence and separates headers from their content, which destroys the context that makes retrieval accurate."*
