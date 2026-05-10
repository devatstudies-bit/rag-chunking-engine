# Strategy 4 — Semantic

> **Best for: transcripts, interviews, chat logs, emails — any conversational text where topics shift without announcement.**

---

## How It Works

1. Split the document into individual sentences.
2. Embed each sentence using the configured embedding model.
3. Compute cosine similarity between every consecutive sentence pair.
4. When similarity drops below the configured percentile threshold, that signals a **topic shift** — insert a chunk boundary.
5. Sentences that remain above the threshold stay in the same chunk.

```
Sentence 1: "We completed the auth module this sprint."         → embed → [0.23, 0.87, …]
Sentence 2: "The token refresh logic was tricky to implement."  → embed → [0.21, 0.85, …]
                                                                   similarity = 0.97 → same topic
Sentence 3: "Moving on to deployment — we cut build time."     → embed → [0.71, 0.12, …]
                                                                   similarity = 0.31 → TOPIC SHIFT ✂
Sentence 4: "We parallelised the test stages."                  → embed → [0.69, 0.14, …]
                                                                   similarity = 0.94 → same topic
```

---

## The Underline Analogy

A careful reader who underlines each sentence and draws a dividing line wherever the topic changes. The lines are not evenly spaced — a topic might last 2 sentences or 20 sentences. Length is entirely determined by topic coherence, not a token budget.

---

## Threshold Configuration

The `semantic_breakpoint_threshold` is a **percentile** of similarity scores within the document:

| Threshold | Effect |
|---|---|
| 70 | Split at the bottom 30% similarity points — produces many small chunks |
| 85 (default) | Split at the bottom 15% — balanced chunks |
| 95 | Only the sharpest topic shifts trigger splits — produces fewer, larger chunks |

The percentile approach is **self-calibrating**: a document about one topic will have naturally high similarities throughout, so the threshold adapts rather than splitting artificially.

---

## Why Not Use This for Code or Tables?

Semantic chunking embeds every sentence independently. For code, a single line (e.g., `return result`) has no meaningful semantic content without its surrounding function. For tables, rows are not "sentences." Use Code-Aware and Row-Aware strategies respectively.

---

## Cost Consideration

Semantic chunking calls the embedding model **once per sentence** at ingestion time — not once per chunk. For a 10,000-word transcript with 500 sentences, that is 500 embedding calls. This is 5–20× more expensive than other strategies.

Mitigation strategies:
- Use a fast, cheap embedding model for ingestion (e.g., Titan Embed v1 instead of v2)
- Reserve semantic chunking only for document types that genuinely need it (transcripts, chat)
- Pre-classify documents and only route transcripts through this strategy

---

## Code Reference

```python
from chunking_engine.chunkers import SemanticChunker, ChunkingConfig
from chunking_engine.models.base import ProviderFactory

provider = ProviderFactory.create()
embeddings = provider.get_embeddings()

config = ChunkingConfig(semantic_threshold=85.0)
chunker = SemanticChunker(embeddings=embeddings, config=config)
chunks = chunker.chunk(transcript_text, {"source": "meeting-2024-09-15", "meeting_id": "M001"})

# Each chunk has:
# chunk.metadata["position"]  → "2 of 7"
# chunk.metadata["chunk_id"]  → 1
# chunk.metadata["doc_type"]  → "transcript"
```

---

## How Semantic Chunking Works Internally

```
Full text
    │
    ▼
Sentence splitter (. ! ?)
    │
    ▼ sentences: [s1, s2, s3, … sN]
    │
    ▼
embed_documents([s1, s2, s3, … sN])  ← one batch call
    │
    ▼ vectors: [v1, v2, v3, … vN]
    │
    ▼
cosine_similarity(vi, vi+1) for all i
    │
    ▼ similarity scores: [0.97, 0.31, 0.94, 0.88, 0.25, …]
    │
    ▼
split where score < percentile(scores, threshold)
    │
    ▼
chunks: [ [s1,s2], [s3,s4], [s5], … ]
```

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `semantic` | |
| `doc_type` | `transcript` | |
| `chunk_id` | Sequential integer | |
| `position` | e.g. `"3 of 8"` | Human-readable position |

---

## Interview Line

> *"For transcripts I use semantic chunking because there are no headers — the conversation flows freely across topics. The chunker embeds each sentence and measures similarity between consecutive sentences. When similarity drops below the 85th percentile threshold, that signals a topic shift and a new chunk begins. This way each chunk discusses one coherent topic even if the speaker never explicitly announced the change."*
