# Strategy 5 — Code-Aware

> **Best for: source code of any programming language — functions, classes, methods must never be split.**

---

## How It Works

Parses code using language-specific rules and splits **exclusively at function, class, or method boundaries**. A function that starts on line 10 and ends on line 50 will always appear as a single, complete chunk.

**Two modes:**

### Mode 1 — LangChain Built-in (high fidelity)

For languages with first-class LangChain support, the built-in `RecursiveCharacterTextSplitter.from_language()` is used. It understands language-specific block delimiters.

Supported: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, Ruby, Kotlin, Scala, Swift

### Mode 2 — Generic Regex Fallback

For all other languages, a regex-based boundary detector finds unit starts using patterns like:
- `func \w+(` — Go-style functions
- `def \w+(` — Python / Ruby
- `class \w+` — any OO language
- `function \w+(` — JavaScript/PHP

The unit spans from its opening line to the start of the next detected unit.

---

## Why a Partial Function Is Useless

```
Bad chunk (fixed-size split mid-function):
──────────────────────────────────────────
    if cache_hit:
        return cached_result
    result = expensive_computation(input_data)
    cache[key] = result
    return result

# The LLM has no idea:
# - What function this is
# - What the parameters are
# - What "cache" refers to
# - What "expensive_computation" does
```

```
Good chunk (code-aware, full function):
──────────────────────────────────────────
def compute_with_cache(input_data: str, cache: dict) -> str:
    """Compute result, using cache to avoid redundant work."""
    key = hash(input_data)
    if key in cache:
        return cache[key]
    result = expensive_computation(input_data)
    cache[key] = result
    return result

# The LLM understands the full contract: signature, parameters, return type, purpose.
```

---

## Multi-Language Support

```python
from chunking_engine.chunkers import CodeAwareChunker, ChunkingConfig

# Python — LangChain built-in AST-aware split
config = ChunkingConfig(language="python", chunk_size=2000, chunk_overlap=100)

# Go — LangChain built-in
config = ChunkingConfig(language="go")

# TypeScript — LangChain built-in
config = ChunkingConfig(language="typescript")

# Any other language — generic regex fallback
config = ChunkingConfig(language="ruby")
config = ChunkingConfig(language="elixir")

chunker = CodeAwareChunker(config)
chunks = chunker.chunk(source_code, {"source": "inventory.py"})
```

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `code_aware` | |
| `doc_type` | `source_code` | |
| `language` | e.g. `"python"` | From config |
| `unit_type` | `function` / `class` / `method` / `block` | Generic mode only |
| `unit_name` | e.g. `"calculate_revenue"` | Generic mode only |
| `chunk_id` | Sequential integer | |

---

## Metadata-Powered Filtering

The `unit_name` and `unit_type` in metadata enable targeted retrieval:

```python
# Find all class definitions in the codebase
results = milvus.search(query_vec, filter_expr='unit_type == "class"')

# Find a specific function by name
results = milvus.search(query_vec, filter_expr='unit_name == "calculate_revenue"')
```

---

## Code + Documentation Pattern

For repositories where source files sit alongside README/docs files:

```python
from chunking_engine.chunkers import CodeAwareChunker, RecursiveCharacterChunker
from chunking_engine.registry import StrategyRegistry

# Route by file extension
if file_path.endswith((".py", ".js", ".ts", ".go", ".java")):
    chunker = CodeAwareChunker(ChunkingConfig(language=detect_language(file_path)))
else:
    chunker = RecursiveCharacterChunker()  # README, docs, comments

chunks = chunker.chunk(content, {"source": file_path})
```

---

## Interview Line

> *"For source code I split at function and class boundaries. Each unit becomes one complete chunk — this is critical because a partial function chunk is meaningless: the LLM needs the full signature, parameters, and body to understand what the code does. The unit name and type are stored in metadata so I can filter to retrieve only specific functions when doing targeted impact analysis."*
