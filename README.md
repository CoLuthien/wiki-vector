# wiki-vector

Local hybrid index layer for a Markdown LLM Wiki. The Markdown wiki remains the
source of truth; this project builds a local LanceDB-backed index and exposes
locator-style search through both a CLI and an MCP server.

Current backend: `lancedb-hybrid` under `<wiki>/.vector/`.

- Dense vector candidate retrieval: LanceDB table at `<wiki>/.vector/lancedb`.
- Local offline embedder: `hashing-ngram-256` for deterministic vectors without
  model downloads.
- Swappable model execution backends: `hashing-ngram` by default, with optional
  `openvino-bge-m3` for `BAAI/bge-m3` on Intel OpenVINO devices such as `NPU`.
- Lexical retrieval: local BM25 over the same Markdown heading chunks.
- Fusion: `0.65 * vector_score + 0.35 * bm25_score` after score normalization.
- For fixed-window neural embedders such as `openvino-bge-m3`, long Markdown
  heading chunks are split into additional vector-only line-range rows before
  embedding. This avoids silently losing everything after
  `--embedding-max-length`; search results point at the matching subchunk line
  range while `read --heading` still returns the full source section.

The embedder is intentionally swappable. `HashingNgramEmbedder` and
`OpenVINOBgeM3Embedder` both implement the same small `Embedder` protocol, so a
later sentence-transformer, remote inference, or other OpenVINO model backend can
be added without changing CLI/MCP tool names or the LanceDB/BM25 fusion layer.
Model-backed embedders are cached by a process-local `EmbeddingRuntimeCache`
outside `WikiIndex`, so long-lived MCP/server processes can reuse the same
OpenVINO/bge runtime across repeated `WikiIndex` instances. A one-shot CLI
process still has to create its process-local runtime on each invocation; use a
long-lived MCP/server process for repeated neural searches.

## Install / run locally

Use `uv` so dependencies such as LanceDB and MCP are present:

```bash
cd /workspace/wiki-vector
uv run python -m pytest -q
uv run wiki-vector --wiki /workspace/llm-wiki index
uv run wiki-vector --wiki /workspace/llm-wiki search "Gemma4 RyzenAI GQO" --json
uv run wiki-vector --wiki /workspace/llm-wiki read concepts/gemma4/runtime/ryzenai-runtime-171-runbook.md --heading "NPU verification"
uv run wiki-vector --wiki /workspace/llm-wiki read concepts/wiki-vector/mcp.md --start-line 141 --end-line 157 --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --semantic --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --readability-model Tymoteusz/distilbert-base-uncased-kaggle-readability --json
uv run wiki-vector --wiki /workspace/llm-wiki verbosity-audit --limit 20 --json
```

To build/search with `BAAI/bge-m3` through OpenVINO on Intel NPU, install the
optional runtime dependencies and select the model backend explicitly:

```bash
cd /workspace/wiki-vector
uv sync --extra openvino
uv run wiki-vector \
  --wiki /workspace/llm-wiki \
  --embedding-backend openvino-bge-m3 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device NPU \
  index
uv run wiki-vector \
  --wiki /workspace/llm-wiki \
  --embedding-backend openvino-bge-m3 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device NPU \
  search "wiki-vector verbosity methodology" --json
```

Equivalent MCP config uses environment variables so tool names stay stable:

```yaml
mcp_servers:
  llm_wiki:
    command: "uv"
    args: ["run", "--project", "/workspace/wiki-vector", "wiki-vector-mcp"]
    env:
      WIKI_PATH: "/workspace/llm-wiki"
      WIKI_VECTOR_EMBEDDING_BACKEND: "openvino-bge-m3"
      WIKI_VECTOR_EMBEDDING_MODEL: "BAAI/bge-m3"
      WIKI_VECTOR_EMBEDDING_DEVICE: "NPU"
      WIKI_VECTOR_EMBEDDING_MAX_LENGTH: "512"
```

OpenVINO/NPU uses static sequence length before compile to avoid unbounded dynamic
shape failures in the Intel NPU compiler. `--embedding-max-length` /
`WIKI_VECTOR_EMBEDDING_MAX_LENGTH` controls the maximum model window and also the
target size for vector-only locator subchunks. Long heading sections are embedded
as multiple rows instead of being represented only by the first model window.
Search must use the same embedding backend/model/dimensions/max-length that built
the index. If these options differ, `wiki-vector search` exits with an embedding
mismatch error instead of falling back to expensive per-chunk re-embedding. Re-run
`wiki-vector ... index` with the desired embedding options before searching.

Index status example:

```json
{
  "backend": "lancedb-hybrid",
  "embedding_backend": "hashing-ngram",
  "embedding_model": "hashing-ngram-256",
  "embedding_dimensions": 256,
  "embedding_device": null,
  "embedding_max_length": null,
  "bm25_weight": 0.35,
  "vector_weight": 0.65
}
```

Search results include component scores and exact read-location hints:

```json
{
  "score": 0.938452,
  "bm25_score": 6.396262,
  "vector_score": 0.394886,
  "path": "concepts/gemma4-ryzenai-runtime-171-runbook.md",
  "heading": "Runtime 1.7.1 findings to treat as provisional",
  "start_line": 42,
  "end_line": 57,
  "read_hint": "concepts/gemma4-ryzenai-runtime-171-runbook.md#Runtime 1.7.1 findings to treat as provisional lines 42-57"
}
```

## MCP server

```bash
WIKI_PATH=/workspace/llm-wiki uv run --project /workspace/wiki-vector wiki-vector-mcp
```

Hermes config shape:

```yaml
mcp_servers:
  llm_wiki:
    command: "uv"
    args: ["run", "--project", "/workspace/wiki-vector", "wiki-vector-mcp"]
    env:
      WIKI_PATH: "/workspace/llm-wiki"
    timeout: 120
    connect_timeout: 60
```

MCP tools:

- `wiki_search(query, limit=8, include_raw=false)` — returns candidate path/heading/snippet locators plus `start_line`, `end_line`, and `read_hint` for where to read.
- `wiki_read(path, heading=null, start_line=null, end_line=null)` — reads Markdown source of truth; `start_line`/`end_line` select an inclusive 1-indexed source-file range.
- `wiki_write(path, content, mode="create", reindex=true)` — creates, overwrites, or appends a Markdown page and optionally rebuilds the index.
- `wiki_reindex(include_raw=false)` — rebuilds the local LanceDB/BM25 hybrid index.
- `wiki_status()` — reports index metadata.
- `wiki_is_verbose(path, include_code=false, compare_to=null, semantic=false, readability_model=null)` — analyzes a page for verbosity and returns `is_verbose`, `score`, `severity`, metric details, exact section line ranges, and restructuring suggestions. With `semantic=true`, it also returns a separate advisory `semantic` block from the configured embedder; this is a semantic-structure proxy, not a readability model. With `readability_model=<HF model id>`, it returns a separate advisory `readability_model` block from a Transformers text-classification/regression model explicitly trained for readability/text complexity. Neither optional block changes the deterministic `score`.
- `wiki_verbosity_audit(limit=20, include_raw=false, severity=null)` — scans curated wiki pages and returns the highest-verbosity candidates sorted by score.

Verbosity policy: `wiki_is_verbose` is advisory, not an automatic rewrite trigger. Agents should inspect `reasons`, `sections[].start_line/end_line`, and `suggestions` before deciding whether to create a hub page, split by heading, archive chronology, or add wikilinks. Semantic mode follows the same policy: embedding/neural proxy scores (`semantic_structure_score`, `coherence_score`, `semantic_redundancy_score`, `rewrite_preservation_score`) are reported under `semantic` as `advisory_only`, `not_used_in_default_score`, and `not_readability_model=true`. Readability model mode is also advisory; raw logits/probabilities need corpus/language calibration before cron automation.

Policy: `wiki_search` results are locators, not authoritative evidence. Agents
should call `wiki_read` on the returned path/heading or returned line range before answering.

## Scope

Designed for `/workspace/llm-wiki`, shared across Hermes profiles and local coding
agents. Raw sources are excluded by default; pass `include_raw=true` only for deep
recall.
