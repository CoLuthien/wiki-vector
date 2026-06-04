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
uv run wiki-vector --wiki /workspace/llm-wiki search "wiki-vector retrieval diagnostics" --json --explain
uv run wiki-vector --wiki /workspace/llm-wiki read concepts/gemma4/runtime/ryzenai-runtime-171-runbook.md --heading "NPU verification"
uv run wiki-vector --wiki /workspace/llm-wiki read concepts/wiki-vector/mcp.md --start-line 141 --end-line 157 --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --semantic --json
uv run wiki-vector --wiki /workspace/llm-wiki is-verbose concepts/wiki-vector/mcp.md --readability-model Tymoteusz/distilbert-base-uncased-kaggle-readability --json
uv run wiki-vector --wiki /workspace/llm-wiki verbosity-audit --limit 20 --json
uv run wiki-vector --wiki /workspace/llm-wiki change-summary --json
uv run wiki-vector --wiki /workspace/llm-wiki change-summary --update --change-count-threshold 10 --line-threshold 200 --json
uv run wiki-vector --wiki /workspace/llm-wiki consistency-audit --json
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

Use `search --explain --json` when debugging why a query did or did not retrieve a page. Explain mode returns `{results, explain}` instead of a bare result list. The `explain` block is deterministic and local: BM25 is decomposed into per-token `keyword_contributions`, while vector retrieval is reported as whole-query stage trace because dense embeddings do not provide honest per-keyword contributions.

```json
{
  "explain": {
    "query_terms": ["wiki-vector", "retrieval", "diagnostics"],
    "weights": {"bm25": 0.35, "vector": 0.65},
    "candidate_counts": {"chunks_after_filters": 239, "bm25_nonzero": 4, "vector_hits": 50, "returned": 8},
    "keyword_contributions": [{"term": "diagnostics", "matching_chunks": 2, "top_hits": [{"path": "concepts/wiki-vector/mcp.md", "bm25_contribution": 1.23}]}],
    "trace": [{"stage": "bm25", "path": "concepts/wiki-vector/mcp.md", "score": 2.4}, {"stage": "vector", "path": "queries/wiki/document-verbosity-methodology.md", "score": 0.72}]
  }
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

- `wiki_search(query, limit=8, include_raw=false, explain=false)` — returns candidate path/heading/snippet locators plus `start_line`, `end_line`, and `read_hint` for where to read. With `explain=true`, also returns deterministic BM25 keyword contributions, candidate counts, and BM25/vector stage trace.
- `wiki_read(path, heading=null, start_line=null, end_line=null)` — reads Markdown source of truth; `start_line`/`end_line` select an inclusive 1-indexed source-file range.
- `wiki_write(path, content, mode="create", heading=null, occurrence=null, reindex=true)` — creates, overwrites, appends, or replaces one ATX Markdown heading section and optionally rebuilds the index. `mode="replace-section"` requires `heading`; it uses hierarchical section spans (nested lower-level headings are replaced with the target), preserves YAML frontmatter, ignores headings inside fenced code blocks, and fails on duplicate headings unless `occurrence` selects the 1-indexed match. Use this for section replacement / heading-scoped write / 섹션별 수정 / 부분 섹션 교체 without overwriting a whole page.
- `wiki_reindex(include_raw=false)` — rebuilds the local LanceDB/BM25 hybrid index.
- `wiki_status()` — reports index metadata.
- `wiki_is_verbose(path, include_code=false, compare_to=null, semantic=false, readability_model=null)` — analyzes a page for verbosity and returns `is_verbose`, `score`, `severity`, metric details, exact section line ranges, and restructuring suggestions. With `semantic=true`, it also returns a separate advisory `semantic` block from the configured embedder; this is a semantic-structure proxy, not a readability model. With `readability_model=<HF model id>`, it returns a separate advisory `readability_model` block from a Transformers text-classification/regression model explicitly trained for readability/text complexity. Neither optional block changes the deterministic `score`.
- `wiki_verbosity_audit(limit=20, include_raw=false, severity=null)` — scans curated wiki pages and returns the highest-verbosity candidates sorted by score.
- `wiki_change_summary(include_raw=false, update=false, since=null, change_count_threshold=1, byte_threshold=1, line_threshold=1)` — uses a SQLite maintenance ledger at `<wiki>/.vector/changes.sqlite` to report per-file `change_count`, last change kind, pending added/modified/deleted events, and aggregate byte/line/diff sizes. `update=false` is a dry-run gate for cron jobs; `update=true` records pending changes and advances the baseline snapshot. Raw files are excluded unless `include_raw=true`.
- `wiki_consistency_audit(include_raw=null)` — read-only audit that compares current Markdown files, manifest metadata, `chunks.jsonl`, expected vector locator rows, and LanceDB row count. It returns `ok`, `summary`, structured `issues`, and `recommendations`; run `wiki_reindex(...)` to repair reported drift.

Verbosity policy: `wiki_is_verbose` is advisory, not an automatic rewrite trigger. Agents should inspect `reasons`, `sections[].start_line/end_line`, and `suggestions` before deciding whether to create a hub page, split by heading, archive chronology, or add wikilinks. Semantic mode follows the same policy: embedding/neural proxy scores (`semantic_structure_score`, `coherence_score`, `semantic_redundancy_score`, `rewrite_preservation_score`) are reported under `semantic` as `advisory_only`, `not_used_in_default_score`, and `not_readability_model=true`. Readability model mode is also advisory; raw logits/probabilities need corpus/language calibration before cron automation.

Change-summary policy: use `wiki_change_summary(update=false, change_count_threshold=N, line_threshold=M)` as the cheap cron preflight. If `significant_change` is false, the scheduled wiki cleanup can stay silent. If true, run the restructuring/audit job and finish by calling `wiki_change_summary(update=true, ...)` or `wiki_reindex()` to record the new baseline. This ledger is operational metadata, not a source of truth; Markdown remains authoritative.

Consistency-audit policy: use `wiki_consistency_audit()` when search results look stale, after manual Markdown edits, after changing embedding options, or before a scheduled maintenance job that depends on fresh locators. The audit is intentionally read-only and should not silently fix anything. High-severity issues such as `chunk_count_mismatch`, `manifest_chunk_count_mismatch`, `indexed_file_missing`, `stale_chunks_in_index`, or `vector_row_count_mismatch` mean the locator layer is stale or corrupt; follow the returned recommendation and run `wiki_reindex(include_raw=<reported include_raw>)`.

Policy: `wiki_search` results are locators, not authoritative evidence. Agents
should call `wiki_read` on the returned path/heading or returned line range before answering.

## Scope

Designed for `/workspace/llm-wiki`, shared across Hermes profiles and local coding
agents. Raw sources are excluded by default; pass `include_raw=true` only for deep
recall.
