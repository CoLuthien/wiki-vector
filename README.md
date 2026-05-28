# wiki-vector

Local hybrid index layer for a Markdown LLM Wiki. The Markdown wiki remains the
source of truth; this project builds a local LanceDB-backed index and exposes
locator-style search through both a CLI and an MCP server.

Current backend: `lancedb-hybrid` under `<wiki>/.vector/`.

- Dense vector candidate retrieval: LanceDB table at `<wiki>/.vector/lancedb`.
- Local offline embedder: `hashing-ngram-256` for deterministic vectors without
  model downloads.
- Lexical retrieval: local BM25 over the same Markdown heading chunks.
- Fusion: `0.65 * vector_score + 0.35 * bm25_score` after score normalization.

The embedder is intentionally swappable. A later `bge-m3`/sentence-transformer
backend can replace `HashingNgramEmbedder` without changing CLI/MCP tool names.

## Install / run locally

Use `uv` so dependencies such as LanceDB and MCP are present:

```bash
cd /workspace/wiki-vector
uv run python -m pytest -q
uv run wiki-vector --wiki /workspace/llm-wiki index
uv run wiki-vector --wiki /workspace/llm-wiki search "Gemma4 RyzenAI GQO" --json
uv run wiki-vector --wiki /workspace/llm-wiki read concepts/gemma4-ryzenai-runtime-171-runbook.md --heading "NPU verification"
```

Index status example:

```json
{
  "backend": "lancedb-hybrid",
  "embedding_model": "hashing-ngram-256",
  "bm25_weight": 0.35,
  "vector_weight": 0.65
}
```

Search results include both component scores:

```json
{
  "score": 0.938452,
  "bm25_score": 6.396262,
  "vector_score": 0.394886,
  "path": "concepts/gemma4-ryzenai-runtime-171-runbook.md",
  "heading": "Runtime 1.7.1 findings to treat as provisional"
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

- `wiki_search(query, limit=8, include_raw=false)` — returns candidate path/heading/snippet locators.
- `wiki_read(path, heading=null)` — reads Markdown source of truth.
- `wiki_reindex(include_raw=false)` — rebuilds the local LanceDB/BM25 hybrid index.
- `wiki_status()` — reports index metadata.

Policy: `wiki_search` results are locators, not authoritative evidence. Agents
should call `wiki_read` on the returned path/heading before answering.

## Scope

Designed for `/workspace/llm-wiki`, shared across Hermes profiles and local coding
agents. Raw sources are excluded by default; pass `include_raw=true` only for deep
recall.
