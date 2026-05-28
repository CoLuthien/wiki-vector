# wiki-vector

Local search/index layer for a Markdown LLM Wiki. The Markdown wiki remains the
source of truth; this project builds a local index and exposes locator-style
search through both a CLI and an MCP server.

Current MVP backend: JSONL sparse lexical/vector scoring under `<wiki>/.vector/`.
The public API is intentionally shaped so LanceDB + local embedding model can be
added later without changing CLI/MCP tool names.

## Install / run locally

```bash
cd /workspace/wiki-vector
python -m pytest -q
python -m wiki_vector.cli --wiki /workspace/llm-wiki index
python -m wiki_vector.cli --wiki /workspace/llm-wiki search "Gemma4 RyzenAI GQO" --json
python -m wiki_vector.cli --wiki /workspace/llm-wiki read concepts/gemma4-ryzenai-runtime-171-runbook.md --heading "NPU verification"
```

## MCP server

```bash
WIKI_PATH=/workspace/llm-wiki python -m wiki_vector.mcp_server
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
- `wiki_reindex(include_raw=false)` — rebuilds the local index.
- `wiki_status()` — reports index metadata.

Policy: `search` results are locators, not authoritative evidence. Agents should
call `read` on the returned path/heading before answering.

## Scope

Designed for `/workspace/llm-wiki`, shared across Hermes profiles and local coding
agents. Raw sources are excluded by default; pass `include_raw=true` only for deep
recall.
