from __future__ import annotations

import os
from typing import Any

from .mcp_tools import wiki_is_verbose, wiki_read, wiki_reindex, wiki_search, wiki_status, wiki_verbosity_audit, wiki_write


def _wiki_path(value: str | None = None) -> str:
    return value or os.environ.get("WIKI_PATH") or os.path.expanduser("~/wiki")


def main() -> None:
    """Run stdio MCP server exposing wiki-vector tools.

    The import is intentionally lazy so the CLI/tests work without the optional
    MCP SDK. Install with `pip install 'wiki-vector[mcp]'` or `pip install mcp`.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit("MCP SDK not installed. Install with: pip install mcp") from exc

    mcp = FastMCP("llm-wiki-vector")

    @mcp.tool(name="wiki_search")
    def wiki_search_tool(query: str, limit: int = 8, include_raw: bool = False, wiki_path: str | None = None) -> dict[str, Any]:
        """Search the local LLM Wiki index. Returns candidate locators/snippets plus start_line, end_line, and read_hint."""
        return wiki_search(_wiki_path(wiki_path), query, limit=limit, include_raw=include_raw)

    @mcp.tool(name="wiki_read")
    def wiki_read_tool(
        path: str,
        heading: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        wiki_path: str | None = None,
    ) -> dict[str, Any]:
        """Read a Markdown page, heading section, or 1-indexed source line range from the wiki source of truth."""
        return wiki_read(_wiki_path(wiki_path), path, heading=heading, start_line=start_line, end_line=end_line)

    @mcp.tool(name="wiki_reindex")
    def wiki_reindex_tool(include_raw: bool = False, wiki_path: str | None = None) -> dict[str, Any]:
        """Rebuild the local index from Markdown wiki files."""
        return wiki_reindex(_wiki_path(wiki_path), include_raw=include_raw)

    @mcp.tool(name="wiki_write")
    def wiki_write_tool(path: str, content: str, mode: str = "create", reindex: bool = True, wiki_path: str | None = None) -> dict[str, Any]:
        """Create, overwrite, or append to a Markdown wiki page, then optionally reindex."""
        return wiki_write(_wiki_path(wiki_path), path=path, content=content, mode=mode, reindex=reindex)

    @mcp.tool(name="wiki_status")
    def wiki_status_tool(wiki_path: str | None = None) -> dict[str, Any]:
        """Return current local index status."""
        return wiki_status(_wiki_path(wiki_path))

    @mcp.tool(name="wiki_is_verbose")
    def wiki_is_verbose_tool(path: str, include_code: bool = False, compare_to: str | None = None, semantic: bool = False, readability_model: str | None = None, wiki_path: str | None = None) -> dict[str, Any]:
        """Analyze whether a wiki page is verbose; optionally include advisory semantic proxy or readability-model metrics."""
        return wiki_is_verbose(_wiki_path(wiki_path), path=path, include_code=include_code, compare_to=compare_to, semantic=semantic, readability_model=readability_model)

    @mcp.tool(name="wiki_verbosity_audit")
    def wiki_verbosity_audit_tool(limit: int = 20, include_raw: bool = False, severity: str | None = None, wiki_path: str | None = None) -> dict[str, Any]:
        """Return highest-verbosity wiki pages sorted by score."""
        return wiki_verbosity_audit(_wiki_path(wiki_path), limit=limit, include_raw=include_raw, severity=severity)

    mcp.run()


if __name__ == "__main__":
    main()
