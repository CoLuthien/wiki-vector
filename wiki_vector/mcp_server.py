from __future__ import annotations

import os
from typing import Any

from .mcp_tools import wiki_read, wiki_reindex, wiki_search, wiki_status


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
        """Search the local LLM Wiki index. Returns candidate locators/snippets only."""
        return wiki_search(_wiki_path(wiki_path), query, limit=limit, include_raw=include_raw)

    @mcp.tool(name="wiki_read")
    def wiki_read_tool(path: str, heading: str | None = None, wiki_path: str | None = None) -> dict[str, Any]:
        """Read a Markdown page or heading section from the wiki source of truth."""
        return wiki_read(_wiki_path(wiki_path), path, heading=heading)

    @mcp.tool(name="wiki_reindex")
    def wiki_reindex_tool(include_raw: bool = False, wiki_path: str | None = None) -> dict[str, Any]:
        """Rebuild the local index from Markdown wiki files."""
        return wiki_reindex(_wiki_path(wiki_path), include_raw=include_raw)

    @mcp.tool(name="wiki_status")
    def wiki_status_tool(wiki_path: str | None = None) -> dict[str, Any]:
        """Return current local index status."""
        return wiki_status(_wiki_path(wiki_path))

    mcp.run()


if __name__ == "__main__":
    main()
