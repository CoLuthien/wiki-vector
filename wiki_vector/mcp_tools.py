from __future__ import annotations

from .index import WikiIndex


def wiki_search(wiki_path: str, query: str, limit: int = 8, include_raw: bool = False, types: list[str] | None = None, tags: list[str] | None = None) -> dict:
    """Return semantic/lexical candidate chunks. Results are locators, not authority."""
    index = WikiIndex(wiki_path)
    return {"results": [r.to_dict() for r in index.search(query, limit=limit, include_raw=include_raw, types=types, tags=tags)]}


def wiki_read(wiki_path: str, path: str, heading: str | None = None) -> dict:
    """Read the Markdown source page/section selected from wiki_search."""
    return WikiIndex(wiki_path).read(path, heading=heading).to_dict()


def wiki_reindex(wiki_path: str, include_raw: bool = False) -> dict:
    """Rebuild the local index. Use include_raw=True only when raw sources should be searchable."""
    return WikiIndex(wiki_path).reindex(include_raw=include_raw).to_dict()


def wiki_status(wiki_path: str) -> dict:
    return WikiIndex(wiki_path).status().to_dict()
