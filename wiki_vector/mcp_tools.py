from __future__ import annotations

from .index import WikiIndex


def wiki_search(wiki_path: str, query: str, limit: int = 8, include_raw: bool = False, types: list[str] | None = None, tags: list[str] | None = None, explain: bool = False) -> dict:
    """Return semantic/lexical candidate chunks with path, heading, snippet, score, and line-range read hints. Results are locators, not authority."""
    index = WikiIndex(wiki_path)
    if explain:
        return index.search_explain(query, limit=limit, include_raw=include_raw, types=types, tags=tags)
    return {"results": [r.to_dict() for r in index.search(query, limit=limit, include_raw=include_raw, types=types, tags=tags)]}


def wiki_read(
    wiki_path: str,
    path: str,
    heading: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read a Markdown page, heading section, or 1-indexed source line range."""
    return WikiIndex(wiki_path).read(path, heading=heading, start_line=start_line, end_line=end_line).to_dict()


def wiki_reindex(wiki_path: str, include_raw: bool = False) -> dict:
    """Rebuild the local index. Use include_raw=True only when raw sources should be searchable."""
    return WikiIndex(wiki_path).reindex(include_raw=include_raw).to_dict()


def wiki_write(wiki_path: str, path: str, content: str, mode: str = "create", reindex: bool = True) -> dict:
    """Create, overwrite, or append to a Markdown wiki page, then optionally reindex."""
    return WikiIndex(wiki_path).write(path, content=content, mode=mode, reindex=reindex).to_dict()


def wiki_status(wiki_path: str) -> dict:
    return WikiIndex(wiki_path).status().to_dict()


def wiki_is_verbose(wiki_path: str, path: str, include_code: bool = False, compare_to: str | None = None, semantic: bool = False, readability_model: str | None = None) -> dict:
    """Analyze whether a Markdown wiki page is verbose and optionally add advisory semantic/readability-model metrics."""
    return WikiIndex(wiki_path).is_verbose(path, include_code=include_code, compare_to=compare_to, semantic=semantic, readability_model=readability_model).to_dict()


def wiki_verbosity_audit(wiki_path: str, limit: int = 20, include_raw: bool = False, severity: str | None = None) -> dict:
    """Return highest-verbosity pages in the wiki with structured diagnostics."""
    results = WikiIndex(wiki_path).verbosity_audit(limit=limit, include_raw=include_raw, severity=severity)
    return {"results": [r.to_dict() for r in results], "count": len(results)}


def wiki_change_summary(
    wiki_path: str,
    include_raw: bool = False,
    update: bool = False,
    since: str | None = None,
    change_count_threshold: int = 1,
    byte_threshold: int = 1,
    line_threshold: int = 1,
) -> dict:
    """Return SQLite-backed per-file change counts and pending diff-size summary for cron gating."""
    return WikiIndex(wiki_path).change_summary(
        include_raw=include_raw,
        update=update,
        since=since,
        change_count_threshold=change_count_threshold,
        byte_threshold=byte_threshold,
        line_threshold=line_threshold,
    )
