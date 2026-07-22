from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Pattern

from .markdown import iter_wiki_markdown_files


def grep_wiki(
    wiki_path: str | Path,
    pattern: str,
    *,
    include_raw: bool = False,
    regex: bool = False,
    case_sensitive: bool = False,
    context: int = 2,
    limit: int = 100,
) -> dict[str, Any]:
    """Search Markdown source directly for literal text or a regular expression.

    This intentionally bypasses the vector index so rare identifiers, filenames,
    and exact error strings remain discoverable even when index artifacts are
    absent or stale. One result is returned per matching source line.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    if context < 0:
        raise ValueError("context must be >= 0")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    matcher = _compile_pattern(pattern, regex=regex, case_sensitive=case_sensitive)
    root = Path(wiki_path).expanduser().resolve()
    matches: list[dict[str, Any]] = []
    total_matches = 0
    searched_files = 0

    for path in iter_wiki_markdown_files(root, include_raw=include_raw):
        searched_files += 1
        rel = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            found = matcher.search(line)
            if found is None:
                continue
            total_matches += 1
            if len(matches) >= limit:
                continue
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            start_line = start + 1
            end_line = end
            matches.append(
                {
                    "path": rel,
                    "line": index + 1,
                    "column": found.start() + 1,
                    "match": found.group(0),
                    "line_text": line,
                    "context_start_line": start_line,
                    "context_end_line": end_line,
                    "context": "\n".join(lines[start:end]),
                    "read_hint": f"{rel} lines {start_line}-{end_line}",
                }
            )

    return {
        "pattern": pattern,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "include_raw": include_raw,
        "context_lines": context,
        "limit": limit,
        "searched_files": searched_files,
        "total_matches": total_matches,
        "count": len(matches),
        "truncated": total_matches > len(matches),
        "matches": matches,
    }


def _compile_pattern(pattern: str, *, regex: bool, case_sensitive: bool) -> Pattern[str]:
    expression = pattern if regex else re.escape(pattern)
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(expression, flags)
    except re.error as exc:
        raise ValueError(f"invalid regular expression: {exc}") from exc
