from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

WIKI_PAGE_DIRS = ("entities", "concepts", "comparisons", "queries")
RAW_DIR = "raw"


@dataclass(frozen=True)
class MarkdownDocument:
    path: str
    slug: str
    title: str
    type: str
    tags: list[str] = field(default_factory=list)
    confidence: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    frontmatter_lines: int = 0
    body: str = ""


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str, int]:
    if not text.startswith("---\n"):
        return {}, text, 0
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text, 0
    raw = text[4:end]
    tail = text[end + len("\n---"):]
    body = tail.lstrip("\n")
    body_start_index = len(text) - len(body)
    frontmatter_lines = text.count("\n", 0, body_start_index)
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        data = {}
    return data, body, frontmatter_lines


def parse_markdown(path: Path, text: str) -> MarkdownDocument:
    fm, body, frontmatter_lines = _split_frontmatter(text)
    rel = path.as_posix()
    title = str(fm.get("title") or _title_from_body(body) or path.stem)
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t) for t in tags]
    doc_type = str(fm.get("type") or _type_from_path(path))
    confidence = fm.get("confidence")
    return MarkdownDocument(
        path=rel,
        slug=path.stem,
        title=title,
        type=doc_type,
        tags=tags,
        confidence=str(confidence) if confidence is not None else None,
        frontmatter=fm,
        frontmatter_lines=frontmatter_lines,
        body=body,
    )


def _title_from_body(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _type_from_path(path: Path) -> str:
    parts = path.parts
    if "entities" in parts:
        return "entity"
    if "concepts" in parts:
        return "concept"
    if "comparisons" in parts:
        return "comparison"
    if "queries" in parts:
        return "query"
    if "raw" in parts:
        return "raw"
    return "page"


def iter_wiki_markdown_files(wiki_path: Path, include_raw: bool = False) -> Iterable[Path]:
    wiki_path = Path(wiki_path)
    for dirname in WIKI_PAGE_DIRS:
        base = wiki_path / dirname
        if base.exists():
            yield from sorted(p for p in base.rglob("*.md") if ".vector" not in p.parts)
    if include_raw:
        base = wiki_path / RAW_DIR
        if base.exists():
            yield from sorted(p for p in base.rglob("*.md") if ".vector" not in p.parts)
