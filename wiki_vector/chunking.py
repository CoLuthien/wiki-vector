from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re

from .markdown import MarkdownDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    slug: str
    title: str
    type: str
    tags: list[str]
    confidence: str | None
    heading: str
    level: int
    text: str
    is_raw: bool
    text_hash: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_document(doc: MarkdownDocument) -> list[Chunk]:
    body = doc.body.strip()
    if not body:
        return []
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [_make_chunk(doc, doc.title, 0, body, 0, 1, _line_count(body))]

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section = body[start:end].strip()
        heading = match.group(2).strip()
        level = len(match.group(1))
        if section:
            start_line = _line_number_at(body, start)
            end_line = _line_number_at(body, max(end - 1, start))
            chunks.append(_make_chunk(doc, heading, level, section, i, start_line, end_line))
    return chunks


def _make_chunk(doc: MarkdownDocument, heading: str, level: int, text: str, ordinal: int, start_line: int, end_line: int) -> Chunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cid = hashlib.sha256(f"{doc.path}\0{heading}\0{ordinal}\0{digest}".encode("utf-8")).hexdigest()[:24]
    line_offset = doc.frontmatter_lines
    return Chunk(
        id=cid,
        path=doc.path,
        slug=doc.slug,
        title=doc.title,
        type=doc.type,
        tags=list(doc.tags),
        confidence=doc.confidence,
        heading=heading,
        level=level,
        text=text,
        is_raw=doc.path.startswith("raw/"),
        text_hash=digest,
        start_line=start_line + line_offset,
        end_line=end_line + line_offset,
    )


def _line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _line_count(text: str) -> int:
    return text.count("\n") + 1 if text else 1
