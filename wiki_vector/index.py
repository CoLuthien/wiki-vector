from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import re
import time
from collections import Counter, defaultdict

from .chunking import Chunk, chunk_document
from .markdown import iter_wiki_markdown_files, parse_markdown

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+|[가-힣]+")


@dataclass(frozen=True)
class IndexStatus:
    wiki_path: str
    backend: str
    pages_indexed: int
    chunks_indexed: int
    include_raw: bool
    last_indexed_at: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    path: str
    title: str
    heading: str
    score: float
    type: str
    tags: list[str]
    confidence: str | None
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReadResult:
    path: str
    title: str
    heading: str | None
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


class WikiIndex:
    """Small local hybrid-ish lexical index for Markdown wiki chunks.

    This MVP stores JSONL under .vector. The public API is intentionally shaped so
    a real embedding backend (LanceDB + bge-m3) can replace/augment scoring later
    without changing CLI/MCP tools.
    """

    def __init__(self, wiki_path: str | Path):
        self.wiki_path = Path(wiki_path).expanduser().resolve()
        self.vector_dir = self.wiki_path / ".vector"
        self.chunks_file = self.vector_dir / "chunks.jsonl"
        self.manifest_file = self.vector_dir / "manifest.json"

    def reindex(self, include_raw: bool = False) -> IndexStatus:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        pages = 0
        files_meta: dict[str, dict] = {}
        for path in iter_wiki_markdown_files(self.wiki_path, include_raw=include_raw):
            rel = path.relative_to(self.wiki_path)
            text = path.read_text(encoding="utf-8")
            doc = parse_markdown(rel, text)
            doc_chunks = chunk_document(doc)
            chunks.extend(doc_chunks)
            pages += 1
            st = path.stat()
            files_meta[rel.as_posix()] = {"mtime": st.st_mtime, "size": st.st_size, "chunks": len(doc_chunks)}
        with self.chunks_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        status = IndexStatus(
            wiki_path=str(self.wiki_path),
            backend="jsonl-lexical-mvp",
            pages_indexed=pages,
            chunks_indexed=len(chunks),
            include_raw=include_raw,
            last_indexed_at=time.time(),
        )
        manifest = status.to_dict() | {"files": files_meta}
        self.manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def status(self) -> IndexStatus:
        if not self.manifest_file.exists():
            return IndexStatus(str(self.wiki_path), "jsonl-lexical-mvp", 0, 0, False, 0.0)
        data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        return IndexStatus(
            wiki_path=data.get("wiki_path", str(self.wiki_path)),
            backend=data.get("backend", "jsonl-lexical-mvp"),
            pages_indexed=int(data.get("pages_indexed", 0)),
            chunks_indexed=int(data.get("chunks_indexed", 0)),
            include_raw=bool(data.get("include_raw", False)),
            last_indexed_at=float(data.get("last_indexed_at", 0.0)),
        )

    def search(self, query: str, limit: int = 8, include_raw: bool = False, types: list[str] | None = None, tags: list[str] | None = None) -> list[SearchResult]:
        chunks = self._load_chunks()
        if not chunks:
            self.reindex(include_raw=include_raw)
            chunks = self._load_chunks()
        q_terms = _tokens(query)
        if not q_terms:
            return []
        doc_freq = Counter()
        chunk_terms = []
        for chunk in chunks:
            terms = set(_tokens(_searchable_text(chunk)))
            chunk_terms.append(terms)
            doc_freq.update(terms)
        n = max(len(chunks), 1)
        results: list[SearchResult] = []
        for chunk, terms in zip(chunks, chunk_terms):
            if chunk.get("is_raw") and not include_raw:
                continue
            if types and chunk.get("type") not in types:
                continue
            if tags and not set(tags).intersection(set(chunk.get("tags") or [])):
                continue
            text = _searchable_text(chunk)
            tf = Counter(_tokens(text))
            score = 0.0
            for term in q_terms:
                if term in tf:
                    idf = math.log((n + 1) / (doc_freq[term] + 0.5)) + 1.0
                    score += (1 + math.log(tf[term])) * idf
            # phrase/substr boost helps exact symbols like xrt-smi, CastAvx, NX_GEMMA4_FULL_CACHE_PRESENT.
            q_lower = query.lower()
            if q_lower in text.lower():
                score += 5.0
            if not chunk.get("is_raw"):
                score += 0.25
            if score <= 0:
                continue
            results.append(SearchResult(
                path=chunk["path"],
                title=chunk.get("title") or chunk["path"],
                heading=chunk.get("heading") or "",
                score=round(score, 6),
                type=chunk.get("type") or "page",
                tags=list(chunk.get("tags") or []),
                confidence=chunk.get("confidence"),
                snippet=_snippet(chunk.get("text", ""), q_terms),
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def read(self, path: str, heading: str | None = None) -> ReadResult:
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("path must be relative to wiki root")
        full = self.wiki_path / rel
        text = full.read_text(encoding="utf-8")
        doc = parse_markdown(rel, text)
        if heading is None:
            return ReadResult(path=rel.as_posix(), title=doc.title, heading=None, content=text)
        for chunk in chunk_document(doc):
            if chunk.heading == heading:
                return ReadResult(path=rel.as_posix(), title=doc.title, heading=heading, content=chunk.text)
        raise ValueError(f"heading not found: {heading}")

    def _load_chunks(self) -> list[dict]:
        if not self.chunks_file.exists():
            return []
        return [json.loads(line) for line in self.chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _searchable_text(chunk: dict) -> str:
    return "\n".join([
        chunk.get("title", ""),
        chunk.get("heading", ""),
        " ".join(chunk.get("tags") or []),
        chunk.get("text", ""),
    ])


def _snippet(text: str, q_terms: list[str], max_len: int = 240) -> str:
    lower = text.lower()
    pos = min([lower.find(t) for t in q_terms if lower.find(t) >= 0] or [0])
    start = max(pos - 80, 0)
    end = min(start + max_len, len(text))
    snippet = text[start:end].replace("\n", " ").strip()
    if start:
        snippet = "…" + snippet
    if end < len(text):
        snippet += "…"
    return snippet
