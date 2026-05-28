from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import re
import time
from collections import Counter
from typing import Any

from .chunking import Chunk, chunk_document
from .embeddings import HashingNgramEmbedder
from .markdown import iter_wiki_markdown_files, parse_markdown

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+|[가-힣]+")
BACKEND = "lancedb-hybrid"
BM25_WEIGHT = 0.35
VECTOR_WEIGHT = 0.65


@dataclass(frozen=True)
class IndexStatus:
    wiki_path: str
    backend: str
    pages_indexed: int
    chunks_indexed: int
    include_raw: bool
    last_indexed_at: float
    embedding_model: str = "hashing-ngram-256"
    bm25_weight: float = BM25_WEIGHT
    vector_weight: float = VECTOR_WEIGHT

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
    bm25_score: float = 0.0
    vector_score: float = 0.0

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


@dataclass(frozen=True)
class WriteResult:
    path: str
    mode: str
    bytes_written: int
    reindexed: bool
    status: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class WikiIndex:
    """LanceDB-backed local hybrid index for Markdown wiki chunks.

    Markdown remains the source of truth. LanceDB stores chunk metadata plus a
    dense vector for candidate retrieval; BM25 is computed locally over the same
    chunk set and fused with vector similarity. Search results are locators for
    `read()`, not authoritative evidence by themselves.
    """

    def __init__(self, wiki_path: str | Path):
        self.wiki_path = Path(wiki_path).expanduser().resolve()
        self.vector_dir = self.wiki_path / ".vector"
        self.lancedb_dir = self.vector_dir / "lancedb"
        self.chunks_file = self.vector_dir / "chunks.jsonl"
        self.manifest_file = self.vector_dir / "manifest.json"
        self.table_name = "chunks"
        self.embedder = HashingNgramEmbedder(dimensions=256)

    def reindex(self, include_raw: bool = False) -> IndexStatus:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        rows: list[dict[str, Any]] = []
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
            for chunk in doc_chunks:
                row = chunk.to_dict()
                row["vector"] = self.embedder.embed(_searchable_text(row))
                rows.append(row)

        with self.chunks_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        self._write_lancedb(rows)
        status = IndexStatus(
            wiki_path=str(self.wiki_path),
            backend=BACKEND,
            pages_indexed=pages,
            chunks_indexed=len(chunks),
            include_raw=include_raw,
            last_indexed_at=time.time(),
            embedding_model=self.embedder.model_name,
        )
        manifest = status.to_dict() | {"files": files_meta}
        self.manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def status(self) -> IndexStatus:
        if not self.manifest_file.exists():
            return IndexStatus(str(self.wiki_path), BACKEND, 0, 0, False, 0.0, self.embedder.model_name)
        data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        return IndexStatus(
            wiki_path=data.get("wiki_path", str(self.wiki_path)),
            backend=data.get("backend", BACKEND),
            pages_indexed=int(data.get("pages_indexed", 0)),
            chunks_indexed=int(data.get("chunks_indexed", 0)),
            include_raw=bool(data.get("include_raw", False)),
            last_indexed_at=float(data.get("last_indexed_at", 0.0)),
            embedding_model=data.get("embedding_model", self.embedder.model_name),
            bm25_weight=float(data.get("bm25_weight", BM25_WEIGHT)),
            vector_weight=float(data.get("vector_weight", VECTOR_WEIGHT)),
        )

    def search(
        self,
        query: str,
        limit: int = 8,
        include_raw: bool = False,
        types: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[SearchResult]:
        chunks = self._load_chunks()
        if not chunks:
            self.reindex(include_raw=include_raw)
            chunks = self._load_chunks()
        q_terms = _tokens(query)
        if not q_terms:
            return []

        filtered = [c for c in chunks if _passes_filters(c, include_raw=include_raw, types=types, tags=tags)]
        if not filtered:
            return []

        bm25_scores = _bm25_scores(query, filtered)
        vector_scores = self._vector_scores(query, filtered, include_raw=include_raw, types=types, tags=tags, limit=max(50, limit * 10))
        bm25_norm = _normalize_scores(bm25_scores)
        vector_norm = _normalize_scores(vector_scores)
        candidate_ids = set(bm25_scores) | set(vector_scores)

        results: list[SearchResult] = []
        by_id = {c["id"]: c for c in filtered}
        for cid in candidate_ids:
            chunk = by_id.get(cid)
            if not chunk:
                continue
            bm25 = bm25_norm.get(cid, 0.0)
            vector = vector_norm.get(cid, 0.0)
            score = BM25_WEIGHT * bm25 + VECTOR_WEIGHT * vector
            q_lower = query.lower()
            text = _searchable_text(chunk)
            if q_lower in text.lower():
                score += 0.05
            if not chunk.get("is_raw"):
                score += 0.01
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    path=chunk["path"],
                    title=chunk.get("title") or chunk["path"],
                    heading=chunk.get("heading") or "",
                    score=round(score, 6),
                    type=chunk.get("type") or "page",
                    tags=list(chunk.get("tags") or []),
                    confidence=chunk.get("confidence"),
                    snippet=_snippet(chunk.get("text", ""), q_terms),
                    bm25_score=round(bm25_scores.get(cid, 0.0), 6),
                    vector_score=round(vector_scores.get(cid, 0.0), 6),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def read(self, path: str, heading: str | None = None) -> ReadResult:
        rel = self._safe_markdown_path(path)
        full = self.wiki_path / rel
        text = full.read_text(encoding="utf-8")
        doc = parse_markdown(rel, text)
        if heading is None:
            return ReadResult(path=rel.as_posix(), title=doc.title, heading=None, content=text)
        for chunk in chunk_document(doc):
            if chunk.heading == heading:
                return ReadResult(path=rel.as_posix(), title=doc.title, heading=heading, content=chunk.text)
        raise ValueError(f"heading not found: {heading}")

    def write(self, path: str, content: str, mode: str = "create", reindex: bool = True) -> WriteResult:
        """Write a Markdown wiki page on the local source of truth.

        `mode` is intentionally small and explicit:
        - `create`: fail if the page already exists.
        - `overwrite`: replace the page completely.
        - `append`: append content to an existing page, creating it if missing.
        """
        rel = self._safe_markdown_path(path)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if mode not in {"create", "overwrite", "append"}:
            raise ValueError("mode must be one of: create, overwrite, append")

        full = self.wiki_path / rel
        if mode == "create" and full.exists():
            raise FileExistsError(f"wiki page already exists: {rel.as_posix()}")
        full.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append" and full.exists():
            existing = full.read_text(encoding="utf-8")
            separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
            final = existing + separator + content.rstrip() + "\n"
        else:
            final = content.rstrip() + "\n"
        full.write_text(final, encoding="utf-8")

        status = self.reindex(include_raw=False).to_dict() if reindex else None
        return WriteResult(
            path=rel.as_posix(),
            mode=mode,
            bytes_written=len(final.encode("utf-8")),
            reindexed=bool(reindex),
            status=status,
        )

    def _safe_markdown_path(self, path: str) -> Path:
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("path must be relative to wiki root")
        if rel.suffix.lower() != ".md":
            raise ValueError("path must point to a Markdown .md file")
        if rel.parts and rel.parts[0] == ".vector":
            raise ValueError("cannot read or write the .vector index directory")
        return rel

    def _load_chunks(self) -> list[dict]:
        if not self.chunks_file.exists():
            return []
        return [json.loads(line) for line in self.chunks_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write_lancedb(self, rows: list[dict[str, Any]]) -> None:
        import lancedb

        db = lancedb.connect(self.lancedb_dir.as_posix())
        if rows:
            db.create_table(self.table_name, data=rows, mode="overwrite")
        else:
            # LanceDB cannot infer a schema from an empty list; leave JSONL +
            # manifest as the status source when the wiki has no pages.
            try:
                db.drop_table(self.table_name)
            except Exception:
                pass

    def _vector_scores(
        self,
        query: str,
        chunks: list[dict],
        include_raw: bool,
        types: list[str] | None,
        tags: list[str] | None,
        limit: int,
    ) -> dict[str, float]:
        try:
            import lancedb

            db = lancedb.connect(self.lancedb_dir.as_posix())
            table = db.open_table(self.table_name)
            rows = table.search(self.embedder.embed(query)).limit(limit).to_list()
        except Exception:
            # Fallback keeps search usable if LanceDB is unavailable/corrupt.
            rows = []
            q_vec = self.embedder.embed(query)
            for chunk in chunks:
                vec = self.embedder.embed(_searchable_text(chunk))
                rows.append(chunk | {"_distance": max(0.0, 1.0 - _dot(q_vec, vec))})
            rows.sort(key=lambda r: r.get("_distance", 999.0))
            rows = rows[:limit]

        allowed = {c["id"] for c in chunks if _passes_filters(c, include_raw=include_raw, types=types, tags=tags)}
        scores: dict[str, float] = {}
        for row in rows:
            cid = row.get("id")
            if cid not in allowed:
                continue
            dist = float(row.get("_distance", 0.0))
            scores[cid] = max(scores.get(cid, 0.0), 1.0 / (1.0 + max(dist, 0.0)))
        return scores


def _passes_filters(chunk: dict, include_raw: bool, types: list[str] | None, tags: list[str] | None) -> bool:
    if chunk.get("is_raw") and not include_raw:
        return False
    if types and chunk.get("type") not in types:
        return False
    if tags and not set(tags).intersection(set(chunk.get("tags") or [])):
        return False
    return True


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _searchable_text(chunk: dict) -> str:
    return "\n".join([
        chunk.get("title", ""),
        chunk.get("heading", ""),
        " ".join(chunk.get("tags") or []),
        chunk.get("text", ""),
    ])


def _bm25_scores(query: str, chunks: list[dict]) -> dict[str, float]:
    q_terms = _tokens(query)
    tokenized = [_tokens(_searchable_text(chunk)) for chunk in chunks]
    doc_freq = Counter()
    for terms in tokenized:
        doc_freq.update(set(terms))
    n = max(len(chunks), 1)
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    for chunk, terms in zip(chunks, tokenized):
        tf = Counter(terms)
        dl = max(len(terms), 1)
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score += idf * (freq * (k1 + 1.0) / denom)
        if query.lower() in _searchable_text(chunk).lower():
            score += 1.0
        if score > 0:
            scores[chunk["id"]] = score
    return scores


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


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

