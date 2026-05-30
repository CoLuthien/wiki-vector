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
from .embeddings import Embedder, create_embedder
from .markdown import iter_wiki_markdown_files, parse_markdown
from .readability import EmbeddingSemanticStructureAnalyzer, ReadabilityAnalysis, ReadabilityAnalysisConfig, TransformersReadabilityModelAnalyzer
from .verbosity import VerbosityResult, analyze_verbosity

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
    embedding_backend: str = "hashing-ngram"
    embedding_dimensions: int = 256
    embedding_device: str | None = None
    embedding_max_length: int | None = None
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
    start_line: int = 0
    end_line: int = 0
    read_hint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReadResult:
    path: str
    title: str
    heading: str | None
    content: str
    start_line: int | None = None
    end_line: int | None = None

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

    def __init__(self, wiki_path: str | Path, embedder: Embedder | None = None):
        self.wiki_path = Path(wiki_path).expanduser().resolve()
        self.vector_dir = self.wiki_path / ".vector"
        self.lancedb_dir = self.vector_dir / "lancedb"
        self.chunks_file = self.vector_dir / "chunks.jsonl"
        self.manifest_file = self.vector_dir / "manifest.json"
        self.table_name = "chunks"
        self.embedder = embedder or create_embedder()

    def reindex(self, include_raw: bool = False) -> IndexStatus:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        rows: list[dict[str, Any]] = []
        vector_rows: list[dict[str, Any]] = []
        vector_texts: list[str] = []
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
                rows.append(row)
                for vector_row in _vector_rows_for_chunk(row, max_tokens=getattr(self.embedder, "max_length", None)):
                    vector_rows.append(vector_row)
                    vector_texts.append(_searchable_text(vector_row))

        if vector_rows:
            vectors = self.embedder.embed_many(vector_texts)
            if len(vectors) != len(vector_rows):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(vector_rows)} rows")
            for row, vector in zip(vector_rows, vectors):
                row["vector"] = vector

        with self.chunks_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        self._write_lancedb(vector_rows)
        status = IndexStatus(
            wiki_path=str(self.wiki_path),
            backend=BACKEND,
            pages_indexed=pages,
            chunks_indexed=len(chunks),
            include_raw=include_raw,
            last_indexed_at=time.time(),
            embedding_model=self.embedder.model_name,
            embedding_backend=getattr(self.embedder, "backend", self.embedder.__class__.__name__),
            embedding_dimensions=int(getattr(self.embedder, "dimensions", 0) or 0),
            embedding_device=getattr(self.embedder, "device", None),
            embedding_max_length=getattr(self.embedder, "max_length", None),
        )
        manifest = status.to_dict() | {"files": files_meta}
        self.manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def status(self) -> IndexStatus:
        if not self.manifest_file.exists():
            return IndexStatus(
                wiki_path=str(self.wiki_path),
                backend=BACKEND,
                pages_indexed=0,
                chunks_indexed=0,
                include_raw=False,
                last_indexed_at=0.0,
                embedding_model=self.embedder.model_name,
                embedding_backend=getattr(self.embedder, "backend", self.embedder.__class__.__name__),
                embedding_dimensions=int(getattr(self.embedder, "dimensions", 0) or 0),
                embedding_device=getattr(self.embedder, "device", None),
                embedding_max_length=getattr(self.embedder, "max_length", None),
            )
        data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        return IndexStatus(
            wiki_path=data.get("wiki_path", str(self.wiki_path)),
            backend=data.get("backend", BACKEND),
            pages_indexed=int(data.get("pages_indexed", 0)),
            chunks_indexed=int(data.get("chunks_indexed", 0)),
            include_raw=bool(data.get("include_raw", False)),
            last_indexed_at=float(data.get("last_indexed_at", 0.0)),
            embedding_model=data.get("embedding_model", self.embedder.model_name),
            embedding_backend=data.get("embedding_backend", getattr(self.embedder, "backend", self.embedder.__class__.__name__)),
            embedding_dimensions=int(data.get("embedding_dimensions", getattr(self.embedder, "dimensions", 0) or 0)),
            embedding_device=data.get("embedding_device", getattr(self.embedder, "device", None)),
            embedding_max_length=data.get("embedding_max_length", getattr(self.embedder, "max_length", None)),
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
        self._ensure_embedder_matches_index()
        q_terms = _tokens(query)
        if not q_terms:
            return []

        filtered = [c for c in chunks if _passes_filters(c, include_raw=include_raw, types=types, tags=tags)]
        if not filtered:
            return []

        bm25_scores = _bm25_scores(query, filtered)
        vector_scores, vector_hits = self._vector_scores(query, filtered, include_raw=include_raw, types=types, tags=tags, limit=max(50, limit * 10))
        bm25_norm = _normalize_scores(bm25_scores)
        vector_norm = _normalize_scores(vector_scores)
        candidate_ids = set(bm25_scores) | set(vector_scores)

        results: list[SearchResult] = []
        by_id = {c["id"]: c for c in filtered} | vector_hits
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
                    start_line=int(chunk.get("start_line") or 0),
                    end_line=int(chunk.get("end_line") or 0),
                    read_hint=_read_hint(chunk),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def read(self, path: str, heading: str | None = None, start_line: int | None = None, end_line: int | None = None) -> ReadResult:
        rel = self._safe_markdown_path(path)
        full = self.wiki_path / rel
        text = full.read_text(encoding="utf-8")
        doc = parse_markdown(rel, text)
        if start_line is not None or end_line is not None:
            if start_line is None or end_line is None:
                raise ValueError("start_line and end_line must be provided together")
            return _read_line_range(rel.as_posix(), doc.title, text, start_line, end_line)
        if heading is None:
            return ReadResult(path=rel.as_posix(), title=doc.title, heading=None, content=text)
        for chunk in chunk_document(doc):
            if chunk.heading == heading:
                return ReadResult(
                    path=rel.as_posix(),
                    title=doc.title,
                    heading=heading,
                    content=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
        raise ValueError(f"heading not found: {heading}")

    def is_verbose(self, path: str, *, include_code: bool = False, compare_to: str | None = None, semantic: bool = False, readability_model: str | None = None) -> VerbosityResult:
        rel = self._safe_markdown_path(path)
        text = (self.wiki_path / rel).read_text(encoding="utf-8")
        compare_text = None
        if compare_to is not None:
            compare_rel = self._safe_markdown_path(compare_to)
            compare_text = (self.wiki_path / compare_rel).read_text(encoding="utf-8")
        analyzers = []
        if semantic:
            analyzers.append(EmbeddingSemanticStructureAnalyzer(self.embedder, ReadabilityAnalysisConfig(compare_to=compare_text)))
        if readability_model:
            analyzers.append(TransformersReadabilityModelAnalyzer(readability_model))
        return analyze_verbosity(rel.as_posix(), text, include_code=include_code, compare_to=compare_text, readability_analyzers=analyzers or None)

    def verbosity_audit(self, *, limit: int = 20, include_raw: bool = False, severity: str | None = None) -> list[VerbosityResult]:
        if severity is not None and severity not in {"ok", "warning", "high"}:
            raise ValueError("severity must be one of: ok, warning, high")
        results: list[VerbosityResult] = []
        for full in iter_wiki_markdown_files(self.wiki_path, include_raw=include_raw):
            rel = full.relative_to(self.wiki_path).as_posix()
            result = analyze_verbosity(rel, full.read_text(encoding="utf-8"))
            if severity is None or result.severity == severity:
                results.append(result)
        _apply_corpus_calibration(results)
        results.sort(key=lambda r: (r.score, r.metrics.get("line_count", 0)), reverse=True)
        return results[:limit]

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

    def _ensure_embedder_matches_index(self) -> None:
        if not self.manifest_file.exists():
            return
        data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        expected = {
            "backend": data.get("embedding_backend"),
            "model": data.get("embedding_model"),
            "dimensions": int(data.get("embedding_dimensions", 0) or 0),
            "max_length": data.get("embedding_max_length"),
        }
        actual = {
            "backend": getattr(self.embedder, "backend", self.embedder.__class__.__name__),
            "model": self.embedder.model_name,
            "dimensions": int(getattr(self.embedder, "dimensions", 0) or 0),
            "max_length": getattr(self.embedder, "max_length", None),
        }
        mismatches = [key for key in expected if expected[key] != actual[key]]
        if mismatches:
            details = ", ".join(
                f"{key}: indexed={expected[key]!r} current={actual[key]!r}" for key in mismatches
            )
            raise ValueError(
                "embedding backend mismatch; reindex with the same embedding options before search "
                f"({details})"
            )

    def _vector_scores(
        self,
        query: str,
        chunks: list[dict],
        include_raw: bool,
        types: list[str] | None,
        tags: list[str] | None,
        limit: int,
    ) -> tuple[dict[str, float], dict[str, dict]]:
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
        hits: dict[str, dict] = {}
        for row in rows:
            cid = row.get("id")
            parent_id = row.get("parent_id", cid)
            if parent_id not in allowed:
                continue
            dist = float(row.get("_distance", 0.0))
            scores[cid] = max(scores.get(cid, 0.0), 1.0 / (1.0 + max(dist, 0.0)))
            hits[cid] = row
        return scores, hits


def _apply_corpus_calibration(results: list[VerbosityResult]) -> None:
    """Add audit-time percentile metrics/reasons in-place via mutable metrics dicts.

    Single-page is_verbose stays absolute-threshold based; audits annotate corpus
    outliers without changing the dataclass object identity.
    """
    if not results:
        return
    keys = ["line_count", "max_section_lines", "word_count"]
    def number_metric(result: VerbosityResult, key: str) -> float:
        value = result.metrics.get(key, 0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    sorted_values = {k: sorted(number_metric(r, k) for r in results) for k in keys}
    for result in results:
        for k in keys:
            v = number_metric(result, k)
            vals = sorted_values[k]
            le = sum(1 for item in vals if item <= v)
            result.metrics[f"{k}_percentile"] = round(le / max(len(vals), 1), 6)


def _read_line_range(path: str, title: str, text: str, start_line: int, end_line: int) -> ReadResult:
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    lines = text.splitlines()
    if end_line > len(lines):
        raise ValueError(f"end_line {end_line} exceeds file length {len(lines)}")
    return ReadResult(
        path=path,
        title=title,
        heading=None,
        content="\n".join(lines[start_line - 1 : end_line]),
        start_line=start_line,
        end_line=end_line,
    )


def _read_hint(chunk: dict) -> str:
    path = chunk.get("path", "")
    heading = chunk.get("heading") or ""
    start_line = int(chunk.get("start_line") or 0)
    end_line = int(chunk.get("end_line") or 0)
    anchor = f"#{heading}" if heading else ""
    if start_line and end_line:
        return f"{path}{anchor} lines {start_line}-{end_line}"
    return f"{path}{anchor}"


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


def _vector_rows_for_chunk(chunk: dict, max_tokens: Any | None = None) -> list[dict[str, Any]]:
    """Return embedding rows for a source chunk.

    Source chunks stay heading-sized for BM25 and read-by-heading.  Neural
    embedders commonly have a fixed sequence length, so long source chunks are
    split into additional line-range locator rows for vector search instead of
    being silently truncated to the first model window.
    """
    limit = _positive_int(max_tokens)
    base = dict(chunk)
    base["parent_id"] = chunk["id"]
    base["vector_ordinal"] = 0
    if limit is None or len(_tokens(_searchable_text(base))) <= limit:
        return [base]

    rows: list[dict[str, Any]] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0
    start_line = int(chunk.get("start_line") or 1)
    lines = str(chunk.get("text", "")).splitlines()

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        ordinal = len(rows)
        first_line = current[0][0]
        last_line = current[-1][0]
        text = "\n".join(line for _, line in current).strip()
        if text:
            row = dict(chunk)
            row["id"] = f"{chunk['id']}:v{ordinal}"
            row["parent_id"] = chunk["id"]
            row["vector_ordinal"] = ordinal
            row["text"] = text
            row["start_line"] = first_line
            row["end_line"] = last_line
            rows.append(row)
        current = []
        current_tokens = 0

    for offset, line in enumerate(lines):
        line_no = start_line + offset
        line_tokens = len(_tokens(line))
        if not line.strip():
            continue
        if current and current_tokens + line_tokens > limit:
            flush()
        current.append((line_no, line))
        current_tokens += line_tokens
        if current_tokens >= limit:
            flush()
    flush()
    return rows or [base]


def _positive_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


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
