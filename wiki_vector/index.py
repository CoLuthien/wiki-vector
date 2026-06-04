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
from .changes import summarize_changes
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
    heading: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    old_section_bytes: int | None = None
    new_section_bytes: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SectionReplaceResult:
    text: str
    heading: str
    start_line: int
    end_line: int
    old_section: str
    new_section: str


@dataclass(frozen=True)
class _HeadingSpan:
    line_index: int
    line_number: int
    level: int
    text: str
    raw_line: str
    start_offset: int


_HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*\r?\n?$")
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _replace_markdown_section(text: str, *, heading: str, content: str, occurrence: int | None = None) -> SectionReplaceResult:
    if occurrence is not None and occurrence < 1:
        raise ValueError("occurrence must be >= 1")
    if not heading:
        raise ValueError("heading is required for replace-section")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")

    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    headings = _scan_atx_headings(lines, offsets)
    matches = [h for h in headings if h.text == heading]
    if not matches:
        raise ValueError(f"heading not found: {heading}")
    if occurrence is None:
        if len(matches) > 1:
            raise ValueError(f"heading is ambiguous: {heading} appears {len(matches)} times; provide occurrence")
        target = matches[0]
    else:
        if occurrence > len(matches):
            raise ValueError(f"heading occurrence not found: {heading} occurrence {occurrence}")
        target = matches[occurrence - 1]

    next_boundary = len(lines)
    for h in headings:
        if h.line_index > target.line_index and h.level <= target.level:
            next_boundary = h.line_index
            break
    section_end_line_index = next_boundary
    while section_end_line_index > target.line_index + 1 and not lines[section_end_line_index - 1].strip():
        section_end_line_index -= 1

    start_offset = offsets[target.line_index]
    end_offset = offsets[section_end_line_index] if section_end_line_index < len(offsets) else len(text)
    old_section = text[start_offset:end_offset].rstrip("\r\n")
    new_section = _build_replacement_section(target, content)
    final = text[:start_offset] + new_section + text[end_offset:]
    final = final.rstrip("\r\n") + "\n"
    return SectionReplaceResult(
        text=final,
        heading=target.text,
        start_line=target.line_number,
        end_line=target.line_number + max(0, section_end_line_index - target.line_index - 1),
        old_section=old_section,
        new_section=new_section.rstrip("\r\n"),
    )


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    current = 0
    for line in lines:
        offsets.append(current)
        current += len(line)
    offsets.append(current)
    return offsets


def _scan_atx_headings(lines: list[str], offsets: list[int]) -> list[_HeadingSpan]:
    headings: list[_HeadingSpan] = []
    start_index = _body_start_line_index(lines)
    in_fence = False
    fence_char = ""
    fence_len = 0
    for i, line in enumerate(lines[start_index:], start=start_index):
        fence = _FENCE_LINE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
                continue
        if in_fence:
            continue
        parsed = _parse_atx_heading_line(line)
        if parsed is None:
            continue
        level, normalized = parsed
        headings.append(_HeadingSpan(i, i + 1, level, normalized, line, offsets[i]))
    return headings


def _body_start_line_index(lines: list[str]) -> int:
    if not lines or lines[0].strip() != "---":
        return 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i + 1
    return 0


def _parse_atx_heading_line(line: str) -> tuple[int, str] | None:
    match = _HEADING_LINE_RE.match(line)
    if not match:
        return None
    text = _normalize_heading_text(match.group(2))
    if not text:
        return None
    return len(match.group(1)), text


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", text.strip()).strip()


def _build_replacement_section(target: _HeadingSpan, content: str) -> str:
    stripped = content.strip()
    first_line = next((line for line in stripped.splitlines() if line.strip()), "")
    first_heading = _parse_atx_heading_line(first_line)
    if first_heading is None:
        return target.raw_line.rstrip("\r\n") + "\n" + stripped + "\n"

    level, normalized = first_heading
    if normalized != target.text:
        raise ValueError(f"replacement heading must match target heading: {target.text}")
    if level != target.level:
        raise ValueError(f"replacement heading level must match target level: {target.level}")
    replacement_lines = stripped.splitlines(keepends=True)
    offsets = _line_offsets(replacement_lines)
    nested_headings = _scan_atx_headings(replacement_lines, offsets)
    for h in nested_headings[1:]:
        if h.level <= target.level:
            raise ValueError("full-section replacement must not contain an additional same-or-higher heading")
    return stripped + "\n"


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
        self.change_db_file = self.vector_dir / "changes.sqlite"
        self.chunks_file = self.vector_dir / "chunks.jsonl"
        self.manifest_file = self.vector_dir / "manifest.json"
        self.table_name = "chunks"
        self.embedder = embedder or create_embedder()

    def reindex(self, include_raw: bool = False) -> IndexStatus:
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)
        self.change_summary(include_raw=include_raw, update=True)
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

    def search_explain(
        self,
        query: str,
        limit: int = 8,
        include_raw: bool = False,
        types: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search plus deterministic retrieval diagnostics.

        Keyword-level explain is intentionally deterministic and local: BM25 is
        decomposable by query term, so we report per-token BM25 contributions.
        Dense vector retrieval is produced by embedding the whole query, so it is
        reported as a stage-level trace rather than pretending individual words
        have independent vector contributions.
        """
        chunks = self._load_chunks()
        if not chunks:
            self.reindex(include_raw=include_raw)
            chunks = self._load_chunks()
        self._ensure_embedder_matches_index()
        q_terms = _unique_tokens(query)
        if not q_terms:
            return {
                "results": [],
                "explain": _empty_explain(query, include_raw=include_raw, types=types, tags=tags),
            }

        filtered = [c for c in chunks if _passes_filters(c, include_raw=include_raw, types=types, tags=tags)]
        if not filtered:
            explain = _empty_explain(query, include_raw=include_raw, types=types, tags=tags)
            explain["candidate_counts"]["chunks_after_filters"] = 0
            return {"results": [], "explain": explain}

        bm25_scores = _bm25_scores(query, filtered)
        vector_scores, vector_hits = self._vector_scores(query, filtered, include_raw=include_raw, types=types, tags=tags, limit=max(50, limit * 10))
        results = self.search(query, limit=limit, include_raw=include_raw, types=types, tags=tags)
        result_dicts = [r.to_dict() for r in results]
        explain = {
            "query": query,
            "query_terms": q_terms,
            "filters": {"include_raw": include_raw, "types": types, "tags": tags},
            "weights": {"bm25": BM25_WEIGHT, "vector": VECTOR_WEIGHT},
            "candidate_counts": {
                "chunks_total": len(chunks),
                "chunks_after_filters": len(filtered),
                "bm25_nonzero": len(bm25_scores),
                "vector_hits": len(vector_scores),
                "returned": len(result_dicts),
            },
            "keyword_contributions": _keyword_contributions(query, filtered),
            "trace": _search_trace(bm25_scores, vector_scores, vector_hits, filtered, limit=max(limit, 3)),
            "fallbacks": [],
            "notes": [
                "BM25 keyword contributions are per-token deterministic scores.",
                "Vector retrieval embeds the whole query, so vector explain is stage-level, not per keyword.",
            ],
        }
        return {"results": result_dicts, "explain": explain}

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

    def change_summary(
        self,
        *,
        include_raw: bool = False,
        update: bool = False,
        since: str | None = None,
        change_count_threshold: int = 1,
        byte_threshold: int = 1,
        line_threshold: int = 1,
    ) -> dict[str, Any]:
        """Return SQLite-backed file change counters and diff-size summary.

        `since` is reserved for future windowing; the current implementation
        reports pending changes against the last recorded snapshot plus total
        per-file counters, which is the useful cron gating signal.
        """
        _ = since
        return summarize_changes(
            self.wiki_path,
            self.change_db_file,
            include_raw=include_raw,
            update=update,
            change_count_threshold=change_count_threshold,
            byte_threshold=byte_threshold,
            line_threshold=line_threshold,
        )


    def consistency_audit(self, include_raw: bool | None = None) -> dict[str, Any]:
        """Audit whether Markdown, manifest, JSONL chunks, and LanceDB rows agree.

        This is read-only diagnostics: it reports drift and corruption without
        rebuilding anything. Use `reindex()` to repair reported issues.
        """
        issues: list[dict[str, Any]] = []
        manifest = _read_json_file(self.manifest_file)
        manifest_include_raw = bool(manifest.get("include_raw", False)) if manifest else False
        effective_include_raw = manifest_include_raw if include_raw is None else bool(include_raw)

        markdown_files: dict[str, dict[str, Any]] = {}
        markdown_chunks: list[dict[str, Any]] = []
        for path in iter_wiki_markdown_files(self.wiki_path, include_raw=effective_include_raw):
            rel_path = path.relative_to(self.wiki_path)
            rel = rel_path.as_posix()
            text = path.read_text(encoding="utf-8")
            doc_chunks = chunk_document(parse_markdown(rel_path, text))
            st = path.stat()
            markdown_files[rel] = {"mtime": st.st_mtime, "size": st.st_size, "chunks": len(doc_chunks)}
            markdown_chunks.extend(chunk.to_dict() for chunk in doc_chunks)

        chunk_rows = self._load_chunks()
        chunk_file_paths = {str(row.get("path", "")) for row in chunk_rows}
        markdown_paths = set(markdown_files)
        manifest_files = manifest.get("files", {}) if isinstance(manifest.get("files", {}), dict) else {}
        manifest_paths = set(manifest_files)

        if not self.manifest_file.exists():
            issues.append(_audit_issue("manifest_missing", "high", "Index manifest is missing.", artifact=self.manifest_file.as_posix()))
        if not self.chunks_file.exists():
            issues.append(_audit_issue("chunks_file_missing", "high", "Index chunks JSONL file is missing.", artifact=self.chunks_file.as_posix()))

        for path in sorted(manifest_paths - markdown_paths):
            issues.append(_audit_issue("indexed_file_missing", "high", "Manifest references a Markdown file that no longer exists.", path=path))
        for path in sorted(markdown_paths - manifest_paths):
            issues.append(_audit_issue("markdown_file_unindexed", "warning", "Markdown file is not present in the manifest.", path=path))
        for path in sorted(chunk_file_paths - markdown_paths):
            issues.append(_audit_issue("chunk_file_missing_source", "high", "chunks.jsonl contains rows for a missing Markdown file.", path=path))

        for path in sorted(markdown_paths & manifest_paths):
            expected = manifest_files.get(path) or {}
            current = markdown_files[path]
            stale_fields: list[str] = []
            if int(expected.get("size", -1)) != int(current["size"]):
                stale_fields.append("size")
            if int(expected.get("chunks", -1)) != int(current["chunks"]):
                stale_fields.append("chunks")
            if abs(float(expected.get("mtime", 0.0) or 0.0) - float(current["mtime"])) > 1e-6:
                stale_fields.append("mtime")
            if stale_fields:
                issues.append(_audit_issue("manifest_file_stale", "warning", "Manifest file metadata no longer matches Markdown source.", path=path, fields=stale_fields))

        manifest_pages = int(manifest.get("pages_indexed", 0) or 0) if manifest else 0
        manifest_chunks = int(manifest.get("chunks_indexed", 0) or 0) if manifest else 0
        if manifest and manifest_pages != len(manifest_files):
            issues.append(_audit_issue("manifest_page_count_mismatch", "warning", "Manifest pages_indexed does not match manifest file entries.", expected=len(manifest_files), actual=manifest_pages))
        if manifest and manifest_chunks != len(chunk_rows):
            issues.append(_audit_issue("manifest_chunk_count_mismatch", "high", "Manifest chunks_indexed does not match chunks.jsonl row count.", expected=len(chunk_rows), actual=manifest_chunks))
        if len(markdown_chunks) != len(chunk_rows):
            issues.append(_audit_issue("chunk_count_mismatch", "high", "Current Markdown chunk count does not match chunks.jsonl row count.", expected=len(markdown_chunks), actual=len(chunk_rows)))

        markdown_ids = {row.get("id") for row in markdown_chunks}
        chunk_ids = [row.get("id") for row in chunk_rows]
        duplicate_chunk_ids = sorted({cid for cid in chunk_ids if cid and chunk_ids.count(cid) > 1})
        if duplicate_chunk_ids:
            issues.append(_audit_issue("duplicate_chunk_ids", "high", "chunks.jsonl contains duplicate chunk ids.", ids=duplicate_chunk_ids[:20], count=len(duplicate_chunk_ids)))
        missing_chunk_ids = sorted(str(cid) for cid in markdown_ids - set(chunk_ids) if cid)
        extra_chunk_ids = sorted(str(cid) for cid in set(chunk_ids) - markdown_ids if cid)
        if missing_chunk_ids:
            issues.append(_audit_issue("markdown_chunks_missing_from_index", "high", "Current Markdown chunks are missing from chunks.jsonl.", ids=missing_chunk_ids[:20], count=len(missing_chunk_ids)))
        if extra_chunk_ids:
            issues.append(_audit_issue("stale_chunks_in_index", "high", "chunks.jsonl has chunk ids not produced by current Markdown.", ids=extra_chunk_ids[:20], count=len(extra_chunk_ids)))

        vector_rows = _lancedb_row_count(self.lancedb_dir, self.table_name)
        expected_vector_rows = sum(len(_vector_rows_for_chunk(row, max_tokens=getattr(self.embedder, "max_length", None))) for row in chunk_rows)
        if vector_rows is None:
            issues.append(_audit_issue("lancedb_table_unavailable", "warning", "LanceDB table could not be opened for row-count audit.", artifact=self.lancedb_dir.as_posix()))
        elif vector_rows != expected_vector_rows:
            issues.append(_audit_issue("vector_row_count_mismatch", "high", "LanceDB row count does not match expected vector locator rows from chunks.jsonl.", expected=expected_vector_rows, actual=vector_rows))

        recommendations: list[str] = []
        if issues:
            recommendations.append(f"Run wiki_reindex(include_raw={str(effective_include_raw).lower()}) to rebuild stale or inconsistent index artifacts.")
        summary = {
            "issue_count": len(issues),
            "markdown_pages": len(markdown_files),
            "markdown_chunks": len(markdown_chunks),
            "manifest_pages": manifest_pages,
            "manifest_chunks": manifest_chunks,
            "manifest_file_entries": len(manifest_files),
            "chunk_file_chunks": len(chunk_rows),
            "lancedb_rows": vector_rows,
            "expected_vector_rows": expected_vector_rows,
        }
        return {
            "wiki_path": str(self.wiki_path),
            "include_raw": effective_include_raw,
            "ok": not issues,
            "summary": summary,
            "issues": issues,
            "recommendations": recommendations,
        }

    def write(
        self,
        path: str,
        content: str,
        mode: str = "create",
        reindex: bool = True,
        heading: str | None = None,
        occurrence: int | None = None,
    ) -> WriteResult:
        """Write a Markdown wiki page on the local source of truth.

        `mode` is intentionally small and explicit:
        - `create`: fail if the page already exists.
        - `overwrite`: replace the page completely.
        - `append`: append content to an existing page, creating it if missing.
        - `replace-section`: replace one ATX heading section in an existing page.
        """
        rel = self._safe_markdown_path(path)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        if mode not in {"create", "overwrite", "append", "replace-section"}:
            raise ValueError("mode must be one of: create, overwrite, append, replace-section")

        full = self.wiki_path / rel
        section_result: SectionReplaceResult | None = None
        if mode == "replace-section":
            if not heading:
                raise ValueError("heading is required for replace-section")
            if not full.exists():
                raise FileNotFoundError(f"wiki page does not exist: {rel.as_posix()}")
            existing = full.read_text(encoding="utf-8")
            section_result = _replace_markdown_section(existing, heading=heading, content=content, occurrence=occurrence)
            final = section_result.text
        else:
            if mode == "create" and full.exists():
                raise FileExistsError(f"wiki page already exists: {rel.as_posix()}")
            full.parent.mkdir(parents=True, exist_ok=True)

            if mode == "append" and full.exists():
                existing = full.read_text(encoding="utf-8")
                separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
                final = existing + separator + content.rstrip() + "\n"
            else:
                final = content.rstrip() + "\n"

        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(f".{full.name}.tmp-{time.time_ns()}")
        try:
            tmp.write_text(final, encoding="utf-8")
            tmp.replace(full)
        finally:
            if tmp.exists():
                tmp.unlink()

        status = self.reindex(include_raw=False).to_dict() if reindex else None
        return WriteResult(
            path=rel.as_posix(),
            mode=mode,
            bytes_written=len(final.encode("utf-8")),
            reindexed=bool(reindex),
            status=status,
            heading=section_result.heading if section_result else None,
            start_line=section_result.start_line if section_result else None,
            end_line=section_result.end_line if section_result else None,
            old_section_bytes=len(section_result.old_section.encode("utf-8")) if section_result else None,
            new_section_bytes=len(section_result.new_section.encode("utf-8")) if section_result else None,
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



def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _audit_issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({k: v for k, v in details.items() if v is not None})
    return issue


def _lancedb_row_count(lancedb_dir: Path, table_name: str) -> int | None:
    try:
        import lancedb
        db = lancedb.connect(lancedb_dir.as_posix())
        table = db.open_table(table_name)
        if hasattr(table, "count_rows"):
            return int(table.count_rows())
        return len(table.to_list())
    except Exception:
        return None

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


def _unique_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in _tokens(text):
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


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
            score += _bm25_term_score(term, tf, dl, doc_freq, n, avgdl, k1=k1, b=b)
        if query.lower() in _searchable_text(chunk).lower():
            score += 1.0
        if score > 0:
            scores[chunk["id"]] = score
    return scores


def _bm25_term_score(term: str, tf: Counter, dl: int, doc_freq: Counter, n: int, avgdl: float, *, k1: float = 1.5, b: float = 0.75) -> float:
    freq = tf.get(term, 0)
    if not freq:
        return 0.0
    idf = math.log(1.0 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
    denom = freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
    return idf * (freq * (k1 + 1.0) / denom)


def _keyword_contributions(query: str, chunks: list[dict]) -> list[dict[str, Any]]:
    q_terms = _unique_tokens(query)
    tokenized = [_tokens(_searchable_text(chunk)) for chunk in chunks]
    doc_freq = Counter()
    for terms in tokenized:
        doc_freq.update(set(terms))
    n = max(len(chunks), 1)
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    rows: list[dict[str, Any]] = []
    for term in q_terms:
        hits: list[dict[str, Any]] = []
        for chunk, terms in zip(chunks, tokenized):
            tf = Counter(terms)
            score = _bm25_term_score(term, tf, max(len(terms), 1), doc_freq, n, avgdl)
            if score > 0:
                hits.append({
                    "path": chunk.get("path", ""),
                    "heading": chunk.get("heading") or "",
                    "bm25_contribution": round(score, 6),
                    "read_hint": _read_hint(chunk),
                })
        hits.sort(key=lambda row: row["bm25_contribution"], reverse=True)
        rows.append({"term": term, "matching_chunks": len(hits), "top_hits": hits[:5]})
    return rows


def _search_trace(bm25_scores: dict[str, float], vector_scores: dict[str, float], vector_hits: dict[str, dict], chunks: list[dict], *, limit: int) -> list[dict[str, Any]]:
    by_id = {c["id"]: c for c in chunks} | vector_hits
    trace: list[dict[str, Any]] = []
    for cid, score in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
        chunk = by_id.get(cid)
        if chunk:
            trace.append({"stage": "bm25", "path": chunk.get("path"), "heading": chunk.get("heading") or "", "score": round(score, 6), "read_hint": _read_hint(chunk)})
    for cid, score in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
        chunk = by_id.get(cid)
        if chunk:
            trace.append({"stage": "vector", "path": chunk.get("path"), "heading": chunk.get("heading") or "", "score": round(score, 6), "read_hint": _read_hint(chunk)})
    return trace


def _empty_explain(query: str, *, include_raw: bool, types: list[str] | None, tags: list[str] | None) -> dict[str, Any]:
    return {
        "query": query,
        "query_terms": _unique_tokens(query),
        "filters": {"include_raw": include_raw, "types": types, "tags": tags},
        "weights": {"bm25": BM25_WEIGHT, "vector": VECTOR_WEIGHT},
        "candidate_counts": {"chunks_total": 0, "chunks_after_filters": 0, "bm25_nonzero": 0, "vector_hits": 0, "returned": 0},
        "keyword_contributions": [],
        "trace": [],
        "fallbacks": [],
        "notes": [],
    }


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
