from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .markdown import iter_wiki_markdown_files, parse_markdown


def audit_index_consistency(
    *,
    wiki_path: Path,
    manifest_file: Path,
    chunks_file: Path,
    lancedb_dir: Path,
    table_name: str,
    expected_vector_rows_for_chunk: Callable[[dict[str, Any]], int],
    include_raw: bool | None = None,
) -> dict[str, Any]:
    """Audit source Markdown and disposable index artifacts without repairing them."""
    issues: list[dict[str, Any]] = []
    manifest, manifest_error = _read_json_object(manifest_file)
    if manifest_error is not None:
        issues.append(
            _issue(
                "manifest_invalid_json",
                "high",
                "Index manifest is not a valid JSON object.",
                artifact=manifest_file.as_posix(),
                error=manifest_error,
            )
        )

    manifest_include_raw_value = manifest.get("include_raw", False)
    if manifest and not isinstance(manifest_include_raw_value, bool):
        issues.append(
            _issue(
                "manifest_invalid_schema",
                "high",
                "Manifest include_raw must be a boolean.",
                field="include_raw",
                actual_type=type(manifest_include_raw_value).__name__,
            )
        )
        manifest_include_raw_value = False
    manifest_include_raw = bool(manifest_include_raw_value)
    effective_include_raw = (
        manifest_include_raw if include_raw is None else bool(include_raw)
    )

    markdown_files: dict[str, dict[str, Any]] = {}
    markdown_chunks: list[dict[str, Any]] = []
    for path in iter_wiki_markdown_files(
        wiki_path, include_raw=effective_include_raw
    ):
        rel_path = path.relative_to(wiki_path)
        rel = rel_path.as_posix()
        text = path.read_text(encoding="utf-8")
        document_chunks = chunk_document(parse_markdown(rel_path, text))
        stat = path.stat()
        markdown_files[rel] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunks": len(document_chunks),
        }
        markdown_chunks.extend(chunk.to_dict() for chunk in document_chunks)

    chunk_rows, chunk_errors = _read_jsonl_objects(chunks_file)
    for line_number, error in chunk_errors:
        issues.append(
            _issue(
                "chunks_file_invalid_json",
                "high",
                "chunks.jsonl contains an invalid JSON object line.",
                artifact=chunks_file.as_posix(),
                line=line_number,
                error=error,
            )
        )

    valid_chunk_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(chunk_rows, start=1):
        schema_errors = _chunk_schema_errors(row)
        if schema_errors:
            issues.append(
                _issue(
                    "chunks_file_invalid_schema",
                    "high",
                    "chunks.jsonl contains an object with invalid chunk fields.",
                    artifact=chunks_file.as_posix(),
                    row_index=row_index,
                    fields=schema_errors,
                )
            )
            continue
        valid_chunk_rows.append(row)

    chunk_file_paths = {str(row["path"]) for row in valid_chunk_rows}
    markdown_paths = set(markdown_files)
    manifest_files_value = manifest.get("files", {})
    if manifest and not isinstance(manifest_files_value, dict):
        issues.append(
            _issue(
                "manifest_invalid_schema",
                "high",
                "Manifest files must be a JSON object.",
                field="files",
                actual_type=type(manifest_files_value).__name__,
            )
        )
        manifest_files_value = {}
    manifest_files = manifest_files_value
    manifest_paths = set(manifest_files)

    if not manifest_file.exists():
        issues.append(
            _issue(
                "manifest_missing",
                "high",
                "Index manifest is missing.",
                artifact=manifest_file.as_posix(),
            )
        )
    if not chunks_file.exists():
        issues.append(
            _issue(
                "chunks_file_missing",
                "high",
                "Index chunks JSONL file is missing.",
                artifact=chunks_file.as_posix(),
            )
        )

    for path in sorted(manifest_paths - markdown_paths):
        issues.append(
            _issue(
                "indexed_file_missing",
                "high",
                "Manifest references a Markdown file that no longer exists.",
                path=path,
            )
        )
    for path in sorted(markdown_paths - manifest_paths):
        issues.append(
            _issue(
                "markdown_file_unindexed",
                "warning",
                "Markdown file is not present in the manifest.",
                path=path,
            )
        )
    for path in sorted(chunk_file_paths - markdown_paths):
        issues.append(
            _issue(
                "chunk_file_missing_source",
                "high",
                "chunks.jsonl contains rows for a missing Markdown file.",
                path=path,
            )
        )

    for path in sorted(markdown_paths & manifest_paths):
        expected = manifest_files.get(path)
        if not isinstance(expected, dict):
            issues.append(
                _issue(
                    "manifest_file_invalid_schema",
                    "high",
                    "Manifest file metadata must be a JSON object.",
                    path=path,
                    actual_type=type(expected).__name__,
                )
            )
            continue
        current = markdown_files[path]
        stale_fields: list[str] = []
        invalid_fields: list[str] = []
        expected_size = _coerce_int(expected.get("size"))
        expected_chunks = _coerce_int(expected.get("chunks"))
        expected_mtime = _coerce_float(expected.get("mtime"))
        if expected_size is None:
            invalid_fields.append("size")
        elif expected_size != int(current["size"]):
            stale_fields.append("size")
        if expected_chunks is None:
            invalid_fields.append("chunks")
        elif expected_chunks != int(current["chunks"]):
            stale_fields.append("chunks")
        if expected_mtime is None:
            invalid_fields.append("mtime")
        elif abs(expected_mtime - float(current["mtime"])) > 1e-6:
            stale_fields.append("mtime")
        if invalid_fields:
            issues.append(
                _issue(
                    "manifest_file_invalid_schema",
                    "high",
                    "Manifest file metadata contains non-numeric fields.",
                    path=path,
                    fields=invalid_fields,
                )
            )
        if stale_fields:
            issues.append(
                _issue(
                    "manifest_file_stale",
                    "warning",
                    "Manifest file metadata no longer matches Markdown source.",
                    path=path,
                    fields=stale_fields,
                )
            )

    manifest_pages = _coerce_int(manifest.get("pages_indexed", 0))
    manifest_chunks = _coerce_int(manifest.get("chunks_indexed", 0))
    for field, value in (
        ("pages_indexed", manifest_pages),
        ("chunks_indexed", manifest_chunks),
    ):
        if manifest and value is None:
            issues.append(
                _issue(
                    "manifest_invalid_schema",
                    "high",
                    f"Manifest {field} must be an integer.",
                    field=field,
                )
            )
    manifest_pages = manifest_pages or 0
    manifest_chunks = manifest_chunks or 0
    if manifest and manifest_pages != len(manifest_files):
        issues.append(
            _issue(
                "manifest_page_count_mismatch",
                "warning",
                "Manifest pages_indexed does not match manifest file entries.",
                expected=len(manifest_files),
                actual=manifest_pages,
            )
        )
    if manifest and manifest_chunks != len(chunk_rows):
        issues.append(
            _issue(
                "manifest_chunk_count_mismatch",
                "high",
                "Manifest chunks_indexed does not match chunks.jsonl row count.",
                expected=len(chunk_rows),
                actual=manifest_chunks,
            )
        )
    if len(markdown_chunks) != len(chunk_rows):
        issues.append(
            _issue(
                "chunk_count_mismatch",
                "high",
                "Current Markdown chunk count does not match chunks.jsonl row count.",
                expected=len(markdown_chunks),
                actual=len(chunk_rows),
            )
        )

    markdown_ids = {str(row.get("id") or "") for row in markdown_chunks}
    chunk_ids = [str(row["id"]) for row in valid_chunk_rows]
    duplicate_chunk_ids = sorted(
        str(chunk_id)
        for chunk_id, count in Counter(chunk_ids).items()
        if chunk_id and count > 1
    )
    if duplicate_chunk_ids:
        issues.append(
            _issue(
                "duplicate_chunk_ids",
                "high",
                "chunks.jsonl contains duplicate chunk ids.",
                ids=duplicate_chunk_ids[:20],
                count=len(duplicate_chunk_ids),
            )
        )
    missing_chunk_ids = sorted(
        str(chunk_id) for chunk_id in markdown_ids - set(chunk_ids) if chunk_id
    )
    extra_chunk_ids = sorted(
        str(chunk_id) for chunk_id in set(chunk_ids) - markdown_ids if chunk_id
    )
    if missing_chunk_ids:
        issues.append(
            _issue(
                "markdown_chunks_missing_from_index",
                "high",
                "Current Markdown chunks are missing from chunks.jsonl.",
                ids=missing_chunk_ids[:20],
                count=len(missing_chunk_ids),
            )
        )
    if extra_chunk_ids:
        issues.append(
            _issue(
                "stale_chunks_in_index",
                "high",
                "chunks.jsonl has chunk ids not produced by current Markdown.",
                ids=extra_chunk_ids[:20],
                count=len(extra_chunk_ids),
            )
        )

    vector_rows = _lancedb_row_count(lancedb_dir, table_name)
    expected_vector_rows = 0
    for row_index, row in enumerate(valid_chunk_rows, start=1):
        try:
            expected_vector_rows += int(expected_vector_rows_for_chunk(row))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "chunks_file_invalid_schema",
                    "high",
                    "A chunk object cannot be converted to vector locator rows.",
                    artifact=chunks_file.as_posix(),
                    row_index=row_index,
                    error=str(exc),
                )
            )
    if vector_rows is None:
        issues.append(
            _issue(
                "lancedb_table_unavailable",
                "warning",
                "LanceDB table could not be opened for row-count audit.",
                artifact=lancedb_dir.as_posix(),
            )
        )
    elif vector_rows != expected_vector_rows:
        issues.append(
            _issue(
                "vector_row_count_mismatch",
                "high",
                "LanceDB row count does not match expected vector locator rows from chunks.jsonl.",
                expected=expected_vector_rows,
                actual=vector_rows,
            )
        )

    recommendations: list[str] = []
    if issues:
        recommendations.append(
            "Run wiki_reindex(include_raw="
            f"{str(effective_include_raw).lower()}) to rebuild stale or inconsistent "
            "index artifacts."
        )
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
        "wiki_path": str(wiki_path),
        "include_raw": effective_include_raw,
        "ok": not issues,
        "summary": summary,
        "issues": issues,
        "recommendations": recommendations,
    }


def _chunk_schema_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        errors.append("id")
    if not isinstance(row.get("path"), str) or not row["path"].strip():
        errors.append("path")

    for field in ("title", "heading", "type", "text"):
        value = row.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(field)

    tags = row.get("tags")
    if tags is not None and (
        not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
    ):
        errors.append("tags")

    for field in ("start_line", "end_line"):
        value = row.get(field)
        if value is not None and _coerce_int(value) is None:
            errors.append(field)
    return errors


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _read_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, str(exc)
    if not isinstance(value, dict):
        return {}, f"expected JSON object, got {type(value).__name__}"
    return value, None


def _read_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [(1, str(exc))]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append((line_number, str(exc)))
            continue
        if not isinstance(value, dict):
            errors.append(
                (line_number, f"expected JSON object, got {type(value).__name__}")
            )
            continue
        rows.append(value)
    return rows, errors


def _issue(
    code: str, severity: str, message: str, **details: Any
) -> dict[str, Any]:
    issue = {"code": code, "severity": severity, "message": message}
    issue.update({key: value for key, value in details.items() if value is not None})
    return issue


def _lancedb_row_count(lancedb_dir: Path, table_name: str) -> int | None:
    try:
        import lancedb

        database = lancedb.connect(lancedb_dir.as_posix())
        table = database.open_table(table_name)
        if hasattr(table, "count_rows"):
            return int(table.count_rows())
        return len(table.to_list())
    except Exception:
        return None
