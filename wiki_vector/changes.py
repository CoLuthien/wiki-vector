from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import sqlite3
import time
from typing import Any, Iterable

from .markdown import iter_wiki_markdown_files


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    sha256: str
    size_bytes: int
    line_count: int
    text: str


def summarize_changes(
    wiki_path: Path,
    db_path: Path,
    *,
    include_raw: bool = False,
    update: bool = False,
    change_count_threshold: int = 1,
    byte_threshold: int = 1,
    line_threshold: int = 1,
) -> dict[str, Any]:
    """Scan wiki Markdown files and summarize changes against SQLite state.

    The SQLite DB is a local maintenance ledger under `.vector/`; Markdown files
    remain the source of truth. `update=False` is a dry-run/pending check for cron
    gating. `update=True` records newly observed added/modified/deleted events and
    updates the file snapshot table.
    """
    wiki_path = Path(wiki_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _init_db(conn)
        previous = _load_state(conn)
        current = _scan_files(wiki_path, include_raw=include_raw)
        pending = _diff_snapshots(previous, current)
        observed_at = time.time()
        if update and pending:
            _record_events(conn, pending, observed_at)
            _replace_state(conn, current, include_raw=include_raw, observed_at=observed_at)
            conn.commit()
            previous = _load_state(conn)
        total_events = _total_events(conn)
        files = _file_rows(conn, previous, pending if not update else [], current if not update else None)
        pending_events = len(pending)
        event_counts = Counter(event["kind"] for event in pending)
        aggregate = _aggregate(pending)
        thresholds = {
            "change_count": max(int(change_count_threshold), 0),
            "bytes": max(int(byte_threshold), 0),
            "lines": max(int(line_threshold), 0),
        }
        significant = (
            pending_events >= thresholds["change_count"]
            or aggregate["abs_byte_delta"] >= thresholds["bytes"]
            or aggregate["abs_line_delta"] >= thresholds["lines"]
        ) if pending_events else False
        return {
            "wiki_path": str(wiki_path),
            "db_path": str(db_path),
            "include_raw": include_raw,
            "update": update,
            "pending_events": pending_events,
            "total_events": total_events,
            "event_counts": {"added": event_counts.get("added", 0), "modified": event_counts.get("modified", 0), "deleted": event_counts.get("deleted", 0)},
            "aggregate": aggregate,
            "thresholds": thresholds,
            "significant_change": significant,
            "files": files,
        }
    finally:
        conn.close()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_state (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            line_count INTEGER NOT NULL,
            content_text TEXT NOT NULL DEFAULT '',
            last_seen_at REAL NOT NULL,
            include_raw INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS file_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('added', 'modified', 'deleted')),
            observed_at REAL NOT NULL,
            old_sha256 TEXT,
            new_sha256 TEXT,
            old_size_bytes INTEGER,
            new_size_bytes INTEGER,
            old_line_count INTEGER,
            new_line_count INTEGER,
            byte_delta INTEGER NOT NULL,
            line_delta INTEGER NOT NULL,
            added_lines INTEGER NOT NULL,
            removed_lines INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_path ON file_changes(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_observed_at ON file_changes(observed_at)")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(file_state)").fetchall()}
    if "content_text" not in columns:
        conn.execute("ALTER TABLE file_state ADD COLUMN content_text TEXT NOT NULL DEFAULT ''")
    conn.commit()


def _scan_files(wiki_path: Path, *, include_raw: bool) -> dict[str, FileSnapshot]:
    snapshots: dict[str, FileSnapshot] = {}
    for full in iter_wiki_markdown_files(wiki_path, include_raw=include_raw):
        rel = full.relative_to(wiki_path).as_posix()
        text = full.read_text(encoding="utf-8")
        snapshots[rel] = FileSnapshot(
            path=rel,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            size_bytes=len(text.encode("utf-8")),
            line_count=len(text.splitlines()),
            text=text,
        )
    return snapshots


def _load_state(conn: sqlite3.Connection) -> dict[str, FileSnapshot]:
    rows = conn.execute("SELECT path, sha256, size_bytes, line_count, content_text FROM file_state").fetchall()
    return {
        row["path"]: FileSnapshot(
            path=row["path"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            line_count=int(row["line_count"]),
            text=row["content_text"] or "",
        )
        for row in rows
    }


def _diff_snapshots(previous: dict[str, FileSnapshot], current: dict[str, FileSnapshot]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(current):
        new = current[path]
        old = previous.get(path)
        if old is None:
            events.append(_event(path, "added", None, new))
        elif old.sha256 != new.sha256:
            events.append(_event(path, "modified", old, new))
    for path in sorted(set(previous) - set(current)):
        events.append(_event(path, "deleted", previous[path], None))
    return events


def _event(path: str, kind: str, old: FileSnapshot | None, new: FileSnapshot | None) -> dict[str, Any]:
    added, removed = _line_diff(old.text if old else "", new.text if new else "")
    old_size = old.size_bytes if old else 0
    new_size = new.size_bytes if new else 0
    old_lines = old.line_count if old else 0
    new_lines = new.line_count if new else 0
    return {
        "path": path,
        "kind": kind,
        "old_sha256": old.sha256 if old else None,
        "new_sha256": new.sha256 if new else None,
        "old_size_bytes": old_size,
        "new_size_bytes": new_size,
        "old_line_count": old_lines,
        "new_line_count": new_lines,
        "byte_delta": new_size - old_size,
        "line_delta": new_lines - old_lines,
        "added_lines": added,
        "removed_lines": removed,
    }


def _line_diff(old_text: str, new_text: str) -> tuple[int, int]:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def _record_events(conn: sqlite3.Connection, events: Iterable[dict[str, Any]], observed_at: float) -> None:
    for event in events:
        conn.execute(
            """
            INSERT INTO file_changes (
                path, kind, observed_at, old_sha256, new_sha256,
                old_size_bytes, new_size_bytes, old_line_count, new_line_count,
                byte_delta, line_delta, added_lines, removed_lines
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["path"], event["kind"], observed_at, event["old_sha256"], event["new_sha256"],
                event["old_size_bytes"], event["new_size_bytes"], event["old_line_count"], event["new_line_count"],
                event["byte_delta"], event["line_delta"], event["added_lines"], event["removed_lines"],
            ),
        )


def _replace_state(conn: sqlite3.Connection, current: dict[str, FileSnapshot], *, include_raw: bool, observed_at: float) -> None:
    conn.execute("DELETE FROM file_state")
    for snapshot in current.values():
        conn.execute(
            """
            INSERT INTO file_state (path, sha256, size_bytes, line_count, content_text, last_seen_at, include_raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (snapshot.path, snapshot.sha256, snapshot.size_bytes, snapshot.line_count, snapshot.text, observed_at, int(include_raw)),
        )


def _total_events(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM file_changes").fetchone()["n"])


def _file_rows(
    conn: sqlite3.Connection,
    state: dict[str, FileSnapshot],
    pending: list[dict[str, Any]],
    current: dict[str, FileSnapshot] | None,
) -> list[dict[str, Any]]:
    paths = set(state)
    paths.update(event["path"] for event in pending)
    if current is not None:
        paths.update(current)
    rows: list[dict[str, Any]] = []
    pending_by_path = {event["path"]: event for event in pending}
    for path in sorted(paths):
        count = int(conn.execute("SELECT COUNT(*) AS n FROM file_changes WHERE path = ?", (path,)).fetchone()["n"])
        last = conn.execute("SELECT * FROM file_changes WHERE path = ? ORDER BY id DESC LIMIT 1", (path,)).fetchone()
        pending_event = pending_by_path.get(path)
        if pending_event:
            count += 1
            kind = pending_event["kind"]
            diff = _diff_dict(pending_event)
            size = pending_event["new_size_bytes"]
            line_count = pending_event["new_line_count"]
        elif last:
            kind = last["kind"]
            diff = _diff_dict(last)
            snap = state.get(path)
            size = snap.size_bytes if snap else None
            line_count = snap.line_count if snap else None
        else:
            kind = None
            diff = None
            snap = state.get(path)
            size = snap.size_bytes if snap else None
            line_count = snap.line_count if snap else None
        rows.append({
            "path": path,
            "change_count": count,
            "last_change_kind": kind,
            "size_bytes": size,
            "line_count": line_count,
            "pending": pending_event is not None,
            "last_diff": diff,
        })
    rows.sort(key=lambda row: (bool(row["pending"]), row["change_count"], row["last_diff"]["abs_byte_delta"] if row["last_diff"] else 0), reverse=True)
    return rows


def _diff_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, int]:
    byte_delta = int(row["byte_delta"])
    line_delta = int(row["line_delta"])
    return {
        "byte_delta": byte_delta,
        "line_delta": line_delta,
        "abs_byte_delta": abs(byte_delta),
        "abs_line_delta": abs(line_delta),
        "added_lines": int(row["added_lines"]),
        "removed_lines": int(row["removed_lines"]),
    }


def _aggregate(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "byte_delta": sum(int(e["byte_delta"]) for e in events),
        "line_delta": sum(int(e["line_delta"]) for e in events),
        "abs_byte_delta": sum(abs(int(e["byte_delta"])) for e in events),
        "abs_line_delta": sum(abs(int(e["line_delta"])) for e in events),
        "added_lines": sum(int(e["added_lines"]) for e in events),
        "removed_lines": sum(int(e["removed_lines"]) for e in events),
    }
