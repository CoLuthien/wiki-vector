import json
from pathlib import Path

from wiki_vector.index import WikiIndex


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "page.md").write_text(
        "# Page\n\n## Fact\n\nA durable fact.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_consistency_audit_reports_invalid_manifest_json_without_crashing(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex()
    index.manifest_file.write_text("{not-json", encoding="utf-8")

    result = index.consistency_audit()

    assert result["ok"] is False
    issue = next(issue for issue in result["issues"] if issue["code"] == "manifest_invalid_json")
    assert issue["severity"] == "high"
    assert issue["artifact"].endswith("/.vector/manifest.json")


def test_consistency_audit_reports_invalid_chunks_jsonl_line_without_crashing(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex()
    with index.chunks_file.open("a", encoding="utf-8") as handle:
        handle.write("{not-json\n")

    result = index.consistency_audit()

    assert result["ok"] is False
    issue = next(issue for issue in result["issues"] if issue["code"] == "chunks_file_invalid_json")
    assert issue["severity"] == "high"
    assert issue["line"] >= 1
    assert result["summary"]["chunk_file_chunks"] >= 1


def test_consistency_audit_reports_non_numeric_manifest_counts_without_crashing(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex()
    manifest = json.loads(index.manifest_file.read_text(encoding="utf-8"))
    manifest["pages_indexed"] = "oops"
    manifest["chunks_indexed"] = {"bad": "type"}
    index.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    result = index.consistency_audit()

    assert result["ok"] is False
    invalid_fields = {
        issue.get("field")
        for issue in result["issues"]
        if issue["code"] == "manifest_invalid_schema"
    }
    assert {"pages_indexed", "chunks_indexed"} <= invalid_fields


def test_consistency_audit_reports_invalid_manifest_file_metadata_without_crashing(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex()
    manifest = json.loads(index.manifest_file.read_text(encoding="utf-8"))
    manifest["files"]["concepts/page.md"] = 7
    index.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    result = index.consistency_audit()

    issue = next(
        issue
        for issue in result["issues"]
        if issue["code"] == "manifest_file_invalid_schema"
    )
    assert result["ok"] is False
    assert issue["path"] == "concepts/page.md"
    assert issue["actual_type"] == "int"


def test_consistency_audit_reports_chunk_schema_errors_without_crashing(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex()
    with index.chunks_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": "concepts/page.md", "tags": 5}) + "\n")

    result = index.consistency_audit()

    issue = next(
        issue
        for issue in result["issues"]
        if issue["code"] == "chunks_file_invalid_schema"
    )
    assert result["ok"] is False
    assert {"id", "tags"} <= set(issue["fields"])
    assert result["summary"]["expected_vector_rows"] >= 1
