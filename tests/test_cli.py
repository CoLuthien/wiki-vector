import json
import subprocess
import sys
from pathlib import Path


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "runbook.md").write_text("""---
title: Gemma4 RyzenAI Runtime 1.7.1 Runbook
type: concept
tags: [gemma4, runtime]
---

# Runbook

## NPU verification

Use xrt-smi to verify NPU activity.
""")
    return tmp_path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wiki_vector.cli", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cli_index_status_search_and_read_json(tmp_path):
    wiki = make_wiki(tmp_path)

    index = run_cli("--wiki", str(wiki), "index")
    status = run_cli("--wiki", str(wiki), "status")
    search = run_cli("--wiki", str(wiki), "search", "verify NPU", "--json")
    read = run_cli("--wiki", str(wiki), "read", "concepts/runbook.md", "--heading", "NPU verification")

    assert index.returncode == 0, index.stderr
    assert "chunks_indexed" in index.stdout
    assert json.loads(status.stdout)["pages_indexed"] == 1
    assert json.loads(status.stdout)["embedding_backend"] == "hashing-ngram"
    search_result = json.loads(search.stdout)[0]
    assert search_result["path"] == "concepts/runbook.md"
    assert search_result["read_hint"] == "concepts/runbook.md#NPU verification lines 9-11"
    assert read.returncode == 0
    assert "xrt-smi" in read.stdout

def test_cli_read_line_range_json(tmp_path):
    wiki = make_wiki(tmp_path)

    read = run_cli("--wiki", str(wiki), "read", "concepts/runbook.md", "--start-line", "9", "--end-line", "11", "--json")

    assert read.returncode == 0, read.stderr
    result = json.loads(read.stdout)
    assert result["start_line"] == 9
    assert result["end_line"] == 11
    assert result["content"] == "## NPU verification\n\nUse xrt-smi to verify NPU activity."


def test_cli_is_verbose_and_verbosity_audit_json(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "long.md").write_text("# Long\n\n" + "\n".join(f"Line {i}." for i in range(240)))

    verbose = run_cli("--wiki", str(wiki), "is-verbose", "concepts/long.md", "--json")
    audit = run_cli("--wiki", str(wiki), "verbosity-audit", "--limit", "2", "--json")
    human = run_cli("--wiki", str(wiki), "is-verbose", "concepts/long.md")

    assert verbose.returncode == 0, verbose.stderr
    data = json.loads(verbose.stdout)
    assert data["is_verbose"] is True
    assert data["score"] > 0
    audit_data = json.loads(audit.stdout)
    assert audit_data["results"][0]["path"] == "concepts/long.md"
    assert "VERBOSE" in human.stdout


def test_cli_is_verbose_semantic_json_preserves_deterministic_fields(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "semantic.md").write_text("""# Semantic

## Cache reuse

Cache reuse keeps operators reusable.

## Cache again

The same cache reuse topic repeats for reusable operators.
""")

    verbose = run_cli("--wiki", str(wiki), "is-verbose", "concepts/semantic.md", "--semantic", "--json")

    assert verbose.returncode == 0, verbose.stderr
    data = json.loads(verbose.stdout)
    assert data["semantic"]["enabled"] is True
    assert data["semantic"]["analyzers"][0]["kind"] == "embedding-semantic-structure"
    assert data["semantic"]["analyzers"][0]["not_readability_model"] is True
    assert data["semantic"].get("ml_readability_score") is None
    assert data["semantic"]["caveats"] == ["advisory_only", "not_used_in_default_score"]
    assert "score" in data and "reasons" in data and "sections" in data

def test_cli_search_reports_embedder_mismatch_without_traceback(tmp_path):
    wiki = make_wiki(tmp_path)
    indexed = run_cli("--wiki", str(wiki), "index")
    search = run_cli(
        "--wiki", str(wiki),
        "--embedding-backend", "hashing-ngram",
        "--embedding-dimensions", "32",
        "search", "verify NPU", "--json",
    )

    assert indexed.returncode == 0, indexed.stderr
    assert search.returncode == 2
    assert "embedding backend mismatch" in search.stderr
    assert "Traceback" not in search.stderr


def test_cli_search_explain_json_returns_results_and_explain(tmp_path):
    wiki = make_wiki(tmp_path)
    indexed = run_cli("--wiki", str(wiki), "index")
    search = run_cli("--wiki", str(wiki), "search", "xrt-smi NPU", "--json", "--explain")

    assert indexed.returncode == 0, indexed.stderr
    assert search.returncode == 0, search.stderr
    data = json.loads(search.stdout)
    assert data["results"][0]["path"] == "concepts/runbook.md"
    assert data["explain"]["query_terms"] == ["xrt-smi", "npu"]
    assert data["explain"]["keyword_contributions"][0]["term"] == "xrt-smi"
    assert any(stage["stage"] == "bm25" for stage in data["explain"]["trace"])


def test_cli_change_summary_json_reports_pending_changes(tmp_path):
    wiki = make_wiki(tmp_path)
    baseline = run_cli("--wiki", str(wiki), "change-summary", "--update", "--json")
    (wiki / "concepts" / "runbook.md").write_text((wiki / "concepts" / "runbook.md").read_text(encoding="utf-8") + "\nCLI changed line.\n", encoding="utf-8")
    pending = run_cli("--wiki", str(wiki), "change-summary", "--json")

    assert baseline.returncode == 0, baseline.stderr
    assert pending.returncode == 0, pending.stderr
    assert json.loads(baseline.stdout)["event_counts"]["added"] == 1
    data = json.loads(pending.stdout)
    assert data["pending_events"] == 1
    assert data["event_counts"]["modified"] == 1
    assert data["files"][0]["path"] == "concepts/runbook.md"


def test_cli_consistency_audit_json_reports_stale_index(tmp_path):
    wiki = make_wiki(tmp_path)
    indexed = run_cli("--wiki", str(wiki), "index")
    (wiki / "concepts" / "runbook.md").write_text((wiki / "concepts" / "runbook.md").read_text(encoding="utf-8") + "\n## Drift\n\nThe index is stale.\n", encoding="utf-8")
    audit = run_cli("--wiki", str(wiki), "consistency-audit", "--json")

    assert indexed.returncode == 0, indexed.stderr
    assert audit.returncode == 0, audit.stderr
    data = json.loads(audit.stdout)
    assert data["ok"] is False
    assert any(issue["code"] == "chunk_count_mismatch" for issue in data["issues"])


def test_cli_write_replace_section_content_file_json(tmp_path):
    wiki = make_wiki(tmp_path)
    content_file = tmp_path / "section.md"
    content_file.write_text("Use CLI replace-section for precise edits.\n", encoding="utf-8")

    result = run_cli(
        "--wiki", str(wiki),
        "write", "concepts/runbook.md",
        "--mode", "replace-section",
        "--heading", "NPU verification",
        "--content-file", str(content_file),
        "--json",
    )
    read = run_cli("--wiki", str(wiki), "read", "concepts/runbook.md", "--heading", "NPU verification")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    data = json.loads(result.stdout)
    assert data["mode"] == "replace-section"
    assert data["path"] == "concepts/runbook.md"
    assert data["heading"] == "NPU verification"
    assert data["start_line"] == 9
    assert data["end_line"] == 11
    assert data["old_section_bytes"] > 0
    assert data["new_section_bytes"] > 0
    assert "precise edits" in read.stdout


def test_cli_write_replace_section_error_has_no_traceback(tmp_path):
    wiki = make_wiki(tmp_path)
    content_file = tmp_path / "section.md"
    content_file.write_text("new\n", encoding="utf-8")

    result = run_cli(
        "--wiki", str(wiki),
        "write", "concepts/runbook.md",
        "--mode", "replace-section",
        "--heading", "Missing",
        "--content-file", str(content_file),
        "--json",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "heading not found" in result.stderr
    assert "Traceback" not in result.stderr
