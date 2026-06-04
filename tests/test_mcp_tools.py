from pathlib import Path
import inspect

import pytest

from wiki_vector import mcp_server
from wiki_vector.mcp_tools import wiki_change_summary, wiki_consistency_audit, wiki_is_verbose, wiki_read, wiki_reindex, wiki_search, wiki_status, wiki_verbosity_audit, wiki_write


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "runbook.md").write_text("""---
title: Runbook
type: concept
tags: [gemma4]
---

# Runbook

## Runtime setup

Use RyzenAI 1.7.1 deployment DLLs.
""")
    return tmp_path


def test_mcp_tool_functions_return_serializable_dicts(tmp_path):
    wiki = make_wiki(tmp_path)

    status = wiki_reindex(str(wiki), include_raw=False)
    search = wiki_search(str(wiki), "deployment DLLs", limit=2)
    read = wiki_read(str(wiki), "concepts/runbook.md", heading="Runtime setup")
    current = wiki_status(str(wiki))

    assert status["pages_indexed"] == 1
    assert search["results"][0]["path"] == "concepts/runbook.md"
    assert search["results"][0]["start_line"] == 9
    assert search["results"][0]["end_line"] == 11
    assert search["results"][0]["read_hint"] == "concepts/runbook.md#Runtime setup lines 9-11"
    assert "RyzenAI 1.7.1" in read["content"]
    assert current["chunks_indexed"] >= 1


def test_mcp_wiki_search_can_include_explain_block(tmp_path):
    wiki = make_wiki(tmp_path)
    wiki_reindex(str(wiki), include_raw=False)

    search = wiki_search(str(wiki), "deployment DLLs", limit=2, explain=True)

    assert search["results"][0]["path"] == "concepts/runbook.md"
    assert search["explain"]["query_terms"] == ["deployment", "dlls"]
    assert search["explain"]["candidate_counts"]["returned"] == len(search["results"])
    assert search["explain"]["keyword_contributions"][0]["top_hits"]


def test_wiki_write_creates_page_and_reindexes(tmp_path):
    wiki = make_wiki(tmp_path)

    result = wiki_write(
        str(wiki),
        "concepts/new-finding.md",
        """---
title: New Finding
type: concept
tags: [gemma4]
---

# New Finding

## Runtime fact

Use wiki_write for durable facts.
""",
    )
    search = wiki_search(str(wiki), "durable facts", limit=2)

    assert result["path"] == "concepts/new-finding.md"
    assert result["reindexed"] is True
    assert (wiki / "concepts" / "new-finding.md").exists()
    assert search["results"][0]["path"] == "concepts/new-finding.md"


def test_wiki_write_replace_section_updates_existing_page_and_reindexes(tmp_path):
    wiki = make_wiki(tmp_path)
    wiki_reindex(str(wiki), include_raw=False)

    result = wiki_write(
        str(wiki),
        "concepts/runbook.md",
        "Use wiki_write replace-section for precise edits.",
        mode="replace-section",
        heading="Runtime setup",
    )

    assert result["path"] == "concepts/runbook.md"
    assert result["mode"] == "replace-section"
    assert result["heading"] == "Runtime setup"
    assert result["start_line"] == 9
    assert result["end_line"] == 11
    assert result["old_section_bytes"] > 0
    assert result["new_section_bytes"] > 0
    assert result["reindexed"] is True
    updated = wiki_read(str(wiki), "concepts/runbook.md", heading="Runtime setup")
    assert "replace-section" in updated["content"]
    assert wiki_search(str(wiki), "precise edits", limit=2)["results"][0]["path"] == "concepts/runbook.md"


def test_wiki_write_replace_section_missing_heading_does_not_modify_file(tmp_path):
    wiki = make_wiki(tmp_path)
    page = wiki / "concepts" / "runbook.md"
    before = page.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="heading not found"):
        wiki_write(str(wiki), "concepts/runbook.md", "new", mode="replace-section", heading="Missing")

    assert page.read_text(encoding="utf-8") == before


def test_wiki_write_replace_section_failure_contracts_do_not_modify_file(tmp_path):
    wiki = make_wiki(tmp_path)
    page = wiki / "concepts" / "runbook.md"
    before = page.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="heading is required"):
        wiki_write(str(wiki), "concepts/runbook.md", "new", mode="replace-section")
    with pytest.raises(ValueError, match="occurrence must be >= 1"):
        wiki_write(str(wiki), "concepts/runbook.md", "new", mode="replace-section", heading="Runtime setup", occurrence=0)
    with pytest.raises(ValueError, match="heading occurrence not found"):
        wiki_write(str(wiki), "concepts/runbook.md", "new", mode="replace-section", heading="Runtime setup", occurrence=2)
    with pytest.raises(ValueError, match="heading.*match"):
        wiki_write(str(wiki), "concepts/runbook.md", "## Renamed\nnew", mode="replace-section", heading="Runtime setup")
    with pytest.raises(ValueError, match="level"):
        wiki_write(str(wiki), "concepts/runbook.md", "### Runtime setup\nnew", mode="replace-section", heading="Runtime setup")
    with pytest.raises(FileNotFoundError):
        wiki_write(str(wiki), "concepts/missing.md", "new", mode="replace-section", heading="Runtime setup")

    assert page.read_text(encoding="utf-8") == before


def test_mcp_server_wiki_write_source_exposes_replace_section_parameters():
    source = inspect.getsource(mcp_server)
    assert "heading: str | None = None" in source
    assert "occurrence: int | None = None" in source
    assert "heading=heading" in source
    assert "occurrence=occurrence" in source

def test_wiki_read_returns_requested_line_range(tmp_path):
    wiki = make_wiki(tmp_path)

    result = wiki_read(str(wiki), "concepts/runbook.md", start_line=9, end_line=11)

    assert result["start_line"] == 9
    assert result["end_line"] == 11
    assert result["content"] == "## Runtime setup\n\nUse RyzenAI 1.7.1 deployment DLLs."


def test_mcp_verbosity_tools_return_serializable_dicts(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "long.md").write_text("# Long\n\n" + "\n".join(f"Line {i}." for i in range(220)))

    result = wiki_is_verbose(str(wiki), "concepts/long.md")
    audit = wiki_verbosity_audit(str(wiki), limit=2)

    assert result["is_verbose"] is True
    assert result["metrics"]["line_count"] >= 200
    assert audit["count"] == 2
    assert audit["results"][0]["path"] == "concepts/long.md"


def test_mcp_wiki_is_verbose_semantic_option_returns_advisory_block(tmp_path):
    wiki = make_wiki(tmp_path)

    result = wiki_is_verbose(str(wiki), "concepts/runbook.md", semantic=True)

    assert result["semantic"]["enabled"] is True
    assert result["semantic"]["analyzers"][0]["kind"] == "embedding-semantic-structure"
    assert result["semantic"]["analyzers"][0]["not_readability_model"] is True
    assert result["semantic"].get("ml_readability_score") is None
    assert result["semantic"]["caveats"] == ["advisory_only", "not_used_in_default_score"]
    assert "score" in result and "reasons" in result


def test_mcp_change_summary_reports_pending_and_recorded_changes(tmp_path):
    wiki = make_wiki(tmp_path)

    baseline = wiki_change_summary(str(wiki), update=True)
    (wiki / "concepts" / "runbook.md").write_text((wiki / "concepts" / "runbook.md").read_text(encoding="utf-8") + "\nNew changed line.\n", encoding="utf-8")
    pending = wiki_change_summary(str(wiki), update=False)
    recorded = wiki_change_summary(str(wiki), update=True)

    assert baseline["event_counts"]["added"] == 1
    assert pending["pending_events"] == 1
    assert pending["event_counts"]["modified"] == 1
    assert recorded["pending_events"] == 1
    assert recorded["files"][0]["last_diff"]["added_lines"] >= 1


def test_mcp_consistency_audit_reports_stale_index(tmp_path):
    wiki = make_wiki(tmp_path)
    wiki_reindex(str(wiki), include_raw=False)
    (wiki / "concepts" / "runbook.md").write_text((wiki / "concepts" / "runbook.md").read_text(encoding="utf-8") + "\n## Drift\n\nThe index is stale.\n", encoding="utf-8")

    audit = wiki_consistency_audit(str(wiki))

    assert audit["ok"] is False
    assert any(issue["code"] == "chunk_count_mismatch" for issue in audit["issues"])
