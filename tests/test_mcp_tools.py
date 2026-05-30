from pathlib import Path

from wiki_vector.mcp_tools import wiki_is_verbose, wiki_read, wiki_reindex, wiki_search, wiki_status, wiki_verbosity_audit, wiki_write


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
