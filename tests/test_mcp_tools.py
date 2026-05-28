from pathlib import Path

from wiki_vector.mcp_tools import wiki_reindex, wiki_search, wiki_read, wiki_status, wiki_write


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
