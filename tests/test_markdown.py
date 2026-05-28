from pathlib import Path

from wiki_vector.markdown import parse_markdown, iter_wiki_markdown_files
from wiki_vector.chunking import chunk_document


def test_parse_markdown_extracts_frontmatter_and_body():
    text = """---
title: Gemma4 RyzenAI Runtime 1.7.1 Runbook
type: concept
tags: [liboptai, gemma4, runtime]
confidence: medium
---

# Gemma4 RyzenAI Runtime 1.7.1 Runbook

Body text.
"""

    doc = parse_markdown(Path("concepts/runbook.md"), text)

    assert doc.path == "concepts/runbook.md"
    assert doc.slug == "runbook"
    assert doc.title == "Gemma4 RyzenAI Runtime 1.7.1 Runbook"
    assert doc.type == "concept"
    assert doc.tags == ["liboptai", "gemma4", "runtime"]
    assert doc.confidence == "medium"
    assert doc.body.startswith("# Gemma4")


def test_chunk_document_splits_by_markdown_headings_with_metadata():
    doc = parse_markdown(Path("concepts/runbook.md"), """---
title: Runbook
type: concept
tags: [gemma4]
---

# Runbook

Intro paragraph.

## Runtime setup

Use deployment DLLs.

## NPU verification

Use xrt-smi.
""")

    chunks = chunk_document(doc)

    assert [c.heading for c in chunks] == ["Runbook", "Runtime setup", "NPU verification"]
    assert chunks[1].path == "concepts/runbook.md"
    assert chunks[1].type == "concept"
    assert chunks[1].tags == ["gemma4"]
    assert "deployment DLLs" in chunks[1].text


def test_iter_wiki_markdown_files_excludes_navigation_and_vector_dir(tmp_path):
    (tmp_path / "concepts").mkdir()
    (tmp_path / "raw" / "transcripts").mkdir(parents=True)
    (tmp_path / ".vector").mkdir()
    (tmp_path / "concepts" / "a.md").write_text("# A")
    (tmp_path / "raw" / "transcripts" / "r.md").write_text("# R")
    (tmp_path / ".vector" / "ignore.md").write_text("# Ignore")
    (tmp_path / "index.md").write_text("# Index")

    no_raw = list(iter_wiki_markdown_files(tmp_path, include_raw=False))
    with_raw = list(iter_wiki_markdown_files(tmp_path, include_raw=True))

    assert [p.relative_to(tmp_path).as_posix() for p in no_raw] == ["concepts/a.md"]
    assert [p.relative_to(tmp_path).as_posix() for p in with_raw] == [
        "concepts/a.md",
        "raw/transcripts/r.md",
    ]
