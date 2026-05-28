from pathlib import Path

from wiki_vector.index import WikiIndex


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "raw" / "transcripts").mkdir(parents=True)
    (tmp_path / "concepts" / "runbook.md").write_text("""---
title: Gemma4 RyzenAI Runtime 1.7.1 Runbook
type: concept
tags: [liboptai, gemma4, runtime]
confidence: medium
---

# Gemma4 RyzenAI Runtime 1.7.1 Runbook

Runtime 1.7.1 only. Use deployment DLLs.

## NPU verification

Use xrt-smi, not Windows GPU counters.
""")
    (tmp_path / "raw" / "transcripts" / "session.md").write_text("""---
title: Raw Session
type: raw
---

# Raw Session

GQO failed in ryzen-4.
""")
    return tmp_path


def test_reindex_changed_creates_manifest_and_searches_wiki_pages_first(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    status = index.reindex(include_raw=False)
    results = index.search("How do I verify RyzenAI NPU activity?", limit=3)

    assert status.pages_indexed == 1
    assert status.chunks_indexed >= 2
    assert (wiki / ".vector" / "manifest.json").exists()
    assert results[0].path == "concepts/runbook.md"
    assert results[0].heading == "NPU verification"
    assert "xrt-smi" in results[0].snippet


def test_raw_files_are_opt_in(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    index.reindex(include_raw=False)
    no_raw = index.search("GQO failed ryzen-4", limit=5)
    index.reindex(include_raw=True)
    with_raw = index.search("GQO failed ryzen-4", limit=5, include_raw=True)

    assert all(not r.path.startswith("raw/") for r in no_raw)
    assert any(r.path == "raw/transcripts/session.md" for r in with_raw)


def test_read_section_returns_requested_heading(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    content = index.read("concepts/runbook.md", heading="NPU verification")

    assert content.path == "concepts/runbook.md"
    assert content.heading == "NPU verification"
    assert content.content.startswith("## NPU verification")
    assert "xrt-smi" in content.content


def test_reindex_uses_lancedb_hybrid_backend(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    status = index.reindex(include_raw=False)

    assert status.backend == "lancedb-hybrid"
    assert status.embedding_model.startswith("hashing-ngram")
    assert (wiki / ".vector" / "lancedb").exists()


def test_hybrid_search_reports_component_scores(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex(include_raw=False)

    results = index.search("xrt-smi NPU verification", limit=1)

    assert results[0].path == "concepts/runbook.md"
    assert results[0].heading == "NPU verification"
    assert results[0].bm25_score > 0
    assert results[0].vector_score > 0
