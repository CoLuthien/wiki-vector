from pathlib import Path

from wiki_vector.index import WikiIndex


class RecordingEmbedder:
    backend = "recording"
    model_name = "recording-3"
    dimensions = 3

    def __init__(self):
        self.embed_many_calls = []
        self.embed_calls = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts):
        self.embed_many_calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


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
    assert (results[0].start_line, results[0].end_line) == (12, 14)
    assert results[0].read_hint == "concepts/runbook.md#NPU verification lines 12-14"


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
    assert status.embedding_backend == "hashing-ngram"
    assert status.embedding_model.startswith("hashing-ngram")
    assert status.embedding_dimensions == 256
    assert (wiki / ".vector" / "lancedb").exists()


def test_reindex_accepts_swappable_embedder_and_batches_rows(tmp_path):
    wiki = make_wiki(tmp_path)
    embedder = RecordingEmbedder()
    index = WikiIndex(wiki, embedder=embedder)

    status = index.reindex(include_raw=False)

    assert status.embedding_backend == "recording"
    assert status.embedding_model == "recording-3"
    assert status.embedding_dimensions == 3
    assert len(embedder.embed_many_calls) == 1
    assert len(embedder.embed_many_calls[0]) == status.chunks_indexed


def test_hybrid_search_reports_component_scores(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex(include_raw=False)

    results = index.search("xrt-smi NPU verification", limit=1)

    assert results[0].path == "concepts/runbook.md"
    assert results[0].heading == "NPU verification"
    assert results[0].bm25_score > 0
    assert results[0].vector_score > 0

def test_read_returns_requested_line_range(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    content = index.read("concepts/runbook.md", start_line=12, end_line=14)

    assert content.path == "concepts/runbook.md"
    assert content.heading is None
    assert content.start_line == 12
    assert content.end_line == 14
    assert content.content == "## NPU verification\n\nUse xrt-smi, not Windows GPU counters."


def test_is_verbose_uses_safe_paths_and_returns_analysis(tmp_path):
    wiki = make_wiki(tmp_path)
    long_body = "\n".join(f"Line {i}." for i in range(305))
    (wiki / "concepts" / "long.md").write_text(f"---\ntitle: Long\ntype: concept\n---\n\n# Long\n\n{long_body}\n")
    index = WikiIndex(wiki)

    result = index.is_verbose("concepts/long.md")

    assert result.is_verbose is True
    assert result.severity == "high"
    assert isinstance(result.metrics["line_count"], int)
    assert result.metrics["line_count"] >= 300


def test_is_verbose_semantic_mode_uses_index_embedder_without_blending_score(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "semantic.md").write_text("""# Semantic

## Cache reuse

Cache reuse keeps operators reusable.

## Cache again

The same cache reuse topic repeats for reusable operators.
""")
    embedder = RecordingEmbedder()
    index = WikiIndex(wiki, embedder=embedder)

    result = index.is_verbose("concepts/semantic.md", semantic=True)

    data = result.to_dict()
    assert data["semantic"]["enabled"] is True
    assert data["semantic"]["analyzers"][0]["backend"] == "recording"
    assert data["semantic"]["analyzers"][0]["kind"] == "embedding-semantic"
    assert "reasons" in data and "sections" in data


def test_verbosity_audit_sorts_verbose_pages_first(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "short.md").write_text("# Short\n\nSmall.")
    (wiki / "concepts" / "long.md").write_text("# Long\n\n" + "\n".join(f"Line {i}." for i in range(230)))
    index = WikiIndex(wiki)

    results = index.verbosity_audit(limit=2)

    assert results[0].path == "concepts/long.md"
    assert "line_count_percentile" in results[0].metrics
