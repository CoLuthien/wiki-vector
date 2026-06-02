from pathlib import Path
import json

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


def test_search_explain_reports_keyword_dependent_bm25_contributions(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex(include_raw=False)

    npu_explained = index.search_explain("xrt-smi NPU verification", limit=2)
    runtime_explained = index.search_explain("deployment DLLs", limit=2)

    assert npu_explained["results"][0]["path"] == "concepts/runbook.md"
    assert npu_explained["explain"]["query_terms"] == ["xrt-smi", "npu", "verification"]
    assert npu_explained["explain"]["weights"] == {"bm25": 0.35, "vector": 0.65}
    assert npu_explained["explain"]["filters"] == {"include_raw": False, "types": None, "tags": None}
    assert npu_explained["explain"]["candidate_counts"]["returned"] == len(npu_explained["results"])

    by_term = {row["term"]: row for row in npu_explained["explain"]["keyword_contributions"]}
    assert by_term["xrt-smi"]["matching_chunks"] >= 1
    assert by_term["xrt-smi"]["top_hits"][0]["path"] == "concepts/runbook.md"
    assert by_term["xrt-smi"]["top_hits"][0]["bm25_contribution"] > 0
    assert any(stage["stage"] == "vector" for stage in npu_explained["explain"]["trace"])
    assert any(stage["stage"] == "bm25" for stage in npu_explained["explain"]["trace"])

    npu_terms = [row["term"] for row in npu_explained["explain"]["keyword_contributions"]]
    runtime_terms = [row["term"] for row in runtime_explained["explain"]["keyword_contributions"]]
    assert npu_terms != runtime_terms


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
    assert data["semantic"]["analyzers"][0]["kind"] == "embedding-semantic-structure"
    assert data["semantic"]["analyzers"][0]["not_readability_model"] is True
    assert data["semantic"].get("ml_readability_score") is None
    assert "reasons" in data and "sections" in data


def test_verbosity_audit_sorts_verbose_pages_first(tmp_path):
    wiki = make_wiki(tmp_path)
    (wiki / "concepts" / "short.md").write_text("# Short\n\nSmall.")
    (wiki / "concepts" / "long.md").write_text("# Long\n\n" + "\n".join(f"Line {i}." for i in range(230)))
    index = WikiIndex(wiki)

    results = index.verbosity_audit(limit=2)

    assert results[0].path == "concepts/long.md"
    assert "line_count_percentile" in results[0].metrics

class LocationAwareEmbedder:
    backend = "location-aware"
    model_name = "location-aware-3"
    dimensions = 3
    max_length = 8

    def __init__(self):
        self.embed_many_calls = []
        self.embed_calls = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if "needle" in text:
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]

    def embed_many(self, texts):
        self.embed_many_calls.append(list(texts))
        return [self.embed(text) for text in texts]


def test_neural_vector_index_splits_long_sections_into_locator_subchunks(tmp_path):
    wiki = make_wiki(tmp_path)
    long_lines = [
        "alpha beta gamma delta epsilon zeta eta theta.",
        "iota kappa lambda mu nu xi omicron pi.",
        "rho sigma tau upsilon phi chi psi omega.",
        "needle cache reuse operator location marker.",
    ]
    (wiki / "concepts" / "long-section.md").write_text("""---
title: Long Section
type: concept
tags: [wiki]
---

# Long Section

## Big Section

""" + "\n".join(long_lines) + "\n")
    embedder = LocationAwareEmbedder()
    index = WikiIndex(wiki, embedder=embedder)

    status = index.reindex(include_raw=False)
    results = index.search("needle", limit=1)

    embedded_texts = embedder.embed_many_calls[0]
    assert len(embedded_texts) > status.chunks_indexed
    assert any("needle cache reuse" in text for text in embedded_texts)
    assert results[0].path == "concepts/long-section.md"
    assert results[0].heading == "Big Section"
    assert results[0].start_line == 14
    assert results[0].end_line == 14
    assert "needle" in results[0].snippet
    assert results[0].read_hint == "concepts/long-section.md#Big Section lines 14-14"

def test_search_rejects_embedder_mismatch_instead_of_reembedding_all_chunks(tmp_path):
    wiki = make_wiki(tmp_path)
    WikiIndex(wiki).reindex(include_raw=False)
    mismatched = RecordingEmbedder()
    index = WikiIndex(wiki, embedder=mismatched)

    try:
        index.search("NPU verification", limit=1)
    except ValueError as exc:
        assert "embedding backend mismatch" in str(exc)
        assert "reindex" in str(exc)
    else:
        raise AssertionError("expected embedding mismatch error")
    assert mismatched.embed_calls == []


def test_change_summary_tracks_file_update_counts_and_diff_sizes(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)

    baseline = index.change_summary(update=True)
    assert baseline["total_events"] == 1
    assert baseline["event_counts"]["added"] == 1
    assert baseline["files"][0]["path"] == "concepts/runbook.md"
    assert baseline["files"][0]["change_count"] == 1

    runbook = wiki / "concepts" / "runbook.md"
    runbook.write_text(runbook.read_text(encoding="utf-8") + "\n## Cache reuse\n\nOperators can be reused after wiki edits.\n", encoding="utf-8")
    changed = index.change_summary(update=True, since="baseline")

    assert changed["total_events"] == 2
    assert changed["event_counts"]["modified"] == 1
    file_row = next(row for row in changed["files"] if row["path"] == "concepts/runbook.md")
    assert file_row["change_count"] == 2
    assert file_row["last_change_kind"] == "modified"
    assert file_row["last_diff"]["added_lines"] >= 3
    assert file_row["last_diff"]["removed_lines"] == 0
    assert changed["aggregate"]["abs_line_delta"] >= 3
    assert changed["significant_change"] is True


def test_change_summary_can_report_without_updating_state_and_ignores_raw_by_default(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.change_summary(update=True)

    (wiki / "concepts" / "new-page.md").write_text("# New Page\n\nImportant active edit.\n", encoding="utf-8")
    (wiki / "raw" / "transcripts" / "session.md").write_text("# Raw Session\n\nRaw edit should be opt-in.\n", encoding="utf-8")
    preview = index.change_summary(update=False, since="baseline")
    after_preview = index.change_summary(update=False, since="baseline")

    assert preview["pending_events"] == 1
    assert preview["event_counts"]["added"] == 1
    assert preview["files"][0]["path"] == "concepts/new-page.md"
    assert after_preview["pending_events"] == 1
    assert all(not row["path"].startswith("raw/") for row in preview["files"])


def test_index_consistency_audit_reports_clean_index_and_markdown_drift(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex(include_raw=False)

    clean = index.consistency_audit()
    assert clean["ok"] is True
    assert clean["summary"]["issue_count"] == 0
    assert clean["summary"]["manifest_pages"] == 1
    assert clean["summary"]["chunk_file_chunks"] >= 2

    runbook = wiki / "concepts" / "runbook.md"
    runbook.write_text(runbook.read_text(encoding="utf-8") + "\n## New section\n\nIndex is now stale.\n", encoding="utf-8")
    stale = index.consistency_audit()

    assert stale["ok"] is False
    assert stale["summary"]["markdown_pages"] == 1
    assert stale["summary"]["markdown_chunks"] > stale["summary"]["chunk_file_chunks"]
    issue_codes = {issue["code"] for issue in stale["issues"]}
    assert "manifest_file_stale" in issue_codes
    assert "chunk_count_mismatch" in issue_codes
    assert stale["recommendations"] == ["Run wiki_reindex(include_raw=false) to rebuild stale or inconsistent index artifacts."]


def test_index_consistency_audit_detects_missing_indexed_files_and_manifest_count_mismatch(tmp_path):
    wiki = make_wiki(tmp_path)
    index = WikiIndex(wiki)
    index.reindex(include_raw=False)

    (wiki / "concepts" / "runbook.md").unlink()
    manifest = json.loads((wiki / ".vector" / "manifest.json").read_text(encoding="utf-8"))
    manifest["chunks_indexed"] = 999
    (wiki / ".vector" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = index.consistency_audit()

    assert audit["ok"] is False
    issue_codes = {issue["code"] for issue in audit["issues"]}
    assert "indexed_file_missing" in issue_codes
    assert "manifest_chunk_count_mismatch" in issue_codes
