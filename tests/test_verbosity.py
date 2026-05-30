import json
from pathlib import Path

from wiki_vector.verbosity import analyze_verbosity


def page(body: str) -> str:
    return f"""---
title: Verbose Page
type: concept
tags: [wiki]
---

{body}
"""


def test_dataclass_to_dict_is_json_serializable_and_strips_frontmatter():
    result = analyze_verbosity("concepts/short.md", page("# Short\n\nConcise body."))

    data = result.to_dict()
    assert data["path"] == "concepts/short.md"
    assert data["metrics"]["body_line_count"] < data["metrics"]["line_count"]
    json.dumps(data)


def test_long_page_and_section_report_source_ranges():
    long_lines = "\n".join(f"Repeated implementation history line {i}." for i in range(125))
    text = page(f"# Long\n\n## Implementation history\n\n{long_lines}")

    result = analyze_verbosity("concepts/long.md", text)

    assert result.is_verbose is True
    assert result.severity == "high"
    assert any(r.code == "section_too_long" for r in result.reasons)
    section = result.sections[0]
    assert section.heading == "Implementation history"
    assert section.start_line > 1
    assert section.end_line >= section.start_line


def test_repeated_phrases_and_near_duplicate_sentences_trigger_redundancy():
    sentence = "The same durable wiki claim repeats across this paragraph for no new reason."
    text = page("# Repetition\n\n" + " ".join([sentence] * 30))

    result = analyze_verbosity("concepts/repetition.md", text)

    codes = {r.code for r in result.reasons}
    assert "repeated_5grams" in codes
    assert "near_duplicate_sentences" in codes
    assert "deduplicate_repeated_claims" in result.suggestions


def test_sentence_extreme_and_long_sentence_ratio():
    long_sentence = " ".join(f"token{i}" for i in range(90)) + "."
    result = analyze_verbosity("concepts/sentence.md", page("# Sentence\n\n" + long_sentence))

    codes = {r.code for r in result.reasons}
    assert "sentence_extreme" in codes
    assert result.metrics["max_sentence_words"] >= 90


def test_code_blocks_are_excluded_by_default():
    repeated_code = "\n".join(["print('same same same same same')" for _ in range(80)])
    text = page(f"# Code\n\n```python\n{repeated_code}\n```\n\nShort prose.")

    without_code = analyze_verbosity("concepts/code.md", text, include_code=False)
    with_code = analyze_verbosity("concepts/code.md", text, include_code=True)

    assert without_code.metrics["word_count"] < with_code.metrics["word_count"]
    assert without_code.metrics["repeated_5gram_ratio"] <= with_code.metrics["repeated_5gram_ratio"]


def test_compare_to_reports_rewrite_metrics_and_preservation():
    original = page("# Original\n\n## Detail\n\n[[wiki-vector-mcp]] " + "word " * 200)
    compact = page("# Original\n\n## Detail\n\n[[wiki-vector-mcp]] compact summary.")

    result = analyze_verbosity("concepts/original.md", original, compare_to=compact)

    assert result.metrics["compression_ratio"] < 0.75
    assert result.metrics["heading_preservation_ratio"] == 1.0
    assert "replace_with_compact_index_and_archive_original" in result.suggestions
