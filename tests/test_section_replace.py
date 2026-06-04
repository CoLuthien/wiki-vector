import pytest

from wiki_vector.index import _replace_markdown_section


def test_replace_markdown_section_replaces_body_and_preserves_sibling():
    text = "# Page\n\n## A\nold\n\n## B\nkeep\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=None)

    assert result.text == "# Page\n\n## A\nnew\n\n## B\nkeep\n"
    assert result.start_line == 3
    assert result.end_line == 4
    assert result.old_section == "## A\nold"
    assert result.new_section == "## A\nnew"


def test_replace_markdown_section_includes_nested_headings():
    text = "# Page\n\n## A\nold\n\n### Child\nold child\n\n## B\nkeep\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=None)

    assert result.text == "# Page\n\n## A\nnew\n\n## B\nkeep\n"
    assert result.start_line == 3
    assert result.end_line == 7


def test_replace_markdown_section_preserves_frontmatter_and_reports_whole_file_lines():
    text = "---\ntitle: Demo\n---\n\n# Demo\n\n## A\nold\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=None)

    assert result.text == "---\ntitle: Demo\n---\n\n# Demo\n\n## A\nnew\n"
    assert result.start_line == 7
    assert result.end_line == 8


def test_replace_markdown_section_ignores_headings_inside_fences():
    text = "# Page\n\n```\n## A\nnot heading\n```\n\n## A\nold\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=None)

    assert "not heading" in result.text
    assert result.text.endswith("## A\nnew\n")


def test_replace_markdown_section_normalizes_closing_hashes():
    text = "# Page\n\n## A ##\nold\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=None)

    assert result.text == "# Page\n\n## A ##\nnew\n"


def test_replace_markdown_section_duplicate_heading_requires_occurrence():
    text = "# Page\n\n## A\none\n\n## A\ntwo\n"

    with pytest.raises(ValueError, match="ambiguous"):
        _replace_markdown_section(text, heading="A", content="new", occurrence=None)


def test_replace_markdown_section_occurrence_replaces_second_match_only():
    text = "# Page\n\n## A\none\n\n## A\ntwo\n"

    result = _replace_markdown_section(text, heading="A", content="new", occurrence=2)

    assert result.text == "# Page\n\n## A\none\n\n## A\nnew\n"


def test_replace_markdown_section_accepts_full_section_with_same_heading_and_level():
    text = "# Page\n\n## A\nold\n\n## B\nkeep\n"

    result = _replace_markdown_section(text, heading="A", content="## A\nnew", occurrence=None)

    assert result.text == "# Page\n\n## A\nnew\n\n## B\nkeep\n"


def test_replace_markdown_section_rejects_full_section_heading_rename():
    text = "# Page\n\n## A\nold\n"

    with pytest.raises(ValueError, match="heading.*match"):
        _replace_markdown_section(text, heading="A", content="## Renamed\nnew", occurrence=None)


def test_replace_markdown_section_rejects_full_section_level_change():
    text = "# Page\n\n## A\nold\n"

    with pytest.raises(ValueError, match="level"):
        _replace_markdown_section(text, heading="A", content="### A\nnew", occurrence=None)


def test_replace_markdown_section_rejects_invalid_occurrence():
    text = "# Page\n\n## A\nold\n"

    with pytest.raises(ValueError, match="occurrence must be >= 1"):
        _replace_markdown_section(text, heading="A", content="new", occurrence=0)


def test_replace_markdown_section_rejects_occurrence_out_of_range():
    text = "# Page\n\n## A\nold\n"

    with pytest.raises(ValueError, match="heading occurrence not found"):
        _replace_markdown_section(text, heading="A", content="new", occurrence=2)


def test_replace_markdown_section_rejects_same_or_higher_heading_in_full_replacement():
    text = "# Page\n\n## A\nold\n\n## B\nkeep\n"

    with pytest.raises(ValueError, match="same-or-higher"):
        _replace_markdown_section(text, heading="A", content="## A\nnew\n\n## Surprise\noops", occurrence=None)
