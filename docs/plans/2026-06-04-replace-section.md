# wiki_write replace-section Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe `mode="replace-section"` write path to wiki-vector so agents can replace one Markdown section in an existing wiki page without overwriting the whole source-of-truth document.

**Architecture:** Keep Markdown files as the source of truth. Extend the existing `WikiIndex.write()` path, then thread the new parameters through `mcp_tools.py`, `mcp_server.py`, and a new CLI `write` subcommand. Implement a small dependency-free ATX heading span helper in `index.py` or a focused helper module; do not introduce a Markdown AST parser.

**Tech Stack:** Python 3.11, pytest, argparse CLI, existing wiki-vector MCP wrappers, existing `WikiIndex.reindex(include_raw=False)` behavior.

---

## Current repository context

Target repo: `/workspace/wiki-vector`

Relevant existing files:
- `wiki_vector/index.py`
  - `WikiIndex.write(path, content, mode="create", reindex=True)` currently supports only `create | overwrite | append`.
  - `WriteResult` currently returns `path`, `mode`, `bytes_written`, `reindexed`, `status`.
  - `_safe_markdown_path()` already validates relative `.md` paths and blocks `.vector`.
- `wiki_vector/mcp_tools.py`
  - `wiki_write(wiki_path, path, content, mode="create", reindex=True)` delegates to `WikiIndex.write()`.
- `wiki_vector/mcp_server.py`
  - MCP `wiki_write` tool exposes `path`, `content`, `mode`, `reindex`, `wiki_path`.
- `wiki_vector/cli.py`
  - Has `index`, `status`, `search`, `read`, `is-verbose`, `verbosity-audit`, `change-summary`, `consistency-audit`.
  - Does **not** currently expose a `write` subcommand.
- `wiki_vector/chunking.py`
  - Existing chunking splits at the next heading regardless of level. `replace-section` intentionally uses hierarchical section semantics instead.
- Tests:
  - `tests/test_mcp_tools.py`
  - `tests/test_cli.py`
  - `tests/test_markdown.py` exists for Markdown helpers and can host parser tests if a helper is moved to `markdown.py`.

Quality gate:
- Use strict TDD: write failing tests first, run and observe RED, implement minimal code, run GREEN.
- Final verification: `pytest tests/ -q` from `/workspace/wiki-vector`.

---

## Finalized behavior contract

### `replace-section` API

Python/MCP call shape:

```python
wiki_write(
    wiki_path,
    path="concepts/foo.md",
    content="new section body or full section",
    mode="replace-section",
    heading="Current status",
    occurrence=None,
    reindex=True,
)
```

`WikiIndex.write()` signature becomes:

```python
def write(
    self,
    path: str,
    content: str,
    mode: str = "create",
    reindex: bool = True,
    heading: str | None = None,
    occurrence: int | None = None,
) -> WriteResult:
```

MCP wrapper/tool signature becomes equivalent.

### Section span semantics

For `mode="replace-section"`:
- `heading` is required.
- Target file must already exist.
- Supported heading syntax is ATX only: `#` through `######` at the start of a line.
- Optional closing hashes are normalized: `## Current status ##` matches `Current status`.
- Heading text matching is exact after stripping heading markers, optional closing hashes, and surrounding whitespace.
- Fenced code blocks are ignored when scanning headings.
  - Support common fences beginning with at least three backticks or tildes.
  - Do not treat `##` lines inside fences as headings.
- Setext headings are out of scope.
- YAML frontmatter at the top of the file is preserved verbatim. Heading scanning begins after the frontmatter block, but returned line numbers are original whole-file line numbers.
- Replacement span is hierarchical: from the target heading line through the line before the next heading whose level is less than or equal to the target heading level. Nested lower-level headings are included in the span.

### Content semantics

`content` may be either:
1. **Body-only replacement:** first non-empty line is not a Markdown heading. Preserve the original heading line and replace only the body.
2. **Full-section replacement:** first non-empty line is a Markdown heading. It must have the same normalized heading text and same heading level as the target section. Rename or heading-level changes are not allowed in this feature.

Replacement-content classification is precise:
- Ignore leading blank lines when deciding whether `content` is body-only or full-section content.
- If the first non-empty line is an ATX heading, classify as full-section content and validate that first heading against the target heading text and level.
- If prose appears before any heading, classify as body-only content even if later lines contain Markdown headings; those later headings become nested content inside the target section.
- Full-section content may contain additional nested lower-level headings. It must not introduce an additional same-or-higher heading after the first line in the replacement section; reject this with `ValueError` to avoid one replacement accidentally creating sibling sections.

For all write modes, keep the existing `content.strip()` non-empty validation.

The final file must end with exactly one trailing newline through the same normalizing style as current `write()`.

All validation must complete before the destination file is opened for writing. Prefer writing to a temporary sibling file and then `Path.replace()` for atomic replacement; at minimum, preserve the existing file untouched until the complete final string has been generated and every `replace-section` validation has passed.

### Failure contract

These failures must not modify the file:
- Missing file for `replace-section`: `FileNotFoundError`.
- Missing `heading`: `ValueError("heading is required for replace-section")` or equivalent clear message.
- Invalid `occurrence < 1`: `ValueError("occurrence must be >= 1")`.
- Heading not found: `ValueError("heading not found: <heading>")`.
- Duplicate heading with no `occurrence`: `ValueError("heading is ambiguous: ...")`.
- Occurrence out of range: `ValueError("heading occurrence not found: ...")`.
- Full replacement heading text mismatch: `ValueError(...)` with no traceback in CLI.
- Full replacement heading level mismatch: `ValueError(...)` with no traceback in CLI.

CLI failures should follow the existing CLI error pattern: return code `2`, human-readable stderr, no Python traceback.

### Result metadata

Extend `WriteResult` with optional fields:

```python
heading: str | None = None
start_line: int | None = None       # original file, 1-indexed inclusive
end_line: int | None = None         # original file, 1-indexed inclusive
old_section_bytes: int | None = None # UTF-8 byte count of exact replaced old span
new_section_bytes: int | None = None # UTF-8 byte count of inserted new section
```

For non-`replace-section` modes, these fields may be `None`.

---

## Task 1: Add RED tests for section span replacement helper

**Objective:** Define exact hierarchical section replacement behavior before production code exists.

**Files:**
- Modify: `tests/test_markdown.py` or create `tests/test_section_replace.py`
- Later production target: `wiki_vector/index.py` or `wiki_vector/markdown.py`

**Step 1: Write failing tests**

Add tests for a helper with a wished-for API. Prefer a small private helper in `wiki_vector.index` initially to keep scope tight:

```python
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
```

If the implementer chooses a different helper name/module, update imports but keep behavior identical.

**Step 2: Run RED**

Run:

```bash
pytest tests/test_section_replace.py::test_replace_markdown_section_replaces_body_and_preserves_sibling tests/test_section_replace.py::test_replace_markdown_section_includes_nested_headings -v
```

Expected: FAIL because `_replace_markdown_section` does not exist.

**Step 3: Commit?**

Do not commit after RED unless the project convention allows red commits. Continue to Task 2 for GREEN.

---

## Task 2: Implement minimal section replacement helper

**Objective:** Pass the first two helper tests with minimal production code.

**Files:**
- Modify: `wiki_vector/index.py` or `wiki_vector/markdown.py`
- Test: `tests/test_section_replace.py`

**Step 1: Implement dataclass and helper**

A minimal internal shape is enough:

```python
@dataclass(frozen=True)
class SectionReplaceResult:
    text: str
    heading: str
    start_line: int
    end_line: int
    old_section: str
    new_section: str
```

Implement `_replace_markdown_section(text, *, heading, content, occurrence)`:
- Parse ATX headings outside fences.
- Find matching heading spans.
- If body-only content, preserve original heading line.
- Build `new_text` and normalize to one trailing newline.

**Step 2: Run GREEN**

Run:

```bash
pytest tests/test_section_replace.py::test_replace_markdown_section_replaces_body_and_preserves_sibling tests/test_section_replace.py::test_replace_markdown_section_includes_nested_headings -v
```

Expected: PASS.

**Step 3: Run nearby tests**

Run:

```bash
pytest tests/test_markdown.py tests/test_mcp_tools.py::test_mcp_tool_functions_return_serializable_dicts -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add wiki_vector/index.py tests/test_section_replace.py
git commit -m "feat: add markdown section replacement helper"
```

---

## Task 3: Add RED tests for Markdown edge semantics

**Objective:** Lock down frontmatter, fences, optional closing hashes, duplicate headings, and occurrence behavior.

**Files:**
- Modify: `tests/test_section_replace.py`

**Step 1: Write failing tests**

Add tests:

```python
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
```

**Step 2: Run RED**

Run:

```bash
pytest tests/test_section_replace.py -v
```

Expected: New tests fail for unimplemented edge semantics.

---

## Task 4: Implement Markdown edge semantics

**Objective:** Make section scanning safe enough for source-of-truth Markdown edits.

**Files:**
- Modify: `wiki_vector/index.py` or helper module selected in Task 2
- Test: `tests/test_section_replace.py`

**Step 1: Implement heading scanner**

Implementation notes:
- Use a line-based scanner, not `re.MULTILINE` over the full text, because fences and line numbers matter.
- Track `in_fence` for lines matching `^\s*(```+|~~~+)` with fence length >= 3. A closing fence must use the same marker char and at least the opening length.
- For first pass, treat headings only when not inside a fence and line starts with `#{1,6} ` after no indentation. If you want up to 3 spaces indentation, add tests first; otherwise keep line-start only.
- Normalize ATX heading text by removing optional closing hashes preceded by whitespace.
- Derive span end by scanning subsequent headings until `level <= target_level`; if none, span ends at EOF.
- Preserve exact original heading line for body-only replacement.

**Step 2: Implement failure checks**

- `occurrence is not None and occurrence < 1` -> `ValueError`.
- no matches -> `ValueError("heading not found: ...")`.
- multiple matches and occurrence is `None` -> `ValueError("heading is ambiguous: ...")`.
- occurrence out of range -> `ValueError("heading occurrence not found: ...")`.

**Step 3: Run GREEN**

Run:

```bash
pytest tests/test_section_replace.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add wiki_vector/index.py tests/test_section_replace.py
git commit -m "test: lock down replace-section markdown semantics"
```

---

## Task 5: Add RED tests for full-section replacement safety

**Objective:** Ensure full replacement cannot accidentally rename a heading or change its level.

**Files:**
- Modify: `tests/test_section_replace.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run RED**

Run:

```bash
pytest tests/test_section_replace.py::test_replace_markdown_section_accepts_full_section_with_same_heading_and_level tests/test_section_replace.py::test_replace_markdown_section_rejects_full_section_heading_rename tests/test_section_replace.py::test_replace_markdown_section_rejects_full_section_level_change -v
```

Expected: At least mismatch/level tests fail until validation is implemented.

---

## Task 6: Implement full-section replacement validation

**Objective:** Support full-section content safely while keeping body-only replacement convenient.

**Files:**
- Modify: `wiki_vector/index.py` or helper module
- Test: `tests/test_section_replace.py`

**Step 1: Detect first non-empty content line**

Rules:
- If first non-empty line is an ATX heading, treat content as full-section replacement.
- Normalize its heading text and level using the same heading parser.
- Require text equals target normalized heading text and level equals target level.
- If first non-empty line is not heading, prepend/preserve the original target heading line.

**Step 2: Normalize final section**

- Avoid adding extra blank lines before the next sibling heading.
- Ensure inserted section has no more than one trailing newline before joining with suffix.
- Ensure whole file ends with one newline.

**Step 3: Run GREEN**

Run:

```bash
pytest tests/test_section_replace.py -v
```

Expected: PASS.

**Step 4: Commit**

```bash
git add wiki_vector/index.py tests/test_section_replace.py
git commit -m "feat: validate full-section replacement content"
```

---

## Task 7: Add RED tests for `WikiIndex.write(mode="replace-section")`

**Objective:** Define integration behavior at the main write API, including result metadata and reindex behavior.

**Files:**
- Modify: `tests/test_mcp_tools.py` or create `tests/test_write_replace_section.py`
- Production target: `wiki_vector/index.py`, `wiki_vector/mcp_tools.py`

**Step 1: Write failing tests**

Add tests using temporary wiki pages:

```python
from wiki_vector.mcp_tools import wiki_read, wiki_reindex, wiki_search, wiki_write


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
    assert "replace-section" in wiki_read(str(wiki), "concepts/runbook.md", heading="Runtime setup")["content"]
    assert wiki_search(str(wiki), "precise edits", limit=2)["results"][0]["path"] == "concepts/runbook.md"


def test_wiki_write_replace_section_missing_heading_does_not_modify_file(tmp_path):
    wiki = make_wiki(tmp_path)
    page = wiki / "concepts" / "runbook.md"
    before = page.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="heading not found"):
        wiki_write(str(wiki), "concepts/runbook.md", "new", mode="replace-section", heading="Missing")

    assert page.read_text(encoding="utf-8") == before
```

Ensure `make_wiki()` includes a `## Runtime setup` section; adjust expected line numbers to the fixture.

**Step 2: Run RED**

Run:

```bash
pytest tests/test_mcp_tools.py::test_wiki_write_replace_section_updates_existing_page_and_reindexes tests/test_mcp_tools.py::test_wiki_write_replace_section_missing_heading_does_not_modify_file -v
```

Expected: FAIL because `wiki_write` does not accept `heading` and mode is unsupported.

---

## Task 8: Extend `WriteResult`, `WikiIndex.write()`, and `mcp_tools.wiki_write()`

**Objective:** Wire the helper into the main Python/MCP wrapper path.

**Files:**
- Modify: `wiki_vector/index.py`
- Modify: `wiki_vector/mcp_tools.py`
- Test: `tests/test_mcp_tools.py`

**Step 1: Extend `WriteResult`**

Add optional fields while preserving JSON serialization via `asdict()`:

```python
heading: str | None = None
start_line: int | None = None
end_line: int | None = None
old_section_bytes: int | None = None
new_section_bytes: int | None = None
```

**Step 2: Extend `WikiIndex.write()` signature**

Add `heading` and `occurrence` optional parameters. Update mode validation:

```python
if mode not in {"create", "overwrite", "append", "replace-section"}:
    raise ValueError("mode must be one of: create, overwrite, append, replace-section")
```

**Step 3: Implement mode branch**

For `replace-section`:
- Validate `heading`.
- Validate file exists.
- Read current file.
- Call `_replace_markdown_section()`.
- Write final text only after all validation passes.
- Reindex with current behavior if requested.
- Return extended metadata.

Do not change behavior for `create`, `overwrite`, or `append` except that their `WriteResult` includes `None` for new optional fields.

**Step 4: Extend `mcp_tools.wiki_write()`**

```python
def wiki_write(..., heading: str | None = None, occurrence: int | None = None) -> dict:
    return WikiIndex(wiki_path).write(..., heading=heading, occurrence=occurrence).to_dict()
```

**Step 5: Run GREEN**

Run:

```bash
pytest tests/test_mcp_tools.py::test_wiki_write_replace_section_updates_existing_page_and_reindexes tests/test_mcp_tools.py::test_wiki_write_replace_section_missing_heading_does_not_modify_file -v
```

Expected: PASS.

**Step 6: Regression tests**

Run:

```bash
pytest tests/test_mcp_tools.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add wiki_vector/index.py wiki_vector/mcp_tools.py tests/test_mcp_tools.py
git commit -m "feat: support replace-section in wiki_write"
```

---

## Task 9: Add RED tests for MCP server signature

**Objective:** Ensure the MCP tool exposes `heading` and `occurrence` parameters.

**Files:**
- Modify: `tests/test_mcp_tools.py` if direct server inspection is hard, or add a focused import/signature test.
- Production target: `wiki_vector/mcp_server.py`

**Step 1: Write failing signature/behavior test**

If FastMCP introspection is too coupled to optional dependencies, test the registered wrapper indirectly by inspecting source is not ideal. Prefer a direct callable extraction only if existing patterns support it. Otherwise, this can be covered by import/type checking and manual MCP tool signature review.

Minimal pragmatic test:

```python
import inspect
from wiki_vector import mcp_server


def test_mcp_server_wiki_write_source_exposes_replace_section_parameters():
    source = inspect.getsource(mcp_server)
    assert "heading: str | None = None" in source
    assert "occurrence: int | None = None" in source
    assert "heading=heading" in source
    assert "occurrence=occurrence" in source
```

This is not beautiful, but it avoids requiring the optional MCP SDK in the test environment. If a better in-repo MCP registration test exists, use that instead.

**Step 2: Run RED**

Run:

```bash
pytest tests/test_mcp_tools.py::test_mcp_server_wiki_write_source_exposes_replace_section_parameters -v
```

Expected: FAIL until `mcp_server.py` is updated.

---

## Task 10: Extend MCP server `wiki_write` tool

**Objective:** Expose the new mode parameters to agents using the MCP server.

**Files:**
- Modify: `wiki_vector/mcp_server.py`
- Test: `tests/test_mcp_tools.py`

**Step 1: Update import wrapper call**

Change tool signature:

```python
def wiki_write_tool(
    path: str,
    content: str,
    mode: str = "create",
    heading: str | None = None,
    occurrence: int | None = None,
    reindex: bool = True,
    wiki_path: str | None = None,
) -> dict[str, Any]:
```

Call:

```python
return wiki_write(
    _wiki_path(wiki_path),
    path=path,
    content=content,
    mode=mode,
    heading=heading,
    occurrence=occurrence,
    reindex=reindex,
)
```

Update docstring to mention replace-section.

**Step 2: Run GREEN**

Run:

```bash
pytest tests/test_mcp_tools.py::test_mcp_server_wiki_write_source_exposes_replace_section_parameters -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add wiki_vector/mcp_server.py tests/test_mcp_tools.py
git commit -m "feat: expose replace-section parameters in MCP tool"
```

---

## Task 11: Add RED tests for CLI `write --mode replace-section`

**Objective:** Define manual/debug CLI behavior.

**Files:**
- Modify: `tests/test_cli.py`
- Production target: `wiki_vector/cli.py`

**Step 1: Write failing CLI smoke test**

```python
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
    data = json.loads(result.stdout)
    assert data["mode"] == "replace-section"
    assert data["heading"] == "NPU verification"
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
    assert "heading not found" in result.stderr
    assert "Traceback" not in result.stderr
```

**Step 2: Run RED**

Run:

```bash
pytest tests/test_cli.py::test_cli_write_replace_section_content_file_json tests/test_cli.py::test_cli_write_replace_section_error_has_no_traceback -v
```

Expected: FAIL because CLI has no `write` subcommand.

---

## Task 12: Implement CLI `write` subcommand

**Objective:** Add a minimal CLI write path with content-file support and replace-section options.

**Files:**
- Modify: `wiki_vector/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Add parser**

In `build_parser()`:

```python
p_write = sub.add_parser("write", help="Create, overwrite, append, or replace a section in a wiki page")
p_write.add_argument("path")
p_write.add_argument("--mode", default="create", choices=["create", "overwrite", "append", "replace-section"])
p_write.add_argument("--heading")
p_write.add_argument("--occurrence", type=int)
p_write.add_argument("--content-file", required=True)
p_write.add_argument("--no-reindex", action="store_true")
p_write.add_argument("--json", action="store_true")
```

Keep `--content` and `--stdin` as non-goals unless explicitly added with tests.

Concrete success smoke shape expected by this task:

```bash
python -m wiki_vector.cli --wiki "$tmp" write concepts/runbook.md \
  --mode replace-section \
  --heading "NPU verification" \
  --content-file "$tmp/section.md" \
  --json
```

Expected: return code `0`; stdout is JSON with at least `mode`, `path`, `heading`, `start_line`, `end_line`, `old_section_bytes`, `new_section_bytes`, `reindexed`; stderr is empty.

Concrete failure smoke shape expected by this task:

```bash
python -m wiki_vector.cli --wiki "$tmp" write concepts/runbook.md \
  --mode replace-section \
  --heading "Missing" \
  --content-file "$tmp/section.md" \
  --json
```

Expected: return code `2`; stderr contains `heading not found`; stderr does not contain `Traceback`; stdout is empty.

**Step 2: Add main branch**

In `main()` after `index = WikiIndex(...)` creation:

```python
if args.command == "write":
    content = Path(args.content_file).read_text(encoding="utf-8")
    result = index.write(
        args.path,
        content=content,
        mode=args.mode,
        heading=args.heading,
        occurrence=args.occurrence,
        reindex=not args.no_reindex,
    ).to_dict()
    if args.json:
        _print_json(result)
    else:
        print(f"{result['mode']} {result['path']} bytes={result['bytes_written']}")
    return 0
```

Ensure existing exception handling catches `ValueError`, `FileNotFoundError`, etc. and returns `2` without traceback. If current handling only catches `RuntimeError`, extend it narrowly.

**Step 3: Run GREEN**

Run:

```bash
pytest tests/test_cli.py::test_cli_write_replace_section_content_file_json tests/test_cli.py::test_cli_write_replace_section_error_has_no_traceback -v
```

Expected: PASS.

**Step 4: Regression tests**

Run:

```bash
pytest tests/test_cli.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add wiki_vector/cli.py tests/test_cli.py
git commit -m "feat: add CLI write replace-section command"
```

---

## Task 13: Add final acceptance tests for all failure cases

**Objective:** Make failure behavior explicitly testable and protect source files from partial writes.

**Files:**
- Modify: `tests/test_section_replace.py`
- Modify: `tests/test_mcp_tools.py`
- Modify: `tests/test_cli.py` if needed

**Step 1: Add missing failure tests**

Cover:
- `mode="replace-section"` without heading fails and does not modify file.
- `occurrence=0` fails.
- occurrence out of range fails.
- replacement full-section heading mismatch does not modify file through `wiki_write`.
- replacement full-section level mismatch does not modify file through `wiki_write`.
- missing file raises `FileNotFoundError`.

Example:

```python
def test_wiki_write_replace_section_rejects_heading_level_change_without_modifying_file(tmp_path):
    wiki = make_wiki(tmp_path)
    page = wiki / "concepts" / "runbook.md"
    before = page.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="level"):
        wiki_write(str(wiki), "concepts/runbook.md", "### Runtime setup\nnew", mode="replace-section", heading="Runtime setup")

    assert page.read_text(encoding="utf-8") == before
```

**Step 2: Run RED if any behavior is not implemented**

Run targeted tests:

```bash
pytest tests/test_section_replace.py tests/test_mcp_tools.py -q
```

Expected: Any missing failure handling fails.

**Step 3: Implement minimal fixes**

Patch helper/write code only as needed.

**Step 4: Run GREEN**

Run:

```bash
pytest tests/test_section_replace.py tests/test_mcp_tools.py tests/test_cli.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add wiki_vector/index.py tests/test_section_replace.py tests/test_mcp_tools.py tests/test_cli.py
git commit -m "test: cover replace-section failure contracts"
```

---

## Task 14: Documentation and wiki retrieval anchors

**Objective:** Document the new mode where future agents/users can discover it.

**Files:**
- Modify: README or docs if present; if no docs exist, update `wiki_vector/mcp_server.py` and tests are enough for repo docs.
- Write to shared LLM Wiki after implementation succeeds.

**Step 1: Find appropriate docs**

Run:

```bash
find . -maxdepth 2 -iname 'readme*' -o -path './docs/*'
```

Use Hermes `search_files(target="files")` instead of shell find if doing this manually.

**Step 2: Update repo docs if available**

Add a concise section:

```md
### Section replacement writes

`wiki_write(..., mode="replace-section", heading="...")` replaces one ATX Markdown section in an existing page. It uses hierarchical section spans, preserves YAML frontmatter, ignores headings inside fenced code blocks, and reindexes by default.
```

Include CLI example:

```bash
wiki-vector --wiki /workspace/llm-wiki write concepts/foo.md \
  --mode replace-section \
  --heading "Current status" \
  --content-file /tmp/new-section.md \
  --json
```

**Step 3: Add LLM Wiki durable note after final verification**

Use `wiki_write` against the shared wiki, likely appending to the existing wiki-vector MCP page or a focused implementation page. Include retrieval anchors:
- `replace-section`
- `section replacement`
- `heading-scoped write`
- `섹션별 수정`
- `부분 섹션 교체`

Do not do this before tests pass.

**Step 4: Commit repo docs**

```bash
git add README* docs/ wiki_vector/mcp_server.py
git commit -m "docs: document replace-section writes"
```

Adjust `git add` paths to actual changed files.

---

## Task 15: Final verification

**Objective:** Prove the feature works end-to-end and no regressions were introduced.

**Files:**
- No planned source changes.

**Step 1: Run full test suite**

Run:

```bash
pytest tests/ -q
```

Expected: all tests pass.

**Step 2: Run CLI smoke manually**

Use a temporary wiki:

```bash
tmp=$(mktemp -d)
mkdir -p "$tmp/concepts"
cat > "$tmp/concepts/demo.md" <<'EOF'
---
title: Demo
---

# Demo

## Current status
old

### Detail
old detail

## Next
keep
EOF
cat > "$tmp/new.md" <<'EOF'
new status body
EOF
python -m wiki_vector.cli --wiki "$tmp" write concepts/demo.md --mode replace-section --heading "Current status" --content-file "$tmp/new.md" --json
python -m wiki_vector.cli --wiki "$tmp" read concepts/demo.md --heading "Current status"
```

Expected:
- JSON write result has `mode="replace-section"`, `heading="Current status"`, non-null line/byte metadata.
- Read output contains `new status body` and does not contain `old detail`.
- `## Next` remains in the file.

**Step 3: Check git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intended source, tests, and docs changed.

**Step 4: Final commit if any uncommitted changes remain**

```bash
git add <intended paths>
git commit -m "feat: add replace-section wiki writes"
```

---

## Acceptance criteria checklist

- [ ] `wiki_write(..., mode="replace-section", heading="...")` replaces only the selected section in an existing page.
- [ ] Body-only content preserves the original heading line.
- [ ] Full-section content is accepted only when heading text and level match the target.
- [ ] Hierarchical section span includes nested headings and stops at next same-or-higher heading.
- [ ] YAML frontmatter is preserved verbatim.
- [ ] ATX optional closing hashes normalize for matching.
- [ ] Headings inside fenced code blocks are ignored.
- [ ] Duplicate headings fail without `occurrence`.
- [ ] `occurrence` is 1-indexed and can replace the selected duplicate.
- [ ] Missing heading, invalid occurrence, missing file, heading mismatch, and level mismatch fail without modifying the file.
- [ ] `WriteResult` includes JSON-serializable replacement metadata.
- [ ] MCP `wiki_write` exposes `heading` and `occurrence`.
- [ ] CLI `write --mode replace-section --heading ... --content-file ... --json` works.
- [ ] CLI failures return nonzero with no traceback.
- [ ] `reindex=True` keeps search results current.
- [ ] Full `pytest tests/ -q` passes.

## Pitfalls

- Do not reuse `chunk_document()` for replacement spans; it splits at any next heading and would not include nested subsections.
- Do not use regex over the full file without fence awareness; it will corrupt sections containing Markdown examples.
- Do not silently replace the first duplicate heading. Require `occurrence`.
- Do not permit full-section replacement to rename headings in this feature; it creates surprising source-of-truth changes.
- Do not write the file until every validation check has passed.
- Keep `reindex(include_raw=False)` behavior identical to current `write()` unless the user separately asks for raw-aware writes.
