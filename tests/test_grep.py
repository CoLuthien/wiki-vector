import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wiki_vector import mcp_server
from wiki_vector.grep import grep_wiki
from wiki_vector.mcp_tools import wiki_grep


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "queries").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "concepts" / "runtime.md").write_text(
        "# Runtime\n\nBefore.\n\nGot negative shape dim bound: '-1'.\n\nAfter.\n",
        encoding="utf-8",
    )
    (tmp_path / "queries" / "other.md").write_text(
        "# Other\n\nPREFIX23_NOMHA_XCLBIN failed.\n",
        encoding="utf-8",
    )
    (tmp_path / "raw" / "secret.md").write_text(
        "# Raw\n\nGot negative shape dim bound: '-1'.\n",
        encoding="utf-8",
    )
    return tmp_path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "wiki_vector.cli", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_grep_literal_reads_source_without_index_and_excludes_raw(tmp_path):
    wiki = make_wiki(tmp_path)

    result = grep_wiki(wiki, "negative shape", context=1)

    assert result["count"] == 1
    assert result["truncated"] is False
    match = result["matches"][0]
    assert match["path"] == "concepts/runtime.md"
    assert match["line"] == 5
    assert match["column"] == 5
    assert match["match"] == "negative shape"
    assert match["context_start_line"] == 4
    assert match["context_end_line"] == 6
    assert "Before." not in match["context"]
    assert "After." not in match["context"]
    assert match["read_hint"] == "concepts/runtime.md lines 4-6"
    assert not (wiki / ".vector").exists()


def test_grep_regex_case_sensitivity_raw_opt_in_and_limit(tmp_path):
    wiki = make_wiki(tmp_path)

    insensitive = grep_wiki(wiki, r"prefix\d+_nomha_xclbin", regex=True)
    sensitive = grep_wiki(wiki, r"prefix\d+_nomha_xclbin", regex=True, case_sensitive=True)
    with_raw = grep_wiki(wiki, "negative shape", include_raw=True, limit=1)

    assert insensitive["count"] == 1
    assert insensitive["matches"][0]["match"] == "PREFIX23_NOMHA_XCLBIN"
    assert sensitive["count"] == 0
    assert with_raw["count"] == 1
    assert with_raw["truncated"] is True


def test_grep_validates_pattern_context_and_limit(tmp_path):
    wiki = make_wiki(tmp_path)

    with pytest.raises(ValueError, match="pattern"):
        grep_wiki(wiki, "")
    with pytest.raises(ValueError, match="invalid regular expression"):
        grep_wiki(wiki, "[", regex=True)
    with pytest.raises(ValueError, match="context"):
        grep_wiki(wiki, "shape", context=-1)
    with pytest.raises(ValueError, match="limit"):
        grep_wiki(wiki, "shape", limit=0)


def test_mcp_wiki_grep_returns_serializable_source_matches(tmp_path):
    wiki = make_wiki(tmp_path)

    result = wiki_grep(str(wiki), "shape dim bound", context=0)

    json.dumps(result)
    assert result["matches"][0]["path"] == "concepts/runtime.md"
    source = inspect.getsource(mcp_server)
    assert '@mcp.tool(name="wiki_grep")' in source
    assert "return wiki_grep(" in source


def test_cli_grep_json_and_regex_error_contract(tmp_path):
    wiki = make_wiki(tmp_path)

    found = run_cli("--wiki", str(wiki), "grep", "shape dim", "--context", "0", "--json")
    invalid = run_cli("--wiki", str(wiki), "grep", "[", "--regex", "--json")

    assert found.returncode == 0, found.stderr
    data = json.loads(found.stdout)
    assert data["count"] == 1
    assert data["matches"][0]["line"] == 5
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert "invalid regular expression" in invalid.stderr
    assert "Traceback" not in invalid.stderr


def test_cli_grep_does_not_initialize_embedding_backend(tmp_path, monkeypatch):
    wiki = make_wiki(tmp_path)
    monkeypatch.setenv("WIKI_VECTOR_EMBEDDING_BACKEND", "definitely-invalid")

    result = run_cli("--wiki", str(wiki), "grep", "shape dim", "--context", "0", "--json")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["count"] == 1
    assert not (wiki / ".vector").exists()
