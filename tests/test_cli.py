import json
import subprocess
import sys
from pathlib import Path


def make_wiki(tmp_path: Path) -> Path:
    (tmp_path / "concepts").mkdir()
    (tmp_path / "concepts" / "runbook.md").write_text("""---
title: Gemma4 RyzenAI Runtime 1.7.1 Runbook
type: concept
tags: [gemma4, runtime]
---

# Runbook

## NPU verification

Use xrt-smi to verify NPU activity.
""")
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


def test_cli_index_status_search_and_read_json(tmp_path):
    wiki = make_wiki(tmp_path)

    index = run_cli("--wiki", str(wiki), "index")
    status = run_cli("--wiki", str(wiki), "status")
    search = run_cli("--wiki", str(wiki), "search", "verify NPU", "--json")
    read = run_cli("--wiki", str(wiki), "read", "concepts/runbook.md", "--heading", "NPU verification")

    assert index.returncode == 0, index.stderr
    assert "chunks_indexed" in index.stdout
    assert json.loads(status.stdout)["pages_indexed"] == 1
    assert json.loads(search.stdout)[0]["path"] == "concepts/runbook.md"
    assert read.returncode == 0
    assert "xrt-smi" in read.stdout
