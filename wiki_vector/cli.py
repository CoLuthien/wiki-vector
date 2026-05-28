from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .index import WikiIndex


def _wiki_default() -> str:
    return os.environ.get("WIKI_PATH") or str(Path.home() / "wiki")


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wiki-vector", description="Local index/search layer for a Markdown LLM Wiki")
    parser.add_argument("--wiki", default=_wiki_default(), help="Wiki root path (default: $WIKI_PATH or ~/wiki)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build or refresh the local index")
    p_index.add_argument("--include-raw", action="store_true", help="Include raw/ markdown files")

    sub.add_parser("status", help="Show index status")

    p_search = sub.add_parser("search", help="Search indexed chunks")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=8)
    p_search.add_argument("--include-raw", action="store_true")
    p_search.add_argument("--type", dest="types", action="append", help="Filter by page type; repeatable")
    p_search.add_argument("--tag", dest="tags", action="append", help="Filter by tag; repeatable")
    p_search.add_argument("--json", action="store_true", help="Emit JSON array")

    p_read = sub.add_parser("read", help="Read a wiki page or heading section")
    p_read.add_argument("path")
    p_read.add_argument("--heading")
    p_read.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = WikiIndex(args.wiki)

    if args.command == "index":
        _print_json(index.reindex(include_raw=args.include_raw).to_dict())
        return 0
    if args.command == "status":
        _print_json(index.status().to_dict())
        return 0
    if args.command == "search":
        results = index.search(args.query, limit=args.limit, include_raw=args.include_raw, types=args.types, tags=args.tags)
        data = [r.to_dict() for r in results]
        if args.json:
            print(json.dumps(data, ensure_ascii=False))
        else:
            for r in results:
                print(f"{r.score:.3f}\t{r.path}#{r.heading}\t{r.snippet}")
        return 0
    if args.command == "read":
        result = index.read(args.path, heading=args.heading)
        if args.json:
            _print_json(result.to_dict())
        else:
            print(result.content)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
