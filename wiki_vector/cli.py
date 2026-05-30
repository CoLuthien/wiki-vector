from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .embeddings import create_embedder, embedding_config_from_args
from .index import WikiIndex


def _wiki_default() -> str:
    return os.environ.get("WIKI_PATH") or str(Path.home() / "wiki")


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_verbosity(result: dict[str, Any]) -> None:
    status = "VERBOSE" if result.get("is_verbose") else "OK"
    print(f"{status} {result.get('severity')} score={float(result.get('score', 0.0)):.2f} {result.get('path')}")
    for reason in result.get("reasons", [])[:5]:
        loc = ""
        if reason.get("start_line") and reason.get("end_line"):
            loc = f" lines {reason['start_line']}-{reason['end_line']}"
        print(f"- {reason.get('code')}: {reason.get('value')} >= {reason.get('threshold')}{loc}")
    suggestions = result.get("suggestions") or []
    if suggestions:
        print("Suggestions: " + ", ".join(suggestions))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wiki-vector", description="Local index/search layer for a Markdown LLM Wiki")
    parser.add_argument("--wiki", default=_wiki_default(), help="Wiki root path (default: $WIKI_PATH or ~/wiki)")
    parser.add_argument("--embedding-backend", choices=["hashing-ngram", "openvino-bge-m3"], help="Embedding backend (default: $WIKI_VECTOR_EMBEDDING_BACKEND or hashing-ngram)")
    parser.add_argument("--embedding-model", help="Embedding model id/path (default for OpenVINO: BAAI/bge-m3)")
    parser.add_argument("--embedding-device", help="Model execution device for runtime backends, e.g. CPU, GPU, NPU (default for OpenVINO: NPU)")
    parser.add_argument("--embedding-dimensions", type=int, help="Vector dimensions for hashing-ngram backend")
    parser.add_argument("--embedding-batch-size", type=int, help="Batch size for model-backed embedders")
    parser.add_argument("--embedding-cache-dir", help="Optional model/cache directory for model-backed embedders")
    parser.add_argument("--embedding-max-length", type=int, help="Static sequence length for model-backed embedders (default: 512)")
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

    p_read = sub.add_parser("read", help="Read a wiki page, heading section, or source line range")
    p_read.add_argument("path")
    p_read.add_argument("--heading")
    p_read.add_argument("--start-line", type=int, help="1-indexed source line where the read range starts")
    p_read.add_argument("--end-line", type=int, help="1-indexed source line where the read range ends")
    p_read.add_argument("--json", action="store_true")

    p_verbose = sub.add_parser("is-verbose", help="Analyze whether a wiki page is too verbose")
    p_verbose.add_argument("path")
    p_verbose.add_argument("--include-code", action="store_true", help="Include code blocks in readability/redundancy metrics")
    p_verbose.add_argument("--compare-to", help="Compare original PATH to a compact rewrite path")
    p_verbose.add_argument("--json", action="store_true")

    p_audit = sub.add_parser("verbosity-audit", help="List highest-verbosity pages in the wiki")
    p_audit.add_argument("--limit", type=int, default=20)
    p_audit.add_argument("--include-raw", action="store_true")
    p_audit.add_argument("--severity", choices=["ok", "warning", "high"])
    p_audit.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = embedding_config_from_args(
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        dimensions=args.embedding_dimensions,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        cache_dir=args.embedding_cache_dir,
        max_length=args.embedding_max_length,
    )
    index = WikiIndex(args.wiki, embedder=create_embedder(config))

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
        result = index.read(args.path, heading=args.heading, start_line=args.start_line, end_line=args.end_line)
        if args.json:
            _print_json(result.to_dict())
        else:
            print(result.content)
        return 0
    if args.command == "is-verbose":
        result = index.is_verbose(args.path, include_code=args.include_code, compare_to=args.compare_to)
        if args.json:
            _print_json(result.to_dict())
        else:
            _print_verbosity(result.to_dict())
        return 0
    if args.command == "verbosity-audit":
        results = index.verbosity_audit(limit=args.limit, include_raw=args.include_raw, severity=args.severity)
        data = {"results": [r.to_dict() for r in results], "count": len(results)}
        if args.json:
            _print_json(data)
        else:
            for result in data["results"]:
                _print_verbosity(result)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
