from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

from .embeddings import Embedder
from .markdown import parse_markdown

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class ReadabilityAnalysisConfig:
    """Configuration shared by optional readability analyzers.

    `compare_to` is the compact rewrite text, when rewrite-preservation scoring is
    requested. Analyzers are advisory; they must not mutate deterministic
    verbosity score/reasons unless a future calibrated policy explicitly opts in.
    """

    compare_to: str | None = None
    min_sections_for_coherence: int = 2


@dataclass(frozen=True)
class ReadabilityAnalysis:
    kind: str
    backend: str
    model: str
    ml_readability_score: float | None = None
    coherence_score: float | None = None
    semantic_redundancy_score: float | None = None
    rewrite_preservation_score: float | None = None
    caveats: list[str] | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReadabilityAnalyzer(Protocol):
    kind: str

    def analyze(
        self,
        *,
        path: str,
        text: str,
        metrics: dict[str, Any],
        sections: Sequence[dict[str, Any]],
        compare_to: str | None = None,
    ) -> ReadabilityAnalysis: ...


class EmbeddingSemanticReadabilityAnalyzer:
    """Embedding-backed advisory readability analyzer.

    With a neural `Embedder` such as OpenVINO bge-m3, this provides a neural
    semantic mode. With the default hashing embedder it still exercises the same
    abstraction deterministically, which keeps tests/MCP startup dependency-light.
    """

    kind = "embedding-semantic"

    def __init__(self, embedder: Embedder, config: ReadabilityAnalysisConfig | None = None):
        self.embedder = embedder
        self.config = config or ReadabilityAnalysisConfig()

    def analyze(
        self,
        *,
        path: str,
        text: str,
        metrics: dict[str, Any],
        sections: Sequence[dict[str, Any]],
        compare_to: str | None = None,
    ) -> ReadabilityAnalysis:
        prose_sections = [_section_prose(s.get("text", "")) for s in sections]
        prose_sections = [s for s in prose_sections if s.strip()]
        section_vectors = self.embedder.embed_many(prose_sections) if prose_sections else []
        adjacent = [_cosine(a, b) for a, b in zip(section_vectors, section_vectors[1:])]
        pairwise = _pairwise_cosines(section_vectors)
        coherence = sum(adjacent) / len(adjacent) if adjacent else None
        # Redundancy is high when non-adjacent sections are semantically near-duplicates.
        redundancy_candidates = [v for i, j, v in pairwise if abs(i - j) > 0]
        redundancy = max(redundancy_candidates) if redundancy_candidates else None
        lexical_difficulty = _lexical_difficulty(metrics)
        semantic_instability = 1.0 - coherence if coherence is not None else 0.0
        ml_readability = _clamp(0.65 * lexical_difficulty + 0.35 * semantic_instability)

        rewrite_text = self.config.compare_to if self.config.compare_to is not None else compare_to
        preservation = None
        if rewrite_text is not None:
            source_body = parse_markdown(Path(path), text).body
            rewrite_body = parse_markdown(Path("rewrite.md"), rewrite_text).body
            src_vec, rew_vec = self.embedder.embed_many([source_body, rewrite_body])
            preservation = _cosine(src_vec, rew_vec)

        return ReadabilityAnalysis(
            kind=self.kind,
            backend=getattr(self.embedder, "backend", self.embedder.__class__.__name__),
            model=getattr(self.embedder, "model_name", self.embedder.__class__.__name__),
            ml_readability_score=_round_or_none(ml_readability),
            coherence_score=_round_or_none(coherence),
            semantic_redundancy_score=_round_or_none(redundancy),
            rewrite_preservation_score=_round_or_none(preservation),
            caveats=["advisory_only", "not_used_in_default_score"],
            details={
                "section_count": len(prose_sections),
                "adjacent_pairs": len(adjacent),
                "pairwise_pairs": len(pairwise),
                "path": path,
            },
        )


def merge_readability_analyses(analyses: Sequence[ReadabilityAnalysis]) -> dict[str, Any] | None:
    if not analyses:
        return None
    fields = ["ml_readability_score", "coherence_score", "semantic_redundancy_score", "rewrite_preservation_score"]
    merged: dict[str, Any] = {
        "enabled": True,
        "analyzers": [a.to_dict() for a in analyses],
        "caveats": ["advisory_only", "not_used_in_default_score"],
    }
    for field in fields:
        values = [getattr(a, field) for a in analyses if getattr(a, field) is not None]
        merged[field] = round(sum(values) / len(values), 6) if values else None
    return merged


def _section_prose(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", " ", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _pairwise_cosines(vectors: Sequence[Sequence[float]]) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for i, a in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            out.append((i, j, _cosine(a, vectors[j])))
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return _clamp(dot / (an * bn))


def _lexical_difficulty(metrics: dict[str, Any]) -> float:
    avg_sentence = float(metrics.get("avg_sentence_words", 0.0) or 0.0)
    long_ratio = float(metrics.get("long_sentence_ratio", 0.0) or 0.0)
    ari = float(metrics.get("ari_score", 0.0) or 0.0)
    sentence_component = min(avg_sentence / 35.0, 1.0)
    ari_component = min(max(ari, 0.0) / 18.0, 1.0)
    return _clamp(0.45 * sentence_component + 0.35 * long_ratio + 0.20 * ari_component)


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return []
    return [p.strip() for p in _SENTENCE_RE.split(compact) if p.strip()]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
