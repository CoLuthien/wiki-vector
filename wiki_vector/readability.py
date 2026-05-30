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
    group: str = "semantic"
    model_role: str | None = None
    not_readability_model: bool = False
    ml_readability_score: float | None = None
    semantic_structure_score: float | None = None
    predicted_label: str | None = None
    predicted_score: float | None = None
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


class EmbeddingSemanticStructureAnalyzer:
    """Embedding-backed advisory semantic-structure analyzer.

    This is not a readability model. With a neural `Embedder` such as OpenVINO
    bge-m3, it provides semantic similarity proxy signals: section coherence,
    semantic redundancy, and original-vs-rewrite preservation. With the default
    hashing embedder it exercises the same abstraction deterministically, keeping
    tests/MCP startup dependency-light.
    """

    kind = "embedding-semantic-structure"

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
        semantic_instability = 1.0 - coherence if coherence is not None else 0.0
        redundancy_pressure = redundancy if redundancy is not None else 0.0
        semantic_structure = _clamp(0.55 * semantic_instability + 0.45 * redundancy_pressure)

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
            group="semantic",
            model_role="semantic_similarity_proxy",
            not_readability_model=True,
            ml_readability_score=None,
            semantic_structure_score=_round_or_none(semantic_structure),
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


# Backward-compatible alias for one release cycle. Prefer
# EmbeddingSemanticStructureAnalyzer: generic embedding models are not
# readability models.
EmbeddingSemanticReadabilityAnalyzer = EmbeddingSemanticStructureAnalyzer


class TransformersReadabilityModelAnalyzer:
    """Opt-in analyzer for models explicitly trained for readability/text complexity.

    This is intentionally separate from embedding semantic-structure proxies. It
    loads Transformers lazily only when used, so MCP startup stays lightweight.
    Tests can inject `predictor` to avoid downloading model artifacts.
    """

    kind = "readability-model"

    def __init__(self, model_name: str, *, predictor: Any | None = None, task: str = "text-classification", max_chars: int = 4000, max_tokens: int = 512):
        self.model_name = model_name
        self.predictor = predictor
        self.task = task
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def analyze(
        self,
        *,
        path: str,
        text: str,
        metrics: dict[str, Any],
        sections: Sequence[dict[str, Any]],
        compare_to: str | None = None,
    ) -> ReadabilityAnalysis:
        body = parse_markdown(Path(path), text).body
        sample = re.sub(r"\s+", " ", body).strip()[: self.max_chars]
        predictor = self.predictor or self._load_predictor()
        raw = _predict_with_truncation(predictor, sample, self.max_tokens)
        label, score = _extract_top_prediction(raw)
        return ReadabilityAnalysis(
            kind=self.kind,
            group="readability_model",
            backend="transformers",
            model=self.model_name,
            model_role=self.task,
            not_readability_model=False,
            predicted_label=label,
            predicted_score=_round_or_none(score),
            caveats=["advisory_only", "not_used_in_default_score"],
            details={
                "path": path,
                "input_chars": len(sample),
                "raw_prediction": raw,
            },
        )

    def _load_predictor(self):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("TransformersReadabilityModelAnalyzer requires transformers; install the openvino extra or add transformers") from exc
        return pipeline(self.task, model=self.model_name, top_k=None)


def merge_readability_analyses(analyses: Sequence[ReadabilityAnalysis]) -> dict[str, dict[str, Any]]:
    if not analyses:
        return {}
    fields = [
        "ml_readability_score",
        "semantic_structure_score",
        "coherence_score",
        "semantic_redundancy_score",
        "rewrite_preservation_score",
        "predicted_score",
    ]
    grouped: dict[str, list[ReadabilityAnalysis]] = {}
    for analysis in analyses:
        grouped.setdefault(analysis.group, []).append(analysis)
    out: dict[str, dict[str, Any]] = {}
    for group, group_analyses in grouped.items():
        merged: dict[str, Any] = {
            "enabled": True,
            "analyzers": [a.to_dict() for a in group_analyses],
            "caveats": ["advisory_only", "not_used_in_default_score"],
        }
        for field in fields:
            values = [getattr(a, field) for a in group_analyses if getattr(a, field) is not None]
            merged[field] = round(sum(values) / len(values), 6) if values else None
        labels = [a.predicted_label for a in group_analyses if a.predicted_label is not None]
        if labels:
            merged["predicted_label"] = labels[0]
        out[group] = merged
    return out


def _extract_top_prediction(raw: Any) -> tuple[str | None, float | None]:
    item = raw
    while isinstance(item, list) and item:
        item = item[0]
    if isinstance(item, dict):
        return item.get("label"), float(item["score"]) if item.get("score") is not None else None
    return None, None


def _predict_with_truncation(predictor: Any, sample: str, max_tokens: int) -> Any:
    try:
        return predictor([sample], truncation=True, max_length=max_tokens)
    except TypeError:
        # Test doubles or non-Transformers predictors may not accept tokenizer
        # kwargs; keep them usable without weakening real pipeline safety.
        return predictor([sample])


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
