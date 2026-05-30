from wiki_vector.readability import EmbeddingSemanticStructureAnalyzer, ReadabilityAnalysisConfig, TransformersReadabilityModelAnalyzer
from wiki_vector.verbosity import analyze_verbosity


class TinySemanticEmbedder:
    backend = "tiny-semantic"
    model_name = "tiny-semantic-test"
    dimensions = 3

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([
                1.0 if "cache" in lower or "reuse" in lower else 0.0,
                1.0 if "compile" in lower or "runtime" in lower else 0.0,
                1.0 if "fruit" in lower or "banana" in lower else 0.0,
            ])
        return vectors


def page(body: str) -> str:
    return f"""---
title: Semantic Page
type: concept
tags: [wiki]
---

{body}
"""


def test_embedding_semantic_structure_reports_proxy_scores_without_claiming_readability():
    text = page("""# Cache Reuse

## Cache operator

Cache reuse keeps compiled operators reusable.

## Same cache topic

The runtime can reuse cached operators for the same compiled cache path.

## Unrelated fruit

Banana fruit salad is unrelated to the runtime cache discussion.
""")
    analyzer = EmbeddingSemanticStructureAnalyzer(TinySemanticEmbedder())

    result = analyze_verbosity("concepts/cache.md", text, readability_analyzers=[analyzer])
    data = result.to_dict()

    assert data["semantic"]["enabled"] is True
    assert data["semantic"]["analyzers"][0]["kind"] == "embedding-semantic-structure"
    assert data["semantic"]["analyzers"][0]["backend"] == "tiny-semantic"
    assert data["semantic"]["analyzers"][0]["not_readability_model"] is True
    assert data["semantic"]["analyzers"][0]["model_role"] == "semantic_similarity_proxy"
    assert data["semantic"].get("ml_readability_score") is None
    assert "semantic_structure_score" in data["semantic"]
    assert 0.0 <= data["semantic"]["coherence_score"] <= 1.0
    assert data["semantic"]["semantic_redundancy_score"] > 0.5
    assert data["semantic"]["caveats"] == ["advisory_only", "not_used_in_default_score"]
    assert "score" in data and "reasons" in data and "sections" in data


def test_embedding_semantic_structure_can_compare_rewrite_preservation():
    source = page("""# Runtime

## Cache

Cache reuse preserves compiled operator state.
""")
    rewrite = page("""# Runtime

## Cache

Compiled operator cache reuse is preserved.
""")
    analyzer = EmbeddingSemanticStructureAnalyzer(TinySemanticEmbedder(), ReadabilityAnalysisConfig(compare_to=rewrite))

    result = analyze_verbosity("concepts/runtime.md", source, readability_analyzers=[analyzer])

    semantic = result.to_dict()["semantic"]
    assert semantic["rewrite_preservation_score"] is not None
    assert semantic["rewrite_preservation_score"] > 0.8


def test_transformers_readability_model_reports_separate_model_block():
    text = page("""# Runtime Note

This concise note explains the runtime cache in simple direct language.
""")

    def fake_predictor(inputs, **kwargs):
        assert isinstance(inputs, list)
        return [[{"label": "easy", "score": 0.82}, {"label": "hard", "score": 0.18}]]

    analyzer = TransformersReadabilityModelAnalyzer(
        model_name="fake/readability-model",
        predictor=fake_predictor,
        task="readability-classification",
    )

    result = analyze_verbosity("concepts/runtime.md", text, readability_analyzers=[analyzer])
    data = result.to_dict()

    assert data["semantic"] is None
    assert data["readability_model"]["enabled"] is True
    assert data["readability_model"]["analyzers"][0]["kind"] == "readability-model"
    assert data["readability_model"]["analyzers"][0]["model"] == "fake/readability-model"
    assert data["readability_model"]["analyzers"][0]["predicted_label"] == "easy"
    assert data["readability_model"]["analyzers"][0]["predicted_score"] == 0.82
    assert data["readability_model"]["caveats"] == ["advisory_only", "not_used_in_default_score"]


def test_transformers_readability_model_requests_token_truncation_for_long_pages():
    text = page("# Long\n\n" + "technical runtime cache details " * 2000)
    seen = {}

    def fake_predictor(inputs, **kwargs):
        seen.update(kwargs)
        return [[{"label": "LABEL_0", "score": 0.5}]]

    analyzer = TransformersReadabilityModelAnalyzer("fake/readability-model", predictor=fake_predictor)

    analyze_verbosity("concepts/long.md", text, readability_analyzers=[analyzer])

    assert seen["truncation"] is True
    assert seen["max_length"] == 512
