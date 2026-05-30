from wiki_vector.readability import EmbeddingSemanticReadabilityAnalyzer, ReadabilityAnalysisConfig
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


def test_embedding_semantic_readability_reports_advisory_scores_without_replacing_deterministic_result():
    text = page("""# Cache Reuse

## Cache operator

Cache reuse keeps compiled operators reusable.

## Same cache topic

The runtime can reuse cached operators for the same compiled cache path.

## Unrelated fruit

Banana fruit salad is unrelated to the runtime cache discussion.
""")
    analyzer = EmbeddingSemanticReadabilityAnalyzer(TinySemanticEmbedder())

    result = analyze_verbosity("concepts/cache.md", text, readability_analyzers=[analyzer])
    data = result.to_dict()

    assert data["semantic"]["enabled"] is True
    assert data["semantic"]["analyzers"][0]["kind"] == "embedding-semantic"
    assert data["semantic"]["analyzers"][0]["backend"] == "tiny-semantic"
    assert 0.0 <= data["semantic"]["coherence_score"] <= 1.0
    assert data["semantic"]["semantic_redundancy_score"] > 0.5
    assert data["semantic"]["caveats"] == ["advisory_only", "not_used_in_default_score"]
    assert "score" in data and "reasons" in data and "sections" in data


def test_embedding_semantic_readability_can_compare_rewrite_preservation():
    source = page("""# Runtime

## Cache

Cache reuse preserves compiled operator state.
""")
    rewrite = page("""# Runtime

## Cache

Compiled operator cache reuse is preserved.
""")
    analyzer = EmbeddingSemanticReadabilityAnalyzer(TinySemanticEmbedder(), ReadabilityAnalysisConfig(compare_to=rewrite))

    result = analyze_verbosity("concepts/runtime.md", source, readability_analyzers=[analyzer])

    semantic = result.to_dict()["semantic"]
    assert semantic["rewrite_preservation_score"] is not None
    assert semantic["rewrite_preservation_score"] > 0.8
