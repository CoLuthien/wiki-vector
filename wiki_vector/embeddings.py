from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+|[가-힣]+")


class HashingNgramEmbedder:
    """Deterministic local dense embedder for the first hybrid backend.

    This is deliberately dependency-free and offline. It is not a replacement for
    a neural embedding model like bge-m3, but it gives LanceDB a stable dense
    vector column and preserves the API/metadata shape needed for a later model
    swap.
    """

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        self.model_name = f"hashing-ngram-{dimensions}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            idx = value % self.dimensions
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def _features(self, text: str) -> list[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        features: list[str] = []
        for token in tokens:
            features.append("tok:" + token)
            padded = f"<{token}>"
            if len(padded) >= 3:
                features.extend("tri:" + padded[i : i + 3] for i in range(len(padded) - 2))
        features.extend("bi:" + " ".join(pair) for pair in zip(tokens, tokens[1:]))
        return features
