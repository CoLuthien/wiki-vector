from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
from typing import Any, Protocol, Sequence, cast

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:\\-]+|[가-힣]+")


class Embedder(Protocol):
    """Small embedding contract used by WikiIndex.

    Backends may be deterministic local code, SentenceTransformers, OpenVINO, or
    any later runtime. Keep the public contract text-in/list-float-out so the
    index, CLI, and MCP surfaces do not change when model runtimes are swapped.
    """

    model_name: str
    backend: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "hashing-ngram"
    model_name: str | None = None
    dimensions: int | None = None
    device: str | None = None
    batch_size: int = 8
    cache_dir: str | None = None
    max_length: int = 512

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        dimensions = _env_int("WIKI_VECTOR_EMBEDDING_DIMENSIONS")
        batch_size = _env_int("WIKI_VECTOR_EMBEDDING_BATCH_SIZE") or 8
        max_length = _env_int("WIKI_VECTOR_EMBEDDING_MAX_LENGTH") or 512
        return cls(
            backend=os.environ.get("WIKI_VECTOR_EMBEDDING_BACKEND", "hashing-ngram"),
            model_name=os.environ.get("WIKI_VECTOR_EMBEDDING_MODEL"),
            dimensions=dimensions,
            device=os.environ.get("WIKI_VECTOR_EMBEDDING_DEVICE"),
            batch_size=batch_size,
            cache_dir=os.environ.get("WIKI_VECTOR_EMBEDDING_CACHE_DIR"),
            max_length=max_length,
        )


def create_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    config = config or EmbeddingConfig.from_env()
    backend = _normalize_backend(config.backend)
    if backend in {"hashing-ngram", "hashing", "hashing-ngram-256"}:
        return HashingNgramEmbedder(dimensions=config.dimensions or 256)
    if backend in {"openvino-bge-m3", "openvino", "bge-m3-openvino"}:
        return OpenVINOBgeM3Embedder(
            model_name=config.model_name or "BAAI/bge-m3",
            device=config.device or "NPU",
            batch_size=config.batch_size,
            cache_dir=config.cache_dir,
            max_length=config.max_length,
        )
    raise ValueError(f"unknown embedding backend: {config.backend}")


def embedding_config_from_args(
    *,
    backend: str | None = None,
    model_name: str | None = None,
    dimensions: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    cache_dir: str | None = None,
    max_length: int | None = None,
) -> EmbeddingConfig:
    env = EmbeddingConfig.from_env()
    return EmbeddingConfig(
        backend=backend or env.backend,
        model_name=model_name or env.model_name,
        dimensions=dimensions if dimensions is not None else env.dimensions,
        device=device or env.device,
        batch_size=batch_size if batch_size is not None else env.batch_size,
        cache_dir=cache_dir or env.cache_dir,
        max_length=max_length if max_length is not None else env.max_length,
    )


class HashingNgramEmbedder:
    """Deterministic local dense embedder for the default hybrid backend.

    This is deliberately dependency-free and offline. It is not a replacement for
    a neural embedding model like bge-m3, but it gives LanceDB a stable dense
    vector column and preserves the API/metadata shape needed for model swaps.
    """

    backend = "hashing-ngram"

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
        return _l2_normalize(vec)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

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


class OpenVINOBgeM3Embedder:
    """bge-m3 dense embedder using OpenVINO as the model execution backend.

    This class keeps OpenVINO/Transformers imports lazy so the default offline
    hashing backend remains lightweight. For Intel NPU use:

        WIKI_VECTOR_EMBEDDING_BACKEND=openvino-bge-m3
        WIKI_VECTOR_EMBEDDING_MODEL=BAAI/bge-m3
        WIKI_VECTOR_EMBEDDING_DEVICE=NPU

    The implementation uses an OpenVINO feature-extraction model and mean-pools
    token embeddings with the attention mask, then L2-normalizes vectors for
    cosine/L2 retrieval. Future OpenVINO or sentence-transformer runtimes can
    implement the same Embedder contract without touching WikiIndex/MCP tools.
    """

    backend = "openvino-bge-m3"

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "NPU",
        batch_size: int = 8,
        cache_dir: str | None = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self.cache_dir = cache_dir
        self.max_length = max(int(max_length), 1)
        self.dimensions = 1024  # BAAI/bge-m3 hidden size; confirmed after first inference when available.
        self._tokenizer = None
        self._model = None

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in optional runtime envs
            raise RuntimeError("OpenVINOBgeM3Embedder requires transformers. Install the openvino extra/dependencies.") from exc
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised in optional runtime envs
            raise RuntimeError("OpenVINOBgeM3Embedder requires optimum-intel[openvino]. Install the openvino extra/dependencies.") from exc

        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = OVModelForFeatureExtraction.from_pretrained(
            self.model_name,
            export=True,
            device=self.device,
            compile=False,
            **kwargs,
        )
        self._model.reshape(self.batch_size, self.max_length)
        self._model.compile()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        tokenizer = cast(Any, self._tokenizer)
        model = cast(Any, self._model)
        original_len = len(texts)
        if original_len < self.batch_size:
            texts = texts + [""] * (self.batch_size - original_len)
        tokens = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = model(**tokens)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        if hidden is None:
            # Some OV wrappers return tuple-like outputs.
            hidden = outputs[0]
        mask = tokens["attention_mask"].astype("float32")
        mask = np.expand_dims(mask, axis=-1)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, a_min=1e-9, a_max=None)
        self.dimensions = int(pooled.shape[1])
        return pooled.astype("float32").tolist()[:original_len]


def _normalize_backend(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]
