import pytest

from wiki_vector.embeddings import EmbeddingConfig, HashingNgramEmbedder, OpenVINOBgeM3Embedder, create_embedder


def test_openvino_bge_m3_default_max_length_is_512():
    config = EmbeddingConfig(backend="openvino-bge-m3")
    embedder = create_embedder(config)

    assert isinstance(embedder, OpenVINOBgeM3Embedder)
    assert config.max_length == 512
    assert embedder.max_length == 512


def test_create_default_hashing_embedder_from_config():
    embedder = create_embedder(EmbeddingConfig(backend="hashing-ngram", dimensions=32))

    assert isinstance(embedder, HashingNgramEmbedder)
    assert embedder.backend == "hashing-ngram"
    assert embedder.model_name == "hashing-ngram-32"
    assert len(embedder.embed("Gemma4 NPU verification")) == 32


def test_create_openvino_bge_m3_embedder_is_lazy():
    embedder = create_embedder(
        EmbeddingConfig(
            backend="openvino-bge-m3",
            model_name="BAAI/bge-m3",
            device="NPU",
            batch_size=2,
            cache_dir="/tmp/wiki-vector-cache",
            max_length=384,
        )
    )

    assert isinstance(embedder, OpenVINOBgeM3Embedder)
    assert embedder.backend == "openvino-bge-m3"
    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.device == "NPU"
    assert embedder.batch_size == 2
    assert embedder.max_length == 384
    # Model runtime is not imported/downloaded until embed/embed_many is called.
    assert embedder._tokenizer is None
    assert embedder._model is None


def test_unknown_embedding_backend_fails_fast():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        create_embedder(EmbeddingConfig(backend="does-not-exist"))
