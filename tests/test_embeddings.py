"""Tests de src/embeddings.py avec un faux modèle : rien à télécharger."""

import numpy as np

from src.embeddings import QUERY_INSTRUCTION, embed_passages, embed_query


class FakeModel:
    """Imite SentenceTransformer.encode et garde une trace des appels."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        self.calls.append({"texts": texts, "normalize": normalize_embeddings, **kwargs})
        if isinstance(texts, str):
            return np.ones(self.dim) / np.sqrt(self.dim)
        return np.ones((len(texts), self.dim)) / np.sqrt(self.dim)


def test_embed_passages_shape_and_type():
    vectors = embed_passages(["a", "b", "c"], model=FakeModel())
    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32


def test_embed_passages_asks_for_normalized_vectors():
    model = FakeModel()
    embed_passages(["a"], model=model)
    assert model.calls[0]["normalize"] is True


def test_embed_passages_has_no_query_instruction():
    model = FakeModel()
    embed_passages(["un passage"], model=model)
    assert model.calls[0]["texts"] == ["un passage"]


def test_embed_passages_empty_list_does_not_load_model():
    vectors = embed_passages([], model=None)
    assert vectors.shape[0] == 0


def test_embed_query_adds_instruction():
    model = FakeModel()
    vector = embed_query("Who wrote Hamlet?", model=model)
    assert model.calls[0]["texts"] == QUERY_INSTRUCTION + "Who wrote Hamlet?"
    assert vector.shape == (4,)
