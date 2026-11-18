"""Calcul des embeddings avec BAAI/bge-base-en-v1.5."""

from functools import lru_cache

import numpy as np

from src.config import EMBEDDING_MODEL, MODELS_DIR

# Pour bge-v1.5, on ajoute cette consigne devant les questions (pas devant les passages) :
# elle améliore la recherche quand la question est courte et le passage long.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model(name: str = EMBEDDING_MODEL):
    """Charge le modèle une seule fois, à la première utilisation."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, cache_folder=str(MODELS_DIR))


def embed_passages(texts: list[str], batch_size: int = 32, model=None) -> np.ndarray:
    """Calcule les vecteurs des passages, par lots.

    Les vecteurs sont normalisés : le produit scalaire est alors égal
    à la similarité cosinus. Renvoie un tableau (nombre de textes, dimension).
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = model or get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 1000,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(question: str, model=None) -> np.ndarray:
    """Calcule le vecteur d'une question, avec la consigne de bge."""
    model = model or get_model()
    vector = model.encode(QUERY_INSTRUCTION + question, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)
