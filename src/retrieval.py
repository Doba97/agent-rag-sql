"""Étape 4 : trois façons de retrouver les passages utiles pour une question.

Chaque moteur de recherche a la même méthode :
    search(question, k) -> liste de (id du passage, score), du meilleur au moins bon.
"""

from pathlib import Path

import numpy as np

from src.data import normalize


def tokenize(text: str) -> list[str]:
    """Découpe un texte en mots, sans majuscules ni ponctuation."""
    return normalize(text).split()


class BM25Retriever:
    """Recherche par mots-clés : favorise les passages qui contiennent les mots rares de la question."""

    def __init__(self, passages: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = [p["id"] for p in passages]
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in passages])

    def search(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        best = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in best]


class DenseRetriever:
    """Recherche par le sens : compare le vecteur de la question à celui de chaque passage.

    Les vecteurs sont normalisés, donc le produit scalaire est la similarité cosinus.
    Avec 3 200 passages, un simple tableau numpy suffit : la recherche est exacte et rapide.
    """

    def __init__(self, ids: list[str], vectors: np.ndarray, embed_query_fn):
        if len(ids) != len(vectors):
            raise ValueError("Il faut autant de vecteurs que de passages")
        self.ids = ids
        self.vectors = vectors
        self.embed_query = embed_query_fn

    @classmethod
    def from_passages(cls, passages: list[dict], cache_path: Path | None = None, embed_passages_fn=None, embed_query_fn=None):
        """Calcule les vecteurs des passages, ou les relit s'ils ont déjà été calculés."""
        from src import embeddings

        embed_passages_fn = embed_passages_fn or embeddings.embed_passages
        embed_query_fn = embed_query_fn or embeddings.embed_query
        ids = [p["id"] for p in passages]

        if cache_path and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            position = {pid: i for i, pid in enumerate(cached["ids"].tolist())}
            # On réutilise le cache si tous les passages demandés y sont déjà,
            # même si on n'en garde qu'une partie (ex. après avoir retiré les passages courts).
            if all(pid in position for pid in ids):
                vectors = cached["vectors"][[position[pid] for pid in ids]]
                return cls(ids, vectors, embed_query_fn)

        vectors = embed_passages_fn([p["text"] for p in passages])
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, ids=np.array(ids), vectors=vectors)
        return cls(ids, vectors, embed_query_fn)

    def search(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.vectors @ self.embed_query(question)
        best = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in best]


def rrf_fuse(rankings: list[list[str]], c: int = 60) -> list[tuple[str, float]]:
    """Fusionne plusieurs classements (Reciprocal Rank Fusion).

    Chaque passage reçoit 1 / (c + rang) dans chaque classement où il apparaît.
    c = 60 est la valeur de l'article d'origine (Cormack et al., 2009) : elle évite
    qu'un passage classé 1er par une seule méthode écrase tout le reste.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, passage_id in enumerate(ranking, start=1):
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Combine plusieurs moteurs avec RRF, à partir de leurs `depth` premiers résultats."""

    def __init__(self, retrievers: list, depth: int = 50, c: int = 60):
        self.retrievers = retrievers
        self.depth = depth
        self.c = c

    def search(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        rankings = [[pid for pid, _ in r.search(question, self.depth)] for r in self.retrievers]
        return rrf_fuse(rankings, self.c)[:k]
