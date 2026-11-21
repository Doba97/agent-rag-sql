"""Étape 4 : trois façons de retrouver les passages utiles pour une question.

Chaque moteur de recherche a la même méthode :
    search(question, k) -> liste de (id du passage, score), du meilleur au moins bon.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.data import normalize


_stemmer = None


def stem_words(words: list[str]) -> list[str]:
    """Réduit chaque mot anglais à sa racine : "died" et "dies" deviennent "die"."""
    global _stemmer
    if _stemmer is None:
        import snowballstemmer

        _stemmer = snowballstemmer.stemmer("english")
    return _stemmer.stemWords(words)


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Découpe un texte en mots, sans majuscules ni ponctuation, avec racinisation si demandé."""
    words = normalize(text).split()
    return stem_words(words) if stem else words


class BM25Retriever:
    """Recherche par mots-clés : favorise les passages qui contiennent les mots rares de la question.

    stem=True applique la racinisation à la question et aux passages, pour que
    "die" dans la question retrouve "died" dans un passage.
    """

    def __init__(self, passages: list[dict], stem: bool = False):
        from rank_bm25 import BM25Okapi

        self.stem = stem
        self.ids = [p["id"] for p in passages]
        self.bm25 = BM25Okapi([tokenize(p["text"], stem) for p in passages])

    def search(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question, self.stem))
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
        self._query_cache: dict[str, np.ndarray] = {}  # une question n'est encodée qu'une fois

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
        if question not in self._query_cache:
            self._query_cache[question] = self.embed_query(question)
        scores = self.vectors @ self._query_cache[question]
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


RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # petit et rapide
    "bge": "BAAI/bge-reranker-base",  # plus gros, souvent plus précis, environ 5 fois plus lent
}


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str):
    """Charge le reranker une seule fois, à la première utilisation."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class RerankRetriever:
    """Deuxième tri : relit chaque passage candidat AVEC la question et le renote.

    1. Le moteur de base (ex. la recherche combinée) donne `candidates` passages.
    2. Un cross-encoder note chaque couple (question, passage) : plus lent mais plus précis,
       car il lit la question et le passage ensemble au lieu de comparer deux vecteurs.
    3. On renvoie les passages dans l'ordre de ces nouvelles notes.

    Le reranking ne fait que réordonner les candidats : Hit@candidates ne peut pas changer.
    """

    def __init__(self, base, passage_text: dict[str, str], score_fn, candidates: int = 10):
        self.base = base
        self.passage_text = passage_text
        self.score_fn = score_fn  # score_fn(question, [textes]) -> liste de notes
        self.candidates = candidates
        self._cache: dict[str, list[tuple[str, float]]] = {}

    @classmethod
    def from_model(cls, base, passage_text: dict[str, str], model_name: str, candidates: int = 10):
        model = load_cross_encoder(model_name)

        def score_fn(question, texts):
            return model.predict([(question, text) for text in texts])

        return cls(base, passage_text, score_fn, candidates)

    def search(self, question: str, k: int = 10) -> list[tuple[str, float]]:
        if question not in self._cache:
            ids = [pid for pid, _ in self.base.search(question, self.candidates)]
            scores = self.score_fn(question, [self.passage_text[pid] for pid in ids])
            ranked = sorted(zip(ids, (float(x) for x in scores)), key=lambda item: item[1], reverse=True)
            self._cache[question] = ranked
        return self._cache[question][:k]


def build_hybrid_retriever(
    passages: list[dict], cache_path: Path | None = None, rerank: str | None = None, candidates: int = 10
):
    """Méthode retenue à l'étape 4 : BM25 avec racinisation + recherche dense, fusionnées par RRF.

    rerank="minilm" ou "bge" ajoute un reranking des `candidates` premiers passages.
    """
    bm25 = BM25Retriever(passages, stem=True)
    dense = DenseRetriever.from_passages(passages, cache_path=cache_path)
    hybrid = HybridRetriever([bm25, dense])
    if rerank is None:
        return hybrid
    passage_text = {p["id"]: p["text"] for p in passages}
    return RerankRetriever.from_model(hybrid, passage_text, RERANKERS[rerank], candidates)
