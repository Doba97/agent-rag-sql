"""Tests de la recherche et des mesures, avec un petit corpus et de faux embeddings."""

import numpy as np
import pytest

from src.evaluation import evaluate_retriever, first_hit_rank, hit_at_k, reciprocal_rank
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, rrf_fuse, tokenize

PASSAGES = [
    {"id": "p1", "text": "Ulysses S. Grant was elected president as a Republican in 1868."},
    {"id": "p2", "text": "Kangaroos live in Australia and can hop very fast."},
    {"id": "p3", "text": "Finland has a life expectancy of 75 years for men."},
    {"id": "p4", "text": "Turtles are divided into three suborders."},
]

# Faux embeddings : chaque passage parle d'un « thème » (une dimension).
THEMES = {"president": 0, "kangaroo": 1, "finland": 2, "turtle": 3}


def fake_embed(text: str) -> np.ndarray:
    vector = np.zeros(4)
    for word, dim in THEMES.items():
        if word in text.lower():
            vector[dim] = 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def fake_embed_passages(texts):
    return np.array([fake_embed(t) for t in texts])


def test_tokenize():
    assert tokenize("Grant, elected in 1868!") == ["grant", "elected", "in", "1868"]


def test_tokenize_with_stemming_merges_word_forms():
    assert tokenize("Monroe died", stem=True) == tokenize("monroe dies", stem=True)


def test_bm25_with_stemming_matches_other_word_forms():
    passages = PASSAGES + [
        {"id": "a", "text": "Monroe died in New York in 1831."},
        {"id": "b", "text": "Monroe was the fifth president of the country."},
    ]
    question = "When did Monroe die?"
    # Sans racinisation, "die" ne correspond pas à "died" : les deux passages sur Monroe sont à égalité.
    plain = dict(BM25Retriever(passages).search(question, k=6))
    assert plain["a"] == pytest.approx(plain["b"], rel=0.2)
    # Avec racinisation, "die" retrouve "died" : le bon passage passe nettement devant.
    stemmed = dict(BM25Retriever(passages, stem=True).search(question, k=6))
    assert stemmed["a"] > 1.5 * stemmed["b"]


def test_bm25_finds_passage_with_rare_words():
    results = BM25Retriever(PASSAGES).search("When was Grant elected?", k=2)
    assert results[0][0] == "p1"
    assert len(results) == 2


def test_dense_finds_passage_by_theme():
    dense = DenseRetriever.from_passages(PASSAGES, embed_passages_fn=fake_embed_passages, embed_query_fn=fake_embed)
    assert dense.search("Which kangaroo is the fastest?", k=1)[0][0] == "p2"


def test_dense_cache_is_reused(tmp_path):
    cache = tmp_path / "vectors.npz"
    DenseRetriever.from_passages(PASSAGES, cache, fake_embed_passages, fake_embed)

    def must_not_be_called(texts):
        raise AssertionError("les vecteurs auraient dû être relus depuis le cache")

    dense = DenseRetriever.from_passages(PASSAGES, cache, must_not_be_called, fake_embed)
    assert dense.search("turtle", k=1)[0][0] == "p4"


def test_dense_cache_is_reused_for_a_subset(tmp_path):
    cache = tmp_path / "vectors.npz"
    DenseRetriever.from_passages(PASSAGES, cache, fake_embed_passages, fake_embed)

    def must_not_be_called(texts):
        raise AssertionError("les vecteurs auraient dû être relus depuis le cache")

    subset = [PASSAGES[3], PASSAGES[0]]  # ordre différent, passages retirés
    dense = DenseRetriever.from_passages(subset, cache, must_not_be_called, fake_embed)
    assert dense.ids == ["p4", "p1"]
    assert dense.search("turtle", k=1)[0][0] == "p4"
    assert dense.search("president", k=1)[0][0] == "p1"


def test_dense_encodes_each_question_once():
    calls = []

    def counting_embed(question):
        calls.append(question)
        return fake_embed(question)

    dense = DenseRetriever.from_passages(PASSAGES, embed_passages_fn=fake_embed_passages, embed_query_fn=counting_embed)
    dense.search("turtle", k=1)
    dense.search("turtle", k=3)
    assert calls == ["turtle"]


def test_dense_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        DenseRetriever(["p1", "p2"], np.zeros((3, 4)), fake_embed)


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], c=60)
    assert fused[0][0] == "b"  # 2e puis 1er : meilleur total
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_keeps_passages_found_by_one_method_only():
    fused = dict(rrf_fuse([["a"], ["b"]]))
    assert set(fused) == {"a", "b"}


def test_hybrid_returns_k_results():
    bm25 = BM25Retriever(PASSAGES)
    dense = DenseRetriever.from_passages(PASSAGES, embed_passages_fn=fake_embed_passages, embed_query_fn=fake_embed)
    results = HybridRetriever([bm25, dense], depth=4).search("Grant president 1868", k=2)
    assert len(results) == 2 and results[0][0] == "p1"


def test_first_hit_rank_and_metrics():
    texts = ["nothing here", "elected in 1868", "also 1868"]
    rank = first_hit_rank("1868", texts)
    assert rank == 2
    assert hit_at_k(rank, 1) == 0.0 and hit_at_k(rank, 3) == 1.0
    assert reciprocal_rank(rank) == 0.5
    assert reciprocal_rank(None) == 0.0 and hit_at_k(None, 10) == 0.0


def test_evaluate_retriever_averages_over_questions():
    questions = [
        {"id": "q1", "question": "When was Grant elected president?", "answer": "1868"},
        {"id": "q2", "question": "How many suborders do turtles have?", "answer": "three"},
    ]
    texts = {p["id"]: p["text"] for p in PASSAGES}
    result = evaluate_retriever(BM25Retriever(PASSAGES), questions, texts, ks=(1, 3))
    assert result["scores"]["Hit@1"] == 1.0
    assert result["scores"]["MRR@3"] == 1.0
    assert len(result["details"]) == 2


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, question, k=10):
        return [(pid, 1.0 / (i + 1)) for i, pid in enumerate(self.ids[:k])]


def test_rerank_reorders_candidates_by_new_scores():
    texts = {p["id"]: p["text"] for p in PASSAGES}

    def score_fn(question, candidate_texts):
        # Le faux reranker préfère les passages qui contiennent "1868".
        return [1.0 if "1868" in t else 0.0 for t in candidate_texts]

    reranker = RerankRetriever(FixedRetriever(["p2", "p3", "p1", "p4"]), texts, score_fn, candidates=4)
    assert reranker.search("When was Grant elected?", k=1)[0][0] == "p1"


def test_rerank_only_reads_the_candidates():
    texts = {p["id"]: p["text"] for p in PASSAGES}
    seen = []

    def score_fn(question, candidate_texts):
        seen.append(len(candidate_texts))
        return [0.0] * len(candidate_texts)

    reranker = RerankRetriever(FixedRetriever(["p1", "p2", "p3", "p4"]), texts, score_fn, candidates=2)
    results = reranker.search("question", k=10)
    assert seen == [2] and len(results) == 2


def test_rerank_scores_each_question_once():
    texts = {p["id"]: p["text"] for p in PASSAGES}
    calls = []

    def score_fn(question, candidate_texts):
        calls.append(question)
        return [0.0] * len(candidate_texts)

    reranker = RerankRetriever(FixedRetriever(["p1", "p2"]), texts, score_fn, candidates=2)
    reranker.search("q", k=1)
    reranker.search("q", k=2)
    assert calls == ["q"]
