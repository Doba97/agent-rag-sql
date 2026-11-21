"""Mesures pour noter la recherche de passages.

Un passage est considéré comme « bon » s'il contient la réponse attendue
(mots entiers, sans tenir compte des majuscules ni de la ponctuation).
"""

from collections import Counter

from src.data import answer_in_text, normalize


def first_hit_rank(answer: str, retrieved_texts: list[str]) -> int | None:
    """Rang (à partir de 1) du premier passage qui contient la réponse, ou None."""
    for rank, text in enumerate(retrieved_texts, start=1):
        if answer_in_text(answer, text):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    """1 si un bon passage est dans les k premiers, sinon 0."""
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """1 si le bon passage est 1er, 1/2 s'il est 2e, ..., 0 s'il n'est pas trouvé."""
    return 0.0 if rank is None else 1.0 / rank


def evaluate_retriever(retriever, questions: list[dict], passage_text: dict[str, str], ks=(1, 3, 5, 10)) -> dict:
    """Note un moteur de recherche sur une liste de questions.

    Renvoie les moyennes (Hit@k pour chaque k, et MRR calculé sur les max(ks)
    premiers passages) et le détail par question, pour lire les échecs.
    """
    depth = max(ks)
    details = []
    for q in questions:
        retrieved = [pid for pid, _ in retriever.search(q["question"], depth)]
        rank = first_hit_rank(q["answer"], [passage_text[pid] for pid in retrieved])
        details.append({"id": q["id"], "question": q["question"], "answer": q["answer"], "rank": rank, "retrieved": retrieved})

    n = len(details)
    scores = {f"Hit@{k}": sum(hit_at_k(d["rank"], k) for d in details) / n for k in ks}
    scores[f"MRR@{depth}"] = sum(reciprocal_rank(d["rank"]) for d in details) / n
    return {"scores": scores, "details": details}


ARTICLES = {"a", "an", "the"}


def answer_tokens(text: str) -> list[str]:
    """Mots d'une réponse, sans majuscules, ponctuation ni articles anglais."""
    return [w for w in normalize(text).split() if w not in ARTICLES]


def contains_answer(prediction: str, gold: str) -> bool:
    """Vrai si tous les mots de la réponse attendue apparaissent, dans l'ordre et à la suite, dans la réponse donnée.

    Exemple : "Federalist Party" contient "The Federalist Party" (les articles sont ignorés).
    """
    pred, ref = answer_tokens(prediction), answer_tokens(gold)
    if not ref:
        return False
    return any(pred[i : i + len(ref)] == ref for i in range(len(pred) - len(ref) + 1))


def token_f1(prediction: str, gold: str) -> float:
    """F1 sur les mots (mesure du jeu SQuAD) : récompense une réponse partiellement juste."""
    pred, ref = answer_tokens(prediction), answer_tokens(gold)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)
