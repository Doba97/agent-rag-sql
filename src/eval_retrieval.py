"""Compare BM25, embeddings et RRF sur les questions de réglage.

Lancer avec :  python -m src.eval_retrieval
Pour retirer les passages de moins de 5 mots :  python -m src.eval_retrieval --min-mots 5
(La première fois, le calcul des vecteurs des 3 200 passages peut être long sur un ordinateur portable ;
ensuite ils sont relus depuis data/passage_vectors.npz.)

Produit :
  results/retrieval_tune.md                tableau des scores (à mettre dans le README)
  data/retrieval_failures_tune.jsonl       questions ratées, pour les lire une par une
"""

import argparse
import sys

from src.config import EMBEDDING_MODEL, ROOT_DIR
from src.data import DATA_DIR, read_jsonl, write_jsonl
from src.evaluation import evaluate_retriever
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

KS = (1, 3, 5, 10)
SHORT_PASSAGE_WORDS = 5


def describe_short_passages(passages: list[dict]) -> None:
    short = [p for p in passages if len(p["text"].split()) < SHORT_PASSAGE_WORDS]
    print(f"Passages de moins de {SHORT_PASSAGE_WORDS} mots : {len(short)} sur {len(passages)}")
    for p in short[:10]:
        print(f"   [{p['id']}] {p['text']!r}")


def to_markdown(results: dict[str, dict], n_questions: int, n_passages: int, min_words: int) -> str:
    columns = list(next(iter(results.values()))["scores"])
    kept = "tous les passages" if min_words <= 1 else f"passages d'au moins {min_words} mots"
    lines = [
        f"Recherche de passages : {n_questions} questions de réglage, {n_passages} passages ({kept}), "
        f"modèle d'embeddings {EMBEDDING_MODEL}",
        "",
        "| Méthode | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name, result in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{result['scores'][c]:.2f}" for c in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-mots", type=int, default=1, help="retire les passages plus courts que ce nombre de mots")
    min_words = parser.parse_args().min_mots

    passages = read_jsonl(DATA_DIR / "wiki_passages.jsonl")
    describe_short_passages(passages)
    passages = [p for p in passages if len(p["text"].split()) >= min_words]
    print(f"Passages utilisés : {len(passages)}")
    questions = read_jsonl(DATA_DIR / "wiki_tune.jsonl")
    passage_text = {p["id"]: p["text"] for p in passages}

    print("\nPréparation des moteurs de recherche...")
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever.from_passages(passages, cache_path=DATA_DIR / "passage_vectors.npz")
    retrievers = {
        "BM25 (mots-clés)": bm25,
        "Embeddings (sens)": dense,
        "Combinée (RRF)": HybridRetriever([bm25, dense]),
    }

    results = {name: evaluate_retriever(r, questions, passage_text, KS) for name, r in retrievers.items()}

    table = to_markdown(results, len(questions), len(passages), min_words)
    suffix = "" if min_words <= 1 else f"_min{min_words}"
    print("\n" + table)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"retrieval_tune{suffix}.md").write_text(table, encoding="utf-8")

    # Échecs : questions où la réponse n'est pas dans les 5 premiers passages, avec ce que chaque méthode a trouvé.
    failures = []
    for i, q in enumerate(questions):
        ranks = {name: res["details"][i]["rank"] for name, res in results.items()}
        if any(rank is None or rank > 5 for rank in ranks.values()):
            top = results["Combinée (RRF)"]["details"][i]["retrieved"][:3]
            failures.append({
                "id": q["id"], "question": q["question"], "answer": q["answer"], "ranks": ranks,
                "rrf_top3": [passage_text[pid][:300] for pid in top],
            })
    failures_path = DATA_DIR / f"retrieval_failures_tune{suffix}.jsonl"
    write_jsonl(failures, failures_path)
    print(f"{len(failures)} questions ratées par au moins une méthode (à 5 passages) : data/{failures_path.name}")


if __name__ == "__main__":
    main()
