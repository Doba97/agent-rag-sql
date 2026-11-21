"""Compare BM25, embeddings et RRF sur les questions de réglage (ou de la réserve).

Sur les questions de la réserve (plus nombreuses, donc résultats plus fiables) :
    python -m src.eval_retrieval --questions reserve

Avec reranking des 10 premiers passages de la recherche combinée :
    python -m src.eval_retrieval --questions reserve --rerank minilm
    python -m src.eval_retrieval --questions reserve --rerank minilm bge --candidats 10

Lancer avec :  python -m src.eval_retrieval
Par défaut, les passages de moins de 5 mots sont retirés. Pour tous les garder :
    python -m src.eval_retrieval --min-mots 1
(La première fois, le calcul des vecteurs des 3 200 passages peut être long sur un ordinateur portable ;
ensuite ils sont relus depuis data/passage_vectors.npz.)

Produit :
  results/retrieval_tune.md                tableau des scores (à mettre dans le README)
  data/retrieval_failures_tune.jsonl       questions ratées, pour les lire une par une
"""

import argparse
import sys
import time

from src.config import EMBEDDING_MODEL, ROOT_DIR
from src.data import DATA_DIR, read_jsonl, write_jsonl
from src.evaluation import evaluate_retriever
from src.retrieval import RERANKERS, BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

KS = (1, 3, 5, 10)
SHORT_PASSAGE_WORDS = 5


def describe_short_passages(passages: list[dict]) -> None:
    short = [p for p in passages if len(p["text"].split()) < SHORT_PASSAGE_WORDS]
    print(f"Passages de moins de {SHORT_PASSAGE_WORDS} mots : {len(short)} sur {len(passages)}")
    for p in short[:10]:
        print(f"   [{p['id']}] {p['text']!r}")


QUESTION_SETS = {"tune": "questions de réglage", "reserve": "questions de la réserve"}


def to_markdown(results: dict[str, dict], n_questions: int, n_passages: int, min_words: int, set_label: str = "questions de réglage") -> str:
    columns = list(next(iter(results.values()))["scores"])
    kept = "tous les passages" if min_words <= 1 else f"passages d'au moins {min_words} mots"
    lines = [
        f"Recherche de passages : {n_questions} {set_label}, {n_passages} passages ({kept}), "
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
    parser.add_argument("--min-mots", type=int, default=5, help="retire les passages plus courts que ce nombre de mots")
    parser.add_argument("--questions", choices=list(QUESTION_SETS), default="tune", help="jeu de questions à utiliser")
    parser.add_argument("--rerank", nargs="*", choices=list(RERANKERS), default=[], help="rerankers à tester")
    parser.add_argument("--candidats", type=int, default=10, help="nombre de passages relus par le reranker")
    parser.add_argument("--limit", type=int, default=None, help="n'utiliser que les N premières questions (essai)")
    args = parser.parse_args()
    min_words, question_set = args.min_mots, args.questions

    passages = read_jsonl(DATA_DIR / "wiki_passages.jsonl")
    describe_short_passages(passages)
    passages = [p for p in passages if len(p["text"].split()) >= min_words]
    print(f"Passages utilisés : {len(passages)}")
    questions = read_jsonl(DATA_DIR / f"wiki_{question_set}.jsonl")[: args.limit]
    passage_text = {p["id"]: p["text"] for p in passages}

    print("\nPréparation des moteurs de recherche...")
    bm25 = BM25Retriever(passages)
    bm25_stem = BM25Retriever(passages, stem=True)
    dense = DenseRetriever.from_passages(passages, cache_path=DATA_DIR / "passage_vectors.npz")
    best = HybridRetriever([bm25_stem, dense])
    retrievers = {
        "BM25 (mots-clés)": bm25,
        "BM25 + racinisation": bm25_stem,
        "Embeddings (sens)": dense,
        "Combinée (RRF)": HybridRetriever([bm25, dense]),
        "Combinée (RRF) + racinisation": best,
    }
    for key in args.rerank:
        name = f"Combinée + racinisation + reranking {key} ({args.candidats} candidats)"
        retrievers[name] = RerankRetriever.from_model(best, passage_text, RERANKERS[key], args.candidats)

    results = {}
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        results[name] = evaluate_retriever(retriever, questions, passage_text, KS)
        print(f"{name} : {time.perf_counter() - start:.0f} s")

    table = to_markdown(results, len(questions), len(passages), min_words, QUESTION_SETS[question_set])
    suffix = ("" if min_words == 5 else f"_min{min_words}") + (f"_{args.limit}" if args.limit else "")
    suffix += "".join(f"_{key}" for key in args.rerank)
    print("\n" + table)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"retrieval_{question_set}{suffix}.md").write_text(table, encoding="utf-8")

    # Échecs : questions où la réponse n'est pas dans les 5 premiers passages, avec ce que chaque méthode a trouvé.
    failures = []
    for i, q in enumerate(questions):
        ranks = {name: res["details"][i]["rank"] for name, res in results.items()}
        if any(rank is None or rank > 5 for rank in ranks.values()):
            top = results["Combinée (RRF) + racinisation"]["details"][i]["retrieved"][:3]
            failures.append({
                "id": q["id"], "question": q["question"], "answer": q["answer"], "ranks": ranks,
                "rrf_top3": [passage_text[pid][:300] for pid in top],
            })
    failures_path = DATA_DIR / f"retrieval_failures_{question_set}{suffix}.jsonl"
    write_jsonl(failures, failures_path)
    print(f"{len(failures)} questions ratées par au moins une méthode (à 5 passages) : data/{failures_path.name}")


if __name__ == "__main__":
    main()
