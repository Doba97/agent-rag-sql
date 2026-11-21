"""Note les réponses du modèle, avec 3 puis 5 passages, sur les questions de réglage (ou de la réserve).

Lancer avec :  python -m src.eval_generation
Options :      --k 3 5   (nombres de passages à comparer)   --limit 5   (essai sur 5 questions)
               --questions reserve --limit 100   (100 questions de la réserve)

Produit :
  results/generation_<jeu>.md            tableau des scores (jeu = tune ou reserve, + _N si --limit)
  data/generation_<jeu>_k3.jsonl, ...    détail question par question, pour lire les erreurs
"""

import argparse
import sys

from src.config import LLM_MODEL, ROOT_DIR
from src.data import DATA_DIR, answer_in_text, load_passages, read_jsonl, write_jsonl
from src.evaluation import contains_answer, token_f1
from src.generation import answer_question
from src.judge import judge_answer
from src.llm import LLMError
from src.retrieval import RERANKERS, build_hybrid_retriever


def score_one(question: dict, result: dict, passage_text: dict[str, str], judge_fn=None) -> dict:
    """Ajoute à une réponse les mesures qui la concernent.

    judge_fn(question, référence, réponse) -> True / False / None ; sans juge, la colonne reste vide.
    """
    passage_has_answer = any(answer_in_text(question["answer"], passage_text[pid]) for pid in result["passage_ids"])
    judged = None
    if judge_fn is not None and not result.get("error"):
        try:
            judged = judge_fn(question["question"], question["answer"], result["answer"])
        except LLMError as error:
            print(f"   juge indisponible pour {question['id']} : {error}")
    return {
        "id": question["id"],
        "expected": question["answer"],
        **result,
        "correct": contains_answer(result["answer"], question["answer"]),
        "f1": round(token_f1(result["answer"], question["answer"]), 3),
        "judge_correct": judged,
        "passage_has_answer": passage_has_answer,
    }


def failed_result(question: str, error: Exception, passage_ids: list[str] | None = None) -> dict:
    """Réponse vide quand l'appel au modèle échoue : la question compte comme ratée, l'évaluation continue."""
    return {
        "question": question, "answer": "", "abstained": False, "passage_ids": passage_ids or [], "cited_ids": [],
        "raw": "", "total_tokens": 0, "latency_s": 0.0, "error": str(error),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    judged = [r for r in rows if r.get("judge_correct") is not None]
    return {
        "Réponse juste": sum(r["correct"] for r in rows) / n,
        "Juste selon le juge": (sum(r["judge_correct"] for r in judged) / len(judged)) if judged else float("nan"),
        "F1 mots": sum(r["f1"] for r in rows) / n,
        "Bon passage fourni": sum(r["passage_has_answer"] for r in rows) / n,
        "Refus (NOT FOUND)": sum(r["abstained"] for r in rows) / n,
        "Refus avec bon passage": sum(r["abstained"] and r["passage_has_answer"] for r in rows),
        "Juste sans bon passage": sum(r["correct"] and not r["passage_has_answer"] for r in rows),
        "Tokens moyens": sum(r["total_tokens"] for r in rows) / n,
        "Latence moyenne (s)": sum(r["latency_s"] for r in rows) / n,
        "Erreurs d'appel": sum(bool(r.get("error")) for r in rows),
    }


QUESTION_SETS = {"tune": "questions de réglage", "reserve": "questions de la réserve"}


def to_markdown(summaries: dict[int, dict], n_questions: int, set_label: str = "questions de réglage", rerank: str | None = None) -> str:
    columns = list(next(iter(summaries.values())))
    lines = [
        f"Génération : {n_questions} {set_label}, modèle {LLM_MODEL}, recherche combinée (RRF) + racinisation"
        + (f" + reranking {rerank}" if rerank else ""),
        "",
        "| Passages | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for k, summary in summaries.items():
        cells = []
        for column in columns:
            value = summary[column]
            if isinstance(value, int):
                cells.append(str(value))
            elif column.startswith(("Tokens", "Latence")):
                cells.append(f"{value:.1f}")
            else:
                cells.append(f"{value:.2f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5], help="nombres de passages à comparer")
    parser.add_argument("--questions", choices=list(QUESTION_SETS), default="tune", help="jeu de questions à utiliser")
    parser.add_argument("--limit", type=int, default=None, help="n'utiliser que les N premières questions")
    parser.add_argument("--sans-juge", action="store_true", help="ne pas appeler le juge LLM")
    parser.add_argument("--rerank", choices=list(RERANKERS), default=None, help="ajoute un reranking des 10 premiers passages")
    args = parser.parse_args()

    passages = load_passages()
    passage_text = {p["id"]: p["text"] for p in passages}
    questions = read_jsonl(DATA_DIR / f"wiki_{args.questions}.jsonl")[: args.limit]
    name = args.questions + (f"_{args.limit}" if args.limit else "") + (f"_{args.rerank}" if args.rerank else "")
    retriever = build_hybrid_retriever(passages, cache_path=DATA_DIR / "passage_vectors.npz", rerank=args.rerank)

    summaries = {}
    for k in args.k:
        rows = []
        for i, q in enumerate(questions, start=1):
            try:
                result = answer_question(q["question"], retriever, passage_text, k=k)
            except LLMError as error:
                print(f"   erreur sur {q['id']} : {error}")
                passage_ids = [pid for pid, _ in retriever.search(q["question"], k)]
                result = failed_result(q["question"], error, passage_ids)
            row = score_one(q, result, passage_text, judge_fn=None if args.sans_juge else judge_answer)
            rows.append(row)
            verdict = {True: "juste", False: "faux", None: "?"}[row["judge_correct"]]
            print(f"k={k}  {i}/{len(questions)}  [{verdict:>5}]  attendu: {q['answer'][:60]!r:<40} obtenu: {result['answer'][:60]!r}")
        write_jsonl(rows, DATA_DIR / f"generation_{name}_k{k}.jsonl")
        summaries[k] = summarize(rows)

    table = to_markdown(summaries, len(questions), QUESTION_SETS[args.questions], args.rerank)
    print("\n" + table)
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"generation_{name}.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
