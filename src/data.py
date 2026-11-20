"""Étape 3 : télécharger les données et les séparer en questions de réglage et de test.

Lancer avec :  python -m src.data

Fichiers produits dans data/ :
  wiki_passages.jsonl     les passages Wikipedia (corpus de la recherche)
  wiki_tune.jsonl         questions sur les documents, pour régler le système
  wiki_test.jsonl         questions sur les documents, à n'utiliser qu'à la fin
  wiki_reserve.jsonl      questions sur les documents non tirées (réglage supplémentaire de la recherche)
  spider_tune.jsonl       questions SQL (Spider), pour régler le système
  spider_test.jsonl       questions SQL (Spider), à n'utiliser qu'à la fin
  router_test.jsonl       les deux jeux de test mélangés, étiquetés "documents" ou "sql"
  spider/database/<db_id>/<db_id>.sqlite   les bases SQLite utilisées
"""

import json
import random
import re
import sqlite3
from pathlib import Path

from src.config import DATA_DIR

SEED = 42
N_TUNE = 30
N_TEST = 100

WIKI_REPO = "rag-datasets/rag-mini-wikipedia"
SPIDER_QUESTIONS_REPO = "xlangai/spider"
SPIDER_DATABASES_REPO = "prem-research/spider"
SPIDER_DIR = DATA_DIR / "spider"


# ---------------------------------------------------------------------------
# Fonctions simples, testées sans réseau
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Répare les accents mal encodés ("SÃ¶derby" devient "Söderby") et retire les espaces en trop."""
    from ftfy import fix_text

    return fix_text(str(text)).strip()


def normalize(text: str) -> str:
    """Minuscules, ponctuation retirée, espaces réduits à un seul."""
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def answer_in_text(answer: str, text: str) -> bool:
    """Vrai si la réponse apparaît comme suite de mots entiers dans le texte.

    Exemple : "no" n'est pas trouvé dans "nothing".
    """
    answer = normalize(answer)
    return bool(answer) and f" {answer} " in f" {normalize(text)} "


def is_usable_doc_question(answer: str, normalized_passages: list[str]) -> bool:
    """Garde une question si sa réponse n'est pas oui/non et figure dans au moins un passage.

    Sans cette condition, on ne pourrait pas vérifier automatiquement si la
    recherche a trouvé le bon passage (mesure Hit@5 : la réponse figure-t-elle
    dans un des 5 passages trouvés ?).
    """
    answer = normalize(answer)
    if not answer or answer in {"yes", "no"}:
        return False
    target = f" {answer} "
    return any(target in f" {passage} " for passage in normalized_passages)


def split_tune_test(items: list, n_tune: int = N_TUNE, n_test: int = N_TEST, seed: int = SEED):
    """Tire au hasard (graine fixe) deux ensembles disjoints : réglage et test."""
    if len(items) < n_tune + n_test:
        raise ValueError(f"{len(items)} éléments disponibles, il en faut {n_tune + n_test}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_tune], shuffled[n_tune : n_tune + n_test]


def remaining(items: list[dict], *used: list[dict]) -> list[dict]:
    """Éléments qui ne figurent dans aucun des ensembles déjà tirés (comparaison par id)."""
    used_ids = {row["id"] for group in used for row in group}
    return [row for row in items if row["id"] not in used_ids]


def build_router_set(doc_questions: list[dict], sql_questions: list[dict], seed: int = SEED) -> list[dict]:
    """Mélange les questions des deux jeux, chacune avec l'outil attendu."""
    rows = [{"id": q["id"], "question": q["question"], "expected_tool": "documents"} for q in doc_questions]
    rows += [
        {"id": q["id"], "question": q["question"], "expected_tool": "sql", "db_id": q["db_id"]}
        for q in sql_questions
    ]
    random.Random(seed).shuffle(rows)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sqlite_path(db_id: str, root: Path = SPIDER_DIR) -> Path:
    return root / "database" / db_id / f"{db_id}.sqlite"


def run_query_readonly(db_path: Path, sql: str) -> list[tuple]:
    """Exécute une requête sur la base ouverte en lecture seule."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Certaines bases Spider contiennent du texte mal encodé : on remplace les caractères illisibles.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _check_columns(dataset, expected: set[str], name: str) -> None:
    missing = expected - set(dataset.column_names)
    if missing:
        raise ValueError(f"{name} : colonnes manquantes {missing}, colonnes présentes {dataset.column_names}")


# ---------------------------------------------------------------------------
# Téléchargements
# ---------------------------------------------------------------------------

def prepare_documents() -> dict:
    from datasets import load_dataset

    corpus = load_dataset(WIKI_REPO, "text-corpus", split="passages")
    qa = load_dataset(WIKI_REPO, "question-answer", split="test")
    _check_columns(corpus, {"passage", "id"}, "corpus")
    _check_columns(qa, {"question", "answer", "id"}, "questions")

    passages = [{"id": f"wiki-p{row['id']}", "text": clean_text(row["passage"])} for row in corpus]
    write_jsonl(passages, DATA_DIR / "wiki_passages.jsonl")

    normalized = [normalize(p["text"]) for p in passages]
    seen, questions = set(), []
    for row in qa:
        question, answer = clean_text(row["question"]), clean_text(row["answer"])
        if question.lower() in seen:
            continue
        seen.add(question.lower())
        if is_usable_doc_question(answer, normalized):
            questions.append({"id": f"wiki-q{row['id']}", "question": question, "answer": answer})

    tune, test = split_tune_test(questions)
    reserve = remaining(questions, tune, test)
    write_jsonl(tune, DATA_DIR / "wiki_tune.jsonl")
    write_jsonl(test, DATA_DIR / "wiki_test.jsonl")
    write_jsonl(reserve, DATA_DIR / "wiki_reserve.jsonl")
    return {
        "passages": len(passages),
        "questions_total": len(qa),
        "questions_gardees": len(questions),
        "reglage": len(tune),
        "test": len(test),
        "reserve": len(reserve),
    }


def prepare_sql() -> dict:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    dev = load_dataset(SPIDER_QUESTIONS_REPO, split="validation")
    _check_columns(dev, {"db_id", "query", "question"}, "spider")
    rows = [
        {"id": f"spider-{i}", "question": r["question"].strip(), "sql": r["query"].strip(), "db_id": r["db_id"]}
        for i, r in enumerate(dev)
    ]
    tune, test = split_tune_test(rows)

    db_ids = sorted({r["db_id"] for r in tune + test})
    snapshot_download(
        repo_id=SPIDER_DATABASES_REPO,
        repo_type="dataset",
        allow_patterns=[f"database/{db}/*" for db in db_ids],
        local_dir=SPIDER_DIR,
    )

    # Vérification : chaque base existe et chaque requête correcte s'exécute.
    problems = []
    for row in tune + test:
        path = sqlite_path(row["db_id"])
        if not path.exists():
            problems.append(f"{row['id']} : base absente {path}")
            continue
        try:
            run_query_readonly(path, row["sql"])
        except sqlite3.Error as error:
            problems.append(f"{row['id']} : la requête correcte échoue ({error})")
    if problems:
        raise RuntimeError("Données SQL incomplètes :\n" + "\n".join(problems))

    write_jsonl(tune, DATA_DIR / "spider_tune.jsonl")
    write_jsonl(test, DATA_DIR / "spider_test.jsonl")
    return {"questions_dev": len(rows), "bases": len(db_ids), "reglage": len(tune), "test": len(test)}


def main() -> None:
    print("Documents :", prepare_documents())
    print("SQL :", prepare_sql())
    router = build_router_set(read_jsonl(DATA_DIR / "wiki_test.jsonl"), read_jsonl(DATA_DIR / "spider_test.jsonl"))
    write_jsonl(router, DATA_DIR / "router_test.jsonl")
    print("Routeur :", {"questions": len(router)})


if __name__ == "__main__":
    main()
