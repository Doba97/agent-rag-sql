"""Exploration rapide des données téléchargées.

Lancer depuis la racine du projet :  python -m notebooks.explorer_donnees
"""

import random
import sys
import sqlite3
from collections import Counter

from src.data import DATA_DIR, SPIDER_DIR, read_jsonl

# Le terminal Windows ne sait pas afficher certains caractères (ex. "ș") : on force l'UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- 1. Les passages Wikipedia -------------------------------------------
passages = read_jsonl(DATA_DIR / "wiki_passages.jsonl")
lengths = sorted(len(p["text"].split()) for p in passages)
print(f"{len(passages)} passages")
print(f"Longueur en mots : min {lengths[0]}, médiane {lengths[len(lengths) // 2]}, max {lengths[-1]}")

print("\n3 passages tirés au hasard :")
for p in random.Random(0).sample(passages, 3):
    print(f"- [{p['id']}] {p['text'][:200]}...")

print("\n5 questions de réglage :")
for q in read_jsonl(DATA_DIR / "wiki_tune.jsonl")[:5]:
    print(f"- {q['question']}  ->  {q['answer']}")

# --- 2. Les bases SQLite --------------------------------------------------
sql_questions = read_jsonl(DATA_DIR / "spider_tune.jsonl") + read_jsonl(DATA_DIR / "spider_test.jsonl")
questions_per_db = Counter(q["db_id"] for q in sql_questions)

print(f"\n{'base':<35}{'tables':>7}{'lignes':>9}{'questions':>11}")
for path in sorted((SPIDER_DIR / "database").glob("*/*.sqlite")):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    ]
    n_rows = sum(conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables)
    conn.close()
    print(f"{path.stem:<35}{len(tables):>7}{n_rows:>9}{questions_per_db[path.stem]:>11}")
    print(f"    tables : {', '.join(tables)}")
