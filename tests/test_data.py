"""Tests des fonctions de src/data.py qui n'ont pas besoin du réseau."""

import sqlite3

import pytest

from src.data import (
    answer_in_text,
    clean_text,
    build_router_set,
    is_usable_doc_question,
    normalize,
    read_jsonl,
    run_query_readonly,
    split_tune_test,
    write_jsonl,
)


def test_normalize_removes_punctuation_and_case():
    assert normalize("  Abraham   LINCOLN, 1861! ") == "abraham lincoln 1861"


def test_answer_in_text_matches_whole_words_only():
    assert answer_in_text("Lincoln", "President Abraham Lincoln was born in 1809.")
    assert not answer_in_text("no", "There is nothing here.")


def test_yes_no_answers_are_excluded():
    passages = [normalize("Yes, he was the sixteenth president.")]
    assert not is_usable_doc_question("yes", passages)


def test_answer_must_appear_in_a_passage():
    passages = [normalize("Lincoln was born in Kentucky.")]
    assert is_usable_doc_question("Kentucky", passages)
    assert not is_usable_doc_question("Illinois", passages)


def test_split_is_reproducible_and_disjoint():
    items = list(range(200))
    tune1, test1 = split_tune_test(items, 30, 100, seed=42)
    tune2, test2 = split_tune_test(items, 30, 100, seed=42)
    assert (tune1, test1) == (tune2, test2)
    assert len(tune1) == 30 and len(test1) == 100
    assert not set(tune1) & set(test1)


def test_split_raises_when_not_enough_items():
    with pytest.raises(ValueError):
        split_tune_test(list(range(50)), 30, 100)


def test_router_set_labels_each_question():
    docs = [{"id": "wiki-q1", "question": "Who?"}]
    sql = [{"id": "spider-1", "question": "How many?", "db_id": "concert_singer"}]
    rows = build_router_set(docs, sql)
    tools = {row["id"]: row["expected_tool"] for row in rows}
    assert tools == {"wiki-q1": "documents", "spider-1": "sql"}


def test_jsonl_round_trip_keeps_accents(tmp_path):
    rows = [{"id": "1", "question": "Où est né Lincoln ?"}]
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows


def test_readonly_query_runs_but_cannot_write(tmp_path):
    db = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE singer (name TEXT)")
    conn.execute("INSERT INTO singer VALUES ('Adele')")
    conn.commit()
    conn.close()

    assert run_query_readonly(db, "SELECT name FROM singer") == [("Adele",)]
    with pytest.raises(sqlite3.OperationalError):
        run_query_readonly(db, "DELETE FROM singer")


def test_clean_text_repairs_garbled_accents():
    assert clean_text("SÃ¶derby, Sweden ") == "Söderby, Sweden"
    assert clean_text("TimiÈ™oara") == "Timișoara"
    assert clean_text("déjà vu") == "déjà vu"
