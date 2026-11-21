"""Tests de la génération, avec un faux moteur de recherche et un faux modèle : aucun appel réseau."""

import pytest

from src.data import load_passages, write_jsonl
from src.eval_generation import failed_result, score_one, summarize
from src.judge import judge_answer
from src.evaluation import contains_answer, token_f1
from src.generation import NOT_FOUND, SYSTEM_PROMPT, answer_question, build_prompt, parse_answer

PASSAGE_TEXT = {
    "p1": "Ulysses S. Grant was elected president as a Republican in 1868.",
    "p2": "Kangaroos live in Australia.",
    "p3": "Finland has a life expectancy of 75 years for men.",
}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, question, k=10):
        return [(pid, 1.0) for pid in self.ids[:k]]


def fake_llm(content):
    calls = []

    def generate(prompt, system=None, max_tokens=None):
        calls.append({"prompt": prompt, "system": system})
        return {"content": content, "total_tokens": 50}

    return generate, calls


def test_build_prompt_numbers_passages():
    prompt = build_prompt("When?", ["first passage", "second passage"])
    assert "[1] first passage" in prompt and "[2] second passage" in prompt
    assert prompt.endswith("Question: When?\nAnswer:")


def test_parse_answer_extracts_answer_and_citations():
    parsed = parse_answer("1868 [1][3]", n_passages=3)
    assert parsed == {"answer": "1868", "cited": [1, 3], "abstained": False}


def test_parse_answer_ignores_impossible_citations():
    assert parse_answer("1868 [7]", n_passages=3)["cited"] == []


def test_parse_answer_detects_refusal():
    parsed = parse_answer("NOT FOUND", n_passages=3)
    assert parsed["abstained"] and parsed["answer"] == ""


def test_answer_question_sends_k_passages_and_maps_citations():
    generate, calls = fake_llm("1868 [2]")
    result = answer_question("When was Grant elected?", FakeRetriever(["p2", "p1", "p3"]), PASSAGE_TEXT, k=2, generate_fn=generate)
    assert result["passage_ids"] == ["p2", "p1"]
    assert result["cited_ids"] == ["p1"]
    assert result["answer"] == "1868"
    assert calls[0]["system"] == SYSTEM_PROMPT
    assert PASSAGE_TEXT["p3"] not in calls[0]["prompt"]


def test_contains_answer_ignores_articles_and_case():
    assert contains_answer("Federalist Party", "The Federalist Party")
    assert contains_answer("He was elected in 1868.", "1868")
    assert not contains_answer("1869", "1868")
    assert not contains_answer("", "1868")


def test_token_f1_rewards_partial_answers():
    assert token_f1("75 years", "75 years") == 1.0
    assert token_f1("75", "75 years") == pytest.approx(2 / 3)
    assert token_f1("80 years", "75 years") == pytest.approx(0.5)
    assert token_f1("", "75 years") == 0.0


def test_score_one_and_summarize():
    question = {"id": "q1", "answer": "1868"}
    good = score_one(question, {"answer": "1868", "abstained": False, "passage_ids": ["p1"], "total_tokens": 40, "latency_s": 1.0}, PASSAGE_TEXT)
    refusal = score_one(question, {"answer": "", "abstained": True, "passage_ids": ["p1"], "total_tokens": 60, "latency_s": 3.0}, PASSAGE_TEXT)
    memory = score_one(question, {"answer": "1868", "abstained": False, "passage_ids": ["p2"], "total_tokens": 50, "latency_s": 2.0}, PASSAGE_TEXT)
    assert good["correct"] and good["passage_has_answer"]
    summary = summarize([good, refusal, memory])
    assert summary["Réponse juste"] == pytest.approx(2 / 3)
    assert summary["Refus avec bon passage"] == 1
    assert summary["Juste sans bon passage"] == 1
    assert summary["Tokens moyens"] == 50


def test_load_passages_removes_short_ones(tmp_path):
    path = tmp_path / "passages.jsonl"
    write_jsonl([{"id": "a", "text": "125px"}, {"id": "b", "text": "Lincoln was born in Kentucky in 1809."}], path)
    assert [p["id"] for p in load_passages(5, path)] == ["b"]


def test_not_found_marker_is_in_the_instructions():
    assert NOT_FOUND in SYSTEM_PROMPT


def test_judge_reads_verdicts():
    assert judge_answer("q", "1868", "1868", generate_fn=lambda p, **k: {"content": "CORRECT", "total_tokens": 1}) is True
    assert judge_answer("q", "1868", "1869", generate_fn=lambda p, **k: {"content": "Incorrect.", "total_tokens": 1}) is False
    assert judge_answer("q", "1868", "x", generate_fn=lambda p, **k: {"content": "maybe", "total_tokens": 1}) is None


def test_judge_does_not_call_model_for_empty_answer():
    def must_not_be_called(prompt, **kwargs):
        raise AssertionError("pas d'appel pour une réponse vide")

    assert judge_answer("q", "1868", "  ", generate_fn=must_not_be_called) is False


def test_judge_sees_question_reference_and_prediction():
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return {"content": "CORRECT", "total_tokens": 1}

    judge_answer("What rate?", "Canada's rate of 5.9% is its lowest", "5.9%", generate_fn=capture)
    assert "What rate?" in seen["prompt"] and "5.9%" in seen["prompt"] and "lowest" in seen["prompt"]


def test_judge_verdict_is_summarized_apart_from_word_match():
    question = {"id": "q1", "question": "Unemployment rate?", "answer": "Canada's national unemployment rate of 5.9% is its lowest in 33 years"}
    result = {"answer": "5.9%", "abstained": False, "passage_ids": ["p1"], "total_tokens": 40, "latency_s": 1.0}
    row = score_one(question, result, PASSAGE_TEXT, judge_fn=lambda q, ref, pred: True)
    assert row["correct"] is False and row["judge_correct"] is True
    assert summarize([row])["Juste selon le juge"] == 1.0


def test_failed_call_counts_as_wrong_and_is_reported():
    row = score_one({"id": "q1", "question": "When?", "answer": "1868"}, failed_result("When?", RuntimeError("vide")), PASSAGE_TEXT, judge_fn=lambda *a: True)
    assert row["correct"] is False and row["judge_correct"] is None
    summary = summarize([row])
    assert summary["Erreurs d'appel"] == 1 and summary["Réponse juste"] == 0.0
