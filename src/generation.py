"""Étape 5 : faire répondre Qwen à partir des passages trouvés par la recherche."""

import re
import time

SYSTEM_PROMPT = """You answer questions using only the numbered passages provided.
Rules:
- Answer in a few words (a name, a date, a number or a short phrase), not a full sentence.
- Use only information from the passages, never your own knowledge.
- After the answer, cite the numbers of the passages you used, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: NOT FOUND"""

NOT_FOUND = "NOT FOUND"
CITATION = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, passages: list[str]) -> str:
    """Numérote les passages pour que le modèle puisse citer ses sources."""
    blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    return f"Passages:\n{blocks}\n\nQuestion: {question}\nAnswer:"


def parse_answer(raw: str, n_passages: int) -> dict:
    """Sépare la réponse, les passages cités et le refus de répondre."""
    cited = sorted({int(n) for n in CITATION.findall(raw) if 1 <= int(n) <= n_passages})
    answer = CITATION.sub("", raw).strip(" .;:,\n")
    abstained = NOT_FOUND in raw.upper()
    return {"answer": "" if abstained else answer, "cited": cited, "abstained": abstained}


def answer_question(
    question: str, retriever, passage_text: dict[str, str], k: int = 3, generate_fn=None, max_tokens: int = 400
) -> dict:
    """Cherche les k meilleurs passages, puis demande la réponse au modèle."""
    if generate_fn is None:
        from src.llm import generate as generate_fn

    passage_ids = [pid for pid, _ in retriever.search(question, k)]
    prompt = build_prompt(question, [passage_text[pid] for pid in passage_ids])

    start = time.perf_counter()
    # Marge large : certains modèles « réfléchissent » avant de répondre, et cette réflexion consomme des tokens.
    output = generate_fn(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
    latency = time.perf_counter() - start

    parsed = parse_answer(output["content"], len(passage_ids))
    return {
        "question": question,
        "answer": parsed["answer"],
        "abstained": parsed["abstained"],
        "passage_ids": passage_ids,
        "cited_ids": [passage_ids[n - 1] for n in parsed["cited"]],
        "raw": output["content"],
        "total_tokens": output["total_tokens"],
        "latency_s": round(latency, 2),
    }
