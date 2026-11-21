"""Juge LLM : décide si une réponse est juste en la comparant à la réponse de référence.

Pourquoi un juge ? Dans rag-mini-wikipedia, certaines réponses de référence sont des phrases
entières (« By 2001, there were 19 raw water reservoirs... »), et le modèle peut répondre juste
avec d'autres mots (« 5.9% » au lieu d'une phrase sur le chômage). Une comparaison mot à mot
compte ces réponses comme fausses. Le juge compare le sens, pas les mots.

Limite : le juge est lui-même un LLM, il peut se tromper. On vérifie son accord avec une
relecture à la main sur les questions de réglage.
"""

JUDGE_SYSTEM = "You grade answers to factual questions. You are strict about facts and lenient about wording."

JUDGE_TEMPLATE = """Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Is the predicted answer correct?
It is CORRECT if it gives the key fact that the question asks for and that the reference contains,
even with different wording, fewer words or extra details, as long as nothing in it contradicts the reference.
It is INCORRECT if it gives a different fact, misses the key fact, or contradicts the reference.
Reply with exactly one word: CORRECT or INCORRECT."""


def judge_answer(question: str, reference: str, prediction: str, generate_fn=None) -> bool | None:
    """True si le juge estime la réponse juste, False sinon, None si sa réponse est illisible."""
    if not prediction.strip():
        return False
    if generate_fn is None:
        from src.llm import generate as generate_fn

    prompt = JUDGE_TEMPLATE.format(question=question, reference=reference, prediction=prediction)
    verdict = generate_fn(prompt, system=JUDGE_SYSTEM, max_tokens=400)["content"].strip().upper()
    if "INCORRECT" in verdict:
        return False
    if "CORRECT" in verdict:
        return True
    return None
