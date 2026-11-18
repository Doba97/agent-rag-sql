"""Appel du modèle de langage (Qwen) par l'API Together."""

import os
import time

from src.config import LLM_MODEL


class LLMError(RuntimeError):
    """Erreur lors de l'appel au modèle de langage."""


_client = None


def get_client():
    """Crée le client Together une seule fois, à la première utilisation."""
    global _client
    if _client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise LLMError("TOGETHER_API_KEY est absente : ajoute-la dans le fichier .env")
        from together import Together

        _client = Together(api_key=api_key)
    return _client


def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str = LLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500,
    retries: int = 3,
    client=None,
) -> dict:
    """Envoie un prompt au modèle et renvoie sa réponse.

    temperature=0.0 par défaut : les évaluations doivent être reproductibles.
    En cas d'erreur réseau, on réessaie `retries` fois en attendant 1 s, 2 s, 4 s...

    Renvoie {"content": str, "total_tokens": int}.
    """
    client = client or get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning={"enabled": False},  # pas de raisonnement : réponses plus courtes et moins chères
            )
            break
        except Exception as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2**attempt)
    else:
        raise LLMError(f"Échec de l'appel au modèle après {retries} essais : {last_error}") from last_error

    content = response.choices[0].message.content
    if not content:
        raise LLMError("Le modèle a renvoyé une réponse vide")

    return {"content": content.strip(), "total_tokens": response.usage.total_tokens}
