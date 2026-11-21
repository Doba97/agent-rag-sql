"""Appel du modèle de langage, quelle que soit la plateforme.

OpenAI, OpenRouter, Together et Groq acceptent tous le format de requête d'OpenAI :
on utilise donc le client `openai`, et on choisit la plateforme dans le fichier .env.

    LLM_API_KEY    la clé de la plateforme
    LLM_BASE_URL   l'adresse de la plateforme (vide pour OpenAI)
    LLM_MODEL      le nom exact du modèle sur cette plateforme
    LLM_MIN_INTERVAL   secondes minimum entre deux appels (ex. 3.5 pour 20 appels par minute)
"""

import os
import time

from src.config import LLM_MODEL


class LLMError(RuntimeError):
    """Erreur lors de l'appel au modèle de langage."""


_client = None
_last_call = 0.0


def get_api_key() -> str | None:
    # TOGETHER_API_KEY est encore accepté pour les anciens fichiers .env.
    return os.getenv("LLM_API_KEY") or os.getenv("TOGETHER_API_KEY")


def get_client():
    """Crée le client une seule fois, à la première utilisation."""
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key or api_key == "ta_cle_ici":
            raise LLMError("LLM_API_KEY est absente : ajoute ta clé dans le fichier .env")
        from openai import OpenAI

        _client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)
    return _client


def _wait_for_rate_limit() -> None:
    """Respecte un délai minimum entre deux appels (limites des offres gratuites)."""
    global _last_call
    interval = float(os.getenv("LLM_MIN_INTERVAL", "0") or 0)
    wait = _last_call + interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _adapt_options(options: dict, error: Exception) -> bool:
    """Certains modèles refusent un paramètre : on l'adapte. Renvoie True si on peut réessayer."""
    message = str(error)
    if "max_tokens" in message and "max_tokens" in options:
        # Modèles récents d'OpenAI : max_tokens est remplacé par max_completion_tokens.
        options["max_completion_tokens"] = options.pop("max_tokens")
        return True
    if "temperature" in message and "temperature" in options:
        # Certains modèles n'acceptent que la température par défaut.
        options.pop("temperature")
        return True
    return False


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
    En cas d'erreur (réseau, limite d'appels), on réessaie `retries` fois en attendant 1 s, 2 s, 4 s...

    Renvoie {"content": str, "total_tokens": int}.
    """
    client = client or get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    last_error = None
    attempt = 0
    while attempt < retries:
        _wait_for_rate_limit()
        try:
            response = client.chat.completions.create(**options)
            break
        except Exception as error:
            last_error = error
            if _adapt_options(options, error):
                continue  # paramètre adapté : nouvel essai immédiat, qui ne compte pas
            if "401" in str(error) or "invalid_api_key" in str(error).lower():
                raise LLMError(f"Clé refusée par la plateforme : vérifie LLM_API_KEY et LLM_BASE_URL ({error})") from error
            attempt += 1
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    else:
        raise LLMError(f"Échec de l'appel au modèle après {retries} essais : {last_error}") from last_error

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError(
                "Réponse vide : la limite de tokens a été atteinte avant la réponse "
                "(le modèle a sans doute « réfléchi » avant de répondre). Augmente max_tokens."
            )
        raise LLMError("Le modèle a renvoyé une réponse vide")

    usage = getattr(response, "usage", None)
    return {"content": content.strip(), "total_tokens": getattr(usage, "total_tokens", 0) or 0}


if __name__ == "__main__":
    # Vérifie la clé, l'adresse et le modèle :  python -m src.llm
    result = generate("Reply with the single word: OK")
    print(f"Plateforme : {os.getenv('LLM_BASE_URL') or 'OpenAI'}")
    print(f"Modèle : {LLM_MODEL}")
    print(f"Réponse : {result['content']!r} ({result['total_tokens']} tokens)")
