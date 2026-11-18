"""Tests de src/llm.py avec un faux client : aucun appel réseau, aucune clé nécessaire."""

from types import SimpleNamespace

import pytest

from src.llm import LLMError, generate


class FakeClient:
    """Imite client.chat.completions.create de Together."""

    def __init__(self, contents, failures=0):
        self.contents = contents
        self.failures = failures
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("réseau indisponible")
        message = SimpleNamespace(content=self.contents)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=42),
        )


def test_generate_returns_content_and_tokens():
    client = FakeClient("  Paris  ")
    result = generate("Capitale de la France ?", client=client)
    assert result == {"content": "Paris", "total_tokens": 42}


def test_generate_puts_system_message_first():
    client = FakeClient("ok")
    generate("question", system="Réponds en une phrase.", client=client)
    roles = [m["role"] for m in client.calls[0]["messages"]]
    assert roles == ["system", "user"]


def test_generate_uses_temperature_zero_by_default():
    client = FakeClient("ok")
    generate("question", client=client)
    assert client.calls[0]["temperature"] == 0.0


def test_generate_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda _: None)
    client = FakeClient("ok", failures=2)
    assert generate("question", client=client, retries=3)["content"] == "ok"
    assert len(client.calls) == 3


def test_generate_raises_after_all_retries(monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda _: None)
    client = FakeClient("ok", failures=5)
    with pytest.raises(LLMError):
        generate("question", client=client, retries=3)


def test_generate_rejects_empty_answer():
    with pytest.raises(LLMError):
        generate("question", client=FakeClient(""))


def test_missing_api_key_gives_clear_error(monkeypatch):
    import src.llm

    monkeypatch.setattr(src.llm, "_client", None)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="TOGETHER_API_KEY"):
        generate("question")
