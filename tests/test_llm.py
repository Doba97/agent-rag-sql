"""Tests de src/llm.py avec un faux client : aucun appel réseau, aucune clé nécessaire."""

from types import SimpleNamespace

import pytest

from src.llm import LLMError, generate


class FakeClient:
    """Imite client.chat.completions.create du client OpenAI."""

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
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        generate("question")


def test_placeholder_key_is_rejected(monkeypatch):
    import src.llm

    monkeypatch.setattr(src.llm, "_client", None)
    monkeypatch.setenv("LLM_API_KEY", "ta_cle_ici")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        generate("question")


class PickyClient(FakeClient):
    """Imite un modèle qui refuse certains paramètres."""

    def __init__(self, refused):
        super().__init__("ok")
        self.refused = refused

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        for name in self.refused:
            if name in kwargs:
                raise ValueError(f"Unsupported parameter: '{name}' is not supported with this model.")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(total_tokens=7),
        )


def test_max_tokens_is_renamed_when_refused():
    client = PickyClient(refused=["max_tokens"])
    assert generate("question", client=client, max_tokens=50)["content"] == "ok"
    assert client.calls[-1]["max_completion_tokens"] == 50


def test_temperature_is_dropped_when_refused():
    client = PickyClient(refused=["max_tokens", "temperature"])
    assert generate("question", client=client)["content"] == "ok"
    assert "temperature" not in client.calls[-1]


def test_wrong_key_stops_immediately(monkeypatch):
    class RejectingClient(FakeClient):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            raise PermissionError("Error code: 401 - invalid_api_key")

    client = RejectingClient("ok")
    with pytest.raises(LLMError, match="Clé refusée"):
        generate("question", client=client)
    assert len(client.calls) == 1


def test_min_interval_spaces_calls(monkeypatch):
    import src.llm

    waits = []
    monkeypatch.setattr(src.llm.time, "sleep", waits.append)
    monkeypatch.setenv("LLM_MIN_INTERVAL", "3")
    monkeypatch.setattr(src.llm, "_last_call", 0.0)
    generate("a", client=FakeClient("ok"))
    generate("b", client=FakeClient("ok"))
    assert waits and 2.5 < waits[-1] <= 3


def test_empty_answer_because_of_token_limit_gives_clear_error():
    class TruncatedClient(FakeClient):
        def _create(self, **kwargs):
            self.calls.append(kwargs)
            choice = SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(total_tokens=100))

    with pytest.raises(LLMError, match="limite de tokens"):
        generate("question", client=TruncatedClient("ok"))
