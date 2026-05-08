"""Tests for pluggable LLM providers: presets, extra_headers, LiteLLM backend."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch

import pytest

from pocket_mem.config import LLMConfig, PROVIDER_PRESETS
from pocket_mem.llm.client import LLMClient


def _mock_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock_resp


# ---------------------------------------------------------------------------
# LLMConfig.for_provider() presets
# ---------------------------------------------------------------------------

def test_preset_ollama_uses_localhost():
    cfg = LLMConfig.for_provider("ollama")
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.model == "qwen2.5:7b"
    assert cfg.api_key == "ollama"
    assert cfg.backend == "openai_compatible"
    assert cfg.json_format_style == "ollama"


def test_preset_ollama_works_without_env(monkeypatch):
    """Ollama preset must succeed even when no LLM env vars are set."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = LLMConfig.for_provider("ollama")
    assert cfg.api_key == "ollama"


def test_preset_openrouter_reads_env_var(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-123")
    cfg = LLMConfig.for_provider("openrouter", model="anthropic/claude-3-haiku")
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model == "anthropic/claude-3-haiku"
    assert cfg.api_key == "or-test-123"
    assert cfg.json_format_style == "openai"


def test_preset_zai_reads_env_var(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
    cfg = LLMConfig.for_provider("zai", model="glm-4.5")
    assert cfg.base_url == "https://api.z.ai/api/coding/paas/v4"
    assert cfg.model == "glm-4.5"
    assert cfg.api_key == "zai-secret"


def test_preset_openai_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    cfg = LLMConfig.for_provider("openai", model="gpt-4o-mini", api_key="explicit")
    assert cfg.api_key == "explicit"


def test_preset_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    cfg = LLMConfig.for_provider("anthropic", model="claude-haiku-4-5-20251001")
    assert cfg.base_url == "https://api.anthropic.com/v1"
    assert cfg.api_key == "a-key"


def test_preset_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    cfg = LLMConfig.for_provider("groq", model="llama-3.1-8b-instant")
    assert cfg.api_key == "g-key"


def test_preset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMConfig.for_provider("not-a-real-provider")


def test_preset_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMConfig.for_provider("openrouter", model="x/y")


def test_preset_overrides_passthrough(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = LLMConfig.for_provider(
        "openrouter", model="x/y", temperature=0.7, max_tokens=2048
    )
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 2048


def test_preset_litellm_backend(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = LLMConfig.for_provider(
        "openrouter", model="anthropic/claude-3-haiku", backend="litellm"
    )
    assert cfg.backend == "litellm"


def test_preset_alias_normalization(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "k")
    # Should accept "z.ai" / "Z.AI" / "z-ai" etc.
    cfg = LLMConfig.for_provider("z.ai", model="glm-4.5")
    assert cfg.base_url == PROVIDER_PRESETS["zai"]["base_url"]


def test_preset_extra_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = LLMConfig.for_provider(
        "openrouter",
        model="x/y",
        extra_headers={"HTTP-Referer": "https://my-app", "X-Title": "MyApp"},
    )
    assert cfg.extra_headers == {
        "HTTP-Referer": "https://my-app",
        "X-Title": "MyApp",
    }


# ---------------------------------------------------------------------------
# Security: api_key redaction in repr
# ---------------------------------------------------------------------------

def test_api_key_is_redacted_in_repr():
    cfg = LLMConfig(api_key="sk-supersecret", answer_api_key="sk-also-secret")
    representation = repr(cfg)
    assert "sk-supersecret" not in representation
    assert "sk-also-secret" not in representation


# ---------------------------------------------------------------------------
# extra_headers injected into HTTP request
# ---------------------------------------------------------------------------

def test_extra_headers_merged_into_openai_compat_request():
    cfg = LLMConfig(
        api_key="k",
        extra_headers={"HTTP-Referer": "https://app", "X-Title": "MyApp"},
    )
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["HTTP-Referer"] == "https://app"
    assert headers["X-Title"] == "MyApp"


def test_authorization_header_cannot_be_overridden_by_extra_headers():
    """Security: extra_headers must not be able to redirect billing."""
    cfg = LLMConfig(
        api_key="real-key",
        extra_headers={"Authorization": "Bearer attacker-key"},
    )
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer real-key"


def test_no_extra_headers_keeps_only_authorization():
    cfg = LLMConfig(api_key="k")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    headers = mock_post.call_args.kwargs["headers"]
    assert set(headers) == {"Authorization"}


# ---------------------------------------------------------------------------
# JSON format wire-format translation
# ---------------------------------------------------------------------------

def test_openai_compat_uses_response_format_for_non_ollama_endpoint(monkeypatch):
    """Non-Ollama endpoints (OpenRouter, OpenAI, etc.) must receive
    response_format, NOT Ollama's `format` field."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = LLMConfig.for_provider("openrouter", model="anthropic/claude-3-haiku")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"x":1}')
        client.complete([{"role": "user", "content": "hi"}], format="json")
    body = mock_post.call_args.kwargs["json"]
    assert body.get("response_format") == {"type": "json_object"}
    assert "format" not in body  # Ollama-style key must NOT be present


def test_openai_compat_wraps_dict_schema_as_json_schema(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    cfg = LLMConfig.for_provider("openai", model="gpt-4o-mini")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"x":"y"}')
        client.complete([{"role": "user", "content": "hi"}], format=schema)
    body = mock_post.call_args.kwargs["json"]
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_openai_compat_keeps_ollama_style_for_ollama_endpoint():
    """Ollama endpoint must keep using body["format"] for backward compat."""
    cfg = LLMConfig()  # default base_url is http://localhost:11434/v1
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"x":1}')
        client.complete([{"role": "user", "content": "hi"}], format="json")
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == "json"
    assert "response_format" not in body


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------

def test_litellm_backend_dispatches_to_litellm_completion():
    """backend='litellm' should call litellm.completion with the right args."""
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "litellm-says-hi"
    fake_response.usage.prompt_tokens = 7
    fake_response.usage.completion_tokens = 11
    fake_litellm.completion.return_value = fake_response

    cfg = LLMConfig(
        backend="litellm",
        model="openrouter/anthropic/claude-3-haiku",
        api_key="or-key",
        max_retries=2,
    )
    client = LLMClient(cfg)

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        out = client.complete([{"role": "user", "content": "hello"}])

    assert out == "litellm-says-hi"
    fake_litellm.completion.assert_called_once()
    kwargs = fake_litellm.completion.call_args.kwargs
    assert kwargs["model"] == "openrouter/anthropic/claude-3-haiku"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["api_key"] == "or-key"
    assert kwargs["num_retries"] == 2
    # Provider-prefixed model strings should NOT pass base_url through.
    assert "base_url" not in kwargs

    # Token stats should accumulate from the litellm response.
    stats = client.token_stats()
    assert stats == {"tokens_in": 7, "tokens_out": 11}


def test_litellm_backend_forwards_base_url_for_plain_model():
    """If model is not provider-prefixed, base_url should be forwarded."""
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    fake_response.usage.prompt_tokens = 0
    fake_response.usage.completion_tokens = 0
    fake_litellm.completion.return_value = fake_response

    cfg = LLMConfig(
        backend="litellm",
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        api_key="ollama",
    )
    client = LLMClient(cfg)

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        client.complete([{"role": "user", "content": "hi"}])

    kwargs = fake_litellm.completion.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:11434/v1"


def test_litellm_backend_translates_json_format():
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = '{"k": 1}'
    fake_response.usage.prompt_tokens = 0
    fake_response.usage.completion_tokens = 0
    fake_litellm.completion.return_value = fake_response

    cfg = LLMConfig(backend="litellm", model="openrouter/x/y", api_key="k")
    client = LLMClient(cfg)

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        client.complete([{"role": "user", "content": "hi"}], format="json")

    kwargs = fake_litellm.completion.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_litellm_backend_translates_schema_format():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = '{"x":"y"}'
    fake_response.usage.prompt_tokens = 0
    fake_response.usage.completion_tokens = 0
    fake_litellm.completion.return_value = fake_response

    cfg = LLMConfig(backend="litellm", model="openrouter/x/y", api_key="k")
    client = LLMClient(cfg)

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        client.complete([{"role": "user", "content": "hi"}], format=schema)

    kwargs = fake_litellm.completion.call_args.kwargs
    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema


def test_litellm_missing_raises_clean_import_error():
    """If litellm is not installed but backend='litellm', surface a clear error."""
    cfg = LLMConfig(backend="litellm", model="x/y", api_key="k")
    client = LLMClient(cfg)

    # Force `import litellm` to fail by removing it from sys.modules and
    # blocking the import via meta_path.
    blocked = {"litellm"}

    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name in blocked:
                raise ImportError(f"blocked: {name}")
            return None

    sys.modules.pop("litellm", None)
    sys.meta_path.insert(0, _Blocker())
    try:
        with pytest.raises(ImportError, match="pocket-mem\\[litellm\\]"):
            client.complete([{"role": "user", "content": "hi"}])
    finally:
        sys.meta_path.pop(0)


def test_litellm_backend_uses_answer_endpoint(monkeypatch):
    """Dual-endpoint routing should also work in the LiteLLM backend."""
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "answered"
    fake_response.usage.prompt_tokens = 0
    fake_response.usage.completion_tokens = 0
    fake_litellm.completion.return_value = fake_response

    cfg = LLMConfig(
        backend="litellm",
        model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        answer_model="claude-haiku-4-5-20251001",
        answer_base_url="https://api.anthropic.com/v1",
        answer_api_key="anth-key",
    )
    client = LLMClient(cfg)

    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        client.complete(
            [{"role": "user", "content": "hi"}],
            model="claude-haiku-4-5-20251001",
        )

    kwargs = fake_litellm.completion.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["api_key"] == "anth-key"
    # Plain (non-prefixed) answer model -> base_url should be forwarded.
    assert kwargs["base_url"] == "https://api.anthropic.com/v1"
