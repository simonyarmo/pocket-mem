import json
import pytest
from unittest.mock import MagicMock, patch
from pocket_mem.config import LLMConfig
from pocket_mem.llm.client import LLMClient


def _mock_response(content: str) -> MagicMock:
    """Build a fake requests.Response returning the given content string."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_resp


def test_complete_returns_content():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("hello")
        result = client.complete([{"role": "user", "content": "hi"}])
    assert result == "hello"


def test_complete_sends_correct_payload():
    cfg = LLMConfig(model="qwen2.5:7b", temperature=0.1, max_tokens=512)
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "test"}])
    _, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 512
    assert body["messages"] == [{"role": "user", "content": "test"}]


def test_complete_with_json_format():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"category": "remember"}')
        client.complete([{"role": "user", "content": "hi"}], format="json")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["format"] == "json"


def test_complete_with_schema_format():
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"label": "David"}')
        client.complete([{"role": "user", "content": "hi"}], format=schema)
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["format"] == schema


def test_complete_omits_format_when_none():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("plain text")
        client.complete([{"role": "user", "content": "hi"}])
    _, kwargs = mock_post.call_args
    assert "format" not in kwargs["json"]


def test_complete_json_parses_clean_json():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"category": "remember"}')
        result = client.complete_json([{"role": "user", "content": "hi"}])
    assert result == {"category": "remember"}


def test_complete_json_strips_markdown_fences():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    raw = '```json\n{"category": "ignore"}\n```'
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response(raw)
        result = client.complete_json([{"role": "user", "content": "hi"}])
    assert result == {"category": "ignore"}


def test_complete_json_strips_plain_fences():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    raw = '```\n{"key": "value"}\n```'
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response(raw)
        result = client.complete_json([{"role": "user", "content": "hi"}])
    assert result == {"key": "value"}


def test_complete_json_raises_on_invalid():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("this is not json at all")
        with pytest.raises(ValueError):
            client.complete_json([{"role": "user", "content": "hi"}])


def test_complete_json_uses_schema_as_format():
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response('{"label": "David"}')
        result = client.complete_json(
            [{"role": "user", "content": "hi"}], schema=schema
        )
    assert result == {"label": "David"}
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["format"] == schema


def test_complete_uses_base_url():
    cfg = LLMConfig(base_url="http://localhost:11434/v1")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    url = mock_post.call_args[0][0]
    assert url == "http://localhost:11434/v1/chat/completions"


def test_complete_sends_authorization_header():
    cfg = LLMConfig(api_key="my-secret-key")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer my-secret-key"


def test_complete_prepends_system_message():
    cfg = LLMConfig()
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete(
            [{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )
    _, kwargs = mock_post.call_args
    messages = kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_complete_model_override():
    cfg = LLMConfig(model="qwen2.5:7b")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}], model="llama3.2:3b")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "llama3.2:3b"


def test_complete_model_falls_back_to_config():
    cfg = LLMConfig(model="qwen2.5:7b")
    client = LLMClient(cfg)
    with patch("pocket_mem.llm.client.requests.post") as mock_post:
        mock_post.return_value = _mock_response("ok")
        client.complete([{"role": "user", "content": "hi"}])
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "qwen2.5:7b"
