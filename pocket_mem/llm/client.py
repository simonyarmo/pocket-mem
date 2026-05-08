from __future__ import annotations
import json
import re
import time

import requests

from pocket_mem.config import LLMConfig

_RETRY_STATUSES = {429, 529}  # 429 = rate limit, 529 = Anthropic overload

# LiteLLM provider prefixes — when a model string starts with one of these,
# LiteLLM picks the endpoint from the prefix and we must NOT override base_url.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "gemini/", "vertex_ai/", "bedrock/",
    "azure/", "azure_ai/", "groq/", "together_ai/", "cohere/",
    "replicate/", "huggingface/", "mistral/", "deepseek/", "xai/",
    "openrouter/", "ollama/", "perplexity/", "fireworks_ai/",
    "cerebras/", "sambanova/", "watsonx/",
)


def _has_litellm_provider_prefix(model: str) -> bool:
    return any(model.startswith(p) for p in _LITELLM_PROVIDER_PREFIXES)


def _json_repair_candidates(text: str):
    """Yield progressively more aggressive repairs of malformed LLM JSON.

    Each candidate is tried with json.loads; the first one to parse wins.
    Targets the failure modes seen in practice from local models:
      - bare (unquoted) keys:           `to: "x"`     -> `"to": "x"`
      - keys missing closing quote:     `"to: "x"`    -> `"to": "x"`
      - trailing commas:                `[1, 2,]`     -> `[1, 2]`
      - smart / curly quotes
      - python-style True/False/None
    """
    yield text

    # 1. Quote bare keys
    bare_keys = re.sub(r'(?<!["\'\w])([A-Za-z_]\w*)(?=\s*:)', r'"\1"', text)
    yield bare_keys

    # 2. Close keys whose closing quote was dropped: `"foo: ` -> `"foo": `
    closed_keys = re.sub(r'"([A-Za-z_][\w\s\-]*?)\s*:', r'"\1":', text)
    yield closed_keys

    # 3. Strip trailing commas before } or ]
    no_trailing = re.sub(r",(\s*[}\]])", r"\1", closed_keys)
    yield no_trailing

    # 4. Normalize curly/smart quotes and Python literals
    normalized = (
        no_trailing
        .replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )
    normalized = re.sub(r"\bTrue\b", "true", normalized)
    normalized = re.sub(r"\bFalse\b", "false", normalized)
    normalized = re.sub(r"\bNone\b", "null", normalized)
    yield normalized


def _apply_json_format(body: dict, fmt, style: str) -> None:
    """Attach a JSON-mode hint to the request body in the correct dialect.

    Ollama uses `body["format"]` (either "json" or a JSON-Schema dict).
    The OpenAI Chat Completions spec uses `body["response_format"]`
    ({"type": "json_object"} for free-form JSON, {"type": "json_schema", ...} for schemas).
    """
    if fmt is None:
        return
    if style == "ollama":
        body["format"] = fmt
        return
    # OpenAI-compatible style.
    if isinstance(fmt, dict):
        # Treat dict as JSON Schema. Wrap per OpenAI's response_format spec.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt, "strict": False},
        }
    elif fmt == "json":
        body["response_format"] = {"type": "json_object"}


class LLMClient:

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._tokens_in = 0
        self._tokens_out = 0

    def complete(
        self,
        messages: list[dict],
        format: str | dict | None = None,
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        """Run a chat completion. Returns the assistant message content string.

        Routes to the configured backend (openai_compatible | litellm). For both
        backends, when `model` matches `config.answer_model` and answer-specific
        endpoint/auth fields are set, those override the primary endpoint/auth.
        """
        if system is not None:
            messages = [{"role": "system", "content": system}] + list(messages)

        if self.config.backend == "litellm":
            return self._complete_litellm(messages, format=format, model=model)
        return self._complete_openai_compat(messages, format=format, model=model)

    def _complete_openai_compat(
        self,
        messages: list[dict],
        format: str | dict | None,
        model: str | None,
    ) -> str:
        is_answer_call = (
            model is not None
            and model == self.config.answer_model
            and self.config.answer_base_url
        )
        if is_answer_call:
            base_url = self.config.answer_base_url
            api_key = self.config.answer_api_key or self.config.api_key
            timeout = self.config.answer_timeout or self.config.timeout
        else:
            base_url = self.config.base_url
            api_key = self.config.api_key
            timeout = self.config.timeout

        body: dict = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        _apply_json_format(body, format, self.config.resolved_format_style(base_url))

        url = f"{base_url}/chat/completions"
        # User-supplied extra_headers go in first; Authorization is set last so
        # callers cannot accidentally overwrite the auth header.
        headers = dict(self.config.extra_headers or {})
        headers["Authorization"] = f"Bearer {api_key}"

        delay = self.config.retry_delay
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            except requests.exceptions.ConnectionError:
                if attempt < self.config.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            if resp.status_code in _RETRY_STATUSES and attempt < self.config.max_retries:
                retry_after = float(resp.headers.get("retry-after", delay))
                time.sleep(retry_after)
                delay *= 2
                continue
            resp.raise_for_status()
            break

        data = resp.json()
        usage = data.get("usage") or {}
        self._tokens_in += usage.get("prompt_tokens", 0)
        self._tokens_out += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _complete_litellm(
        self,
        messages: list[dict],
        format: str | dict | None,
        model: str | None,
    ) -> str:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "backend='litellm' requires the litellm package. "
                "Install with: pip install 'pocket-mem[litellm]' "
                "or: pip install litellm"
            ) from e

        is_answer_call = (
            model is not None
            and model == self.config.answer_model
            and self.config.answer_base_url
        )
        if is_answer_call:
            base_url = self.config.answer_base_url
            api_key = self.config.answer_api_key or self.config.api_key
            timeout = self.config.answer_timeout or self.config.timeout
        else:
            base_url = self.config.base_url
            api_key = self.config.api_key
            timeout = self.config.timeout

        resolved_model = model or self.config.model
        kwargs: dict = {
            "model": resolved_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key": api_key,
            "timeout": timeout,
            "num_retries": self.config.max_retries,
        }
        # When the model uses a LiteLLM provider prefix (e.g. "openrouter/...",
        # "gemini/..."), LiteLLM picks the endpoint from the prefix; passing
        # base_url would override it incorrectly. Otherwise forward base_url.
        if base_url and not _has_litellm_provider_prefix(resolved_model):
            kwargs["base_url"] = base_url
        if self.config.extra_headers:
            kwargs["extra_headers"] = dict(self.config.extra_headers)
        if format is not None:
            # LiteLLM accepts OpenAI-shaped response_format. Map our format
            # parameter accordingly: dict -> json_schema, "json" -> json_object.
            if isinstance(format, dict):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": format, "strict": False},
                }
            elif format == "json":
                kwargs["response_format"] = {"type": "json_object"}

        resp = litellm.completion(**kwargs)
        # LiteLLM returns OpenAI-shaped responses (ModelResponse); accept both
        # attribute and dict-style access for forward compat / mocking.
        try:
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self._tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        except AttributeError:
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage") or {}
            self._tokens_in += usage.get("prompt_tokens", 0)
            self._tokens_out += usage.get("completion_tokens", 0)
        return content

    def token_stats(self) -> dict:
        """Return cumulative token counts across all calls."""
        return {"tokens_in": self._tokens_in, "tokens_out": self._tokens_out}

    def complete_json(
        self,
        messages: list[dict],
        schema: dict | None = None,
    ) -> dict:
        """Like complete(), but parses and returns the JSON response as a dict.

        Uses schema as the format value if provided, else format='json'.
        On JSON parse failure, strips markdown code fences and retries once.
        Raises ValueError if the response cannot be parsed as JSON.
        """
        fmt: str | dict = schema if schema is not None else "json"
        raw = self.complete(messages, format=fmt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown fences, then extract the first complete JSON object/array.
            # This handles models that append explanations after the closing brace.
            cleaned = re.sub(r"```json|```", "", raw).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            # Last resort: pull out the first {...} or [...] block by brace matching
            match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
            if match:
                extracted = match.group(1)
                for repaired in _json_repair_candidates(extracted):
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue
            raise ValueError(
                f"LLM response could not be parsed as JSON: {raw!r}"
            )
