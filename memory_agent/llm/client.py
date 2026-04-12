from __future__ import annotations
import json
import re
import time

import requests

from memory_agent.config import LLMConfig

_RETRY_STATUSES = {429, 529}  # 429 = rate limit, 529 = Anthropic overload


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
        """POST to /v1/chat/completions. Returns the assistant message content string.

        Retries up to `config.max_retries` times on rate-limit (429/529) responses,
        using exponential backoff starting at `config.retry_delay` seconds.
        """
        if system is not None:
            messages = [{"role": "system", "content": system}] + list(messages)
        body: dict = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if format is not None:
            body["format"] = format

        is_answer_call = (
            model is not None
            and model == self.config.answer_model
            and self.config.answer_base_url
        )
        if is_answer_call:
            url = f"{self.config.answer_base_url}/chat/completions"
            api_key = self.config.answer_api_key or self.config.api_key
            timeout = self.config.answer_timeout or self.config.timeout
        else:
            url = f"{self.config.base_url}/chat/completions"
            api_key = self.config.api_key
            timeout = self.config.timeout

        delay = self.config.retry_delay
        for attempt in range(self.config.max_retries + 1):
            resp = requests.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
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
            cleaned = re.sub(r"```json|```", "", raw).strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM response could not be parsed as JSON: {raw!r}"
                ) from exc
