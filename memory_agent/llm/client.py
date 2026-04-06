from __future__ import annotations
import json
import re

import requests

from memory_agent.config import LLMConfig


class LLMClient:

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict],
        format: str | dict | None = None,
    ) -> str:
        """POST to /v1/chat/completions. Returns the assistant message content string."""
        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if format is not None:
            body["format"] = format

        url = f"{self.config.base_url}/chat/completions"
        resp = requests.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

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
