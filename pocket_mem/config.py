from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal


# Built-in provider presets. Each entry maps a short name to its
# OpenAI-compatible base_url, the env var that holds the API key, and
# (optionally) a default model.
PROVIDER_PRESETS: dict[str, dict] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,            # Ollama ignores auth; default to "ollama"
        "api_key_default": "ollama",
        "default_model": "qwen2.5:7b",
        "json_format_style": "ollama",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "zai": {
        # z.ai's OpenAI-compatible coding endpoint.
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key_env": "ZAI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "gemini": {
        # Google's OpenAI-compatible endpoint.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
}


Backend = Literal["openai_compatible", "litellm"]
JsonFormatStyle = Literal["auto", "openai", "ollama"]


def _detect_format_style(base_url: str) -> str:
    """Auto-detect whether a base_url speaks Ollama-style `format=` or OpenAI-style
    `response_format=`. Defaults to OpenAI-style for unknown endpoints."""
    bu = (base_url or "").lower()
    if ":11434" in bu or "/ollama" in bu:
        return "ollama"
    return "openai"


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    answer_model: str | None = None
    answer_base_url: str | None = None   # if set, answer calls route here instead of base_url
    answer_api_key: str | None = field(default=None, repr=False)  # repr-redacted
    answer_timeout: int | None = None    # timeout for answer calls (falls back to timeout)
    api_key: str = field(default="ollama", repr=False)             # repr-redacted
    timeout: int = 45
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 3                 # retries on 429/529 rate-limit responses
    retry_delay: float = 5.0             # initial backoff in seconds (doubles each retry)
    # Backend selector. "openai_compatible" (default) uses the built-in
    # requests-based client and works with any OpenAI-compatible endpoint
    # (Ollama, OpenRouter, z.ai, OpenAI, Anthropic, Groq, Gemini, Together,
    # vLLM, LM Studio, etc.). "litellm" routes through the LiteLLM SDK,
    # which adds unified model strings (e.g. "openrouter/anthropic/claude-3-haiku",
    # "gemini/gemini-1.5-pro", "bedrock/...") and broader provider coverage.
    backend: Backend = "openai_compatible"
    # JSON-mode wire format for the openai_compatible backend.
    #   "auto"   -> infer from base_url (Ollama if :11434 / "/ollama")
    #   "ollama" -> body["format"] = "json" | <schema>           (Ollama / Ollama-compat)
    #   "openai" -> body["response_format"] = {"type": "json_object"} | json_schema (OpenAI/OpenRouter/z.ai/...)
    json_format_style: JsonFormatStyle = "auto"
    # Extra headers attached to every openai_compatible HTTP request.
    # Useful for e.g. OpenRouter's optional ranking headers
    # ({"HTTP-Referer": "...", "X-Title": "..."}). Note: the Authorization
    # header is set last and cannot be overridden via extra_headers.
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_format_style(self, base_url: str | None = None) -> str:
        if self.json_format_style != "auto":
            return self.json_format_style
        return _detect_format_style(base_url or self.base_url)

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        backend: Backend = "openai_compatible",
        extra_headers: dict[str, str] | None = None,
        **overrides,
    ) -> "LLMConfig":
        """Build an LLMConfig for a known provider preset.

        Examples:
            LLMConfig.for_provider("openrouter", model="anthropic/claude-3-haiku")
            LLMConfig.for_provider("zai", model="glm-4.5")
            LLMConfig.for_provider("ollama")  # uses default model qwen2.5:7b

        If api_key is None, the matching env var (e.g. OPENROUTER_API_KEY) is read.
        Pass backend="litellm" to route through the LiteLLM SDK instead of the
        built-in HTTP client. Any other LLMConfig field can be set via **overrides.
        """
        key = provider.lower().replace("-", "").replace("_", "").replace(".", "")
        # Map a few common aliases.
        aliases = {"zaibsubscription": "zai", "z": "zai", "openrouterai": "openrouter"}
        key = aliases.get(key, key)
        if key not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown provider preset: {provider!r}. "
                f"Known: {sorted(PROVIDER_PRESETS)}"
            )
        preset = PROVIDER_PRESETS[key]

        resolved_key = api_key
        if resolved_key is None:
            env_var = preset.get("api_key_env")
            if env_var:
                resolved_key = os.environ.get(env_var)
            if resolved_key is None:
                resolved_key = preset.get("api_key_default")
        if resolved_key is None:
            env_var = preset.get("api_key_env")
            hint = f"set ${env_var}" if env_var else "pass api_key="
            raise ValueError(
                f"No API key for provider {provider!r}. "
                f"Pass api_key= or {hint}."
            )

        resolved_model = model or preset.get("default_model")
        if resolved_model is None:
            raise ValueError(
                f"Provider {provider!r} has no default model — pass model=..."
            )

        # Pick up preset's format style (Ollama uses "format=", others use OpenAI style).
        format_style = preset.get("json_format_style", "openai")
        if "json_format_style" in overrides:
            format_style = overrides.pop("json_format_style")

        return cls(
            base_url=preset["base_url"],
            model=resolved_model,
            api_key=resolved_key,
            backend=backend,
            json_format_style=format_style,
            extra_headers=dict(extra_headers or {}),
            **overrides,
        )


@dataclass
class StorageConfig:
    path: str = "./memory"


@dataclass
class MemoryConfig:
    working_memory_turns: int = 10
    compaction_threshold: int = 100
    importance_prune_threshold: float = 0.1
    prune_after_days: int = 30
    prune_min_access_count: int = 2
