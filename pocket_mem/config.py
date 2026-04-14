from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    answer_model: str | None = None
    answer_base_url: str | None = None   # if set, answer calls route here instead of base_url
    answer_api_key: str | None = None    # api key for answer_base_url (falls back to api_key)
    answer_timeout: int | None = None    # timeout for answer calls (falls back to timeout)
    api_key: str = "ollama"
    timeout: int = 45
    temperature: float = 0.1
    max_tokens: int = 4096
    max_retries: int = 3                 # retries on 429/529 rate-limit responses
    retry_delay: float = 5.0            # initial backoff in seconds (doubles each retry)


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
