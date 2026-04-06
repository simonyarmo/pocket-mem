from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    api_key: str = "ollama"
    timeout: int = 30
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class StorageConfig:
    path: str = "./memory"


@dataclass
class MemoryConfig:
    working_memory_turns: int = 10
    compaction_threshold: int = 20
    importance_prune_threshold: float = 0.1
    prune_after_days: int = 30
    prune_min_access_count: int = 2
