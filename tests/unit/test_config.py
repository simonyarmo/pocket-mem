from memory_agent.config import LLMConfig, StorageConfig, MemoryConfig


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.model == "qwen2.5:7b"
    assert cfg.api_key == "ollama"
    assert cfg.timeout == 45
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 1024


def test_llm_config_override():
    cfg = LLMConfig(model="gpt-4o-mini", api_key="sk-abc")
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key == "sk-abc"
    assert cfg.base_url == "http://localhost:11434/v1"  # unchanged


def test_storage_config_defaults():
    cfg = StorageConfig()
    assert cfg.path == "./memory"


def test_storage_config_override():
    cfg = StorageConfig(path="/data/mem")
    assert cfg.path == "/data/mem"


def test_memory_config_defaults():
    cfg = MemoryConfig()
    assert cfg.working_memory_turns == 10
    assert cfg.compaction_threshold == 20
    assert cfg.importance_prune_threshold == 0.1
    assert cfg.prune_after_days == 30
    assert cfg.prune_min_access_count == 2
