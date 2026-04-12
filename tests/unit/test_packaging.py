"""Tests that pyproject.toml is correctly configured for pip install pocket-mem."""
import tomllib
import importlib
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent.parent / "pyproject.toml"


def _meta():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


# ── pyproject.toml metadata ───────────────────────────────────────────────────

def test_package_name():
    assert _meta()["project"]["name"] == "pocket-mem"


def test_requires_python():
    assert _meta()["project"]["requires-python"] == ">=3.10"


def test_required_deps():
    deps = _meta()["project"]["dependencies"]
    assert any("requests" in d for d in deps)
    assert any("sentence-transformers" in d for d in deps)


def test_cloud_extra_removed_for_v1():
    # Cloud/Supabase support is deferred to v2; not part of v1 package.
    extras = _meta()["project"]["optional-dependencies"]
    assert "cloud" not in extras, "[cloud] extra should not be in v1 pyproject.toml"


def test_dev_optional_extra_exists():
    extras = _meta()["project"]["optional-dependencies"]
    assert "dev" in extras


def test_package_discovery_configured():
    """find directive must be present so subpackages are included."""
    data = _meta()
    tool = data.get("tool", {}).get("setuptools", {})
    assert "packages" in tool, "setuptools must have packages.find configured"


# ── subpackage importability ──────────────────────────────────────────────────

def test_memory_agent_importable():
    import memory_agent
    assert memory_agent.__file__ is not None


def test_store_subpackage_importable():
    from memory_agent.store.local import SQLiteStore
    assert SQLiteStore is not None


def test_llm_subpackage_importable():
    from memory_agent.llm.client import LLMClient
    assert LLMClient is not None


def test_public_api_importable():
    from memory_agent import MemoryAgent, LLMConfig, StorageConfig, MemoryConfig
    assert all(x is not None for x in [MemoryAgent, LLMConfig, StorageConfig, MemoryConfig])
