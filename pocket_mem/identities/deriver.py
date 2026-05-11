from __future__ import annotations
import hashlib
import json
import logging
import time
from datetime import datetime

from pocket_mem.identities.prebuilt import match_prebuilt

_log = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "role", "domain", "seed_topics", "priority_entity_types",
    "priority_relations", "high_importance_signals", "low_importance_signals",
    "extraction_focus", "compaction_preserve", "compaction_compress",
    "question_complexity_target",
}
_MAX_RETRIES = 3


def _description_hash(description: str) -> str:
    return hashlib.sha256(description.encode()).hexdigest()[:16]


def _load_from_cache(description: str, store) -> dict | None:
    """Check nodes table for a previously derived identity config."""
    cache_key = _description_hash(description)
    try:
        candidates = store.get_nodes_by_type("identity_config")
        for node in candidates:
            if node.label.startswith(cache_key):
                derived = node.data.get("derived")
                if derived and _REQUIRED_KEYS.issubset(derived.keys()):
                    _log.debug("Identity cache hit for hash %s", cache_key)
                    return derived
    except Exception as e:
        _log.warning("Identity cache lookup failed: %s", e)
    return None


def _save_to_cache(description: str, derived: dict, model_name: str, store) -> None:
    """Persist derived identity config as an identity_config node."""
    from pocket_mem.models import Node

    cache_key = _description_hash(description)
    node = Node(
        id=cache_key,
        node_type="identity_config",
        label=f"{cache_key}:{description[:80]}",
        data={
            "description": description,
            "derived": derived,
            "model_used": model_name,
            "derived_at": datetime.utcnow().isoformat(),
        },
        importance=1.0,
    )
    now = datetime.utcnow().isoformat()
    node.created_at = now
    node.updated_at = now
    try:
        store.write_node(node)
    except Exception as e:
        _log.warning("Failed to cache identity config: %s", e)


def _derive_via_llm(description: str, llm_client, model_name: str) -> dict | None:
    """Call LLM to derive a structured identity config from description."""
    from pocket_mem.llm.prompts import IDENTITY_DERIVE_SCHEMA, IDENTITY_DERIVE_SYSTEM

    prompt = (
        f"Derive a structured memory configuration for an AI assistant used by:\n\n"
        f"{description}\n\n"
        "Return the JSON config exactly as specified in your instructions."
    )

    for attempt in range(_MAX_RETRIES):
        try:
            result = llm_client.complete_json(
                [{"role": "user", "content": prompt}],
                schema=IDENTITY_DERIVE_SCHEMA,
            )
            if _REQUIRED_KEYS.issubset(result.keys()):
                return result
            missing = _REQUIRED_KEYS - result.keys()
            _log.warning(
                "Identity derivation attempt %d missing keys: %s", attempt + 1, missing
            )
        except Exception as e:
            _log.warning("Identity derivation attempt %d failed: %s", attempt + 1, e)

        if attempt < _MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    return None


def resolve_identity(description: str, identity_config, store, llm) -> dict | None:
    """Resolve identity using: prebuilt → cache → LLM derivation → fallback None.

    Returns the derived config dict, or None if all resolution methods fail.
    identity_config is an IdentityConfig instance.
    llm is an LLMClient instance (used as fallback when no derivation_api_key set).
    """
    # Layer 1: prebuilt match (skipped when skip_prebuilt=True)
    if not identity_config.skip_prebuilt:
        prebuilt = match_prebuilt(description)
        if prebuilt is not None:
            _log.debug("Identity resolved from prebuilt: %s", prebuilt.get("role"))
            return prebuilt

    # Layer 2: SQLite cache
    cached = _load_from_cache(description, store)
    if cached is not None:
        return cached

    # Layer 3: LLM derivation
    from pocket_mem.config import LLMConfig
    from pocket_mem.llm.client import LLMClient

    if identity_config.derivation_api_key:
        derive_llm_config = LLMConfig(
            base_url=identity_config.derivation_base_url,
            model=identity_config.derivation_model,
            api_key=identity_config.derivation_api_key,
        )
        derive_llm = LLMClient(derive_llm_config)
        model_name = identity_config.derivation_model
    else:
        derive_llm = llm
        model_name = llm.config.model

    derived = _derive_via_llm(description, derive_llm, model_name)
    if derived is not None:
        _save_to_cache(description, derived, model_name, store)
        return derived

    _log.warning(
        "Identity derivation failed for description: %.60s — falling back to generic mode",
        description,
    )
    return None
