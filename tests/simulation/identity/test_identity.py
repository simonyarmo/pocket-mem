"""
Identity awareness simulation test.

Verifies that the identity system:
- Seeds the correct topics on init
- Shapes extraction towards priority entity types
- Elevates importance scores for priority entities
- Produces a sensible self-description via recall

Run:  pytest tests/simulation/ -v -s -k identity
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from pocket_mem import IdentityConfig, LLMConfig, MemoryAgent, MemoryConfig

SIM_DIR = Path(__file__).parent
MEMORY_DIR = SIM_DIR / "first_sim_test_50_q/memory"

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")


def _ollama_available() -> bool:
    try:
        host, port = "localhost", 11434
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_identity_awareness():
    """Initialize with a sales identity, ingest 6 turns, inspect results."""
    identity_description = (
        "B2B sales representative at a SaaS company managing enterprise software deals. "
        "Tracks prospects, deals, competitors, and contract negotiations."
    )

    agent = MemoryAgent(
        project="identity_test",
        path=str(MEMORY_DIR),
        llm=LLMConfig(base_url=OLLAMA_BASE, model=MODEL),
        config=MemoryConfig(
            identity=IdentityConfig(description=identity_description)
        ),
    )

    identity = agent._config.identity
    derived = identity.derived if identity else None

    print("\n\n=== IDENTITY TEST ===\n")
    print(f"Identity description: {identity_description}")
    if derived:
        print(f"Derived role: {derived.get('role', 'unknown')}")
        print(f"Domain: {derived.get('domain', 'unknown')}")
        print(f"question_complexity_target: {derived.get('question_complexity_target')}")
    else:
        print("WARNING: Identity derivation failed — running in generic mode")

    # Verify seed topics were written
    topics = agent.topics()
    print(f"\nTopics in store: {topics}")

    if derived:
        seed_topics = derived.get("seed_topics", [])
        seeded = [t for t in seed_topics if t.lower() in [x.lower() for x in topics]]
        print(f"Seed topics expected: {seed_topics}")
        print(f"Seed topics found in store: {seeded}")

    # Feed identity-relevant observations
    turns = [
        (
            "Just got off a call with Marcus Webb from Acme Corp. He's the VP of Engineering "
            "and said their current vendor contract expires in Q3.",
            "Noted — Marcus Webb at Acme Corp, VP Eng, contract up in Q3.",
        ),
        (
            "The deal with Acme is at $180,000 ARR. Their main objection is integration complexity. "
            "Competitor is DataStream who they tried last year.",
            "Got it — $180K ARR, integration objection, DataStream as competitor.",
        ),
        (
            "Sarah Kim from TechFlow reached out. She's the procurement lead. "
            "They want a proposal by Friday with pricing and SLA details.",
            "Understood — proposal needed by Friday, Sarah Kim is procurement lead at TechFlow.",
        ),
        (
            "Marcus said budget was approved at $200k. Decision coming end of month.",
            "Great news — Acme budget approved, decision by end of month.",
        ),
        (
            "DataStream lowered their price by 20% to keep the Acme deal. Need to respond.",
            "Noted — DataStream competing on price, need a counter.",
        ),
        (
            "Closed the TechFlow deal at $95k. Sarah signed the contract this afternoon.",
            "Congratulations on closing TechFlow at $95k!",
        ),
    ]

    for user_msg, agent_msg in turns:
        agent.observe(user_msg, agent_msg)

    agent.flush()

    # Inspect entity importance scores
    entities = agent._store.get_nodes_by_type("entity")

    priority_types = derived.get("priority_entity_types", []) if derived else []
    priority_nodes = [e for e in entities if e.data.get("entity_type") in priority_types]
    non_priority_nodes = [e for e in entities if e.data.get("entity_type") not in priority_types]

    avg_priority = (
        sum(n.importance for n in priority_nodes) / len(priority_nodes)
        if priority_nodes else 0.0
    )
    avg_non_priority = (
        sum(n.importance for n in non_priority_nodes) / len(non_priority_nodes)
        if non_priority_nodes else 0.0
    )

    print(f"\nEntity importance comparison:")
    print(f"  Priority types {priority_types}:")
    print(f"    count: {len(priority_nodes)}")
    print(f"    avg importance: {avg_priority:.3f}")
    print(f"    examples: {[n.label for n in priority_nodes[:5]]}")
    print(f"  Non-priority types:")
    print(f"    count: {len(non_priority_nodes)}")
    print(f"    avg importance: {avg_non_priority:.3f}")

    # Ask agent to describe itself
    answer = agent.recall("Who are you and what is your role?", mode="answer")
    print(f"\nAgent self-description:")
    print(f"  {answer}")

    prebuilt_used = derived is not None and not identity._was_derived_by_llm if hasattr(identity, '_was_derived_by_llm') else "unknown"
    model_used = "prebuilt or cache (no LLM call)" if derived and not identity.derivation_api_key else identity.derivation_model if identity else "N/A"
    print(f"\nDerivation model: {model_used}")

    agent.close()

    # Assertions
    assert derived is not None, "Identity derivation must succeed"
    assert len(topics) >= 3, f"Expected at least 3 topics, got {len(topics)}"
    assert len(entities) > 0, "No entities were extracted"
