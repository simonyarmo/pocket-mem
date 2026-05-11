"""
Question loop inspection simulation test.

Loads an existing memory database, manually triggers the self-questioning loop,
and prints every generated Q/A pair with verification status.

Run (local model only):
    pytest tests/simulation/ -v -s -k question_loop

Run (model comparison, requires GEMINI_API_KEY):
    GEMINI_API_KEY=your-key pytest tests/simulation/ -v -s -k question_loop_compare
"""
from __future__ import annotations

import os
import socket
import time
from datetime import datetime
from pathlib import Path

import pytest

from pocket_mem import LLMConfig, MemoryConfig
from pocket_mem.config import StorageConfig
from pocket_mem.questioner import calculate_question_count, run_question_loop, verify_qa_pair
from pocket_mem.store.local import SQLiteStore

SIM_DIR = Path(__file__).parent
VELORIS_DB = SIM_DIR.parent / "identity_sim_test" / "memory" / "veloris-benchmark.db"

# Persistent output directory — survives pytest, inspectable with pocket-mem show
IDENTITY_MEM_DIR = SIM_DIR.parent / "identity_memory_test" / "memory"
PROJECT = "veloris-benchmark"

RESULTS_DIR = SIM_DIR
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Set FORCE_COMPACTION=1 to compress working memory into episodic summaries first.
# Without it, the question loop runs directly on raw email chunks.
FORCE_COMPACTION = os.environ.get("FORCE_COMPACTION", "").strip() == "1"


def _ollama_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=2)
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


def _open_veloris_store(tmp_path: Path) -> SQLiteStore:
    """Copy benchmark DB to tmp dir (compare test only — discarded after run)."""
    import shutil
    if not VELORIS_DB.exists():
        pytest.skip(f"Benchmark DB not found: {VELORIS_DB}")
    dest = tmp_path / f"{PROJECT}.db"
    shutil.copy(VELORIS_DB, dest)
    return SQLiteStore(PROJECT, StorageConfig(path=str(tmp_path)))


def _open_persistent_store() -> SQLiteStore:
    """Copy benchmark DB to the persistent identity_memory_test directory.

    Fresh copy each run so qa_cache and compaction results don't accumulate
    across runs. Inspect the result with:
        pocket-mem show --path tests/simulation/identity_memory_test/memory
    """
    import shutil
    if not VELORIS_DB.exists():
        pytest.skip(f"Benchmark DB not found: {VELORIS_DB}")
    IDENTITY_MEM_DIR.mkdir(parents=True, exist_ok=True)
    dest = IDENTITY_MEM_DIR / f"{PROJECT}.db"
    shutil.copy(VELORIS_DB, dest)
    print(f"\n[setup] Copied benchmark DB → {dest}")
    return SQLiteStore(PROJECT, StorageConfig(path=str(IDENTITY_MEM_DIR)))


def _run_and_print(store: SQLiteStore, llm, model_name: str, identity=None) -> dict:
    """Run question loop and print detailed output. Returns stats dict."""
    from pocket_mem.llm.client import LLMClient
    from pocket_mem.questioner import (
        QUESTION_GEN_SCHEMA,
        _write_qa_cache_node,
        calculate_question_count,
        verify_qa_pair,
    )
    from pocket_mem.llm.prompts import build_question_gen_system
    from pocket_mem.embedding import embed

    all_chunks = store.get_nodes_by_type("memory_chunk")
    episodic = [n for n in all_chunks if n.data.get("memory_tier") == "episodic"]
    chunks = episodic if episodic else [
        n for n in all_chunks if not n.data.get("processed", False)
    ]
    entities = store.get_nodes_by_type("entity")
    all_nodes = chunks + entities

    node_count = len(all_nodes)
    q_count = calculate_question_count(node_count, identity)

    print(f"\n=== QUESTION LOOP INSPECTION ===")
    print(f"Database: {VELORIS_DB.parent}")
    chunk_label = f"{len(episodic)} episodic" if episodic else f"{len(chunks)} working"
    print(f"Nodes in graph: {node_count} ({chunk_label}, {len(entities)} entities)")
    print(f"Model: {model_name}")
    print(f"Questions to generate: {q_count}")

    # Generate questions
    context_nodes = all_nodes[:50]
    node_context = "\n\n".join(
        f"[{n.id[:8]}] {n.node_type}: {n.label}\n"
        + (n.data.get("summary") or n.data.get("raw", "")[:300])
        for n in context_nodes
    )

    system_msg = build_question_gen_system(identity)
    user_msg = (
        f"Generate {q_count} questions from these memory nodes:\n\n"
        f"{node_context}\n\n"
        f"Use the exact node ID prefixes shown in brackets as source_node_ids."
    )

    try:
        result = llm.complete_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            schema=QUESTION_GEN_SCHEMA,
        )
    except Exception as e:
        print(f"ERROR: Question generation failed: {e}")
        return {"generated": 0, "verified": 0, "rejected": 0, "tier_counts": {}}

    questions = result if isinstance(result, list) else result.get("questions", [])
    prefix_to_id = {n.id[:8]: n.id for n in context_nodes}

    print(f"\n{'─' * 65}")

    verified_count = 0
    rejected_count = 0
    tier_counts: dict[str, int] = {"T1": 0, "T2": 0, "T3": 0}
    rejection_reasons: list[str] = []

    for i, qa in enumerate(questions, 1):
        if not isinstance(qa, dict):
            continue

        question = qa.get("question", "")
        answer = qa.get("answer", "")
        confidence = float(qa.get("confidence", 0.0))
        tier = qa.get("question_tier", "T1")
        raw_source_ids = qa.get("source_node_ids", [])
        source_ids = [
            full_id
            for sid in raw_source_ids
            if sid and (full_id := prefix_to_id.get(sid.strip("[]")[:8]))
        ]

        # Verify
        if confidence < 0.65:
            verified = False
            reason = "confidence < 0.65"
        else:
            real_nodes = [n for sid in source_ids if (n := store.read_node(sid)) is not None]
            if not real_nodes:
                verified = False
                reason = "source node IDs not found in store"
            else:
                source_content = " ".join(
                    n.data.get("raw", "") + " " + n.data.get("summary", "") + " " + n.label
                    for n in real_nodes
                ).lower()
                terms = [t for t in answer.lower().split() if len(t) > 3]
                if not terms:
                    verified = False
                    reason = "answer has no meaningful terms"
                else:
                    hits = sum(1 for t in terms if t in source_content)
                    coverage = hits / len(terms)
                    verified = coverage >= 0.40
                    reason = f"answer term coverage {coverage:.0%} < 40%" if not verified else ""

        status = "YES" if verified else f"NO  ← {reason}"
        tier_label = tier if tier in tier_counts else "T1"
        tier_counts[tier_label] += 1

        print(f"Q{i:02d} [{tier}] confidence: {confidence:.2f}  verified: {status}")
        print(f"Question: {question}")
        print(f"Answer:   {answer[:200]}")
        print(f"Sources:  {', '.join(s[:8] for s in source_ids[:4])}")
        if not verified and reason:
            rejection_reasons.append(reason)
        print(f"{'─' * 65}")

        if verified:
            _write_qa_cache_node(
                question, answer, confidence, source_ids, tier, model_name, store
            )
            verified_count += 1
        else:
            rejected_count += 1

    print(f"\nSUMMARY")
    print(f"  Generated:  {len(questions)}")
    print(f"  Verified:    {verified_count}  (stored in cache)")
    print(f"  Rejected:    {rejected_count}  (low confidence or failed verification)")
    print(f"  Tier breakdown: T1: {tier_counts['T1']}  T2: {tier_counts['T2']}  T3: {tier_counts['T3']}")

    if rejection_reasons:
        from collections import Counter
        reason_counts = Counter(rejection_reasons)
        print(f"\n  Rejection reasons:")
        for reason, count in reason_counts.most_common():
            print(f"    - {reason}: {count}")

    return {
        "generated": len(questions),
        "verified": verified_count,
        "rejected": rejected_count,
        "tier_counts": tier_counts,
    }


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_question_loop_inspection():
    """Run the question loop against a persistent DB so you can inspect the results.

    The DB is saved to tests/simulation/identity_memory_test/memory/ and
    includes the qa_cache nodes written during this run. Inspect with:
        pocket-mem show --path tests/simulation/identity_memory_test/memory

    Flags:
        FORCE_COMPACTION=1   Compress working memory → episodic before running
                             the question loop. Produces richer, topic-grouped
                             summaries as the question context.
    """
    import shutil
    from pocket_mem.llm.client import LLMClient

    store = _open_persistent_store()
    llm = LLMClient(LLMConfig(base_url=OLLAMA_BASE, model=MODEL))

    if FORCE_COMPACTION:
        from pocket_mem.compactor import compress, prune
        from pocket_mem.config import MemoryConfig
        print("\n[compaction] Compressing working memory into episodic summaries...")
        # Patch: suppress internal question loop during compress so _run_and_print
        # is the single question-loop run and its output is fully visible.
        import pocket_mem.compactor as _compactor
        _orig = _compactor.run_question_loop
        _compactor.run_question_loop = lambda *a, **kw: 0
        created = compress(store, llm)
        _compactor.run_question_loop = _orig
        chunks_after = [
            n for n in store.get_nodes_by_type("memory_chunk")
            if n.data.get("memory_tier") == "episodic"
        ]
        print(f"[compaction] {created} episodic nodes created from working chunks")
        print(f"[compaction] Episodic nodes in store: {len(chunks_after)}")

    stats = _run_and_print(store, llm, MODEL)
    store.close()

    mode = "with compaction" if FORCE_COMPACTION else "without compaction (working chunks)"
    print(f"\n{'─' * 65}")
    print(f"Run mode  : {mode}")
    print(f"DB saved  : {IDENTITY_MEM_DIR / (PROJECT + '.db')}")
    print(f"Inspect   : pocket-mem show --path {IDENTITY_MEM_DIR}")

    assert stats["generated"] >= 0, "Question generation should not error"


@pytest.mark.skipif(
    not _ollama_available() or not GEMINI_API_KEY,
    reason="Ollama not running or GEMINI_API_KEY not set",
)
def test_question_loop_compare(tmp_path):
    """Side-by-side comparison of local model vs Gemini question quality."""
    from pocket_mem.llm.client import LLMClient

    # Local model run
    store_local = _open_veloris_store(tmp_path / "local")
    llm_local = LLMClient(LLMConfig(base_url=OLLAMA_BASE, model=MODEL))
    local_stats = _run_and_print(store_local, llm_local, MODEL)
    store_local.close()

    # Gemini run (fresh copy of DB)
    store_gemini = _open_veloris_store(tmp_path / "gemini")
    llm_gemini = LLMClient(LLMConfig(
        base_url=GEMINI_BASE,
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
    ))
    gemini_stats = _run_and_print(store_gemini, llm_gemini, GEMINI_MODEL)
    store_gemini.close()

    # Print comparison
    print(f"\n\n=== MODEL COMPARISON ===")
    print(f"\n{'LOCAL MODEL (' + MODEL + ')':<40}{'GEMINI 2.5 FLASH'}")
    print("─" * 70)

    def row(label, local_val, gemini_val):
        print(f"{label + ': ' + str(local_val):<40}{label + ': ' + str(gemini_val)}")

    row("Generated", local_stats["generated"], gemini_stats["generated"])
    row(
        "Verified",
        f"{local_stats['verified']}  ({local_stats['verified']/max(local_stats['generated'],1):.0%})",
        f"{gemini_stats['verified']}  ({gemini_stats['verified']/max(gemini_stats['generated'],1):.0%})",
    )
    for tier in ["T1", "T2", "T3"]:
        row(f"{tier} questions", local_stats["tier_counts"].get(tier, 0), gemini_stats["tier_counts"].get(tier, 0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"question_loop_comparison_{ts}.txt"
    print(f"\nComparison complete.")
