from __future__ import annotations
import logging
import uuid
from datetime import datetime, timedelta, timezone

from pocket_mem.embedding import cosine_similarity, embed
from pocket_mem.llm.client import LLMClient
from pocket_mem.llm.prompts import QUESTION_GEN_SCHEMA, build_question_gen_system
from pocket_mem.models import Node
from pocket_mem.store.base import StoreInterface

_log = logging.getLogger(__name__)

_QA_CACHE_TTL_DAYS = 7
_CONFIDENCE_THRESHOLD = 0.65   # local models (qwen2.5:7b) return lower confidence scores
_COVERAGE_THRESHOLD = 0.40


def calculate_question_count(node_count: int, identity=None) -> int:
    base = min(node_count // 3, 15)
    if identity and identity.derived:
        multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        target = identity.derived.get("question_complexity_target", "medium")
        base = int(base * multipliers.get(target, 1.0))
    return max(3, min(base, 20))


def verify_qa_pair(
    question: str,
    answer: str,
    source_node_ids: list[str],
    confidence: float,
    store: StoreInterface,
) -> bool:
    """Return True only if the QA pair is grounded in the graph.

    Layer 1 — confidence floor.
    Layer 2 — source node IDs must resolve to real nodes (model didn't fabricate them).
    Layer 3 — answer terms must appear in those specific source nodes.
    """
    if confidence < _CONFIDENCE_THRESHOLD:
        return False

    # Resolve source node IDs — check they actually exist in the store
    real_source_nodes = [
        node for node_id in source_node_ids
        if (node := store.read_node(node_id)) is not None
    ]
    if not real_source_nodes:
        return False

    # Answer terms must appear in the content of the cited source nodes
    source_content = " ".join(
        n.data.get("raw", "") + " " + n.data.get("summary", "") + " " + n.label
        for n in real_source_nodes
    ).lower()

    meaningful_terms = [t for t in answer.lower().split() if len(t) > 3]
    if not meaningful_terms:
        return False

    hits = sum(1 for t in meaningful_terms if t in source_content)
    coverage = hits / len(meaningful_terms)

    return coverage >= _COVERAGE_THRESHOLD


def _write_qa_cache_node(
    question: str,
    answer: str,
    confidence: float,
    source_node_ids: list[str],
    tier: str,
    model_name: str,
    store: StoreInterface,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=_QA_CACHE_TTL_DAYS)).isoformat()
    node = Node(
        id=str(uuid.uuid4()),
        node_type="qa_cache",
        label=question,
        data={
            "answer": answer,
            "confidence": confidence,
            "source_node_ids": source_node_ids,
            "question_tier": tier,
            "verified": True,
            "expires_at": expires_at,
            "hit_count": 0,
            "last_hit_at": None,
            "generated_by_model": model_name,
        },
        embedding=embed(question),
        importance=confidence,
    )
    node.created_at = node.updated_at = now
    store.write_node(node)


def run_question_loop(
    store: StoreInterface,
    llm: LLMClient,
    identity=None,
) -> int:
    """Generate QA pairs from recent graph content and cache verified ones.

    Triggered after compaction. Returns count of verified pairs stored.
    """
    all_chunks = store.get_nodes_by_type("memory_chunk")
    episodic = [n for n in all_chunks if n.data.get("memory_tier") == "episodic"]
    # Fall back to working chunks when compaction has never run (e.g. small datasets)
    chunks = episodic if episodic else [
        n for n in all_chunks if not n.data.get("processed", False)
    ]
    entities = store.get_nodes_by_type("entity")

    # Episodic/working chunks first — they contain the narrative; entities add structure
    all_nodes = chunks + entities
    if not all_nodes:
        return 0

    count = calculate_question_count(len(all_nodes), identity)

    # Build compact context from nodes (limit to avoid token overflow)
    context_nodes = all_nodes[:50]
    node_context = "\n\n".join(
        f"[{n.id[:8]}] {n.node_type}: {n.label}\n"
        + (n.data.get("summary") or n.data.get("raw", "")[:300])
        for n in context_nodes
    )

    system_msg = build_question_gen_system(identity)
    user_msg = (
        f"Generate {count} questions from these memory nodes:\n\n"
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
        _log.warning("Question generation failed: %s", e)
        return 0

    questions = result if isinstance(result, list) else result.get("questions", [])
    if not questions:
        return 0

    # Map 8-char prefixes back to full node IDs
    prefix_to_id = {n.id[:8]: n.id for n in context_nodes}

    stored = 0
    for qa in questions:
        if not isinstance(qa, dict):
            continue

        question = qa.get("question", "")
        answer = qa.get("answer", "")
        confidence = float(qa.get("confidence", 0.0))
        tier = qa.get("question_tier", "T1")
        raw_source_ids = qa.get("source_node_ids", [])

        # Resolve abbreviated IDs to full IDs — strip brackets the model may copy
        # from the context format "[abc12345]"
        source_ids = [
            full_id
            for sid in raw_source_ids
            if sid
            and (full_id := prefix_to_id.get(sid.strip("[]")[:8]))
        ]

        if not question or not answer:
            continue

        if verify_qa_pair(question, answer, source_ids, confidence, store):
            _write_qa_cache_node(
                question, answer, confidence, source_ids, tier,
                llm.config.model, store,
            )
            stored += 1
            _log.debug(
                "QA cached [%s] conf=%.2f: %s", tier, confidence, question[:60]
            )
        else:
            _log.debug(
                "QA rejected [%s] conf=%.2f: %s", tier, confidence, question[:60]
            )

    if stored or len(questions) - stored:
        _log.info(
            "Question loop: %d generated, %d verified and cached, %d rejected",
            len(questions), stored, len(questions) - stored,
        )

    return stored
