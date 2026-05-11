from __future__ import annotations
import uuid
from datetime import datetime, timedelta, timezone

from pocket_mem.config import MemoryConfig
from pocket_mem.llm.client import LLMClient
from pocket_mem.llm.prompts import SUMMARIZE_GROUP_SCHEMA, build_compact_prompt
from pocket_mem.models import Edge, MemoryChunk, Node
from pocket_mem.questioner import run_question_loop
from pocket_mem.store.base import StoreInterface


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_topic_for_chunk(chunk: Node, store: StoreInterface) -> str | None:
    """Return topic_id for a chunk by traversing derived_from edges to entities."""
    for edge in store.get_edges(chunk.id):
        if edge.from_id == chunk.id and edge.relation == "derived_from":
            entity = store.read_node(edge.to_id)
            if entity and entity.node_type == "entity":
                topic_id = entity.data.get("topic_id")
                if topic_id:
                    return topic_id
    return None


def _get_topic_label(topic_id: str, store: StoreInterface) -> str | None:
    """Return label of topic node, or None if not found."""
    node = store.read_node(topic_id)
    return node.label if node else None


def _group_by_topic(
    chunks: list[Node], store: StoreInterface
) -> dict[str | None, list[Node]]:
    """Group chunk nodes by topic_id. Chunks with no topic go under key None."""
    groups: dict[str | None, list[Node]] = {}
    for chunk in chunks:
        topic_id = _find_topic_for_chunk(chunk, store)
        groups.setdefault(topic_id, []).append(chunk)
    return groups


def _summarize_group(
    chunks: list[Node], topic_label: str, llm: LLMClient, identity=None
) -> str:
    """Call LLM to summarise a group of chunks. Returns summary string."""
    combined = "\n\n---\n\n".join(
        n.data.get("raw") or n.data.get("summary", "") for n in chunks
    )
    prompt = build_compact_prompt(topic_label, combined, identity)
    result = llm.complete_json(
        [{"role": "user", "content": prompt}],
        schema=SUMMARIZE_GROUP_SCHEMA,
    )
    return result.get("summary", combined[:200])


def compress(store: StoreInterface, llm: LLMClient, identity=None) -> int:
    """Compress unprocessed working chunks into topic-grouped episodic summaries.

    Steps:
    1. Fetch all unprocessed working memory_chunk nodes
    2. Group by topic via entity edges
    3. LLM-summarise each group → write one episodic MemoryChunk per group
    4. Mark original chunks processed=True (never deleted)

    Returns number of episodic chunks created.
    """
    now = _now()

    all_chunks = store.get_nodes_by_type("memory_chunk")
    working = [
        n for n in all_chunks
        if n.data.get("memory_tier") == "working"
        and not n.data.get("processed", False)
    ]
    if not working:
        return 0

    groups = _group_by_topic(working, store)
    created = 0

    for topic_id, chunks in groups.items():
        topic_label = (
            _get_topic_label(topic_id, store) if topic_id else "General"
        ) or "General"

        summary = _summarize_group(chunks, topic_label, llm, identity)

        # Write episodic chunk
        episodic = MemoryChunk(
            id=str(uuid.uuid4()),
            label=f"Summary: {topic_label}",
            raw="\n\n".join(n.data.get("raw", "") for n in chunks),
            summary=summary,
            source="compaction",
            memory_tier="episodic",
        )
        episodic_node = episodic.to_node()
        episodic_node.created_at = episodic_node.updated_at = now
        store.write_node(episodic_node)

        # Link episodic chunk to topic node
        if topic_id:
            store.write_edge(Edge(
                id=str(uuid.uuid4()),
                from_id=episodic_node.id,
                to_id=topic_id,
                relation="derived_from",
                created_at=now,
            ))

        # Mark originals processed
        for chunk in chunks:
            chunk.data["processed"] = True
            chunk.updated_at = now
            chunk.embedding = None  # re-embed with updated data
            store.write_node(chunk)

        created += 1

    if created:
        run_question_loop(store, llm, identity)

    return created


def prune(store: StoreInterface, config: MemoryConfig) -> int:
    """Delete old low-importance episodic chunks.

    Criteria (ALL must be true to prune):
    - memory_tier == "episodic"
    - importance < config.importance_prune_threshold
    - created_at older than config.prune_after_days days

    Edges connected to pruned nodes are invalidated before deletion.
    Returns number of nodes deleted.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=config.prune_after_days)
    ).isoformat()
    now = _now()

    all_chunks = store.get_nodes_by_type("memory_chunk")
    count = 0
    for node in all_chunks:
        if (
            node.data.get("memory_tier") == "episodic"
            and node.importance < config.importance_prune_threshold
            and node.created_at < cutoff
        ):
            for edge in store.get_edges(node.id):
                edge.invalid_at = now
                store.write_edge(edge)
            store.delete_node(node.id)
            count += 1
    return count
