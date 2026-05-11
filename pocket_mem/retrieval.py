from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from pocket_mem.embedding import cosine_similarity, embed
from pocket_mem.llm.client import LLMClient
from pocket_mem.llm.prompts import ANSWER, ANSWER_SYSTEM
from pocket_mem.models import Edge, Node
from pocket_mem.store.base import StoreInterface


def search(query: str, store: StoreInterface, limit: int = 10) -> list[Node]:
    """Hybrid BM25+vector search. Embeds query and calls search_hybrid()."""
    embedding = embed(query)
    return store.search_hybrid(query=query, embedding=embedding, limit=limit)


def traverse(
    seed_nodes: list[Node],
    store: StoreInterface,
    hops: int = 2,
) -> list[Node]:
    """Walk the graph up to `hops` edges from seed_nodes, following valid edges only.

    Returns deduplicated list of all nodes reachable (including seeds).
    """
    visited_ids: set[str] = {n.id for n in seed_nodes}
    frontier: list[Node] = list(seed_nodes)
    all_nodes: list[Node] = list(seed_nodes)

    for _ in range(hops):
        next_frontier: list[Node] = []
        for node in frontier:
            for edge in store.get_edges(node.id):
                neighbor_id = (
                    edge.to_id if edge.from_id == node.id else edge.from_id
                )
                if neighbor_id not in visited_ids:
                    neighbor = store.read_node(neighbor_id)
                    if neighbor is not None:
                        visited_ids.add(neighbor_id)
                        next_frontier.append(neighbor)
                        all_nodes.append(neighbor)
        frontier = next_frontier

    return all_nodes


def format_context(nodes: list[Node], edges: list[Edge] | None = None) -> str:
    """Render nodes (and optional edges) into a readable text block for prompt injection."""
    lines: list[str] = []
    included_ids: set[str] = set()

    for node in nodes:
        if node.node_type == "entity":
            entity_type = node.data.get("entity_type", "")
            attrs = node.data.get("attributes", {})
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            line = f"[entity/{entity_type}] {node.label}"
            if attr_str:
                line += f" — {attr_str}"
        elif node.node_type == "memory_chunk":
            text = node.data.get("summary") or node.data.get("raw", "")
            line = f"[memory] {text}"
        elif node.node_type == "tone":
            tone_type = node.data.get("tone_type", "")
            valence = node.data.get("valence", "")
            context = node.data.get("context", "")
            line = f"[tone] {node.label} ({tone_type}, {valence})"
            if context:
                line += f' — "{context}"'
        elif node.node_type == "topic":
            line = f"[topic] {node.label}"
        else:
            line = f"[{node.node_type}] {node.label}"
        lines.append(line)
        included_ids.add(node.id)

    if edges:
        node_labels: dict[str, str] = {n.id: n.label for n in nodes}
        rel_lines: list[str] = []
        for edge in edges:
            if edge.from_id not in included_ids and edge.to_id not in included_ids:
                continue
            from_label = node_labels.get(edge.from_id, edge.from_id)
            to_label = node_labels.get(edge.to_id, edge.to_id)
            rel = f"  {from_label} -[{edge.relation}]-> {to_label}"
            if edge.context:
                rel += f" ({edge.context})"
            rel_lines.append(rel)
        if rel_lines:
            lines.append("Relations:")
            lines.extend(rel_lines)

    return "\n".join(lines)


def synthesize_answer(query: str, context: str, llm: LLMClient) -> str:
    """Ask the LLM to answer `query` using `context`. Returns plain-text answer."""
    prompt = ANSWER.format(query=query, context=context)
    return llm.complete(
        [{"role": "user", "content": prompt}],
        system=ANSWER_SYSTEM,
        model=llm.config.answer_model,
    )


def check_qa_cache(query: str, store: StoreInterface) -> str | None:
    """Return cached answer if a verified, non-expired QA pair matches query closely enough."""
    q_embedding = embed(query)
    cache_nodes = store.get_nodes_by_type("qa_cache")
    best_score = 0.0
    best_node: Node | None = None

    for node in cache_nodes:
        expires = node.data.get("expires_at")
        if expires and datetime.fromisoformat(expires) < datetime.now(timezone.utc):
            continue
        if not node.data.get("verified"):
            continue
        if node.embedding is None:
            continue
        score = cosine_similarity(q_embedding, node.embedding)
        if score > best_score:
            best_score = score
            best_node = node

    if best_score >= 0.88 and best_node is not None:
        store.increment_cache_hit(best_node.id)
        return best_node.data["answer"]

    return None


def recall(
    query: str,
    store: StoreInterface,
    llm: LLMClient | None = None,
    mode: str = "context",
    limit: int = 10,
    hops: int = 2,
) -> list[dict] | str:
    """Entry point for memory retrieval.

    Modes:
      "raw"       → list[dict]  — raw nodes as plain dicts (JSON-serializable, embedding excluded)
      "context"   → str         — formatted context string (full retrieved nodes)
      "answer"    → str         — LLM synthesizes an answer from retrieved context (requires llm=)
    """
    # Cache check for context and answer modes (raw always queries live graph)
    if mode != "raw":
        cached = check_qa_cache(query, store)
        if cached is not None:
            return cached

    nodes = search(query, store, limit=limit)
    nodes = traverse(nodes, store, hops=hops)

    if mode == "raw":
        # Return plain dicts (embedding excluded — binary, not useful to callers)
        return [
            {k: v for k, v in dataclasses.asdict(n).items() if k != "embedding"}
            for n in nodes
        ]

    # Collect entity-to-entity edges between retrieved nodes for context rendering
    node_ids = {n.id for n in nodes}
    relation_edges: list[Edge] = []
    seen_edge_ids: set[str] = set()
    for node in nodes:
        if node.node_type == "entity":
            for edge in store.get_edges(node.id):
                if (
                    edge.id not in seen_edge_ids
                    and edge.from_id in node_ids
                    and edge.to_id in node_ids
                    and edge.relation not in ("derived_from", "belongs_to", "expresses")
                ):
                    relation_edges.append(edge)
                    seen_edge_ids.add(edge.id)

    context = format_context(nodes, edges=relation_edges if relation_edges else None)

    if mode == "context":
        return context

    if mode == "answer":
        if llm is None:
            raise ValueError("mode='answer' requires an LLMClient (llm= argument)")
        return synthesize_answer(query, context, llm)

    raise ValueError(f"Unknown mode {mode!r}. Must be 'raw', 'context', or 'answer'.")


def as_tool() -> dict:
    """Return an OpenAI-compatible function schema for the recall tool (Pattern B)."""
    return {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search memory for information relevant to a query. "
                "Returns formatted context from the stored knowledge graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search phrase to look up in memory",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["raw", "context", "answer"],
                        "description": (
                            "'context' returns formatted text (default), "
                            "'answer' synthesizes an LLM response"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of nodes to retrieve (default 10)",
                    },
                },
                "required": ["query"],
            },
        },
    }
