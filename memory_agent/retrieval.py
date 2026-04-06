from __future__ import annotations

from memory_agent.embedding import embed
from memory_agent.llm.client import LLMClient
from memory_agent.llm.prompts import ANSWER
from memory_agent.models import Edge, Node
from memory_agent.store.base import StoreInterface


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

    for node in nodes:
        if node.node_type == "entity":
            entity_type = node.data.get("entity_type", "")
            attrs = node.data.get("attributes", {})
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            line = f"[entity/{entity_type}] {node.label}"
            if attr_str:
                line += f" — {attr_str}"
        elif node.node_type == "memory_chunk":
            text = node.data.get("summary") or node.data.get("raw", "")[:200]
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

    if edges:
        node_labels: dict[str, str] = {n.id: n.label for n in nodes}
        rel_lines: list[str] = []
        for edge in edges:
            from_label = node_labels.get(edge.from_id, edge.from_id)
            to_label = node_labels.get(edge.to_id, edge.to_id)
            rel_lines.append(f"  {from_label} -[{edge.relation}]-> {to_label}")
        if rel_lines:
            lines.append("Relations:")
            lines.extend(rel_lines)

    return "\n".join(lines)


def synthesize_answer(query: str, context: str, llm: LLMClient) -> str:
    """Ask the LLM to answer `query` using `context`. Returns plain-text answer."""
    prompt = ANSWER.format(query=query, context=context)
    return llm.complete([{"role": "user", "content": prompt}])


def recall(
    query: str,
    store: StoreInterface,
    llm: LLMClient | None = None,
    mode: str = "context",
    limit: int = 10,
    hops: int = 2,
) -> list[Node] | str:
    """Entry point for memory retrieval.

    Modes:
      "raw"     → list[Node]  — raw nodes, no formatting
      "context" → str         — formatted context string
      "answer"  → str         — LLM-synthesized answer (requires llm)
    """
    nodes = search(query, store, limit=limit)
    nodes = traverse(nodes, store, hops=hops)

    if mode == "raw":
        return nodes

    context = format_context(nodes)

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
