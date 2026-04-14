import pytest
from unittest.mock import MagicMock
from pocket_mem.config import LLMConfig, StorageConfig
from pocket_mem.store.local import SQLiteStore
from pocket_mem.llm.client import LLMClient
from pocket_mem.models import Entity, MemoryChunk, Tone, Edge
from pocket_mem.retrieval import (
    search, traverse, format_context, synthesize_answer, recall, as_tool,
)


@pytest.fixture
def tmp_store(tmp_path):
    store = SQLiteStore("test", StorageConfig(path=str(tmp_path)))
    yield store
    store.close()


def _write_entity(store, id, label, entity_type, attributes=None):
    entity = Entity(id=id, label=label, entity_type=entity_type,
                    attributes=attributes or {})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-05T00:00:00"
    store.write_node(node)
    return node


def _write_edge(store, id, from_id, to_id, relation, weight=1.0):
    edge = Edge(id=id, from_id=from_id, to_id=to_id,
                relation=relation, weight=weight,
                created_at="2026-04-05T00:00:00")
    store.write_edge(edge)
    return edge


# ── search() ─────────────────────────────────────────────────────────────────

def test_search_returns_nodes(tmp_store):
    _write_entity(tmp_store, "n1", "David", "person", {"role": "boss"})
    results = search("David", tmp_store)
    assert any(n.id == "n1" for n in results)


def test_search_semantic_boss(tmp_store):
    """Canonical: 'my boss' finds David without keyword match."""
    _write_entity(tmp_store, "n1", "David", "person", {"role": "boss"})
    _write_entity(tmp_store, "n2", "httpx", "tool", {})
    results = search("my boss", tmp_store, limit=5)
    ids = [n.id for n in results]
    assert "n1" in ids  # David found via vector


def test_search_respects_limit(tmp_store):
    for i in range(5):
        _write_entity(tmp_store, f"n{i}", f"Entity {i}", "concept")
    results = search("entity", tmp_store, limit=2)
    assert len(results) <= 2


def test_search_returns_list(tmp_store):
    results = search("anything", tmp_store)
    assert isinstance(results, list)


# ── traverse() ───────────────────────────────────────────────────────────────

def test_traverse_returns_seed_when_no_edges(tmp_store):
    node = _write_entity(tmp_store, "n1", "David", "person")
    results = traverse([node], tmp_store, hops=2)
    assert any(n.id == "n1" for n in results)


def test_traverse_follows_one_hop(tmp_store):
    n1 = _write_entity(tmp_store, "n1", "David", "person")
    n2 = _write_entity(tmp_store, "n2", "httpx", "tool")
    _write_edge(tmp_store, "e1", "n1", "n2", "recommended")
    results = traverse([n1], tmp_store, hops=1)
    ids = [n.id for n in results]
    assert "n1" in ids
    assert "n2" in ids


def test_traverse_follows_two_hops(tmp_store):
    """Multi-hop: chunk → David → httpx chain."""
    chunk = MemoryChunk(id="c1", label="turn", raw="David recommended httpx")
    chunk_node = chunk.to_node()
    chunk_node.created_at = chunk_node.updated_at = "2026-04-05T00:00:00"
    tmp_store.write_node(chunk_node)

    david = _write_entity(tmp_store, "n1", "David", "person")
    httpx = _write_entity(tmp_store, "n2", "httpx", "tool")

    _write_edge(tmp_store, "e1", "c1", "n1", "derived_from")
    _write_edge(tmp_store, "e2", "c1", "n2", "derived_from")
    _write_edge(tmp_store, "e3", "n1", "n2", "recommended")

    # Start from httpx, traverse 2 hops — should reach chunk and David
    results = traverse([httpx], tmp_store, hops=2)
    ids = [n.id for n in results]
    assert "n1" in ids  # David reachable
    assert "c1" in ids  # chunk reachable


def test_traverse_no_duplicate_nodes(tmp_store):
    n1 = _write_entity(tmp_store, "n1", "A", "concept")
    n2 = _write_entity(tmp_store, "n2", "B", "concept")
    _write_edge(tmp_store, "e1", "n1", "n2", "related_to")
    _write_edge(tmp_store, "e2", "n2", "n1", "related_to")  # cycle
    results = traverse([n1], tmp_store, hops=3)
    ids = [n.id for n in results]
    assert ids.count("n1") == 1
    assert ids.count("n2") == 1


def test_traverse_excludes_invalid_edges(tmp_store):
    n1 = _write_entity(tmp_store, "n1", "A", "concept")
    n2 = _write_entity(tmp_store, "n2", "B", "concept")
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="related_to",
                invalid_at="2020-01-01T00:00:00", created_at="2026-04-05T00:00:00")
    tmp_store.write_edge(edge)
    results = traverse([n1], tmp_store, hops=1)
    ids = [n.id for n in results]
    assert "n2" not in ids


# ── format_context() ─────────────────────────────────────────────────────────

def test_format_context_includes_entity_label(tmp_store):
    node = _write_entity(tmp_store, "n1", "David", "person", {"role": "boss"})
    ctx = format_context([node])
    assert "David" in ctx


def test_format_context_includes_entity_type(tmp_store):
    node = _write_entity(tmp_store, "n1", "David", "person", {})
    ctx = format_context([node])
    assert "person" in ctx or "entity" in ctx


def test_format_context_includes_attributes(tmp_store):
    node = _write_entity(tmp_store, "n1", "David", "person", {"role": "boss"})
    ctx = format_context([node])
    assert "boss" in ctx


def test_format_context_handles_memory_chunk():
    chunk = MemoryChunk(id="c1", label="turn", raw="David is my boss",
                        summary="David is the boss")
    chunk_node = chunk.to_node()
    chunk_node.created_at = chunk_node.updated_at = "2026-04-05T00:00:00"
    ctx = format_context([chunk_node])
    assert "David" in ctx


def test_format_context_handles_tone():
    tone = Tone(id="t1", label="Frustration", tone_type="frustration",
                intensity=0.8, valence="negative", context="ugh this breaks")
    tone_node = tone.to_node()
    tone_node.created_at = tone_node.updated_at = "2026-04-05T00:00:00"
    ctx = format_context([tone_node])
    assert "Frustration" in ctx or "frustration" in ctx


def test_format_context_returns_string(tmp_store):
    node = _write_entity(tmp_store, "n1", "David", "person")
    ctx = format_context([node])
    assert isinstance(ctx, str)
    assert len(ctx) > 0


def test_format_context_empty_nodes():
    ctx = format_context([])
    assert isinstance(ctx, str)


def test_format_context_includes_edge_relations(tmp_store):
    n1 = _write_entity(tmp_store, "n1", "David", "person")
    n2 = _write_entity(tmp_store, "n2", "httpx", "tool")
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="recommended",
                created_at="2026-04-05T00:00:00")
    ctx = format_context([n1, n2], edges=[edge])
    assert "recommended" in ctx


# ── synthesize_answer() ──────────────────────────────────────────────────────

def test_synthesize_answer_calls_llm():
    llm = MagicMock(spec=LLMClient)
    llm.config = LLMConfig()
    llm.complete.return_value = "David is your boss."
    result = synthesize_answer("Who is my boss?", "David — role: boss", llm)
    assert result == "David is your boss."
    assert llm.complete.called


def test_synthesize_answer_includes_query_in_prompt():
    llm = MagicMock(spec=LLMClient)
    llm.config = LLMConfig()
    llm.complete.return_value = "answer"
    synthesize_answer("Who is my boss?", "context text", llm)
    messages = llm.complete.call_args[0][0]
    assert "Who is my boss?" in messages[0]["content"]


def test_synthesize_answer_includes_context_in_prompt():
    llm = MagicMock(spec=LLMClient)
    llm.config = LLMConfig()
    llm.complete.return_value = "answer"
    synthesize_answer("query", "David is the boss", llm)
    messages = llm.complete.call_args[0][0]
    assert "David is the boss" in messages[0]["content"]


# ── recall() ─────────────────────────────────────────────────────────────────

def test_recall_raw_returns_list(tmp_store):
    _write_entity(tmp_store, "n1", "David", "person")
    result = recall("David", tmp_store, mode="raw")
    assert isinstance(result, list)


def test_recall_context_returns_string(tmp_store):
    _write_entity(tmp_store, "n1", "David", "person", {"role": "boss"})
    result = recall("David", tmp_store, mode="context")
    assert isinstance(result, str)
    assert "David" in result



def test_recall_answer_calls_llm(tmp_store):
    llm = MagicMock()
    llm.complete.return_value = "Marcus Webb is the CTO."
    result = recall("job title", tmp_store, llm=llm, mode="answer")
    assert result == "Marcus Webb is the CTO."
    llm.complete.assert_called_once()


def test_recall_answer_requires_llm(tmp_store):
    with pytest.raises(ValueError, match="llm"):
        recall("job title", tmp_store, llm=None, mode="answer")


# ── as_tool() ────────────────────────────────────────────────────────────────

def test_as_tool_returns_dict():
    tool = as_tool()
    assert isinstance(tool, dict)


def test_as_tool_has_function_type():
    tool = as_tool()
    assert tool["type"] == "function"


def test_as_tool_has_name():
    tool = as_tool()
    assert tool["function"]["name"] == "recall"


def test_as_tool_has_query_parameter():
    tool = as_tool()
    params = tool["function"]["parameters"]["properties"]
    assert "query" in params


def test_as_tool_query_is_required():
    tool = as_tool()
    assert "query" in tool["function"]["parameters"]["required"]
