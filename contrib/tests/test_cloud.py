"""Unit tests for SupabaseAdapter — all Supabase calls mocked, no network."""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from memory_agent.models import Node, Edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(n: int = 4) -> bytes:
    return struct.pack(f"{n}f", *([0.1] * n))


def _make_node(**kwargs) -> Node:
    defaults = dict(
        id="n1", node_type="entity", label="David",
        data={"entity_type": "person", "attributes": {"role": "boss"}},
        importance=0.8,
        created_at="2026-04-08T00:00:00",
        updated_at="2026-04-08T00:00:00",
    )
    defaults.update(kwargs)
    return Node(**defaults)


def _make_edge(**kwargs) -> Edge:
    defaults = dict(
        id="e1", from_id="n1", to_id="n2", relation="knows",
        weight=0.9, source_chunk_id=None,
        created_at="2026-04-08T00:00:00",
        valid_at=None, invalid_at=None,
    )
    defaults.update(kwargs)
    return Edge(**defaults)


def _mock_response(data: list) -> MagicMock:
    resp = MagicMock()
    resp.data = data
    return resp


# ---------------------------------------------------------------------------
# Fixture: SupabaseAdapter with fully mocked supabase client
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    with patch("memory_agent.store.cloud._load_credentials",
               return_value=("https://fake.supabase.co", "fake-key")):
        with patch("supabase.create_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            from memory_agent.store.cloud import SupabaseAdapter
            a = SupabaseAdapter("test-project", user_id="alice")
            a._mock_client = mock_client
            return a


def _chain(adapter, data):
    """Make the fluent builder chain on adapter._client.table() return data."""
    builder = MagicMock()
    builder.select.return_value = builder
    builder.upsert.return_value = builder
    builder.delete.return_value = builder
    builder.eq.return_value = builder
    builder.not_.return_value = builder
    builder.is_.return_value = builder
    builder.ilike.return_value = builder
    builder.or_.return_value = builder
    builder.in_.return_value = builder
    builder.limit.return_value = builder
    builder.execute.return_value = _mock_response(data)
    adapter._mock_client.table.return_value = builder
    return builder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_node_calls_upsert(adapter):
    builder = _chain(adapter, [])
    node = _make_node()
    node.embedding = _make_embedding()
    adapter.write_node(node)
    builder.upsert.assert_called_once()
    record = builder.upsert.call_args[0][0]
    assert record["id"] == "n1"
    assert record["project"] == "test-project"
    assert record["user_id"] == "alice"
    assert isinstance(record["embedding"], list)


def test_write_node_embeds_if_missing(adapter):
    builder = _chain(adapter, [])
    node = _make_node()
    assert node.embedding is None
    with patch("memory_agent.store.cloud.embed", return_value=_make_embedding()):
        adapter.write_node(node)
    assert node.embedding is not None


def test_write_edge_calls_upsert(adapter):
    builder = _chain(adapter, [])
    edge = _make_edge()
    adapter.write_edge(edge)
    builder.upsert.assert_called_once()
    record = builder.upsert.call_args[0][0]
    assert record["id"] == "e1"
    assert record["project"] == "test-project"


def test_read_node_returns_none_when_missing(adapter):
    _chain(adapter, [])
    result = adapter.read_node("nonexistent")
    assert result is None


def test_read_node_returns_node(adapter):
    emb_list = [0.1, 0.2, 0.3, 0.4]
    row = {
        "id": "n1", "node_type": "entity", "label": "David",
        "data": {"entity_type": "person", "attributes": {}},
        "embedding": emb_list, "importance": 0.8,
        "created_at": "2026-04-08T00:00:00",
        "updated_at": "2026-04-08T00:00:00",
    }
    _chain(adapter, [row])
    node = adapter.read_node("n1")
    assert node is not None
    assert node.label == "David"
    assert node.embedding is not None


def test_delete_node(adapter):
    builder = _chain(adapter, [])
    adapter.delete_node("n1")
    builder.delete.assert_called_once()


def test_list_topics(adapter):
    _chain(adapter, [{"label": "Work"}, {"label": "People"}])
    topics = adapter.list_topics()
    assert topics == ["Work", "People"]


def test_get_edges_deduplicates(adapter):
    edge_row = {
        "id": "e1", "from_id": "n1", "to_id": "n2",
        "relation": "knows", "weight": 0.9,
        "source_chunk_id": None, "created_at": "2026-04-08T00:00:00",
        "valid_at": None, "invalid_at": None,
    }
    # Both from and to queries return the same edge
    builder = MagicMock()
    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.is_.return_value = builder
    builder.execute.return_value = _mock_response([edge_row])
    adapter._mock_client.table.return_value = builder

    edges = adapter.get_edges("n1")
    assert len(edges) == 1
    assert edges[0].id == "e1"


def test_stats_returns_dict(adapter):
    _chain(adapter, [
        {"node_type": "entity"},
        {"node_type": "entity"},
        {"node_type": "memory_chunk"},
        {"node_type": "topic"},
    ])
    # Second call (for edges) needs to return something
    builder = MagicMock()
    builder.select.return_value = builder
    builder.eq.return_value = builder
    builder.is_.return_value = builder
    call_count = 0

    def execute_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response([
                {"node_type": "entity"},
                {"node_type": "entity"},
                {"node_type": "memory_chunk"},
                {"node_type": "topic"},
            ])
        return _mock_response([{"id": "e1"}, {"id": "e2"}])

    builder.execute.side_effect = execute_side_effect
    adapter._mock_client.table.return_value = builder

    s = adapter.stats()
    assert s["node_count"] == 4
    assert s["entity_count"] == 2
    assert s["chunk_count"] == 1
    assert s["topic_count"] == 1


def test_search_keyword(adapter):
    row = {
        "id": "n1", "node_type": "entity", "label": "David",
        "data": {"entity_type": "person", "attributes": {}},
        "embedding": None, "importance": 0.8,
        "created_at": "2026-04-08T00:00:00",
        "updated_at": "2026-04-08T00:00:00",
    }
    _chain(adapter, [row])
    results = adapter.search_keyword("David", limit=5)
    assert len(results) == 1
    assert results[0].label == "David"


def test_search_keyword_empty_for_short_words(adapter):
    _chain(adapter, [])
    results = adapter.search_keyword("it", limit=5)
    assert results == []


def test_path_routing_creates_supabase_adapter():
    with patch("memory_agent.store.cloud._load_credentials",
               return_value=("https://x.supabase.co", "key")):
        with patch("supabase.create_client", return_value=MagicMock()):
            from memory_agent import MemoryAgent, LLMConfig
            from memory_agent.store.cloud import SupabaseAdapter
            agent = MemoryAgent(
                "proj",
                path="supabase://",
                llm=LLMConfig(base_url="http://localhost:11434/v1"),
            )
            assert isinstance(agent._store, SupabaseAdapter)
