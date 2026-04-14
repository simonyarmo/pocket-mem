import pytest
from unittest.mock import MagicMock
from pocket_mem.config import StorageConfig, MemoryConfig
from pocket_mem.store.local import SQLiteStore
from pocket_mem.llm.client import LLMClient
from pocket_mem.models import Entity, MemoryChunk, Topic, Edge
from pocket_mem.compactor import compress, prune


@pytest.fixture
def tmp_store(tmp_path):
    store = SQLiteStore("test", StorageConfig(path=str(tmp_path)))
    yield store
    store.close()


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"summary": "David is the boss and uses httpx."}
    return llm


def _write_working_chunk(store, id, raw, label="turn"):
    chunk = MemoryChunk(id=id, label=label, raw=raw, memory_tier="working")
    node = chunk.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    store.write_node(node)
    return node


def _write_entity_with_topic(store, entity_id, label, topic_id):
    entity = Entity(id=entity_id, label=label, entity_type="person",
                    topic_id=topic_id)
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    store.write_node(node)
    return node


def _write_topic(store, topic_id, label):
    topic = Topic(id=topic_id, label=label)
    node = topic.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    store.write_node(node)
    return node


def _link_chunk_to_entity(store, chunk_id, entity_id):
    import uuid
    edge = Edge(id=str(uuid.uuid4()), from_id=chunk_id, to_id=entity_id,
                relation="derived_from", created_at="2026-04-06T00:00:00")
    store.write_edge(edge)


# ── compress() ───────────────────────────────────────────────────────────────

def test_compress_returns_zero_when_no_chunks(tmp_store, mock_llm):
    result = compress(tmp_store, mock_llm)
    assert result == 0
    mock_llm.complete_json.assert_not_called()


def test_compress_creates_episodic_chunk(tmp_store, mock_llm):
    _write_working_chunk(tmp_store, "c1", "David recommended httpx")
    result = compress(tmp_store, mock_llm)
    assert result == 1
    episodic = [
        n for n in tmp_store.get_nodes_by_type("memory_chunk")
        if n.data.get("memory_tier") == "episodic"
    ]
    assert len(episodic) == 1
    assert episodic[0].data["summary"] == "David is the boss and uses httpx."


def test_compress_marks_originals_processed(tmp_store, mock_llm):
    _write_working_chunk(tmp_store, "c1", "Some content")
    compress(tmp_store, mock_llm)
    node = tmp_store.read_node("c1")
    assert node.data.get("processed") is True


def test_compress_does_not_delete_originals(tmp_store, mock_llm):
    _write_working_chunk(tmp_store, "c1", "Some content")
    compress(tmp_store, mock_llm)
    assert tmp_store.read_node("c1") is not None


def test_compress_skips_already_processed_chunks(tmp_store, mock_llm):
    chunk = MemoryChunk(id="c1", label="turn", raw="old content",
                        memory_tier="working")
    node = chunk.to_node()
    node.data["processed"] = True
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    tmp_store.write_node(node)
    result = compress(tmp_store, mock_llm)
    assert result == 0
    mock_llm.complete_json.assert_not_called()


def test_compress_groups_by_topic(tmp_store, mock_llm):
    """Two chunks in different topics → two episodic chunks."""
    _write_topic(tmp_store, "t1", "People")
    _write_topic(tmp_store, "t2", "Tools")
    _write_entity_with_topic(tmp_store, "e1", "David", "t1")
    _write_entity_with_topic(tmp_store, "e2", "httpx", "t2")

    _write_working_chunk(tmp_store, "c1", "David is my boss")
    _write_working_chunk(tmp_store, "c2", "httpx is a tool")
    _link_chunk_to_entity(tmp_store, "c1", "e1")
    _link_chunk_to_entity(tmp_store, "c2", "e2")

    result = compress(tmp_store, mock_llm)
    assert result == 2
    episodic = [
        n for n in tmp_store.get_nodes_by_type("memory_chunk")
        if n.data.get("memory_tier") == "episodic"
    ]
    assert len(episodic) == 2


def test_compress_episodic_chunk_has_embedding(tmp_store, mock_llm):
    _write_working_chunk(tmp_store, "c1", "Some content")
    compress(tmp_store, mock_llm)
    episodic = [
        n for n in tmp_store.get_nodes_by_type("memory_chunk")
        if n.data.get("memory_tier") == "episodic"
    ]
    assert len(episodic) == 1
    assert episodic[0].embedding is not None


def test_compress_calls_llm_once_per_group(tmp_store, mock_llm):
    """Two chunks in same topic → one LLM call."""
    _write_working_chunk(tmp_store, "c1", "Chunk one")
    _write_working_chunk(tmp_store, "c2", "Chunk two")
    compress(tmp_store, mock_llm)
    assert mock_llm.complete_json.call_count == 1


def test_compress_episodic_source_is_compaction(tmp_store, mock_llm):
    _write_working_chunk(tmp_store, "c1", "Some content")
    compress(tmp_store, mock_llm)
    episodic = [
        n for n in tmp_store.get_nodes_by_type("memory_chunk")
        if n.data.get("memory_tier") == "episodic"
    ]
    assert episodic[0].data["source"] == "compaction"


# ── prune() ──────────────────────────────────────────────────────────────────

def test_prune_returns_zero_when_nothing_to_prune(tmp_store):
    config = MemoryConfig()
    result = prune(tmp_store, config)
    assert result == 0


def test_prune_deletes_old_low_importance_episodic(tmp_store):
    config = MemoryConfig(importance_prune_threshold=0.5, prune_after_days=30)
    chunk = MemoryChunk(id="e1", label="old", raw="old", memory_tier="episodic")
    node = chunk.to_node()
    node.importance = 0.05
    node.created_at = "2020-01-01T00:00:00"
    node.updated_at = "2020-01-01T00:00:00"
    tmp_store.write_node(node)
    result = prune(tmp_store, config)
    assert result == 1
    assert tmp_store.read_node("e1") is None


def test_prune_keeps_high_importance_episodic(tmp_store):
    config = MemoryConfig(importance_prune_threshold=0.1, prune_after_days=30)
    chunk = MemoryChunk(id="e1", label="important", raw="x", memory_tier="episodic")
    node = chunk.to_node()
    node.importance = 0.9
    node.created_at = "2020-01-01T00:00:00"
    node.updated_at = "2020-01-01T00:00:00"
    tmp_store.write_node(node)
    result = prune(tmp_store, config)
    assert result == 0
    assert tmp_store.read_node("e1") is not None


def test_prune_keeps_recent_episodic(tmp_store):
    config = MemoryConfig(importance_prune_threshold=0.5, prune_after_days=30)
    chunk = MemoryChunk(id="e1", label="new", raw="x", memory_tier="episodic")
    node = chunk.to_node()
    node.importance = 0.05
    node.created_at = "2099-01-01T00:00:00"
    node.updated_at = "2099-01-01T00:00:00"
    tmp_store.write_node(node)
    result = prune(tmp_store, config)
    assert result == 0


def test_prune_does_not_delete_working_chunks(tmp_store):
    config = MemoryConfig(importance_prune_threshold=0.9, prune_after_days=0)
    chunk = MemoryChunk(id="w1", label="turn", raw="old", memory_tier="working")
    node = chunk.to_node()
    node.importance = 0.01
    node.created_at = "2020-01-01T00:00:00"
    node.updated_at = "2020-01-01T00:00:00"
    tmp_store.write_node(node)
    result = prune(tmp_store, config)
    assert result == 0
    assert tmp_store.read_node("w1") is not None


def test_prune_invalidates_edges_before_deleting(tmp_store):
    config = MemoryConfig(importance_prune_threshold=0.5, prune_after_days=30)
    chunk = MemoryChunk(id="e1", label="old", raw="x", memory_tier="episodic")
    node = chunk.to_node()
    node.importance = 0.05
    node.created_at = "2020-01-01T00:00:00"
    node.updated_at = "2020-01-01T00:00:00"
    tmp_store.write_node(node)

    _write_topic(tmp_store, "t1", "People")
    edge = Edge(id="edge1", from_id="e1", to_id="t1", relation="derived_from",
                created_at="2020-01-01T00:00:00")
    tmp_store.write_edge(edge)

    prune(tmp_store, config)

    row = tmp_store._conn.execute(
        "SELECT invalid_at FROM edges WHERE id='edge1'"
    ).fetchone()
    assert row is not None
    assert row["invalid_at"] is not None
