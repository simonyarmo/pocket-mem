import time
import pytest
from unittest.mock import MagicMock, patch
from memory_agent.config import LLMConfig, MemoryConfig, StorageConfig
from memory_agent.agent import MemoryAgent


@pytest.fixture
def agent(tmp_path):
    ag = MemoryAgent("test", path=str(tmp_path))
    yield ag
    ag._executor.shutdown(wait=False)
    ag._store.close()


# ── __init__ ─────────────────────────────────────────────────────────────────

def test_agent_creates_store(tmp_path):
    ag = MemoryAgent("test", path=str(tmp_path))
    assert ag._store is not None
    ag._executor.shutdown(wait=False)
    ag._store.close()


def test_agent_creates_llm(tmp_path):
    ag = MemoryAgent("test", path=str(tmp_path))
    assert ag._llm is not None
    ag._executor.shutdown(wait=False)
    ag._store.close()


def test_agent_creates_session_id(tmp_path):
    ag = MemoryAgent("test", path=str(tmp_path))
    assert ag._session_id is not None
    assert isinstance(ag._session_id, str)
    ag._executor.shutdown(wait=False)
    ag._store.close()


def test_agent_default_path_uses_memory(tmp_path, monkeypatch):
    """If path=None, defaults to './memory'."""
    monkeypatch.chdir(tmp_path)
    ag = MemoryAgent("test")
    assert "memory" in str(ag._store.db_path)
    ag._executor.shutdown(wait=False)
    ag._store.close()


# ── observe() ────────────────────────────────────────────────────────────────

def test_observe_is_nonblocking(agent):
    """observe() must return before ingest finishes even if ingest is slow."""
    def slow_ingest(*args, **kwargs):
        time.sleep(0.3)

    with patch("memory_agent.agent.ingest", side_effect=slow_ingest):
        start = time.monotonic()
        agent.observe("hello", "world")
        elapsed = time.monotonic() - start

    assert elapsed < 0.1, f"observe() blocked for {elapsed:.3f}s"


def test_observe_calls_ingest(agent):
    """observe() submits ingest() with correct arguments."""
    with patch("memory_agent.agent.ingest") as mock_ingest:
        agent.observe("user said hi", "agent said hi back")
        agent._executor.shutdown(wait=True)

    mock_ingest.assert_called_once()
    args = mock_ingest.call_args[0]
    assert args[0] == "user said hi"
    assert args[1] == "agent said hi back"
    assert args[2] is agent._store
    assert args[3] is agent._llm
    assert args[4] == agent._session_id


# ── recall() ─────────────────────────────────────────────────────────────────

def test_recall_context_returns_string(agent):
    from memory_agent.models import Entity
    entity = Entity(id="n1", label="David", entity_type="person",
                    attributes={"role": "boss"})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    agent._store.write_node(node)

    result = agent.recall("David")
    assert isinstance(result, str)
    assert "David" in result


def test_recall_raw_returns_list(agent):
    from memory_agent.models import Entity
    entity = Entity(id="n1", label="David", entity_type="person")
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    agent._store.write_node(node)

    result = agent.recall("David", mode="raw")
    assert isinstance(result, list)



def test_recall_answer_uses_llm(agent):
    agent._llm = MagicMock()
    agent._llm.complete.return_value = "The answer."
    result = agent.recall("something", mode="answer")
    assert result == "The answer."


# ── forget() ─────────────────────────────────────────────────────────────────

def test_forget_deletes_matching_node(agent):
    from memory_agent.models import Entity
    entity = Entity(id="n1", label="David", entity_type="person")
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    agent._store.write_node(node)

    agent.forget("David")

    assert agent._store.read_node("n1") is None


def test_forget_invalidates_edges(agent):
    from memory_agent.models import Entity, Edge
    n1 = Entity(id="n1", label="David", entity_type="person").to_node()
    n1.created_at = n1.updated_at = "2026-04-06T00:00:00"
    n2 = Entity(id="n2", label="httpx", entity_type="tool").to_node()
    n2.created_at = n2.updated_at = "2026-04-06T00:00:00"
    agent._store.write_node(n1)
    agent._store.write_node(n2)
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="recommended",
                created_at="2026-04-06T00:00:00")
    agent._store.write_edge(edge)

    agent.forget("David")

    # Edge row must still exist (never hard-deleted) but be marked invalid
    row = agent._store._conn.execute(
        "SELECT invalid_at FROM edges WHERE id='e1'"
    ).fetchone()
    assert row is not None, "Edge row must not be hard-deleted"
    assert row["invalid_at"] is not None, "Edge must be invalidated via invalid_at"
    # And get_edges() (which filters invalid_at IS NULL) should no longer return it
    assert not any(e.id == "e1" for e in agent._store.get_edges("n2"))


def test_forget_returns_count(agent):
    from memory_agent.models import Entity
    for i in range(3):
        e = Entity(id=f"n{i}", label=f"David {i}", entity_type="person")
        n = e.to_node()
        n.created_at = n.updated_at = "2026-04-06T00:00:00"
        agent._store.write_node(n)

    count = agent.forget("David")
    assert isinstance(count, int)
    assert count > 0


# ── topics() ─────────────────────────────────────────────────────────────────

def test_topics_returns_list(agent):
    result = agent.topics()
    assert isinstance(result, list)


def test_topics_returns_topic_labels(agent):
    from memory_agent.models import Topic
    import uuid
    for label in ["People", "Tools"]:
        t = Topic(id=str(uuid.uuid4()), label=label)
        tn = t.to_node()
        tn.created_at = tn.updated_at = "2026-04-06T00:00:00"
        agent._store.write_node(tn)

    result = agent.topics()
    assert set(result) == {"People", "Tools"}


# ── stats() ──────────────────────────────────────────────────────────────────

def test_stats_returns_dict(agent):
    result = agent.stats()
    assert isinstance(result, dict)


def test_stats_includes_required_keys(agent):
    s = agent.stats()
    for key in ("node_count", "entity_count", "chunk_count", "topic_count",
                "edge_count"):
        assert key in s, f"Missing key: {key}"


def test_stats_reflects_written_nodes(agent):
    from memory_agent.models import Entity
    entity = Entity(id="n1", label="David", entity_type="person")
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    agent._store.write_node(node)

    s = agent.stats()
    assert s["entity_count"] == 1
    assert s["node_count"] >= 1


# ── as_tool() ────────────────────────────────────────────────────────────────

def test_as_tool_returns_dict(agent):
    assert isinstance(agent.as_tool(), dict)


def test_as_tool_has_recall_name(agent):
    assert agent.as_tool()["function"]["name"] == "recall"


# ── export / import_pack ─────────────────────────────────────────────────────

def test_export_creates_mempack(tmp_path):
    from memory_agent.models import Entity
    ag = MemoryAgent("test", path=str(tmp_path / "agent"))
    entity = Entity(id="e1", label="David", entity_type="person")
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    ag._store.write_node(node)

    pack = str(tmp_path / "out.mempack")
    ag.export(pack)

    import zipfile
    assert (tmp_path / "out.mempack").exists()
    assert zipfile.is_zipfile(pack)
    ag.close()


def test_import_pack_merges_into_agent(tmp_path):
    from memory_agent.models import Entity
    ag_a = MemoryAgent("src", path=str(tmp_path / "a"))
    ag_b = MemoryAgent("dst", path=str(tmp_path / "b"))

    entity = Entity(id="e1", label="David", entity_type="person")
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    ag_a._store.write_node(node)

    pack = str(tmp_path / "out.mempack")
    ag_a.export(pack)
    ag_b.import_pack(pack)

    assert ag_b._store.read_node("e1") is not None
    ag_a.close()
    ag_b.close()


def test_close_shuts_down_executor_and_store(tmp_path):
    ag = MemoryAgent("test", path=str(tmp_path))
    ag.close()  # Must not raise
    # Executor is shut down — submitting new work should raise RuntimeError
    import pytest
    with pytest.raises(RuntimeError):
        ag._executor.submit(lambda: None)


def test_observe_triggers_compaction_at_threshold(tmp_path):
    """When chunk_count >= threshold, observe() submits a compaction job."""
    from memory_agent.config import MemoryConfig
    from unittest.mock import patch, MagicMock
    ag = MemoryAgent("test", path=str(tmp_path),
                     config=MemoryConfig(compaction_threshold=1))

    with patch("memory_agent.agent._run_compaction") as mock_compact, \
         patch("memory_agent.agent.ingest"):
        ag._store.stats = MagicMock(return_value={
            "chunk_count": 1, "node_count": 1, "entity_count": 0,
            "topic_count": 0, "tone_count": 0, "edge_count": 0,
            "session_count": 0,
        })
        ag.observe("hello", "world")
        ag._executor.shutdown(wait=True)

    mock_compact.assert_called_once()
    ag._store.close()


def test_observe_does_not_compact_below_threshold(tmp_path):
    """When chunk_count < threshold, observe() does NOT submit compaction."""
    from memory_agent.config import MemoryConfig
    from unittest.mock import patch, MagicMock
    ag = MemoryAgent("test", path=str(tmp_path),
                     config=MemoryConfig(compaction_threshold=100))

    with patch("memory_agent.agent._run_compaction") as mock_compact, \
         patch("memory_agent.agent.ingest"):
        ag._store.stats = MagicMock(return_value={
            "chunk_count": 0, "node_count": 0, "entity_count": 0,
            "topic_count": 0, "tone_count": 0, "edge_count": 0,
            "session_count": 0,
        })
        ag.observe("hello", "world")
        ag._executor.shutdown(wait=True)

    mock_compact.assert_not_called()
    ag._store.close()
