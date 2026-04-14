import pytest
from pocket_mem.store.base import StoreInterface


def test_store_interface_is_abstract():
    """StoreInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StoreInterface()


def test_store_interface_has_required_methods():
    methods = [
        "write_node", "write_edge", "read_node",
        "search_keyword", "search_vector",
        "delete_node", "list_topics",
        "export_pack", "import_pack",
    ]
    for m in methods:
        assert hasattr(StoreInterface, m), f"Missing method: {m}"


from pathlib import Path
from pocket_mem.config import StorageConfig
from pocket_mem.store.local import SQLiteStore


@pytest.fixture
def tmp_store(tmp_path):
    cfg = StorageConfig(path=str(tmp_path))
    store = SQLiteStore(project="test", config=cfg)
    yield store
    store.close()


def test_db_file_created(tmp_path):
    cfg = StorageConfig(path=str(tmp_path))
    store = SQLiteStore(project="myproject", config=cfg)
    store.close()
    assert (tmp_path / "myproject.db").exists()


def test_schema_has_nodes_table(tmp_store):
    tables = {
        row[0]
        for row in tmp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "nodes" in tables
    assert "edges" in tables
    assert "sessions" in tables


def test_schema_has_fts_table(tmp_store):
    tables = {
        row[0]
        for row in tmp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "nodes_fts" in tables


def test_schema_idempotent(tmp_path):
    """Opening the same DB twice must not raise."""
    cfg = StorageConfig(path=str(tmp_path))
    s1 = SQLiteStore(project="test", config=cfg)
    s1.close()
    s2 = SQLiteStore(project="test", config=cfg)
    s2.close()


def test_schema_has_indexes(tmp_store):
    indexes = {
        row[0]
        for row in tmp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_nodes_type" in indexes
    assert "idx_edges_from" in indexes
    assert "idx_edges_to" in indexes
    assert "idx_edges_valid" in indexes


import uuid
from pocket_mem.models import Node


def _make_node(**kwargs) -> Node:
    defaults = dict(
        id=str(uuid.uuid4()),
        node_type="entity",
        label="David",
        data={"entity_type": "person"},
        created_at="2026-04-02T00:00:00",
        updated_at="2026-04-02T00:00:00",
    )
    defaults.update(kwargs)
    return Node(**defaults)


def test_write_and_read_node(tmp_store):
    node = _make_node(id="n1", label="David")
    tmp_store.write_node(node)
    result = tmp_store.read_node("n1")
    assert result is not None
    assert result.id == "n1"
    assert result.label == "David"
    assert result.node_type == "entity"
    assert result.data == {"entity_type": "person"}
    assert result.importance == 0.5


def test_read_node_not_found(tmp_store):
    assert tmp_store.read_node("nonexistent") is None


def test_write_node_upserts(tmp_store):
    node = _make_node(id="n1", label="David")
    tmp_store.write_node(node)
    updated = _make_node(id="n1", label="David Smith")
    tmp_store.write_node(updated)
    result = tmp_store.read_node("n1")
    assert result.label == "David Smith"
    fts_count = tmp_store._conn.execute(
        "SELECT COUNT(*) FROM nodes_fts WHERE id = 'n1'"
    ).fetchone()[0]
    assert fts_count == 1


def test_write_node_with_importance(tmp_store):
    node = _make_node(id="n2", importance=0.9)
    tmp_store.write_node(node)
    result = tmp_store.read_node("n2")
    assert result.importance == 0.9


def test_write_node_data_roundtrip(tmp_store):
    node = _make_node(id="n3", data={"key": "value", "nested": {"a": 1}})
    tmp_store.write_node(node)
    result = tmp_store.read_node("n3")
    assert result.data == {"key": "value", "nested": {"a": 1}}


from pocket_mem.models import Edge


def _make_edge(**kwargs) -> Edge:
    defaults = dict(
        id=str(uuid.uuid4()),
        from_id="n1",
        to_id="n2",
        relation="recommended",
        created_at="2026-04-02T00:00:00",
    )
    defaults.update(kwargs)
    return Edge(**defaults)


def test_write_edge(tmp_store):
    tmp_store.write_node(_make_node(id="n1", label="David"))
    tmp_store.write_node(_make_node(id="n2", label="httpx"))
    edge = _make_edge(id="e1", from_id="n1", to_id="n2", relation="recommended")
    returned_id = tmp_store.write_edge(edge)
    assert returned_id == "e1"
    row = tmp_store._conn.execute("SELECT * FROM edges WHERE id='e1'").fetchone()
    assert row is not None
    assert row["relation"] == "recommended"
    assert row["weight"] == 1.0
    assert row["invalid_at"] is None


def test_write_edge_upserts(tmp_store):
    edge = _make_edge(id="e1", relation="recommended")
    tmp_store.write_edge(edge)
    updated = _make_edge(id="e1", relation="supersedes")
    tmp_store.write_edge(updated)
    row = tmp_store._conn.execute("SELECT * FROM edges WHERE id='e1'").fetchone()
    assert row["relation"] == "supersedes"


def test_write_edge_with_invalid_at(tmp_store):
    ts = "2026-04-02T12:00:00"
    edge = _make_edge(id="e1", invalid_at=ts)
    tmp_store.write_edge(edge)
    row = tmp_store._conn.execute("SELECT * FROM edges WHERE id='e1'").fetchone()
    assert row["invalid_at"] == ts


def test_write_edge_retirement_pattern(tmp_store):
    """Write a live edge, then retire it by upserting with invalid_at set."""
    edge = _make_edge(id="e1", relation="recommended")
    tmp_store.write_edge(edge)
    # Verify it starts as live
    row = tmp_store._conn.execute("SELECT * FROM edges WHERE id='e1'").fetchone()
    assert row["invalid_at"] is None

    # Retire the edge by upserting with invalid_at
    retired = _make_edge(id="e1", relation="recommended", invalid_at="2026-04-02T12:00:00")
    tmp_store.write_edge(retired)
    row = tmp_store._conn.execute("SELECT * FROM edges WHERE id='e1'").fetchone()
    assert row["invalid_at"] == "2026-04-02T12:00:00"


def test_search_keyword_finds_by_label(tmp_store):
    tmp_store.write_node(_make_node(id="n1", label="Cursor IDE",
                                    data={"entity_type": "tool"}))
    tmp_store.write_node(_make_node(id="n2", label="David",
                                    data={"entity_type": "person"}))
    results = tmp_store.search_keyword("Cursor")
    ids = [r.id for r in results]
    assert "n1" in ids
    assert "n2" not in ids


def test_search_keyword_finds_in_data(tmp_store):
    tmp_store.write_node(_make_node(
        id="n1", label="SomeNode",
        data={"description": "recommended httpx for HTTP clients"},
    ))
    results = tmp_store.search_keyword("httpx")
    assert any(r.id == "n1" for r in results)


def test_search_keyword_limit(tmp_store):
    for i in range(5):
        tmp_store.write_node(_make_node(
            id=f"n{i}", label=f"Alpha {i}", data={"tag": "alpha"}
        ))
    results = tmp_store.search_keyword("Alpha", limit=3)
    assert len(results) <= 3


def test_search_keyword_no_results(tmp_store):
    tmp_store.write_node(_make_node(id="n1", label="David", data={}))
    results = tmp_store.search_keyword("zzznomatch")
    assert results == []


from pocket_mem.models import Topic


def test_delete_node(tmp_store):
    node = _make_node(id="n1", label="David")
    tmp_store.write_node(node)
    assert tmp_store.read_node("n1") is not None
    tmp_store.delete_node("n1")
    assert tmp_store.read_node("n1") is None


def test_delete_node_removes_from_fts(tmp_store):
    node = _make_node(id="n1", label="Cursor IDE", data={})
    tmp_store.write_node(node)
    tmp_store.delete_node("n1")
    results = tmp_store.search_keyword("Cursor")
    assert results == []


def test_list_topics_empty(tmp_store):
    assert tmp_store.list_topics() == []


def test_list_topics(tmp_store):
    for label in ["People I Know", "Dev Tools", "Projects"]:
        t = Topic(id=str(uuid.uuid4()), label=label)
        node = t.to_node()
        node.created_at = "2026-04-02T00:00:00"
        node.updated_at = "2026-04-02T00:00:00"
        tmp_store.write_node(node)
    # entity node should NOT appear
    tmp_store.write_node(_make_node(id="e1", label="David", node_type="entity"))
    topics = tmp_store.list_topics()
    assert set(topics) == {"People I Know", "Dev Tools", "Projects"}


def test_write_node_stores_embedding(tmp_store):
    node = _make_node(id="n1", label="David", data={"entity_type": "person"})
    tmp_store.write_node(node)
    result = tmp_store.read_node("n1")
    assert result.embedding is not None
    assert isinstance(result.embedding, bytes)
    assert len(result.embedding) == 384 * 4  # 384 float32 values


def test_search_vector_returns_nodes(tmp_store):
    tmp_store.write_node(_make_node(id="n1", label="David", data={"entity_type": "person"}))
    from pocket_mem.embedding import embed
    results = tmp_store.search_vector(embed("person named David"))
    assert any(r.id == "n1" for r in results)


def test_search_vector_ranks_by_similarity(tmp_store):
    tmp_store.write_node(_make_node(id="n1", label="David", data={"role": "boss manager"}))
    tmp_store.write_node(_make_node(id="n2", label="httpx", data={"entity_type": "tool"}))
    from pocket_mem.embedding import embed
    results = tmp_store.search_vector(embed("my boss manager"), limit=2)
    ids = [r.id for r in results]
    assert ids[0] == "n1"  # David (boss/manager) should rank above httpx


def test_search_vector_respects_limit(tmp_store):
    for i in range(5):
        tmp_store.write_node(_make_node(id=f"n{i}", label=f"Node {i}", data={}))
    from pocket_mem.embedding import embed
    results = tmp_store.search_vector(embed("node"), limit=2)
    assert len(results) <= 2


def test_search_vector_semantic_boss(tmp_store):
    """Vector search finds David for 'supervisor' even though neither label nor
    attributes contain that word — purely semantic via embedding."""
    from pocket_mem.models import Entity
    entity = Entity(id="n1", label="David", entity_type="person",
                    topic_id=None, attributes={"role": "boss"})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-03T00:00:00"
    tmp_store.write_node(node)

    from pocket_mem.embedding import embed
    # BM25 should NOT find David for "supervisor" — word not in any indexed text
    bm25_results = tmp_store.search_keyword("supervisor")
    assert not any(r.id == "n1" for r in bm25_results)

    # Vector search SHOULD find David for "supervisor" (semantic similarity to "boss")
    vec_results = tmp_store.search_vector(embed("supervisor"), limit=5)
    assert any(r.id == "n1" for r in vec_results)


def test_search_hybrid_combines_results(tmp_store):
    # n1: keyword match but low semantic similarity
    tmp_store.write_node(_make_node(id="n1", label="httpx library",
                                    data={"description": "my boss recommended this"}))
    # n2: no keyword match but high semantic similarity
    from pocket_mem.models import Entity
    entity = Entity(id="n2", label="David", entity_type="person",
                    attributes={"role": "boss"})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-03T00:00:00"
    tmp_store.write_node(node)

    from pocket_mem.embedding import embed
    results = tmp_store.search_hybrid(
        query="my boss", embedding=embed("my boss"), limit=5
    )
    ids = [r.id for r in results]
    # Both should appear — hybrid merges keyword + vector
    assert "n1" in ids  # keyword match: "my boss recommended"
    assert "n2" in ids  # vector match: David the boss


def test_search_hybrid_respects_limit(tmp_store):
    for i in range(6):
        tmp_store.write_node(_make_node(id=f"n{i}", label=f"Item {i}",
                                        data={"tag": "searchable"}))
    from pocket_mem.embedding import embed
    results = tmp_store.search_hybrid(
        query="Item", embedding=embed("Item"), limit=3
    )
    assert len(results) <= 3


def test_store_interface_has_search_hybrid():
    assert hasattr(StoreInterface, "search_hybrid")


def test_get_edges_returns_outgoing(tmp_store):
    from pocket_mem.models import Edge
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="recommended",
                created_at="2026-04-05T00:00:00")
    tmp_store.write_edge(edge)
    edges = tmp_store.get_edges("n1")
    assert len(edges) == 1
    assert edges[0].id == "e1"
    assert edges[0].relation == "recommended"


def test_get_edges_returns_incoming(tmp_store):
    from pocket_mem.models import Edge
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="derived_from",
                created_at="2026-04-05T00:00:00")
    tmp_store.write_edge(edge)
    edges = tmp_store.get_edges("n2")
    assert len(edges) == 1
    assert edges[0].id == "e1"


def test_get_edges_excludes_invalid(tmp_store):
    from pocket_mem.models import Edge
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="mentioned",
                invalid_at="2026-01-01T00:00:00", created_at="2026-04-05T00:00:00")
    tmp_store.write_edge(edge)
    edges = tmp_store.get_edges("n1")
    assert len(edges) == 0


def test_get_edges_empty_when_no_edges(tmp_store):
    edges = tmp_store.get_edges("nonexistent-node")
    assert edges == []


def test_store_interface_has_get_edges():
    assert hasattr(StoreInterface, "get_edges")


def test_stats_returns_dict(tmp_store):
    result = tmp_store.stats()
    assert isinstance(result, dict)


def test_stats_counts_nodes_by_type(tmp_store):
    from pocket_mem.models import Entity, Topic
    import uuid
    entity = Entity(id="e1", label="David", entity_type="person")
    n = entity.to_node()
    n.created_at = n.updated_at = "2026-04-06T00:00:00"
    tmp_store.write_node(n)

    topic = Topic(id="t1", label="People")
    tn = topic.to_node()
    tn.created_at = tn.updated_at = "2026-04-06T00:00:00"
    tmp_store.write_node(tn)

    s = tmp_store.stats()
    assert s["entity_count"] == 1
    assert s["topic_count"] == 1
    assert s["node_count"] == 2


def test_stats_counts_edges(tmp_store):
    from pocket_mem.models import Edge
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="related_to",
                created_at="2026-04-06T00:00:00")
    tmp_store.write_edge(edge)
    s = tmp_store.stats()
    assert s["edge_count"] >= 1


def test_stats_empty_store(tmp_store):
    s = tmp_store.stats()
    assert s["node_count"] == 0
    assert s["edge_count"] == 0


def test_store_interface_has_stats():
    assert hasattr(StoreInterface, "stats")


def test_stats_excludes_invalid_edges(tmp_store):
    from pocket_mem.models import Edge
    edge = Edge(id="e1", from_id="n1", to_id="n2", relation="related",
                invalid_at="2026-04-06T00:00:00", created_at="2026-04-06T00:00:00")
    tmp_store.write_edge(edge)
    s = tmp_store.stats()
    assert s["edge_count"] == 0


def test_get_nodes_by_type_returns_matching(tmp_store):
    from pocket_mem.models import Entity, Topic
    entity = Entity(id="e1", label="David", entity_type="person")
    n = entity.to_node()
    n.created_at = n.updated_at = "2026-04-06T00:00:00"
    tmp_store.write_node(n)
    topic = Topic(id="t1", label="People")
    tn = topic.to_node()
    tn.created_at = tn.updated_at = "2026-04-06T00:00:00"
    tmp_store.write_node(tn)

    entities = tmp_store.get_nodes_by_type("entity")
    assert len(entities) == 1
    assert entities[0].id == "e1"


def test_get_nodes_by_type_empty_when_none(tmp_store):
    result = tmp_store.get_nodes_by_type("memory_chunk")
    assert result == []


def test_store_interface_has_get_nodes_by_type():
    assert hasattr(StoreInterface, "get_nodes_by_type")
