import json
import zipfile
import pytest
from memory_agent.config import StorageConfig
from memory_agent.store.local import SQLiteStore
from memory_agent.models import Entity, Topic, Edge, MemoryChunk


@pytest.fixture
def store_a(tmp_path):
    s = SQLiteStore("project_a", StorageConfig(path=str(tmp_path / "a")))
    yield s
    s.close()


@pytest.fixture
def store_b(tmp_path):
    s = SQLiteStore("project_b", StorageConfig(path=str(tmp_path / "b")))
    yield s
    s.close()


def _write_entity(store, id, label, entity_type="person"):
    entity = Entity(id=id, label=label, entity_type=entity_type)
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    store.write_node(node)
    return node


def _write_topic(store, id, label):
    topic = Topic(id=id, label=label)
    node = topic.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    store.write_node(node)


# ── export_pack() ─────────────────────────────────────────────────────────────

def test_export_creates_file(tmp_path, store_a):
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    assert (tmp_path / "out.mempack").exists()


def test_export_pack_is_valid_zip(tmp_path, store_a):
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    assert zipfile.is_zipfile(pack)


def test_export_contains_manifest(tmp_path, store_a):
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    with zipfile.ZipFile(pack) as zf:
        assert "manifest.json" in zf.namelist()


def test_export_contains_db(tmp_path, store_a):
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    with zipfile.ZipFile(pack) as zf:
        assert "memory.db" in zf.namelist()


def test_export_manifest_has_required_keys(tmp_path, store_a):
    _write_entity(store_a, "e1", "David")
    _write_topic(store_a, "t1", "People")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    with zipfile.ZipFile(pack) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    for key in ("version", "project", "exported_at", "node_count", "topics"):
        assert key in manifest, f"Missing manifest key: {key}"


def test_export_manifest_node_count(tmp_path, store_a):
    _write_entity(store_a, "e1", "David")
    _write_entity(store_a, "e2", "httpx", "tool")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    with zipfile.ZipFile(pack) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["node_count"] == 2


def test_export_manifest_topics(tmp_path, store_a):
    _write_topic(store_a, "t1", "People")
    _write_topic(store_a, "t2", "Dev Tools")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)
    with zipfile.ZipFile(pack) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert set(manifest["topics"]) == {"People", "Dev Tools"}


# ── import_pack() ─────────────────────────────────────────────────────────────

def test_import_merges_nodes(tmp_path, store_a, store_b):
    _write_entity(store_a, "e1", "David")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)

    store_b.import_pack(pack)
    assert store_b.read_node("e1") is not None
    assert store_b.read_node("e1").label == "David"


def test_import_merges_edges(tmp_path, store_a, store_b):
    _write_entity(store_a, "e1", "David")
    _write_entity(store_a, "e2", "httpx", "tool")
    edge = Edge(id="ed1", from_id="e1", to_id="e2", relation="recommended",
                created_at="2026-04-06T00:00:00")
    store_a.write_edge(edge)
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)

    store_b.import_pack(pack)
    edges = store_b.get_edges("e1")
    assert any(e.id == "ed1" for e in edges)


def test_import_no_duplicate_nodes(tmp_path, store_a, store_b):
    _write_entity(store_a, "e1", "David")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)

    store_b.import_pack(pack)
    store_b.import_pack(pack)  # import twice

    nodes = store_b.get_nodes_by_type("entity")
    ids = [n.id for n in nodes]
    assert ids.count("e1") == 1


def test_import_preserves_embeddings(tmp_path, store_a, store_b):
    _write_entity(store_a, "e1", "David")
    original_embedding = store_a.read_node("e1").embedding
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)

    store_b.import_pack(pack)
    imported_embedding = store_b.read_node("e1").embedding
    assert imported_embedding == original_embedding


def test_import_makes_nodes_searchable(tmp_path, store_a, store_b):
    """Imported nodes appear in keyword search (FTS synced)."""
    _write_entity(store_a, "e1", "UniqueNameXYZ")
    pack = str(tmp_path / "out.mempack")
    store_a.export_pack(pack)

    store_b.import_pack(pack)
    results = store_b.search_keyword("UniqueNameXYZ")
    assert any(r.id == "e1" for r in results)


def test_export_import_recall_roundtrip(tmp_path):
    """Canonical: export A → import B → recall from B finds same nodes."""
    from memory_agent.embedding import embed
    sa = SQLiteStore("proj", StorageConfig(path=str(tmp_path / "a")))
    sb = SQLiteStore("proj", StorageConfig(path=str(tmp_path / "b")))

    entity = Entity(id="e1", label="David", entity_type="person",
                    attributes={"role": "boss"})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-06T00:00:00"
    sa.write_node(node)

    pack = str(tmp_path / "roundtrip.mempack")
    sa.export_pack(pack)
    sb.import_pack(pack)

    results = sb.search_hybrid("my boss", embed("my boss"), limit=5)
    assert any(r.id == "e1" for r in results)

    sa.close()
    sb.close()
