import pytest
from unittest.mock import MagicMock
from pocket_mem.config import StorageConfig
from pocket_mem.store.local import SQLiteStore
from pocket_mem.llm.client import LLMClient
from pocket_mem.ingestion import (
    ingest, _classify, _extract, _resolve_entity, _get_or_create_topic,
)


@pytest.fixture
def tmp_store(tmp_path):
    store = SQLiteStore("test", StorageConfig(path=str(tmp_path)))
    yield store
    store.close()


def _mock_llm(classify_result: dict, extract_result: dict) -> MagicMock:
    """LLMClient mock that returns classify then extract results in order."""
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.side_effect = [classify_result, extract_result]
    return llm


# ── _classify ────────────────────────────────────────────────────────────────

def test_classify_remember():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"category": "remember", "topics": ["People I Know"]}
    result = _classify("David is my boss", [], llm)
    assert result["category"] == "remember"
    assert "People I Know" in result["topics"]


def test_classify_ignore():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"category": "ignore", "topics": []}
    result = _classify("Hello!", [], llm)
    assert result["category"] == "ignore"
    assert result["topics"] == []


def test_classify_uses_schema():
    from pocket_mem.llm.prompts import CLASSIFY_SCHEMA
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"category": "remember", "topics": []}
    _classify("some turn", [], llm)
    _, kwargs = llm.complete_json.call_args
    assert kwargs.get("schema") == CLASSIFY_SCHEMA


def test_classify_formats_existing_topics():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"category": "remember", "topics": ["Dev Tools"]}
    _classify("use httpx", ["Dev Tools", "People I Know"], llm)
    messages = llm.complete_json.call_args[0][0]
    prompt_text = messages[0]["content"]
    assert "Dev Tools" in prompt_text
    assert "People I Know" in prompt_text


# ── _extract ─────────────────────────────────────────────────────────────────

def test_extract_returns_entities():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {
        "entities": [{"label": "David", "type": "person", "attributes": {"role": "boss"}}],
        "relationships": [],
    }
    result = _extract("David is my boss", "context", llm)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["label"] == "David"
    assert result["tone"] is None


def test_extract_includes_tone_when_present():
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {
        "entities": [],
        "relationships": [],
        "tone": {
            "label": "Frustration", "tone_type": "frustration",
            "intensity": 0.8, "valence": "negative", "context": "ugh",
        },
    }
    result = _extract("Ugh this is broken", "context", llm)
    assert result["tone"] is not None
    assert result["tone"]["tone_type"] == "frustration"


def test_extract_uses_extraction_schema():
    from pocket_mem.llm.prompts import EXTRACTION_SCHEMA
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"entities": [], "relationships": []}
    _extract("some turn", "context", llm)
    _, kwargs = llm.complete_json.call_args
    assert kwargs.get("schema") == EXTRACTION_SCHEMA


# ── _resolve_entity ──────────────────────────────────────────────────────────

def test_resolve_entity_returns_none_when_missing(tmp_store):
    result = _resolve_entity("David", "person", tmp_store)
    assert result is None


def test_resolve_entity_finds_existing(tmp_store):
    from pocket_mem.models import Entity
    entity = Entity(id="e1", label="David", entity_type="person",
                    attributes={"role": "boss"})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-05T00:00:00"
    tmp_store.write_node(node)

    result = _resolve_entity("David", "person", tmp_store)
    assert result is not None
    assert result.id == "e1"


def test_resolve_entity_case_insensitive(tmp_store):
    from pocket_mem.models import Entity
    entity = Entity(id="e1", label="David", entity_type="person", attributes={})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-05T00:00:00"
    tmp_store.write_node(node)

    result = _resolve_entity("david", "person", tmp_store)
    assert result is not None
    assert result.id == "e1"


def test_resolve_entity_ignores_wrong_type(tmp_store):
    from pocket_mem.models import Entity
    entity = Entity(id="e1", label="Python", entity_type="tool", attributes={})
    node = entity.to_node()
    node.created_at = node.updated_at = "2026-04-05T00:00:00"
    tmp_store.write_node(node)

    result = _resolve_entity("Python", "project", tmp_store)
    assert result is None


# ── _get_or_create_topic ─────────────────────────────────────────────────────

def test_get_or_create_topic_creates_new(tmp_store):
    topic_id = _get_or_create_topic("People I Know", tmp_store)
    assert topic_id is not None
    assert "People I Know" in tmp_store.list_topics()


def test_get_or_create_topic_returns_existing_id(tmp_store):
    id1 = _get_or_create_topic("Dev Tools", tmp_store)
    id2 = _get_or_create_topic("Dev Tools", tmp_store)
    assert id1 == id2
    assert tmp_store.list_topics().count("Dev Tools") == 1


# ── ingest() end-to-end ──────────────────────────────────────────────────────

def test_ingest_ignore_stores_nothing(tmp_store):
    llm = MagicMock(spec=LLMClient)
    llm.complete_json.return_value = {"category": "ignore", "topics": []}
    ingest("Hello!", "Hi there!", tmp_store, llm)
    assert tmp_store.list_topics() == []
    assert llm.complete_json.call_count == 1  # only classify, no extract


def test_ingest_remember_creates_entity(tmp_store):
    llm = _mock_llm(
        classify_result={"category": "remember", "topics": ["People I Know"]},
        extract_result={
            "entities": [{"label": "David", "type": "person",
                          "attributes": {"role": "boss"}}],
            "relationships": [],
        },
    )
    ingest("David is my boss", "Got it", tmp_store, llm)
    results = [n for n in tmp_store.search_keyword("David")
               if n.node_type == "entity"]
    assert len(results) == 1
    assert results[0].label == "David"


def test_ingest_creates_topic_node(tmp_store):
    llm = _mock_llm(
        classify_result={"category": "remember", "topics": ["People I Know"]},
        extract_result={
            "entities": [{"label": "David", "type": "person", "attributes": {}}],
            "relationships": [],
        },
    )
    ingest("David is my boss", "Got it", tmp_store, llm)
    assert "People I Know" in tmp_store.list_topics()


def test_ingest_makes_two_llm_calls(tmp_store):
    llm = _mock_llm(
        classify_result={"category": "remember", "topics": ["Dev Tools"]},
        extract_result={
            "entities": [{"label": "httpx", "type": "tool", "attributes": {}}],
            "relationships": [],
        },
    )
    ingest("Use httpx instead of requests", "Sure", tmp_store, llm)
    assert llm.complete_json.call_count == 2


def test_ingest_stores_tone_node(tmp_store):
    llm = _mock_llm(
        classify_result={"category": "remember", "topics": ["General"]},
        extract_result={
            "entities": [],
            "relationships": [],
            "tone": {
                "label": "Frustration", "tone_type": "frustration",
                "intensity": 0.8, "valence": "negative", "context": "ugh",
            },
        },
    )
    ingest("Ugh this keeps breaking!", "I see", tmp_store, llm)
    results = tmp_store.search_keyword("Frustration")
    tone_nodes = [r for r in results if r.node_type == "tone"]
    assert len(tone_nodes) == 1
    assert tone_nodes[0].data["tone_type"] == "frustration"


def test_ingest_deduplicates_entities(tmp_store):
    llm1 = _mock_llm(
        classify_result={"category": "remember", "topics": ["People I Know"]},
        extract_result={
            "entities": [{"label": "David", "type": "person",
                          "attributes": {"role": "boss"}}],
            "relationships": [],
        },
    )
    ingest("David is my boss", "Got it", tmp_store, llm1)

    llm2 = _mock_llm(
        classify_result={"category": "remember", "topics": ["People I Know"]},
        extract_result={
            "entities": [{"label": "David", "type": "person",
                          "attributes": {"email": "david@acme.com"}}],
            "relationships": [],
        },
    )
    ingest("David's email is david@acme.com", "Noted", tmp_store, llm2)

    entity_nodes = [n for n in tmp_store.search_keyword("David")
                    if n.node_type == "entity"]
    assert len(entity_nodes) == 1
    attrs = entity_nodes[0].data.get("attributes", {})
    assert attrs.get("email") == "david@acme.com"
    assert attrs.get("role") == "boss"  # original attribute preserved


def test_ingest_stores_relationships(tmp_store):
    llm = _mock_llm(
        classify_result={"category": "remember", "topics": ["Dev Tools"]},
        extract_result={
            "entities": [
                {"label": "David", "type": "person", "attributes": {}},
                {"label": "httpx", "type": "tool", "attributes": {}},
            ],
            "relationships": [
                {"from": "David", "relation": "recommended", "to": "httpx", "weight": 0.9},
            ],
        },
    )
    ingest("David recommended httpx", "Got it", tmp_store, llm)
    david_nodes = [n for n in tmp_store.search_keyword("David") if n.node_type == "entity"]
    assert len(david_nodes) == 1
    httpx_nodes = [n for n in tmp_store.search_keyword("httpx") if n.node_type == "entity"]
    assert len(httpx_nodes) == 1
