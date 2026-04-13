import pytest
from dataclasses import fields as dc_fields
from memory_agent.models import Node, Edge
from memory_agent.models import Topic, Entity, MemoryChunk, Event, Tone


def test_node_fields():
    node = Node(
        id="n1",
        node_type="entity",
        label="David",
        data={"entity_type": "person"},
    )
    assert node.id == "n1"
    assert node.node_type == "entity"
    assert node.label == "David"
    assert node.data == {"entity_type": "person"}
    assert node.embedding is None
    assert node.importance == 0.5
    assert node.created_at == ""
    assert node.updated_at == ""


def test_edge_fields():
    edge = Edge(
        id="e1",
        from_id="n1",
        to_id="n2",
        relation="recommended",
    )
    assert edge.id == "e1"
    assert edge.from_id == "n1"
    assert edge.to_id == "n2"
    assert edge.relation == "recommended"
    assert edge.weight == 1.0
    assert edge.source_chunk_id is None
    assert edge.created_at == ""
    assert edge.valid_at is None
    assert edge.invalid_at is None


def test_node_has_correct_field_count():
    assert len(dc_fields(Node)) == 8


def test_edge_has_correct_field_count():
    assert len(dc_fields(Edge)) == 10


def test_node_with_all_fields():
    node = Node(
        id="n1",
        node_type="entity",
        label="David",
        data={"entity_type": "person", "nested": {"a": 1}},
        embedding=b"\x00\x01\x02",
        importance=0.9,
        created_at="2026-04-02T10:00:00",
        updated_at="2026-04-02T10:00:00",
    )
    assert node.embedding == b"\x00\x01\x02"
    assert node.importance == 0.9
    assert node.created_at == "2026-04-02T10:00:00"


def test_edge_with_all_fields():
    edge = Edge(
        id="e1",
        from_id="n1",
        to_id="n2",
        relation="recommended",
        weight=0.8,
        source_chunk_id="c1",
        created_at="2026-04-02T10:00:00",
        valid_at="2026-04-02T10:00:00",
        invalid_at="2026-04-02T12:00:00",
    )
    assert edge.weight == 0.8
    assert edge.source_chunk_id == "c1"
    assert edge.valid_at == "2026-04-02T10:00:00"
    assert edge.invalid_at == "2026-04-02T12:00:00"


def test_topic_to_node():
    topic = Topic(id="t1", label="People I Know", aliases=["people"], description="Humans")
    node = topic.to_node()
    assert node.node_type == "topic"
    assert node.label == "People I Know"
    assert node.data["aliases"] == ["people"]
    assert node.data["description"] == "Humans"


def test_topic_from_node():
    topic = Topic(id="t1", label="People I Know", aliases=["people"], description="Humans")
    node = topic.to_node()
    restored = Topic.from_node(node)
    assert restored.id == "t1"
    assert restored.aliases == ["people"]
    assert restored == topic


def test_entity_to_node():
    entity = Entity(
        id="e1", label="David", entity_type="person",
        topic_id="t1", attributes={"role": "boss"}, importance=0.8, access_count=3,
    )
    node = entity.to_node()
    assert node.node_type == "entity"
    assert node.importance == 0.8
    assert node.data["entity_type"] == "person"
    assert node.data["topic_id"] == "t1"
    assert node.data["attributes"] == {"role": "boss"}
    assert node.data["access_count"] == 3


def test_entity_from_node():
    entity = Entity(id="e1", label="David", entity_type="person", topic_id="t1",
                    attributes={"role": "boss"}, importance=0.8, access_count=3)
    restored = Entity.from_node(entity.to_node())
    assert restored.entity_type == "person"
    assert restored.attributes == {"role": "boss"}
    assert restored.access_count == 3
    assert restored == entity


def test_memory_chunk_to_node():
    chunk = MemoryChunk(
        id="c1", label="turn-001", raw="David said use httpx",
        summary="httpx recommendation", source="observe", memory_tier="working", session_id="s1",
    )
    node = chunk.to_node()
    assert node.node_type == "memory_chunk"
    assert node.data["raw"] == "David said use httpx"
    assert node.data["memory_tier"] == "working"


def test_memory_chunk_from_node():
    chunk = MemoryChunk(id="c1", label="turn-001", raw="David said use httpx",
                        summary="httpx recommendation", source="observe",
                        memory_tier="working", session_id="s1")
    restored = MemoryChunk.from_node(chunk.to_node())
    assert restored.raw == "David said use httpx"
    assert restored.session_id == "s1"
    assert restored == chunk


def test_event_to_node():
    event = Event(id="ev1", label="httpx recommendation", event_type="recommendation",
                  timestamp="2026-04-02T10:00:00", source_chunk_id="c1")
    node = event.to_node()
    assert node.node_type == "event"
    assert node.data["event_type"] == "recommendation"
    assert node.data["source_chunk_id"] == "c1"


def test_event_from_node():
    event = Event(id="ev1", label="httpx recommendation", event_type="recommendation",
                  timestamp="2026-04-02T10:00:00", source_chunk_id="c1")
    restored = Event.from_node(event.to_node())
    assert restored.event_type == "recommendation"
    assert restored.timestamp == "2026-04-02T10:00:00"
    assert restored == event


def test_entity_importance_roundtrip():
    entity = Entity(id="e1", label="David", entity_type="person", importance=0.8)
    restored = Entity.from_node(entity.to_node())
    assert restored.importance == 0.8


def test_topic_empty_aliases_default():
    topic = Topic(id="t1", label="Solo")
    node = topic.to_node()
    restored = Topic.from_node(node)
    assert restored.aliases == []
    assert restored.description == ""


def test_entity_no_topic_id():
    entity = Entity(id="e1", label="David", entity_type="person")
    restored = Entity.from_node(entity.to_node())
    assert restored.topic_id is None


def test_tone_to_node():
    tone = Tone(id="t1", label="Frustration", tone_type="frustration",
                intensity=0.8, valence="negative",
                context="this is taking too long", source_chunk_id="c1")
    node = tone.to_node()
    assert node.node_type == "tone"
    assert node.label == "Frustration"
    assert node.importance == 0.8
    assert node.data["tone_type"] == "frustration"
    assert node.data["valence"] == "negative"
    assert node.data["source_chunk_id"] == "c1"


def test_tone_from_node():
    tone = Tone(id="t1", label="Frustration", tone_type="frustration",
                intensity=0.8, valence="negative",
                context="this is taking too long", source_chunk_id="c1")
    restored = Tone.from_node(tone.to_node())
    assert restored == tone


def test_tone_defaults():
    tone = Tone(id="t1", label="Neutral", tone_type="neutral")
    node = tone.to_node()
    assert node.importance == 0.5
    assert node.data["valence"] == "neutral"
    assert node.data["source_chunk_id"] is None
