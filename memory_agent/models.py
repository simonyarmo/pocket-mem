from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    node_type: str
    label: str
    data: dict[str, Any]
    embedding: bytes | None = None
    importance: float = 0.5
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Edge:
    id: str
    from_id: str
    to_id: str
    relation: str
    weight: float = 1.0
    context: str = ""
    source_chunk_id: str | None = None
    created_at: str = ""
    valid_at: str | None = None
    invalid_at: str | None = None


@dataclass
class Topic:
    id: str
    label: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""

    def to_node(self) -> Node:
        return Node(
            id=self.id,
            node_type="topic",
            label=self.label,
            data={"aliases": self.aliases, "description": self.description},
        )

    @classmethod
    def from_node(cls, node: Node) -> Topic:
        return cls(
            id=node.id,
            label=node.label,
            aliases=node.data.get("aliases", []),
            description=node.data.get("description", ""),
        )


@dataclass
class Entity:
    id: str
    label: str
    entity_type: str  # person | tool | project | company | concept
    topic_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0

    def to_node(self) -> Node:
        return Node(
            id=self.id,
            node_type="entity",
            label=self.label,
            data={
                "entity_type": self.entity_type,
                "topic_id": self.topic_id,
                "attributes": self.attributes,
                "access_count": self.access_count,
            },
            importance=self.importance,
        )

    @classmethod
    def from_node(cls, node: Node) -> Entity:
        return cls(
            id=node.id,
            label=node.label,
            entity_type=node.data["entity_type"],
            topic_id=node.data.get("topic_id"),
            attributes=node.data.get("attributes", {}),
            importance=node.importance,
            access_count=node.data.get("access_count", 0),
        )


@dataclass
class MemoryChunk:
    id: str
    label: str
    raw: str
    summary: str = ""
    source: str = ""
    memory_tier: str = "working"  # working | episodic | semantic
    session_id: str | None = None

    def to_node(self) -> Node:
        return Node(
            id=self.id,
            node_type="memory_chunk",
            label=self.label,
            data={
                "raw": self.raw,
                "summary": self.summary,
                "source": self.source,
                "memory_tier": self.memory_tier,
                "session_id": self.session_id,
            },
        )

    @classmethod
    def from_node(cls, node: Node) -> MemoryChunk:
        return cls(
            id=node.id,
            label=node.label,
            raw=node.data["raw"],
            summary=node.data.get("summary", ""),
            source=node.data.get("source", ""),
            memory_tier=node.data.get("memory_tier", "working"),
            session_id=node.data.get("session_id"),
        )


@dataclass
class Event:
    id: str
    label: str
    event_type: str
    timestamp: str = ""
    source_chunk_id: str | None = None

    def to_node(self) -> Node:
        return Node(
            id=self.id,
            node_type="event",
            label=self.label,
            data={
                "event_type": self.event_type,
                "timestamp": self.timestamp,
                "source_chunk_id": self.source_chunk_id,
            },
        )

    @classmethod
    def from_node(cls, node: Node) -> Event:
        return cls(
            id=node.id,
            label=node.label,
            event_type=node.data["event_type"],
            timestamp=node.data.get("timestamp", ""),
            source_chunk_id=node.data.get("source_chunk_id"),
        )


@dataclass
class Tone:
    id: str
    label: str                        # e.g. "Frustration", "Excitement", "Curiosity"
    tone_type: str                    # joy | frustration | curiosity | urgency | anxiety | gratitude | excitement | neutral
    intensity: float = 0.5           # 0.0 (subtle) → 1.0 (strong)
    valence: str = "neutral"         # "positive" | "negative" | "neutral"
    context: str = ""                # text snippet or summary that revealed the tone
    source_chunk_id: str | None = None  # which MemoryChunk this was detected from

    def to_node(self) -> Node:
        return Node(
            id=self.id,
            node_type="tone",
            label=self.label,
            data={
                "tone_type": self.tone_type,
                "intensity": self.intensity,
                "valence": self.valence,
                "context": self.context,
                "source_chunk_id": self.source_chunk_id,
            },
            importance=self.intensity,  # stronger emotion = higher retention priority
        )

    @classmethod
    def from_node(cls, node: Node) -> Tone:
        return cls(
            id=node.id,
            label=node.label,
            tone_type=node.data["tone_type"],
            intensity=node.data.get("intensity", 0.5),
            valence=node.data.get("valence", "neutral"),
            context=node.data.get("context", ""),
            source_chunk_id=node.data.get("source_chunk_id"),
        )
