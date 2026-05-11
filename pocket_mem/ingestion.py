from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

from pocket_mem.config import MemoryConfig
from pocket_mem.llm.client import LLMClient
from pocket_mem.llm.prompts import (
    CLASSIFY, CLASSIFY_SCHEMA,
    EXTRACTION_SCHEMA,
    build_extract_prompt,
)
from pocket_mem.models import Edge, Entity, MemoryChunk, Tone, Topic
from pocket_mem.store.base import StoreInterface


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify(turn: str, topics: list[str], llm: LLMClient) -> dict:
    """Classify turn and assign topic labels. Returns {"category": str, "topics": list[str]}."""
    topics_str = ", ".join(topics) if topics else "none yet"
    prompt = CLASSIFY.format(turn=turn, topics=topics_str)
    result = llm.complete_json([{"role": "user", "content": prompt}], schema=CLASSIFY_SCHEMA)
    return {
        "category": result.get("category", "ignore"),
        "topics": result.get("topics", []),
    }


def _extract(turn: str, context: str, llm: LLMClient, identity=None) -> dict:
    """Extract entities, relationships, and optional tone. Returns structured dict."""
    prompt = build_extract_prompt(identity).format(turn=turn, context=context)
    result = llm.complete_json([{"role": "user", "content": prompt}], schema=EXTRACTION_SCHEMA)
    return {
        "entities": result.get("entities", []),
        "relationships": result.get("relationships", []),
        "tone": result.get("tone"),
    }


def _get_or_create_topic(label: str, store: StoreInterface) -> str:
    """Return id of topic node with this label, creating it if needed."""
    candidates = [
        n for n in store.search_keyword(label, limit=10)
        if n.node_type == "topic" and n.label.lower() == label.lower()
    ]
    if candidates:
        return candidates[0].id
    now = _now()
    topic = Topic(id=str(uuid.uuid4()), label=label)
    node = topic.to_node()
    node.created_at = node.updated_at = now
    store.write_node(node)
    return node.id


def _resolve_entity(label: str, entity_type: str, store: StoreInterface):
    """Find existing entity node by label (case-insensitive) and type. Returns Node or None."""
    candidates = [
        n for n in store.search_keyword(label, limit=10)
        if n.node_type == "entity"
        and n.label.lower() == label.lower()
        and n.data.get("entity_type") == entity_type
    ]
    return candidates[0] if candidates else None


def score_importance(node, identity=None) -> float:
    """Score node importance, boosted by identity priority signals."""
    import json as _json
    base = 0.5
    if not identity or not identity.derived:
        return base
    boost = 0.0
    content = f"{node.label} {_json.dumps(node.data)}".lower()
    if node.data.get("entity_type") in identity.derived.get("priority_entity_types", []):
        boost += 0.2
    for signal in identity.derived.get("high_importance_signals", []):
        if signal.lower() in content:
            boost += 0.1
            break
    for signal in identity.derived.get("low_importance_signals", []):
        if signal.lower() in content:
            boost -= 0.1
            break
    return max(0.1, min(1.0, base + boost))


def verify_extraction(raw_input: str, extracted: dict) -> dict:
    """Remove extracted entities and relationships whose key terms are not in the raw input.

    Heuristic grounding check — catches fabrications of unstated entities but not
    subtle term substitutions. Words ≤ 2 chars are skipped (stop words, abbreviations).
    """
    raw_lower = raw_input.lower()
    verified_entities = []
    rejected_entities = []

    for entity in extracted.get("entities", []):
        if not isinstance(entity, dict):
            continue
        label = entity.get("label", "")
        words = [w for w in label.lower().split() if len(w) > 2]
        if not words or any(w in raw_lower for w in words):
            verified_entities.append(entity)
        else:
            rejected_entities.append(entity)
            _log.debug("Extraction rejected: '%s' not found in input", label)

    verified_labels = {e["label"].lower() for e in verified_entities}
    verified_rels = []
    rejected_rels = []

    for rel in extracted.get("relationships", []):
        if not isinstance(rel, dict):
            continue
        from_words = [w for w in rel.get("from", "").lower().split() if len(w) > 2]
        to_words = [w for w in rel.get("to", "").lower().split() if len(w) > 2]
        from_ok = not from_words or any(w in raw_lower for w in from_words)
        to_ok = not to_words or any(w in raw_lower for w in to_words)
        if from_ok and to_ok:
            verified_rels.append(rel)
        else:
            rejected_rels.append(rel)
            _log.debug(
                "Relationship rejected: %s → %s not grounded in input",
                rel.get("from"), rel.get("to"),
            )

    if rejected_entities or rejected_rels:
        _log.warning(
            "Grounding check removed %d entities and %d relationships not found in input text",
            len(rejected_entities), len(rejected_rels),
        )

    return {
        "entities": verified_entities,
        "relationships": verified_rels,
        "_rejected": {"entities": rejected_entities, "relationships": rejected_rels},
    }


def ingest(
    user_input: str,
    agent_response: str,
    store: StoreInterface,
    llm: LLMClient,
    session_id: str | None = None,
    config: MemoryConfig | None = None,
    identity=None,
) -> None:
    """Extract knowledge from one conversation turn and persist it to the store.

    Synchronous — the caller (agent.py) is responsible for running this in a
    background thread via ThreadPoolExecutor.
    """
    turn = f"User: {user_input}\nAssistant: {agent_response}"
    existing_topics = store.list_topics()

    # Step 1: Classify
    classification = _classify(turn, existing_topics, llm)
    if classification["category"] != "remember":
        return

    topic_names: list[str] = classification["topics"]
    context = ", ".join(existing_topics) if existing_topics else "none"

    # Step 2: Extract
    extraction = _extract(turn, context, llm, identity)

    # Step 2a: Grounding check — remove entities/relationships not present in raw text
    verified = verify_extraction(turn, extraction)

    # Step 2b: Weight threshold — discard low-confidence relationships
    min_weight = config.relationship_min_weight if config else 0.6
    verified["relationships"] = [
        r for r in verified["relationships"]
        if r.get("weight", 1.0) >= min_weight
    ]

    if not verified["entities"]:
        _log.warning("Entity extraction returned 0 entities for turn: %.80s...", turn)

    now = _now()

    # Step 3: Store MemoryChunk
    chunk_id = str(uuid.uuid4())
    chunk = MemoryChunk(
        id=chunk_id,
        label=turn[:60],
        raw=turn,
        source="observe",
        session_id=session_id,
    )
    chunk_node = chunk.to_node()
    chunk_node.created_at = chunk_node.updated_at = now
    store.write_node(chunk_node)

    # Step 4: Ensure topic nodes exist; map label → id
    topic_id_map: dict[str, str] = {
        t: _get_or_create_topic(t, store) for t in topic_names
    }
    primary_topic_id: str | None = next(iter(topic_id_map.values()), None)

    # Link chunk → each topic it belongs to
    for tid in topic_id_map.values():
        store.write_edge(Edge(
            id=str(uuid.uuid4()),
            from_id=chunk_id,
            to_id=tid,
            relation="belongs_to",
            source_chunk_id=chunk_id,
            created_at=now,
        ))

    # Step 5: Resolve + store entities, link chunk → entity
    entity_id_map: dict[str, str] = {}
    for ent_data in verified["entities"]:
        if not isinstance(ent_data, dict):
            continue  # skip if model returned a string instead of an object
        label = ent_data["label"]
        etype = ent_data.get("type", "concept")
        attrs = ent_data.get("attributes", {})

        existing = _resolve_entity(label, etype, store)
        if existing is not None:
            existing.data["attributes"] = {
                **existing.data.get("attributes", {}), **attrs
            }
            existing.data["access_count"] = existing.data.get("access_count", 0) + 1
            existing.updated_at = now
            existing.embedding = None  # re-embed with updated data
            store.write_node(existing)
            entity_id_map[label] = existing.id
        else:
            entity = Entity(
                id=str(uuid.uuid4()),
                label=label,
                entity_type=etype,
                topic_id=primary_topic_id,
                attributes=attrs,
            )
            node = entity.to_node()
            node.importance = score_importance(node, identity)
            node.created_at = node.updated_at = now
            store.write_node(node)
            entity_id_map[label] = node.id

        store.write_edge(Edge(
            id=str(uuid.uuid4()),
            from_id=chunk_id,
            to_id=entity_id_map[label],
            relation="derived_from",
            source_chunk_id=chunk_id,
            created_at=now,
        ))

    # Step 6: Store entity → entity relationships
    for rel in verified["relationships"]:
        from_id = entity_id_map.get(rel["from"])
        to_id = entity_id_map.get(rel["to"])
        if from_id and to_id:
            store.write_edge(Edge(
                id=str(uuid.uuid4()),
                from_id=from_id,
                to_id=to_id,
                relation=rel["relation"],
                weight=float(rel.get("weight", 1.0)),
                context=rel.get("context", ""),
                source_chunk_id=chunk_id,
                created_at=now,
            ))

    # Step 7: Store tone node if detected
    tone_data = extraction.get("tone")
    if tone_data:
        tone = Tone(
            id=str(uuid.uuid4()),
            label=tone_data.get("label", "Unnamed Tone"),
            tone_type=tone_data.get("tone_type", "neutral"),
            intensity=float(tone_data.get("intensity", 0.5)),
            valence=tone_data.get("valence", "neutral"),
            context=tone_data.get("context", ""),
            source_chunk_id=chunk_id,
        )
        tone_node = tone.to_node()
        tone_node.created_at = tone_node.updated_at = now
        store.write_node(tone_node)
        store.write_edge(Edge(
            id=str(uuid.uuid4()),
            from_id=chunk_id,
            to_id=tone_node.id,
            relation="expresses",
            source_chunk_id=chunk_id,
            created_at=now,
        ))
