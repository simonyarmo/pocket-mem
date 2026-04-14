import numpy as np
from pocket_mem.embedding import embed, cosine_similarity, _embed_text
from pocket_mem.models import Node, Entity, Tone


def test_embed_returns_bytes():
    result = embed("hello world")
    assert isinstance(result, bytes)
    assert len(result) == 384 * 4  # 384 float32 values


def test_embed_is_normalized():
    result = embed("hello world")
    vec = np.frombuffer(result, dtype="float32")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


def test_cosine_similarity_identical():
    a = embed("hello world")
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-5


def test_cosine_similarity_different():
    a = embed("I love cats")
    b = embed("quantum physics equations")
    score = cosine_similarity(a, b)
    assert -1.0 <= score <= 1.0
    assert score < 0.9  # unrelated texts should not be near-identical


def test_cosine_similarity_semantic():
    boss = embed("my boss")
    manager = embed("my manager")
    unrelated = embed("quantum physics")
    assert cosine_similarity(boss, manager) > cosine_similarity(boss, unrelated)


def test_embed_text_entity():
    node = Node(id="n1", node_type="entity", label="David",
                data={"entity_type": "person", "attributes": {"role": "boss"}},
                created_at="2026-04-03T00:00:00", updated_at="2026-04-03T00:00:00")
    text = _embed_text(node)
    assert "entity" in text
    assert "David" in text


def test_embed_text_uses_summary_field():
    node = Node(id="n1", node_type="memory_chunk", label="turn-001",
                data={"raw": "David said use httpx", "summary": "httpx recommendation"},
                created_at="2026-04-03T00:00:00", updated_at="2026-04-03T00:00:00")
    text = _embed_text(node)
    assert "httpx recommendation" in text


def test_embed_text_tone():
    tone = Tone(id="t1", label="Frustration", tone_type="frustration",
                intensity=0.8, valence="negative", context="this keeps breaking")
    node = tone.to_node()
    node.created_at = node.updated_at = "2026-04-03T00:00:00"
    text = _embed_text(node)
    assert "Frustration" in text
    assert "this keeps breaking" in text
