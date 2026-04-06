from __future__ import annotations
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

from memory_agent.models import Node

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed_text(node: Node) -> str:
    """Build the text to embed from a node. Format: '{node_type}: {label}. {summary}'"""
    summary = (
        node.data.get("summary")
        or node.data.get("description")
        or node.data.get("context")
        or node.data.get("raw", "")[:200]
        or ""
    )
    base = f"{node.node_type}: {node.label}"
    return f"{base}. {summary}" if summary else base


def embed(text: str) -> bytes:
    """Encode text to a 384-dim float32 BLOB, pre-normalized (dot product = cosine)."""
    vec = _get_model().encode(text, normalize_embeddings=True)
    return vec.astype("float32").tobytes()


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Dot product of two pre-normalized float32 BLOBs (= cosine similarity)."""
    va = np.frombuffer(a, dtype="float32")
    vb = np.frombuffer(b, dtype="float32")
    return float(np.dot(va, vb))
