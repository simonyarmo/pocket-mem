from __future__ import annotations
from abc import ABC, abstractmethod
from pocket_mem.models import Node, Edge


class StoreInterface(ABC):

    @abstractmethod
    def write_node(self, node: Node) -> str:
        """Upsert a node. Returns its id."""

    @abstractmethod
    def write_edge(self, edge: Edge) -> str:
        """Upsert an edge. Returns its id."""

    @abstractmethod
    def read_node(self, id: str) -> Node | None:
        """Return Node by id, or None if not found."""

    @abstractmethod
    def search_keyword(self, query: str, limit: int = 10) -> list[Node]:
        """BM25 full-text search over label + data. Returns ranked nodes."""

    @abstractmethod
    def search_vector(self, embedding: bytes, limit: int = 10) -> list[Node]:
        """Cosine similarity search. Returns ranked nodes. Implemented in Phase 2."""

    @abstractmethod
    def delete_node(self, id: str) -> None:
        """Remove node and its FTS entry. Connected edges are NOT deleted — per
        design, edges are invalidated via invalid_at (never hard-deleted). Callers
        that want to retire edges should set invalid_at before calling delete_node."""

    @abstractmethod
    def search_hybrid(
        self, query: str, embedding: bytes, limit: int = 10, alpha: float = 0.5
    ) -> list[Node]:
        """Merge BM25 and vector rankings via Reciprocal Rank Fusion.
        alpha=1.0 → pure vector; alpha=0.0 → pure BM25. Default 0.5 = equal weight."""

    @abstractmethod
    def get_edges(self, node_id: str) -> list[Edge]:
        """Return all valid (non-invalidated) edges connected to node_id,
        both outgoing (from_id=node_id) and incoming (to_id=node_id)."""

    @abstractmethod
    def list_topics(self) -> list[str]:
        """Return labels of all topic nodes."""

    @abstractmethod
    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Return all nodes of the given node_type."""

    @abstractmethod
    def stats(self) -> dict:
        """Return counts: node_count, entity_count, chunk_count, topic_count,
        tone_count, edge_count, session_count."""

    def export_pack(self, path: str) -> None:
        """Export memory to .mempack zip. Implemented in Phase 8."""
        raise NotImplementedError("Implemented in Phase 8")

    def import_pack(self, path: str) -> None:
        """Merge .mempack into this store. Implemented in Phase 8."""
        raise NotImplementedError("Implemented in Phase 8")
