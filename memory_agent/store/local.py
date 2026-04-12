from __future__ import annotations
import json
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from memory_agent.config import StorageConfig
from memory_agent.embedding import embed, _embed_text, cosine_similarity
from memory_agent.models import Node, Edge
from memory_agent.store.base import StoreInterface


_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "how", "who",
    "when", "where", "why", "which", "did", "does", "do", "for", "and",
    "or", "but", "in", "on", "at", "to", "from", "by", "with", "his",
    "her", "their", "of", "has", "have", "been", "this", "that", "it",
    "its", "not", "any", "all", "one", "two", "about", "into", "than",
    "then", "them", "they", "also", "much", "many", "some", "would",
    "could", "should",
}


def _fts_body(node: Node) -> str:
    """Return clean plain text for FTS5 indexing (no JSON structure)."""
    if node.node_type == "memory_chunk":
        return node.data.get("raw", "") or node.data.get("summary", "")
    if node.node_type == "entity":
        attrs = node.data.get("attributes", {})
        attr_text = " ".join(f"{k} {v}" for k, v in attrs.items())
        extra = " ".join(filter(None, [
            node.data.get("description", ""),
            node.data.get("summary", ""),
        ]))
        return f"{node.label} {attr_text} {extra}".strip()
    # Generic fallback: label + any known text fields
    extras = " ".join(filter(None, [
        node.data.get("description", ""),
        node.data.get("summary", ""),
        node.data.get("context", ""),
    ]))
    return f"{node.label} {extras}".strip()


def _to_fts_query(query: str) -> str:
    """Convert a natural language query to an FTS5 OR query over content words."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return " OR ".join(content) if content else query


class SQLiteStore(StoreInterface):

    def __init__(self, project: str, config: StorageConfig | None = None) -> None:
        cfg = config or StorageConfig()
        base = Path(cfg.path)
        base.mkdir(parents=True, exist_ok=True)
        self.db_path = base / f"{project}.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                node_type   TEXT NOT NULL,
                label       TEXT NOT NULL,
                data        TEXT NOT NULL,
                embedding   BLOB,
                importance  REAL DEFAULT 0.5,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                id              TEXT PRIMARY KEY,
                from_id         TEXT,
                to_id           TEXT,
                relation        TEXT,
                weight          REAL DEFAULT 1.0,
                source_chunk_id TEXT,
                created_at      TEXT,
                valid_at        TEXT,
                invalid_at      TEXT
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                project     TEXT,
                started_at  TEXT,
                ended_at    TEXT,
                turn_count  INTEGER DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
                USING fts5(id UNINDEXED, label, body);
            CREATE INDEX IF NOT EXISTS idx_nodes_type  ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_from  ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to    ON edges(to_id);
            CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(invalid_at);
        """)
        self._conn.commit()

    # ── stubs (implemented in later tasks) ────────────────────────────────

    def write_node(self, node: Node) -> str:
        if node.embedding is None:
            node.embedding = embed(_embed_text(node))
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes "
            "(id, node_type, label, data, embedding, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node.id, node.node_type, node.label,
                json.dumps(node.data), node.embedding,
                node.importance, node.created_at, node.updated_at,
            ),
        )
        # Keep FTS in sync (body column — NOT content)
        self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node.id,))
        self._conn.execute(
            "INSERT INTO nodes_fts(id, label, body) VALUES (?, ?, ?)",
            (node.id, node.label, _fts_body(node)),
        )
        self._conn.commit()
        return node.id

    def write_edge(self, edge: Edge) -> str:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges "
            "(id, from_id, to_id, relation, weight, source_chunk_id, "
            "created_at, valid_at, invalid_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                edge.id, edge.from_id, edge.to_id, edge.relation,
                edge.weight, edge.source_chunk_id,
                edge.created_at, edge.valid_at, edge.invalid_at,
            ),
        )
        self._conn.commit()
        return edge.id

    def read_node(self, id: str) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return Node(
            id=row["id"],
            node_type=row["node_type"],
            label=row["label"],
            data=json.loads(row["data"]),
            embedding=row["embedding"],
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def search_keyword(self, query: str, limit: int = 10) -> list[Node]:
        try:
            rows = self._conn.execute(
                "SELECT n.* FROM nodes_fts "
                "JOIN nodes n ON n.id = nodes_fts.id "
                "WHERE nodes_fts MATCH ? "
                "LIMIT ?",
                (_to_fts_query(query), limit),
            ).fetchall()
        except Exception:
            return []
        return [
            Node(
                id=r["id"],
                node_type=r["node_type"],
                label=r["label"],
                data=json.loads(r["data"]),
                embedding=r["embedding"],
                importance=r["importance"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def search_vector(self, embedding: bytes, limit: int = 10) -> list[Node]:
        rows = self._conn.execute(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
        ).fetchall()
        scored: list[tuple[float, str]] = []
        for row in rows:
            score = cosine_similarity(embedding, bytes(row["embedding"]))
            scored.append((score, row["id"]))
        scored.sort(reverse=True)
        return [
            node
            for _, node_id in scored[:limit]
            if (node := self.read_node(node_id)) is not None
        ]

    def search_hybrid(
        self, query: str, embedding: bytes, limit: int = 10, alpha: float = 0.5
    ) -> list[Node]:
        """Reciprocal Rank Fusion of BM25 and vector results."""
        fetch = limit * 2
        bm25 = self.search_keyword(query, limit=fetch)
        vec = self.search_vector(embedding, limit=fetch)
        bm25_rank = {n.id: i for i, n in enumerate(bm25)}
        vec_rank = {n.id: i for i, n in enumerate(vec)}
        all_ids = set(bm25_rank) | set(vec_rank)
        k = 60  # standard RRF constant
        scores = {
            node_id: (
                alpha * (1.0 / (k + vec_rank.get(node_id, fetch)))
                + (1 - alpha) * (1.0 / (k + bm25_rank.get(node_id, fetch)))
            )
            for node_id in all_ids
        }
        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]
        return [
            node
            for node_id in top_ids
            if (node := self.read_node(node_id)) is not None
        ]

    def delete_node(self, id: str) -> None:
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (id,))
        self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (id,))
        self._conn.commit()

    def list_topics(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT label FROM nodes WHERE node_type = 'topic'"
        ).fetchall()
        return [r["label"] for r in rows]

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM nodes GROUP BY node_type"
        ).fetchall()
        type_counts = {r["node_type"]: r["cnt"] for r in rows}
        edge_count = self._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE invalid_at IS NULL"
        ).fetchone()[0]
        session_count = self._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        return {
            "node_count": sum(type_counts.values()),
            "entity_count": type_counts.get("entity", 0),
            "chunk_count": type_counts.get("memory_chunk", 0),
            "topic_count": type_counts.get("topic", 0),
            "tone_count": type_counts.get("tone", 0),
            "edge_count": edge_count,
            "session_count": session_count,
        }

    def get_edges(self, node_id: str) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges "
            "WHERE (from_id = ? OR to_id = ?) AND invalid_at IS NULL",
            (node_id, node_id),
        ).fetchall()
        return [
            Edge(
                id=r["id"],
                from_id=r["from_id"],
                to_id=r["to_id"],
                relation=r["relation"],
                weight=r["weight"],
                source_chunk_id=r["source_chunk_id"],
                created_at=r["created_at"] or "",
                valid_at=r["valid_at"],
                invalid_at=r["invalid_at"],
            )
            for r in rows
        ]

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE node_type = ?", (node_type,)
        ).fetchall()
        return [
            Node(
                id=r["id"],
                node_type=r["node_type"],
                label=r["label"],
                data=json.loads(r["data"]),
                embedding=r["embedding"],
                importance=r["importance"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def export_pack(self, path: str) -> None:
        """Export all nodes, edges, and metadata to a .mempack zip file."""
        # Flush WAL to main DB file before copying
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.commit()

        s = self.stats()
        manifest = {
            "version": "1.0",
            "project": self.db_path.stem,
            "exported_at": datetime.utcnow().isoformat(),
            "node_count": s["node_count"],
            "topics": self.list_topics(),
        }

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(self.db_path), "memory.db")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    def import_pack(self, path: str) -> None:
        """Merge a .mempack into this store. Existing nodes/edges are upserted."""
        with zipfile.ZipFile(path, "r") as zf:
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract("memory.db", tmp)
                src_conn = sqlite3.connect(
                    str(Path(tmp) / "memory.db"), check_same_thread=False
                )
                src_conn.row_factory = sqlite3.Row
                try:
                    # Merge nodes (preserve original embeddings — do NOT re-embed)
                    for row in src_conn.execute("SELECT * FROM nodes").fetchall():
                        self._conn.execute(
                            "INSERT OR REPLACE INTO nodes "
                            "(id, node_type, label, data, embedding, importance, "
                            "created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                row["id"], row["node_type"], row["label"],
                                row["data"], row["embedding"], row["importance"],
                                row["created_at"], row["updated_at"],
                            ),
                        )
                        # Keep FTS in sync
                        self._conn.execute(
                            "DELETE FROM nodes_fts WHERE id = ?", (row["id"],)
                        )
                        self._conn.execute(
                            "INSERT INTO nodes_fts(id, label, body) VALUES (?, ?, ?)",
                            (row["id"], row["label"], row["data"]),
                        )

                    # Merge edges
                    for row in src_conn.execute("SELECT * FROM edges").fetchall():
                        self._conn.execute(
                            "INSERT OR REPLACE INTO edges "
                            "(id, from_id, to_id, relation, weight, source_chunk_id, "
                            "created_at, valid_at, invalid_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                row["id"], row["from_id"], row["to_id"],
                                row["relation"], row["weight"], row["source_chunk_id"],
                                row["created_at"], row["valid_at"], row["invalid_at"],
                            ),
                        )
                finally:
                    src_conn.close()
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()
