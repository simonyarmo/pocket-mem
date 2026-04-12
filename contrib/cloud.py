"""
cloud.py — Supabase backend for pocket-mem.

Usage:
    agent = MemoryAgent("myproject", path="supabase://")

Credentials are read from environment variables (or a .env file in the
current working directory):
    SUPABASE_URL  — your project URL, e.g. https://xxxx.supabase.co
    SUPABASE_KEY  — your anon/service-role key

Run this SQL in the Supabase SQL editor once to create the required tables:

    CREATE TABLE IF NOT EXISTS pm_nodes (
        id           TEXT PRIMARY KEY,
        project      TEXT NOT NULL,
        node_type    TEXT NOT NULL,
        label        TEXT NOT NULL,
        data         JSONB NOT NULL DEFAULT '{}',
        embedding    JSONB,
        importance   FLOAT DEFAULT 0.5,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        user_id      TEXT
    );
    CREATE INDEX IF NOT EXISTS pm_nodes_project ON pm_nodes(project);
    CREATE INDEX IF NOT EXISTS pm_nodes_type    ON pm_nodes(project, node_type);

    CREATE TABLE IF NOT EXISTS pm_edges (
        id               TEXT PRIMARY KEY,
        project          TEXT NOT NULL,
        from_id          TEXT,
        to_id            TEXT,
        relation         TEXT,
        weight           FLOAT DEFAULT 1.0,
        source_chunk_id  TEXT,
        created_at       TEXT,
        valid_at         TEXT,
        invalid_at       TEXT,
        user_id          TEXT
    );
    CREATE INDEX IF NOT EXISTS pm_edges_project  ON pm_edges(project);
    CREATE INDEX IF NOT EXISTS pm_edges_from     ON pm_edges(project, from_id);
    CREATE INDEX IF NOT EXISTS pm_edges_to       ON pm_edges(project, to_id);
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

from memory_agent.embedding import embed, _embed_text, cosine_similarity
from memory_agent.models import Node, Edge
from memory_agent.store.base import StoreInterface

_NODES_TABLE = "pm_nodes"
_EDGES_TABLE = "pm_edges"

_SCHEMA_SQL = """\
-- Run this once in your Supabase dashboard → SQL Editor
CREATE TABLE IF NOT EXISTS pm_nodes (
    id           TEXT PRIMARY KEY,
    project      TEXT NOT NULL,
    node_type    TEXT NOT NULL,
    label        TEXT NOT NULL,
    data         JSONB NOT NULL DEFAULT '{}',
    embedding    JSONB,
    importance   FLOAT DEFAULT 0.5,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    user_id      TEXT
);
CREATE INDEX IF NOT EXISTS pm_nodes_project ON pm_nodes(project);
CREATE INDEX IF NOT EXISTS pm_nodes_type    ON pm_nodes(project, node_type);

CREATE TABLE IF NOT EXISTS pm_edges (
    id               TEXT PRIMARY KEY,
    project          TEXT NOT NULL,
    from_id          TEXT,
    to_id            TEXT,
    relation         TEXT,
    weight           FLOAT DEFAULT 1.0,
    source_chunk_id  TEXT,
    created_at       TEXT,
    valid_at         TEXT,
    invalid_at       TEXT,
    user_id          TEXT
);
CREATE INDEX IF NOT EXISTS pm_edges_project ON pm_edges(project);
CREATE INDEX IF NOT EXISTS pm_edges_from    ON pm_edges(project, from_id);
CREATE INDEX IF NOT EXISTS pm_edges_to      ON pm_edges(project, to_id);
"""


# ---------------------------------------------------------------------------
# Credential loader
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str]:
    """Return (SUPABASE_URL, SUPABASE_KEY) from env or .env file."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("SUPABASE_URL=") and not url:
                    url = line.split("=", 1)[1].strip()
                elif line.startswith("SUPABASE_KEY=") and not key:
                    key = line.split("=", 1)[1].strip()

    if not url:
        raise ValueError(
            "SUPABASE_URL not set. Add it to your environment or .env file."
        )
    if not key:
        raise ValueError(
            "SUPABASE_KEY not set. Add it to your environment or .env file."
        )
    return url, key


# ---------------------------------------------------------------------------
# Embedding conversion (bytes ↔ JSON float list)
# ---------------------------------------------------------------------------

def _emb_to_list(emb: bytes) -> list[float]:
    n = len(emb) // 4
    return list(struct.unpack(f"{n}f", emb))


def _list_to_emb(lst: list[float]) -> bytes:
    return struct.pack(f"{len(lst)}f", *lst)


# ---------------------------------------------------------------------------
# SupabaseAdapter
# ---------------------------------------------------------------------------

class SupabaseAdapter(StoreInterface):
    """StoreInterface backed by Supabase (PostgreSQL via REST API).

    All records are scoped to `project` so multiple agents can share
    the same Supabase project without collision.
    """

    def __init__(self, project: str, user_id: str | None = None) -> None:
        try:
            from supabase import create_client
        except ImportError as e:
            raise ImportError(
                "supabase package required for cloud storage. "
                "Install it with: pip install pocket-mem[cloud]"
            ) from e

        url, key = _load_credentials()
        self._client = create_client(url, key)
        self._project = project
        self._user_id = user_id
        self._check_tables()

    def _check_tables(self) -> None:
        """Verify pm_nodes exists; raise a helpful error with setup SQL if not."""
        try:
            self._client.table(_NODES_TABLE).select("id").limit(1).execute()
        except Exception as exc:
            msg = str(exc)
            if "PGRST205" in msg or _NODES_TABLE in msg:
                raise RuntimeError(
                    f"Supabase tables not found. "
                    f"Open your Supabase dashboard → SQL Editor and run:\n\n"
                    f"{_SCHEMA_SQL}"
                ) from None
            raise

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _exec(self, builder):
        """Execute a PostgREST query with retry on transient network errors."""
        import time
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return builder.execute()
            except Exception as exc:
                msg = str(exc)
                # Don't retry API-level errors (auth, schema, etc.) — only network
                if "PGRST" in msg or "JWT" in msg:
                    raise
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def _nodes(self):
        return self._client.table(_NODES_TABLE)

    def _edges(self):
        return self._client.table(_EDGES_TABLE)

    def _row_to_node(self, row: dict) -> Node:
        emb_raw = row.get("embedding")
        emb = _list_to_emb(emb_raw) if emb_raw else None
        return Node(
            id=row["id"],
            node_type=row["node_type"],
            label=row["label"],
            data=row["data"] if isinstance(row["data"], dict) else {},
            embedding=emb,
            importance=row.get("importance") or 0.5,
            created_at=row.get("created_at") or "",
            updated_at=row.get("updated_at") or "",
        )

    def _row_to_edge(self, row: dict) -> Edge:
        return Edge(
            id=row["id"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            relation=row.get("relation") or "",
            weight=row.get("weight") or 1.0,
            source_chunk_id=row.get("source_chunk_id"),
            created_at=row.get("created_at") or "",
            valid_at=row.get("valid_at"),
            invalid_at=row.get("invalid_at"),
        )

    def _node_to_record(self, node: Node) -> dict:
        record: dict = {
            "id": node.id,
            "project": self._project,
            "node_type": node.node_type,
            "label": node.label,
            "data": node.data,
            "importance": node.importance,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
        }
        if node.embedding:
            record["embedding"] = _emb_to_list(node.embedding)
        if self._user_id:
            record["user_id"] = self._user_id
        return record

    def _edge_to_record(self, edge: Edge) -> dict:
        record: dict = {
            "id": edge.id,
            "project": self._project,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "relation": edge.relation,
            "weight": edge.weight,
            "source_chunk_id": edge.source_chunk_id,
            "created_at": edge.created_at,
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
        }
        if self._user_id:
            record["user_id"] = self._user_id
        return record

    # ── StoreInterface ────────────────────────────────────────────────────────

    def write_node(self, node: Node) -> str:
        if node.embedding is None:
            node.embedding = embed(_embed_text(node))
        self._exec(self._nodes().upsert(self._node_to_record(node)))
        return node.id

    def write_edge(self, edge: Edge) -> str:
        self._exec(self._edges().upsert(self._edge_to_record(edge)))
        return edge.id

    def read_node(self, id: str) -> Node | None:
        resp = self._exec(
            self._nodes()
            .select("*")
            .eq("project", self._project)
            .eq("id", id)
            .limit(1)
        )
        if not resp.data:
            return None
        return self._row_to_node(resp.data[0])

    def search_keyword(self, query: str, limit: int = 10) -> list[Node]:
        """Case-insensitive search across label and key data fields.

        Uses PostgREST or_() to match on:
          - label (entity/topic names)
          - data->>raw (full email body on working memory_chunks)
          - data->>summary (compact summary on episodic chunks)
        This mirrors what SQLite FTS5 does via _fts_body().
        """
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return []
        term = f"%{words[0]}%"
        or_filter = (
            f"label.ilike.{term},"
            f"data->>raw.ilike.{term},"
            f"data->>summary.ilike.{term}"
        )
        resp = self._exec(
            self._nodes()
            .select("*")
            .eq("project", self._project)
            .or_(or_filter)
            .limit(limit)
        )
        return [self._row_to_node(r) for r in (resp.data or [])]

    def search_vector(self, embedding: bytes, limit: int = 10) -> list[Node]:
        """Fetch all embeddings for this project, rank by cosine similarity.

        Top-K nodes are fetched in a single batched in_() query instead of
        N individual read_node() calls, cutting recall latency from O(N) HTTP
        round-trips to 2 total (embeddings fetch + batch node fetch).
        """
        resp = self._exec(
            self._nodes()
            .select("id, embedding")
            .eq("project", self._project)
            .not_.is_("embedding", "null")
        )
        rows = resp.data or []
        scored: list[tuple[float, str]] = []
        for row in rows:
            emb_list = row.get("embedding")
            if not emb_list:
                continue
            emb_bytes = _list_to_emb(emb_list)
            score = cosine_similarity(embedding, emb_bytes)
            scored.append((score, row["id"]))
        scored.sort(reverse=True)
        top_ids = [nid for _, nid in scored[:limit]]
        if not top_ids:
            return []
        resp2 = self._exec(
            self._nodes()
            .select("*")
            .eq("project", self._project)
            .in_("id", top_ids)
        )
        id_to_node = {r["id"]: self._row_to_node(r) for r in (resp2.data or [])}
        return [id_to_node[nid] for nid in top_ids if nid in id_to_node]

    def search_hybrid(
        self, query: str, embedding: bytes, limit: int = 10, alpha: float = 0.5
    ) -> list[Node]:
        """Reciprocal Rank Fusion of keyword + vector results."""
        fetch = limit * 2
        bm25 = self.search_keyword(query, limit=fetch)
        vec = self.search_vector(embedding, limit=fetch)
        bm25_rank = {n.id: i for i, n in enumerate(bm25)}
        vec_rank = {n.id: i for i, n in enumerate(vec)}
        all_ids = set(bm25_rank) | set(vec_rank)
        k = 60
        scores = {
            nid: (
                alpha * (1.0 / (k + vec_rank.get(nid, fetch)))
                + (1 - alpha) * (1.0 / (k + bm25_rank.get(nid, fetch)))
            )
            for nid in all_ids
        }
        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]
        return [n for nid in top_ids if (n := self.read_node(nid)) is not None]

    def delete_node(self, id: str) -> None:
        self._exec(self._nodes().delete().eq("project", self._project).eq("id", id))

    def list_topics(self) -> list[str]:
        resp = self._exec(
            self._nodes()
            .select("label")
            .eq("project", self._project)
            .eq("node_type", "topic")
        )
        return [r["label"] for r in (resp.data or [])]

    def get_edges(self, node_id: str) -> list[Edge]:
        resp_from = self._exec(
            self._edges()
            .select("*")
            .eq("project", self._project)
            .eq("from_id", node_id)
            .is_("invalid_at", "null")
        )
        resp_to = self._exec(
            self._edges()
            .select("*")
            .eq("project", self._project)
            .eq("to_id", node_id)
            .is_("invalid_at", "null")
        )
        edges = []
        seen: set[str] = set()
        for row in (resp_from.data or []) + (resp_to.data or []):
            if row["id"] not in seen:
                seen.add(row["id"])
                edges.append(self._row_to_edge(row))
        return edges

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        resp = self._exec(
            self._nodes()
            .select("*")
            .eq("project", self._project)
            .eq("node_type", node_type)
        )
        return [self._row_to_node(r) for r in (resp.data or [])]

    def stats(self) -> dict:
        resp = self._exec(
            self._nodes()
            .select("node_type")
            .eq("project", self._project)
        )
        type_counts: dict[str, int] = {}
        for r in (resp.data or []):
            t = r["node_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        resp_e = self._exec(
            self._edges()
            .select("id")
            .eq("project", self._project)
            .is_("invalid_at", "null")
        )
        edge_count = len(resp_e.data or [])
        return {
            "node_count": sum(type_counts.values()),
            "entity_count": type_counts.get("entity", 0),
            "chunk_count": type_counts.get("memory_chunk", 0),
            "topic_count": type_counts.get("topic", 0),
            "tone_count": type_counts.get("tone", 0),
            "edge_count": edge_count,
            "session_count": 0,  # sessions not tracked in cloud store
        }

    def export_pack(self, path: str) -> None:
        """Dump all nodes/edges to a local .mempack via a temporary SQLiteStore."""
        from memory_agent.store.local import SQLiteStore
        from memory_agent.config import StorageConfig
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(self._project, StorageConfig(path=tmp))
            # Copy nodes
            resp = self._exec(
                self._nodes()
                .select("*")
                .eq("project", self._project)
            )
            for row in (resp.data or []):
                node = self._row_to_node(row)
                store.write_node(node)
            # Copy edges
            resp_e = self._exec(
                self._edges()
                .select("*")
                .eq("project", self._project)
            )
            for row in (resp_e.data or []):
                store.write_edge(self._row_to_edge(row))
            store.export_pack(path)

    def import_pack(self, path: str) -> None:
        """Import a .mempack, upserting all nodes and edges into Supabase."""
        from memory_agent.store.local import SQLiteStore
        from memory_agent.config import StorageConfig

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(self._project, StorageConfig(path=tmp))
            store.import_pack(path)
            for node in store.get_nodes_by_type("entity") + \
                         store.get_nodes_by_type("memory_chunk") + \
                         store.get_nodes_by_type("topic") + \
                         store.get_nodes_by_type("tone"):
                self.write_node(node)
            resp = store._conn.execute("SELECT * FROM edges").fetchall()
            import sqlite3
            store._conn.row_factory = sqlite3.Row
            for row in store._conn.execute("SELECT * FROM edges").fetchall():
                self.write_edge(Edge(
                    id=row["id"],
                    from_id=row["from_id"],
                    to_id=row["to_id"],
                    relation=row["relation"] or "",
                    weight=row["weight"] or 1.0,
                    source_chunk_id=row["source_chunk_id"],
                    created_at=row["created_at"] or "",
                    valid_at=row["valid_at"],
                    invalid_at=row["invalid_at"],
                ))

    def close(self) -> None:
        pass  # supabase client has no explicit close
