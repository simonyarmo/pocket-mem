"""
query.py — reads SQLite directly, no LLM required.

Builds the graph JSON consumed by the D3 visualizer.

Usage:
  pocket-mem show --path ./memory --project myproject
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_graph(
    path: str = "./memory",
    project: str | None = None,
    topics: list[str] | None = None,
    types: list[str] | None = None,
    since: str | None = None,
    search: str | None = None,
) -> dict:
    """Query memory and return graph dict {nodes[], edges[], meta{}}."""
    db_path = _resolve_db(path, project)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return _build_from_conn(conn, project or db_path.stem, str(db_path),
                                topics, types, since, search)
    finally:
        conn.close()


def _build_from_conn(
    conn: sqlite3.Connection,
    proj_name: str,
    db_label: str,
    topics: list[str] | None,
    types: list[str] | None,
    since: str | None,
    search: str | None,
) -> dict:
    """Build graph dict from an open SQLite connection."""
    total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    node_ids = _filter_node_ids(conn, topics=topics, types=types,
                                since=since, search=search)
    nodes = _fetch_nodes(conn, node_ids)
    edges = _fetch_edges(conn, node_ids)

    filters_applied = []
    if topics:
        filters_applied += [f"topic:{t}" for t in topics]
    if types:
        filters_applied += [f"type:{t}" for t in types]
    if since:
        filters_applied.append(f"since:{since}")
    if search:
        filters_applied.append(f"search:{search}")

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "project": proj_name,
            "db_path": db_label,
            "total_nodes": total_nodes,
            "filtered_nodes": len(nodes),
            "filters_applied": filters_applied,
        },
    }


# ---------------------------------------------------------------------------
# DB discovery
# ---------------------------------------------------------------------------

def _resolve_db(path: str, project: str | None) -> Path:
    base = Path(path)
    if project:
        db = base / f"{project}.db"
        if not db.exists():
            raise FileNotFoundError(
                f"No database found at {db}. "
                f"Run MemoryAgent(project={project!r}, path={path!r}) first."
            )
        return db

    if not base.exists():
        raise FileNotFoundError(
            f"Memory directory not found: {base}. "
            "Run your agent first to create a database."
        )

    dbs = sorted(base.glob("*.db"))
    if not dbs:
        raise FileNotFoundError(f"No .db files found in {base}.")
    if len(dbs) == 1:
        return dbs[0]

    names = ", ".join(d.stem for d in dbs)
    raise ValueError(
        f"Multiple databases found in {base}: {names}. "
        "Specify one with --project."
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_node_ids(
    conn: sqlite3.Connection,
    topics: list[str] | None,
    types: list[str] | None,
    since: str | None,
    search: str | None,
) -> set[str]:
    """Return the set of node IDs that pass all filters. Empty set = all nodes."""

    # Start with all node IDs
    all_ids: set[str] = {
        r[0] for r in conn.execute("SELECT id FROM nodes").fetchall()
    }

    active_filters: list[set[str]] = []

    # --- topic filter ---
    if topics:
        ph = ",".join("?" * len(topics))
        topic_node_ids = {
            r[0] for r in conn.execute(
                f"SELECT id FROM nodes WHERE node_type='topic' AND label IN ({ph})",
                topics,
            ).fetchall()
        }
        if topic_node_ids:
            # find all nodes connected to those topic nodes via edges
            connected: set[str] = set(topic_node_ids)
            for tid in topic_node_ids:
                rows = conn.execute(
                    "SELECT from_id, to_id FROM edges WHERE from_id=? OR to_id=?",
                    (tid, tid),
                ).fetchall()
                for r in rows:
                    connected.add(r[0])
                    connected.add(r[1])
            active_filters.append(connected)
        else:
            # topics not found → empty result
            active_filters.append(set())

    # --- type filter ---
    if types:
        ph = ",".join("?" * len(types))
        type_ids = {
            r[0] for r in conn.execute(
                f"SELECT id FROM nodes WHERE node_type IN ({ph})", types
            ).fetchall()
        }
        active_filters.append(type_ids)

    # --- since filter ---
    if since:
        cutoff = _parse_since(since)
        since_ids = {
            r[0] for r in conn.execute(
                "SELECT id FROM nodes WHERE updated_at >= ?",
                (cutoff.isoformat(),),
            ).fetchall()
        }
        active_filters.append(since_ids)

    # --- search filter ---
    if search:
        try:
            fts_term = _to_fts_query(search)
            search_ids = {
                r[0] for r in conn.execute(
                    "SELECT id FROM nodes_fts WHERE nodes_fts MATCH ?",
                    (fts_term,),
                ).fetchall()
            }
        except sqlite3.OperationalError:
            # FTS not available or bad query — fallback to label ilike
            search_ids = {
                r[0] for r in conn.execute(
                    "SELECT id FROM nodes WHERE label LIKE ?",
                    (f"%{search}%",),
                ).fetchall()
            }
        active_filters.append(search_ids)

    if not active_filters:
        return all_ids

    result = active_filters[0]
    for s in active_filters[1:]:
        result = result & s
    return result


def _fetch_nodes(conn: sqlite3.Connection, node_ids: set[str]) -> list[dict]:
    if not node_ids:
        return []
    ph = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT id, node_type, label, data, importance, created_at, updated_at "
        f"FROM nodes WHERE id IN ({ph})",
        list(node_ids),
    ).fetchall()

    nodes = []
    for r in rows:
        data = json.loads(r["data"]) if r["data"] else {}
        node: dict = {
            "id": r["id"],
            "label": r["label"],
            "node_type": r["node_type"],
            "entity_type": data.get("entity_type", ""),
            "importance": r["importance"] or 0.5,
            "created_at": r["created_at"] or "",
            "updated_at": r["updated_at"] or "",
            "attributes": data.get("attributes", {}),
            "summary": data.get("summary") or data.get("raw", "")[:200],
            "tone_type": data.get("tone_type", ""),
            "valence": data.get("valence", ""),
        }
        nodes.append(node)
    return nodes


def _fetch_edges(conn: sqlite3.Connection, node_ids: set[str]) -> list[dict]:
    if not node_ids:
        return []
    ph = ",".join("?" * len(node_ids))
    ids = list(node_ids)
    rows = conn.execute(
        f"SELECT id, from_id, to_id, relation, weight, invalid_at "
        f"FROM edges WHERE from_id IN ({ph}) AND to_id IN ({ph})",
        ids + ids,
    ).fetchall()

    return [
        {
            "id": r["id"],
            "from": r["from_id"],
            "to": r["to_id"],
            "relation": r["relation"] or "",
            "weight": r["weight"] or 0.5,
            "valid": r["invalid_at"] is None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "how", "who",
    "when", "where", "why", "which", "did", "does", "do", "for", "and",
    "or", "but", "in", "on", "at", "to", "from", "by", "with",
}


def _to_fts_query(query: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    content = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return " OR ".join(content) if content else query


def _parse_since(since: str) -> datetime:
    """Parse '7d', '30d', or ISO date string into a datetime."""
    m = re.match(r"^(\d+)d$", since.strip())
    if m:
        return datetime.now(timezone.utc) - timedelta(days=int(m.group(1)))
    return datetime.fromisoformat(since.strip())
