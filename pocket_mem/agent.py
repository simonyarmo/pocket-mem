from __future__ import annotations
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

from pocket_mem.config import LLMConfig, MemoryConfig, StorageConfig
from pocket_mem.compactor import compress, prune
from pocket_mem.identities import resolve_identity
from pocket_mem.ingestion import ingest
from pocket_mem.llm.client import LLMClient
from pocket_mem.retrieval import as_tool as _as_tool
from pocket_mem.retrieval import recall as _recall
from pocket_mem.retrieval import search
from pocket_mem.store.local import SQLiteStore


class MemoryAgent:

    def __init__(
        self,
        project: str,
        path: str | None = None,
        user_id: str | None = None,
        llm: LLMConfig | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self._project = project
        self._user_id = user_id
        self._config = config or MemoryConfig()
        storage = StorageConfig(path=path or "./memory")
        self._store = SQLiteStore(project, storage)
        self._llm = LLMClient(llm or LLMConfig())
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._compaction_in_progress = False
        self._session_id = str(uuid.uuid4())

        if self._config.identity is not None:
            self._config.identity.derived = resolve_identity(
                self._config.identity.description,
                self._config.identity,
                self._store,
                self._llm,
            )
            self._seed_identity_topics()

    def _seed_identity_topics(self) -> None:
        """Write identity seed_topics as topic nodes if they don't already exist."""
        from pocket_mem.models import Node, Topic
        from pocket_mem.ingestion import _get_or_create_topic

        identity = self._config.identity
        if not identity or not identity.derived:
            return

        existing = {t.lower() for t in self._store.list_topics()}
        now = datetime.now(timezone.utc).isoformat()

        for label in identity.derived.get("seed_topics", []):
            if label.lower() in existing:
                continue
            topic = Topic(id=str(uuid.uuid4()), label=label)
            node = topic.to_node()
            node.data["source"] = "identity_seed"
            node.importance = 0.8
            node.created_at = node.updated_at = now
            self._store.write_node(node)
            existing.add(label.lower())

    def observe(self, user_input: str, agent_response: str) -> None:
        """Queue a conversation turn for background ingestion. Never blocks."""
        def _done_final(future):
            exc = future.exception()
            if exc:
                _log.error("observe() retry also failed, dropping turn: %s", exc, exc_info=exc)
            else:
                self._maybe_compact()

        def _done(future):
            exc = future.exception()
            if exc:
                _log.warning("observe() ingestion failed, retrying once: %s", exc)
                try:
                    retry = self._executor.submit(
                        ingest,
                        user_input,
                        agent_response,
                        self._store,
                        self._llm,
                        self._session_id,
                        self._config,
                        self._config.identity,
                    )
                    retry.add_done_callback(_done_final)
                except RuntimeError:
                    _log.error(
                        "observe() ingestion failed and executor is shut down, dropping turn: %s", exc
                    )
            else:
                self._maybe_compact()

        future = self._executor.submit(
            ingest,
            user_input,
            agent_response,
            self._store,
            self._llm,
            self._session_id,
            self._config,
            self._config.identity,
        )
        future.add_done_callback(_done)

    def _maybe_compact(self) -> None:
        """Fire background compaction if chunk count exceeds threshold."""
        if self._compaction_in_progress:
            return
        if self._store.stats()["chunk_count"] >= self._config.compaction_threshold:
            self._compaction_in_progress = True
            future = self._executor.submit(
                _run_compaction, self._store, self._llm, self._config
            )
            future.add_done_callback(self._on_compact_done)

    def _on_compact_done(self, future) -> None:
        self._compaction_in_progress = False
        exc = future.exception()
        if exc:
            _log.error("compaction failed: %s", exc, exc_info=exc)

    def recall(self, query: str, mode: str = "context") -> list | str:
        """Search memory and return results in the requested mode.

        mode="context"   → formatted context string (default) — full retrieved nodes
        mode="raw"       → list of dicts (JSON-serializable, embedding excluded)
        mode="answer"    → LLM synthesizes an answer from retrieved context
        """
        return _recall(query, self._store, llm=self._llm, mode=mode)

    def forget(self, query: str) -> int:
        """Delete memory nodes that match query. Returns count of deleted nodes.

        Edges connected to deleted nodes are invalidated (invalid_at set) rather
        than hard-deleted, preserving history per the schema design.

        Note: not safe to call concurrently with observe() — a background ingest
        job may write a new node for the same entity between the search and delete.
        For v1, callers should treat forget() as a best-effort operation.
        """
        now = datetime.now(timezone.utc).isoformat()
        nodes = search(query, self._store, limit=5)
        count = 0
        for node in nodes:
            for edge in self._store.get_edges(node.id):
                edge.invalid_at = now
                self._store.write_edge(edge)
            self._store.delete_node(node.id)
            count += 1
        return count

    def topics(self) -> list[str]:
        """Return labels of all topic nodes in memory."""
        return self._store.list_topics()

    def stats(self) -> dict:
        """Return counts of stored nodes, edges, and sessions."""
        return self._store.stats()

    def token_stats(self) -> dict:
        """Return cumulative LLM token counts (tokens_in, tokens_out)."""
        return self._llm.token_stats()

    def as_tool(self) -> dict:
        """Return OpenAI function schema for tool-calling integration (Pattern B)."""
        return _as_tool()

    def export(self, path: str) -> None:
        """Export memory to a .mempack zip file."""
        self._store.export_pack(path)

    def import_pack(self, path: str) -> None:
        """Merge a .mempack into this agent's store. Upserts — no duplicates."""
        self._store.import_pack(path)

    def flush(self) -> None:
        """Block until all pending observe() background tasks have completed."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=2)

    def close(self) -> None:
        """Shut down the background executor and close the store connection."""
        self._executor.shutdown(wait=True)
        self._store.close()


def _run_compaction(store, llm, config: MemoryConfig) -> None:
    """Top-level function for executor.submit() — runs compress then prune."""
    compress(store, llm, identity=config.identity)
    prune(store, config)
