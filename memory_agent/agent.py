from __future__ import annotations
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

_log = logging.getLogger(__name__)

from memory_agent.config import LLMConfig, MemoryConfig, StorageConfig
from memory_agent.compactor import compress, prune
from memory_agent.ingestion import ingest
from memory_agent.llm.client import LLMClient
from memory_agent.retrieval import as_tool as _as_tool
from memory_agent.retrieval import recall as _recall
from memory_agent.retrieval import search
from memory_agent.store.local import SQLiteStore


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

    def observe(self, user_input: str, agent_response: str) -> None:
        """Queue a conversation turn for background ingestion. Never blocks."""
        def _done(future):
            exc = future.exception()
            if exc:
                _log.error("observe() ingestion failed: %s", exc, exc_info=exc)
            else:
                # Run compaction check on the background thread so that cloud
                # stores (Supabase) don't block the HTTP request with a stats()
                # round-trip on every observe() call.
                self._maybe_compact()

        future = self._executor.submit(
            ingest,
            user_input,
            agent_response,
            self._store,
            self._llm,
            self._session_id,
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
        mode="raw"       → list of Node objects
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
        now = datetime.utcnow().isoformat()
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
    compress(store, llm)
    prune(store, config)
