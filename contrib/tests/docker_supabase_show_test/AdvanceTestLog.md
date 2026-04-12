# Supabase / Docker Test Log

Tracks test runs, issues found, and fixes applied for the Docker+Supabase integration test.

---

## Run 1 — 2026-04-09 · qwen2.5:7b · first successful Docker run

**Result:** 20 emails, 130 nodes, session persistence PASS
**Observe:** p50=176ms, p99=807ms
**Recall:** p50=22,958ms, p99=31,527ms

**Issues found:**
- Hallucinated email summaries instead of direct answers (T1-01, T1-13, U-05)
- 23s p50 recall latency (vs ~4s for local SQLite)
- `pocket-mem show` not supported for supabase:// path

**Root causes identified:**
1. `search_vector`: N+1 HTTP round-trips — calls `read_node()` individually for each top-K result (10 sequential Supabase requests per recall)
2. `search_keyword`: only searches node `label` field — memory chunks have UUID labels and are never found by keyword search; hybrid degrades to vector-only
3. `_maybe_compact()`: called synchronously in `observe()`, makes 2 Supabase HTTP calls per observe for stats check (~200ms overhead each)
4. qwen2.5:7b synthesis: generates hallucinated email responses when context contains full email bodies (documented failure in local benchmark Runs 1–9; fixed there by switching to Claude Haiku for synthesis)
5. `query.py`: reads SQLite directly with no Supabase support

**Fixes applied before Run 2:**
1. `search_vector` — batch top-K fetch via single `in_()` query (2 HTTP calls total instead of N+1)
2. `search_keyword` — `or_()` filter on `label`, `data->>raw`, `data->>summary` to mirror SQLite FTS5 coverage
3. `_maybe_compact()` — moved from synchronous call in `observe()` to the ingest `_done` callback (runs on background thread after ingest completes, never blocks HTTP response)
4. `query.py` — added `supabase://` path support via temp SQLite export (`_build_graph_from_supabase()`)

---
