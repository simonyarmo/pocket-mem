# pocket-mem versions

## v1.0 — current release

### Core memory
- **Knowledge graph storage** — entities, relationships, memory chunks, topics, and tone nodes stored in a local SQLite database
- **Background ingestion** — `observe()` returns immediately; extraction runs on a background thread and never blocks your agent
- **Entity extraction** — people, tools, projects, and typed relationships extracted from conversation turns via LLM
- **Topic classification** — conversations automatically grouped into topics (e.g. "People I Know", "Dev Tools", "Decisions")
- **Memory compaction** — background compression and pruning keeps the graph from growing unbounded over long sessions

### Retrieval
- **Hybrid search** — BM25 full-text search (FTS5) combined with semantic vector similarity for accurate recall
- **Graph traversal** — multi-hop retrieval follows edges across the knowledge graph to surface connected facts
- **Three recall modes** — `context` (formatted string for system prompts), `answer` (LLM-synthesized response), `raw` (graph data for debugging)

### LLM flexibility
- Works with any OpenAI-compatible API — Ollama (local), Claude, OpenAI, Groq, Mistral, Together AI, and others
- Separate configuration for ingestion model and answer model — use a fast local model for extraction and a cloud model for synthesis
- Default model: `qwen2.5:7b` via Ollama (free, local, no API key required)

### Persistence and portability
- **Session persistence** — memory survives full process restarts; new sessions automatically load prior memory
- **Export / import** — pack any project's memory into a `.mempack` zip file and share or restore it on any machine
- **User isolation** — `user_id` parameter scopes memory within a shared project

### Visualizer
- **`pocket-mem show`** — browser-based graph explorer with filtering by topic, node type, date range, and keyword search

### Testing
- 218 unit tests covering every layer of the stack (storage, retrieval, ingestion, compaction, LLM client, config, packaging)
- Veloris benchmark: 60-question simulation across direct lookup, single-hop, multi-hop, and unanswerable categories
- 98% accuracy on answerable questions using Claude Haiku as the answer model

---

## v2.0 — planned

### Cloud storage backend
- Supabase adapter — store the knowledge graph in a hosted Postgres database instead of local SQLite
- Multi-user shared memory — multiple agents on different machines write to and read from a single shared graph
- `path="supabase://"` routing in `MemoryAgent`

### Docker HTTP sidecar
- FastAPI HTTP server wrapping pocket-mem, so any language (Node, Go, Ruby, etc.) can use memory over HTTP
- `POST /observe`, `POST /recall`, `GET /stats`, `GET /health` endpoints
- Docker Compose setup pairing the sidecar with Ollama for a fully self-hosted stack

### Cloud retrieval performance
- Batch graph traversal — collapse the N+1 Supabase HTTP calls in `traverse()` into bulk queries
- Automatic DDL — create required tables on first use rather than requiring manual SQL setup
- JSONB full-text search — keyword search across memory chunk content (not just entity labels)

### Visualizer improvements
- `pocket-mem show --path supabase://` — export cloud data to temp SQLite and visualize it
