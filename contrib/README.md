# contrib — future features (not part of v1)

This directory contains code that was removed from the v1 release because it
needs more work before it's ready for production.

## Contents

| File / Dir | What it is |
|---|---|
| `cloud.py` | Supabase cloud storage backend (`SupabaseAdapter`) |
| `server.py` | FastAPI HTTP sidecar — exposes pocket-mem over HTTP for non-Python callers |
| `Dockerfile` | Container image definition for the HTTP sidecar |
| `docker-compose.yml` | Wires Ollama + the HTTP sidecar together |
| `tests/docker_supabase_show_test/` | End-to-end tests for the Docker + Supabase stack |

## Known issues (why these were deferred)

- **`traverse()` N+1 latency**: Supabase recall makes 40–80 sequential HTTP calls per query (19 s p50).
- **No DDL automation**: `pm_nodes`/`pm_edges` tables must be created manually in Supabase SQL Editor.
- **Hallucination with qwen2.5:7b synthesis**: Fixed by routing to Claude Haiku, but adds external API dependency.

These will be addressed in a future cloud release.
