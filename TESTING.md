# TESTING.md

## North star metric
**Cross-session recall score** — fraction of questions only answerable via stored memory (not in-context history) that the agent gets right. Target: beat Mem0's ~70–80% baseline on LoCoMo benchmark.

---

## Layer 1 — Unit tests (`tests/unit/`)

No LLM required. All external calls are mocked. Suite runs in under 15 seconds.

The unit tests cover every layer of the stack:

- **Data model** — node and edge schema validation, field defaults, type constraints
- **Storage (SQLite)** — write/read nodes and edges, FTS5 full-text search, vector similarity queries, topic listing, stats counting, export/import `.mempack`
- **Embedding** — vector generation, dimensionality, determinism across calls
- **LLM client** — prompt formatting, response parsing, token tracking, mock injection
- **Ingestion pipeline** — entity extraction, memory chunk creation, deduplication, session tagging
- **Retrieval** — hybrid search (BM25 + vector), graph traversal, context formatting, `as_tool()` schema
- **Compactor** — grouping logic for compression and pruning thresholds
- **Agent facade** — `observe()` queues to background thread, `flush()` drains executor, `_maybe_compact()` fires on background callback
- **Config** — `LLMConfig`, `StorageConfig`, `MemoryConfig` defaults and env-var overrides
- **Packaging** — `pyproject.toml` has correct name, Python requirement, required deps, and that the `[cloud]` extra is absent from v1

```bash
pytest tests/unit/ -v
```

---

## Layer 2 — Behavioral tests (`tests/behavioral/`)

Real LLM, targeted scenarios. Each test injects a small number of `observe()` turns then immediately queries.

```bash
pytest tests/behavioral/ -v
```

---

## Layer 3 — Simulation tests (`tests/simulation/`)

Full end-to-end multi-session simulation. The agent is shut down completely between ingestion and recall so no in-context state can leak — only the SQLite graph bridges sessions.

### Veloris benchmark (`tests/simulation/first_sim_test_50_q/`)

The primary simulation test. Uses a synthetic but realistic dataset:

**Dataset:** 20 emails from a fictional B2B SaaS company (Veloris Technologies), covering four concurrent engineering initiatives (Conduit rewrite, Nightwatch monitoring migration, Atlas mobile app, security audit). Eight named people. Facts are deliberately spread across multiple emails — no single email contains all the information needed to answer any multi-hop question.

**Test structure:**
1. **Ingestion** — all 20 emails fed to `observe()`, then `agent.flush()` to drain the background queue
2. **Restart** — agent is re-instantiated with the same project name (no shared Python state)
3. **Query** — all 60 questions asked via `recall(mode="answer")`
4. **Results** — written to `last_run_results.txt` with the answer and latency for every question

**Question tiers:**

| Tier | Count | What it tests |
|------|-------|---------------|
| T1 — Direct lookup | 20 | Single facts stated explicitly in one email |
| T2 — Single hop | 20 | Connecting two pieces of information from different emails |
| T3 — Multi-hop | 10 | Traversing 3+ connected facts; graph traversal advantage |
| U — Unanswerable | 10 | Correct "I don't know" responses; one trap question (U-05, cloud provider = AWS) |

**Scoring:**
- Correct (1.0): answer contains the key facts
- Partial (0.5): some correct facts, missing others
- Incorrect (0.0): wrong or confidently fabricated
- False positive (flagged): confident wrong answer on a genuinely unanswerable question
- Trap test (U-05): agent must answer "AWS" — saying "I don't know" is a failure

```bash
pytest tests/simulation/ -v -s
```

Results are written to `tests/simulation/first_sim_test_50_q/last_run_results.txt`.

---

## Benchmark run history

All benchmark runs, results, model configurations, and root cause analyses are tracked in:

**[`tests/simulation/first_sim_test_50_q/BENCHMARK.md`](tests/simulation/first_sim_test_50_q/BENCHMARK.md)**

This file documents every trial run against the Veloris dataset — what model was used, the scores per tier, latency, and what changed between runs. Consult it before making changes to retrieval or ingestion to understand what has already been tried and what the current baseline is.

---

## Inspecting memory state

```python
print(agent.recall("David", mode="raw"))
```

Use `mode="raw"` as your debugger — see exactly what nodes the graph returned.
