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

**Test run logs:** Every `pytest` run creates a timestamped WARNING-level log file in `logs/` at the repo root (e.g. `logs/2026-04-13_09-19-34.log`). These capture ingestion pipeline warnings — retry attempts, zero-entity extractions, and JSON parse fallbacks — without cluttering the terminal output. The `logs/` directory is created automatically on first run.

### Multi-dataset V2 benchmark (`tests/simulation/complex_sim_test/`)

A more demanding simulation suite that tests three independent domains loaded either separately or together into a single memory store.

**Datasets:**

| Dataset | Source | Documents | Domain |
|---------|--------|-----------|--------|
| Veloris Technologies | 20 emails | Engineering communications | B2B SaaS |
| Hargrove Family Medicine | 20 clinical notes | Patient records, specialist referrals | Medical |
| Castellan & Briggs LLP | 20 case notes | Four active legal matters | Legal |

**Key design property:** The three domains share zero vocabulary. Any fact about Arthur Pemberton (medical) cannot be confused with a fact about Harlan Voss (legal). Cross-dataset contamination in answers is a hallucination, not a retrieval ambiguity.

**Intentional trap:** The name "Marcus Webb" appears across all three datasets as a different person each time (Veloris CTO, Terravast general counsel, Kellerman & Drape supervisor). Conflating them is a measured failure mode.

**Three test functions:**

| Test | Command | What it does |
|------|---------|--------------|
| `test_medical_benchmark` | `-k medical` | Ingests 20 clinical notes, asks 60 questions (T1/T2/T3 + unanswerable) |
| `test_legal_benchmark` | `-k legal` | Ingests 20 case notes, asks 60 questions (T1/T2/T3 + unanswerable) |
| `test_combined_benchmark` | `-k combined` | Ingests all 60 documents into one store, asks all 180 questions (60 per dataset) |

**Question tiers per dataset (60 questions each):**

| Tier | Count | Notes |
|------|-------|-------|
| T1 — Direct lookup | 20 | Single explicit facts |
| T2 — Single hop | 20 | Connecting facts across documents |
| T3 — Multi-hop | 10 | 3+ step chains; tests graph traversal advantage |
| U — Unanswerable | 10 | Correct decline expected; each dataset has one trap question |

**Trap questions:**
- `M-U-04`: "What caused Raymond Chu's small vessel disease?" — answerable as "likely hypertension-related per radiology"
- `L-U-06`: "What is Derek Briggs's area of specialty?" — inferable as employment law from the Voss matter
- `V-U-05`: "Which cloud provider does Veloris use?" — answerable as AWS (mentioned in email 003)

**Persistent memory:** Each test writes its SQLite database to `tests/simulation/complex_sim_test/memory/{medical,legal,combined}/`. Databases persist after the run for inspection with the built-in visualizer:

```bash
pocket-mem show --path tests/simulation/complex_sim_test/memory/legal
pocket-mem show --path tests/simulation/complex_sim_test/memory/combined
```

Delete the relevant subfolder before a run to start fresh.

**Results files** written after each run:
- `last_run_results_medical.txt`
- `last_run_results_legal.txt`
- `last_run_results_combined.txt` — single file with all 180 answers in three labelled sections

```bash
# Run all three
pytest tests/simulation/complex_sim_test/ -v -s

# Run combined only (180 questions)
pytest tests/simulation/complex_sim_test/ -v -s -k combined
```

**Benchmark results:** All run history, scores, failure analysis, and root cause notes are tracked in:

**[`tests/simulation/complex_sim_test/BENCHMARK_V2.md`](tests/simulation/complex_sim_test/BENCHMARK_V2.md)**

**Run 2 headline result (2026-04-13):** 160.5/180 = 89% overall across all three domains in a single combined store. Medical 96% · Veloris 90% · Legal 82%. Zero cross-dataset contamination across 401 nodes.

---

## Benchmark run history

All Veloris single-dataset benchmark runs, results, model configurations, and root cause analyses are tracked in:

**[`tests/simulation/first_sim_test_50_q/BENCHMARK.md`](tests/simulation/first_sim_test_50_q/BENCHMARK.md)**

This file documents every trial run against the Veloris dataset — what model was used, the scores per tier, latency, and what changed between runs. Consult it before making changes to retrieval or ingestion to understand what has already been tried and what the current baseline is.

---

## Inspecting memory state

```python
print(agent.recall("David", mode="raw"))
```

Use `mode="raw"` as your debugger — see exactly what nodes the graph returned.
