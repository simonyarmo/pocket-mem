# TESTING.md

## North star metric
**Cross-session recall score** — fraction of questions only answerable via stored memory (not in-context history) that the agent gets right. Target: beat Mem0's ~70–80% baseline on LoCoMo benchmark.

## Three layers

**Layer 1 — Unit (no LLM, <5s)**
Mock `LLMClient` at class level. Test: schema init, node write/read, edge traversal, FTS5 search, vector similarity, recall formatting, compactor grouping.
```bash
pytest tests/unit/ -v
```

**Layer 2 — Behavioral (real LLM, fixed scenarios)**
30 synthetic scenarios, each with setup turns and graded questions.
```python
SCENARIOS = [
  {
    "id": "person_tool",
    "setup": [("user","My boss David recommended Cursor IDE")],
    "questions": [
      {"q": "What did David recommend?", "must_contain": ["Cursor"]},
      {"q": "Who is David?",             "must_contain": ["boss"]},
    ]
  },
  # ... 29 more covering: person+attr, tool rec, decision, project, preference, multi-hop
]
```
Scoring: exact match (must_contain), F1 word overlap, LLM judge 0–1.
```bash
pytest tests/behavioral/ -v
```

**Layer 3 — Simulation (end-to-end, multi-session)**
Fake host agent. Each session starts fresh — only memory bridges sessions.
```python
class MemoryTestHarness:
    def run_session(self, session_id, turns):
        agent = MemoryAgent(project="test", session_id=session_id)
        for role, content in turns:
            if role == "user":
                agent.observe(user_input=content, agent_response="(noted)")

    def ask(self, question):
        agent = MemoryAgent(project="test")   # new session, no shared context
        return agent.recall(question, mode="answer")

    def score(self, answer, expected_terms):
        return sum(t.lower() in answer.lower() for t in expected_terms) / len(expected_terms)
```
```bash
pytest tests/simulation/ -v
```

## Scenario categories (30 total)
| Category | Count | Multi-hop? |
|----------|-------|-----------|
| Person + attribute | 5 | No |
| Person + tool rec | 5 | No |
| Decision memory | 5 | No |
| Project context | 5 | No |
| Preference memory | 5 | No |
| Multi-hop (person→event→tool) | 5 | **Yes** — hardest |

Multi-hop is the graph's key advantage over flat vector search.

## Benchmark tracking
Append to `tests/benchmarks/results.json` after each run:
```json
{"2025-01-15": {"model":"qwen2.5:7b","cross_session_recall":0.71,"f1":0.78,"passed":21,"total":30}}
```

Targets: v0.1 > 0.50 · v0.2 > 0.65 · v0.3 > 0.75 · v1.0 > 0.80

## Inspecting memory state
```python
print(json.dumps(agent.recall("David", mode="raw"), indent=2))
```
Use `mode="raw"` as your debugger — see exactly what's in the graph.
