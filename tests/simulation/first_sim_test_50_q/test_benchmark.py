"""
Veloris Technologies benchmark — full end-to-end simulation.

Ingests 20 emails, restarts the agent (session persistence), queries 60 questions,
auto-scores unanswerable/trap questions, writes human-readable results.

Run:  pytest tests/simulation/ -v -s
Out:  tests/simulation/last_run_results.txt
"""
from __future__ import annotations

import os
import re
import time
import socket
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pytest

from memory_agent import MemoryAgent, LLMConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR = Path(__file__).parent
_DATA_DIR = SIM_DIR.parent / "test_data"
EMAILS_FILE = _DATA_DIR / "test_emails.txt"
ANSWER_KEY_FILE = _DATA_DIR / "answer_key.txt"
RESULTS_FILE = SIM_DIR / "last_run_results.txt"

MODEL = "qwen2.5:7b"
OLLAMA_BASE = "http://localhost:11434/v1"

CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

def _load_claude_key() -> str | None:
    """Read CLAUDE_API_KEY from .env at repo root, falling back to environment."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLAUDE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("CLAUDE_API_KEY")

# ---------------------------------------------------------------------------
# Question set (from BENCHMARK.md)
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Tier 1 — Direct lookup
    {"id": "T1-01", "tier": 1, "q": "What is Marcus Webb's job title?"},
    {"id": "T1-02", "tier": 1, "q": "Who is Leo Reyes's manager?"},
    {"id": "T1-03", "tier": 1, "q": "What is the name of Veloris's main product?"},
    {"id": "T1-04", "tier": 1, "q": "How much was Veloris paying for Datadog per month?"},
    {"id": "T1-05", "tier": 1, "q": "What city is Veloris headquartered in?"},
    {"id": "T1-06", "tier": 1, "q": "What message queue technology did Darnell propose for the Conduit rewrite?"},
    {"id": "T1-07", "tier": 1, "q": "What analytics store did Priya originally suggest for Conduit?"},
    {"id": "T1-08", "tier": 1, "q": "What analytics store did Darnell propose instead, and why?"},
    {"id": "T1-09", "tier": 1, "q": "What is the test coverage percentage of the Atlas payments module?"},
    {"id": "T1-10", "tier": 1, "q": "How much money does Veloris process monthly through the payments module?"},
    {"id": "T1-11", "tier": 1, "q": "What build toolchain is Suki using for the React Native app?"},
    {"id": "T1-12", "tier": 1, "q": "What is the name of the internal monitoring project?"},
    {"id": "T1-13", "tier": 1, "q": "What security firm did Marcus choose for the audit?"},
    {"id": "T1-14", "tier": 1, "q": "What was the other security firm Marcus considered?"},
    {"id": "T1-15", "tier": 1, "q": "When does the security audit start?"},
    {"id": "T1-16", "tier": 1, "q": "What crash reporting tool does the Atlas mobile app use?"},
    {"id": "T1-17", "tier": 1, "q": "How many enterprise customers' users signed up for the Atlas mobile beta?"},
    {"id": "T1-18", "tier": 1, "q": "What is the actual monthly cost of the self-hosted Nightwatch stack?"},
    {"id": "T1-19", "tier": 1, "q": "Who is Tobias Hunt's direct manager?"},
    {"id": "T1-20", "tier": 1, "q": "What training did Marcus require all engineers to complete by end of February?"},
    # Tier 2 — Single hop
    {"id": "T2-01", "tier": 2, "q": "Who did Marcus assign to lead the Conduit rewrite?"},
    {"id": "T2-02", "tier": 2, "q": "What specific part of the Conduit system is Tobias responsible for?"},
    {"id": "T2-03", "tier": 2, "q": "What technology did Priya explicitly ban from the Conduit rewrite, and why?"},
    {"id": "T2-04", "tier": 2, "q": "What was Darnell's preferred timeline for Conduit if the backfill feature was dropped?"},
    {"id": "T2-05", "tier": 2, "q": "Why did Priya insist on keeping the backfill feature?"},
    {"id": "T2-06", "tier": 2, "q": "What log aggregation tool is part of the new Nightwatch stack?"},
    {"id": "T2-07", "tier": 2, "q": "What is the projected annual saving from the Datadog migration?"},
    {"id": "T2-08", "tier": 2, "q": "What was Leo's fallback option if Loki had problems?"},
    {"id": "T2-09", "tier": 2, "q": "What phase of the Atlas mobile app includes offline sync?"},
    {"id": "T2-10", "tier": 2, "q": "Why is the Atlas backend off-limits for mobile development in Phase 1?"},
    {"id": "T2-11", "tier": 2, "q": "What compliance issue did Leo discover in the payment logging?"},
    {"id": "T2-12", "tier": 2, "q": "Who discovered the race condition in the payment refund logic?"},
    {"id": "T2-13", "tier": 2, "q": "Where are the Stripe test credentials stored?"},
    {"id": "T2-14", "tier": 2, "q": "What problem did Suki encounter with Expo related to Meridian Group's requirements?"},
    {"id": "T2-15", "tier": 2, "q": "What solution did Suki recommend for the Expo biometric bug?"},
    {"id": "T2-16", "tier": 2, "q": "What Conduit throughput did Darnell achieve in load testing, and how did it compare to the target?"},
    {"id": "T2-17", "tier": 2, "q": "What p99 latency did Conduit achieve in load testing?"},
    {"id": "T2-18", "tier": 2, "q": "Where did Leo document the LogQL learning guide?"},
    {"id": "T2-19", "tier": 2, "q": "What is the mobile beta launch date?"},
    {"id": "T2-20", "tier": 2, "q": "What known issue does Sentry's React Native SDK have?"},
    # Tier 3 — Multi-hop
    {"id": "T3-01", "tier": 3, "q": "Which engineer was originally assigned to write tests for the payments module but was pulled away, and what were they pulled onto?"},
    {"id": "T3-02", "tier": 3, "q": "Who flagged the PCI compliance issue, and what three actions did Marcus take as a result?"},
    {"id": "T3-03", "tier": 3, "q": "What technology stack replaced Datadog, and which team member was involved in both the replacement and the Conduit rewrite?"},
    {"id": "T3-04", "tier": 3, "q": "The mobile beta includes a customer with special authentication requirements. What is that customer, what was their requirement, and how was it solved?"},
    {"id": "T3-05", "tier": 3, "q": "How many dashboards did the new monitoring system have after migration, and how does that compare to before?"},
    {"id": "T3-06", "tier": 3, "q": "Who reported a concern to Priya about test coverage, what module was the concern about, and who was ultimately assigned to fix it?"},
    {"id": "T3-07", "tier": 3, "q": "Which project involves both Suki Tanaka and Camille Russo, and what did Camille change about Suki's original scope?"},
    {"id": "T3-08", "tier": 3, "q": "Tobias Hunt worked on two separate things in Q1. What were they, and on which one did his manager say he deserved a performance note?"},
    {"id": "T3-09", "tier": 3, "q": "What was the actual monthly cost saving from the Nightwatch migration, and how did it compare to Leo's original estimate?"},
    {"id": "T3-10", "tier": 3, "q": "Which engineer discovered a bug while doing a different task, what was the task, and what was the bug?"},
    # Unanswerable
    {"id": "U-01", "tier": 0, "q": "What is Priya Nair's salary?"},
    {"id": "U-02", "tier": 0, "q": "What programming language is the Atlas backend written in?"},
    {"id": "U-03", "tier": 0, "q": "How many total employees does Veloris have?"},
    {"id": "U-04", "tier": 0, "q": "What did Marcus think about Python as a language?"},
    {"id": "U-05", "tier": 0, "q": "Which cloud provider does Veloris use?"},  # TRAP — answer is AWS
    {"id": "U-06", "tier": 0, "q": "What is Hana Bergström's salary?"},
    {"id": "U-07", "tier": 0, "q": "Has Veloris ever had a data breach?"},
    {"id": "U-08", "tier": 0, "q": "What is Tobias Hunt's favorite programming language?"},
    {"id": "U-09", "tier": 0, "q": "Does Veloris have offices outside Austin?"},
    {"id": "U-10", "tier": 0, "q": "What database does the Atlas backend use?"},
]

_UNCERTAINTY_MARKERS = [
    "not mentioned", "don't know", "no information", "cannot determine",
    "not in the", "i don't", "no data", "not available", "not provided",
    "not stated", "not specified", "unable to find", "no record",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=2)
        s.close()
        return True
    except OSError:
        return False


def parse_emails(text: str) -> list[dict]:
    blocks = re.split(r"\n---\n", text)
    emails = []
    for block in blocks:
        block = block.strip()
        if not re.match(r"^EMAIL \d+", block):
            continue
        lines = block.split("\n")
        h: dict = {}
        body_start = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.startswith("From:"):
                h["from"] = line[5:].strip()
            elif line.startswith("To:"):
                h["to"] = line[3:].strip()
            elif line.startswith("Date:"):
                h["date"] = line[5:].strip()
            elif line.startswith("Subject:"):
                h["subject"] = line[8:].strip()
            elif line == "" and "subject" in h:
                body_start = i + 1
                break
        h["body"] = "\n".join(lines[body_start:]).strip()
        emails.append(h)
    return emails


def parse_answer_key(text: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    current_id: str | None = None
    for line in text.split("\n"):
        m = re.match(r"^(T[123]-\d+|U-\d+): ", line)
        if m:
            current_id = m.group(1)
        elif line.startswith("ANSWER: ") and current_id:
            answers[current_id] = line[8:].strip()
            current_id = None
    return answers


def auto_score(q_id: str, answer: str) -> str:
    a = answer.lower()
    if q_id == "U-05":
        return "PASS (trap)" if "aws" in a else "FAIL — missed trap (expected AWS)"
    if q_id.startswith("U-"):
        uncertain = any(m in a for m in _UNCERTAINTY_MARKERS)
        return "PASS" if uncertain else "FAIL? (possible false positive — review)"
    return "REVIEW"


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_veloris_benchmark(tmp_path):
    claude_key = _load_claude_key()
    llm_cfg = LLMConfig(
        base_url=OLLAMA_BASE,
        model=MODEL,
        answer_model=CLAUDE_MODEL if claude_key else None,
        answer_base_url=CLAUDE_BASE if claude_key else None,
        answer_api_key=claude_key,
        timeout=120,
    )
    raw = EMAILS_FILE.read_text(encoding="utf-8")
    header = raw.split("\n---\n")[0].strip()
    emails = parse_emails(raw)
    answer_key = parse_answer_key(ANSWER_KEY_FILE.read_text(encoding="utf-8"))
    project = "veloris-benchmark"

    # ------------------------------------------------------------------
    # Phase 1: Ingestion
    # ------------------------------------------------------------------
    print(f"\n[benchmark] Ingesting header + {len(emails)} emails...")
    agent = MemoryAgent(project, path=str(tmp_path), llm=llm_cfg)
    observe_times: list[float] = []

    # Ingest the company/people/projects header first — it contains facts
    # (roles, reporting lines, HQ, project names) not repeated in email bodies.
    if header:
        agent.observe(user_input="Company reference document", agent_response=header)

    for i, email in enumerate(emails, 1):
        t0 = perf_counter()
        agent.observe(
            user_input=f"Email from {email['from']} to {email['to']}: {email['subject']}",
            agent_response=email["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{i:02d}/{len(emails)}] {email['subject'][:60]}")

    print("[benchmark] Flushing background ingestion...")
    agent.flush()
    ingest_tokens = agent.token_stats()
    stats_after = agent.stats()
    agent.close()
    print(f"[benchmark] Ingestion complete. nodes={stats_after['node_count']} tokens={ingest_tokens}")

    assert stats_after["node_count"] > 0, "No nodes stored after ingestion"

    # ------------------------------------------------------------------
    # Phase 2: Restart (session persistence)
    # ------------------------------------------------------------------
    print("[benchmark] Restarting agent (session persistence test)...")
    agent = MemoryAgent(project, path=str(tmp_path), llm=llm_cfg)
    stats_after_restart = agent.stats()
    session_persistence = stats_after_restart["node_count"] > 0
    print(f"[benchmark] Post-restart nodes={stats_after_restart['node_count']}  persistence={'PASS' if session_persistence else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 3: Query
    # ------------------------------------------------------------------
    print(f"[benchmark] Querying {len(QUESTIONS)} questions...")
    results: dict[str, str] = {}
    recall_times: list[float] = []

    for q in QUESTIONS:
        t0 = perf_counter()
        answer = agent.recall(q["q"], mode="answer")
        elapsed = perf_counter() - t0
        recall_times.append(elapsed)
        results[q["id"]] = answer
        score = auto_score(q["id"], answer)
        print(f"  {q['id']} [{elapsed*1000:.0f}ms] {score}")

    total_tokens = agent.token_stats()
    agent.close()

    # ------------------------------------------------------------------
    # Phase 4: Write results file
    # ------------------------------------------------------------------
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w(f"=== VELORIS BENCHMARK — pocket-mem ===")
    w(f"Run: {ts}   Ingest model: {MODEL}   Answer model: {llm_cfg.answer_model or MODEL}")
    w()
    w("--- INGESTION ---")
    w(f"Emails ingested : {len(emails)}")
    w(f"Observe p50     : {percentile(observe_times, 50)*1000:.0f}ms")
    w(f"Observe p99     : {percentile(observe_times, 99)*1000:.0f}ms")
    w(f"Tokens (ingest) : in={ingest_tokens['tokens_in']}  out={ingest_tokens['tokens_out']}")
    w(f"Nodes stored    : {stats_after['node_count']}")
    w()
    w("--- SESSION PERSISTENCE ---")
    w("PASS" if session_persistence else "FAIL")
    w()

    tier_labels = {1: "TIER 1 — DIRECT LOOKUP", 2: "TIER 2 — SINGLE HOP",
                   3: "TIER 3 — MULTI-HOP", 0: "UNANSWERABLE (U-01 to U-10)"}
    current_tier = None
    u_auto_pass = 0
    u_auto_total = 0

    for q in QUESTIONS:
        if q["tier"] != current_tier:
            current_tier = q["tier"]
            w()
            w(f"--- {tier_labels[current_tier]} ---")
            w()

        answer = results[q["id"]]
        expected = answer_key.get(q["id"], "—")
        score = auto_score(q["id"], answer)

        if q["tier"] == 0:
            u_auto_total += 1
            if score.startswith("PASS"):
                u_auto_pass += 1

        w(f"{q['id']} | {q['q']}")
        w(f"  EXPECTED : {expected}")
        w(f"  ANSWER   : {answer.strip()}")
        w(f"  SCORE    : {score}")
        w()

    # Latency summary
    w()
    w("--- LATENCY ---")
    w(f"Recall p50 : {percentile(recall_times, 50)*1000:.0f}ms")
    w(f"Recall p99 : {percentile(recall_times, 99)*1000:.0f}ms")
    w()
    w("--- TOKEN TOTALS ---")
    w(f"tokens_in  : {total_tokens['tokens_in']}")
    w(f"tokens_out : {total_tokens['tokens_out']}")
    haiku_cost = total_tokens["tokens_in"] * 0.00000025 + total_tokens["tokens_out"] * 0.00000125
    w(f"Est. cost (Haiku pricing) : ${haiku_cost:.4f}")
    w()
    w("--- SCORE SUMMARY (fill in manually) ---")
    w("T1 accuracy : __/20")
    w("T2 accuracy : __/20")
    w("T3 accuracy : __/10")
    w(f"U  auto-scored: {u_auto_pass}/{u_auto_total - 1} pass (excl. trap)")
    u05_score = auto_score("U-05", results.get("U-05", ""))
    w(f"Trap (U-05) : {u05_score}")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[benchmark] Results written to {RESULTS_FILE}")
    print(f"[benchmark] U auto-scored: {u_auto_pass}/{u_auto_total - 1}  Trap: {u05_score}")
