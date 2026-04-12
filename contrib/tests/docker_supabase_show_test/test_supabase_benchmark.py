"""
Veloris benchmark — Supabase / Docker backend.

Two separate tests so ingestion and recall can be run independently:

  Ingest once:
    pytest tests/simulation/docker_supabase_show_test/test_supabase_benchmark.py::test_ingest -v -s

  Iterate on recall (no re-ingestion):
    pytest tests/simulation/docker_supabase_show_test/test_supabase_benchmark.py::test_recall -v -s

Results written to: tests/simulation/docker_supabase_show_test/last_benchmark_results.txt

Local mode (for comparison baseline):
    POCKET_MEM_BACKEND=local pytest ... ::test_ingest -v -s
    POCKET_MEM_BACKEND=local pytest ... ::test_recall -v -s
"""
from __future__ import annotations

import os
import re
import socket
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pytest
import requests

# ── Load .env at repo root ────────────────────────────────────────────────────
def _load_dotenv() -> None:
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() not in os.environ:
            os.environ[key.strip()] = val.strip()

_load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND: str = os.environ.get("POCKET_MEM_BACKEND", "supabase")

TEST_DIR        = Path(__file__).parent
DATA_DIR        = TEST_DIR.parent / "test_data"
EMAILS_FILE     = DATA_DIR / "test_emails.txt"
ANSWER_KEY_FILE = DATA_DIR / "answer_key.txt"
RESULTS_FILE    = TEST_DIR / "last_benchmark_results.txt"
TEST_MEMORY     = TEST_DIR / "test_memory"

DOCKER_URL  = "http://localhost:8000"
PROJECT     = "veloris-docker-test"
MODEL       = "qwen2.5:7b"
OLLAMA_BASE = "http://localhost:11434/v1"
CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ── Question set ──────────────────────────────────────────────────────────────
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
    # Unanswerable (U-05 is the trap — answer IS in data: AWS)
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
    "not explicitly", "not directly", "i do not have", "does not provide",
    "not found", "no mention",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _docker_healthy() -> bool:
    try:
        return requests.get(f"{DOCKER_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _ollama_reachable() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=2)
        s.close()
        return True
    except OSError:
        return False


def _load_claude_key() -> str | None:
    env_file = TEST_DIR.parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLAUDE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("CLAUDE_API_KEY")


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
            if line.startswith("From:"):    h["from"] = line[5:].strip()
            elif line.startswith("To:"):    h["to"] = line[3:].strip()
            elif line.startswith("Date:"):  h["date"] = line[5:].strip()
            elif line.startswith("Subject:"): h["subject"] = line[8:].strip()
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
        return "PASS" if any(m in a for m in _UNCERTAINTY_MARKERS) else "FAIL? (possible false positive — review)"
    return "REVIEW"


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _observe_http(user_input: str, agent_response: str) -> None:
    requests.post(f"{DOCKER_URL}/observe", json={
        "project": PROJECT, "user_input": user_input,
        "agent_response": agent_response, "path": "supabase://",
    }, timeout=120).raise_for_status()


def _recall_http(query: str) -> str:
    r = requests.post(f"{DOCKER_URL}/recall", json={
        "project": PROJECT, "query": query,
        "mode": "answer", "path": "supabase://",
    }, timeout=120)
    r.raise_for_status()
    return r.json()["result"]


def _flush_http() -> None:
    requests.post(f"{DOCKER_URL}/flush", json={
        "project": PROJECT, "path": "supabase://",
    }, timeout=600).raise_for_status()


def _stats_http() -> dict:
    r = requests.get(f"{DOCKER_URL}/stats",
                     params={"project": PROJECT, "path": "supabase://"},
                     timeout=10)
    r.raise_for_status()
    return r.json()


def _config_http() -> dict:
    r = requests.get(f"{DOCKER_URL}/config", timeout=5)
    r.raise_for_status()
    return r.json()


# ── Skip markers ──────────────────────────────────────────────────────────────

_skip_supabase = pytest.mark.skipif(
    BACKEND == "supabase" and not _docker_healthy(),
    reason="Docker sidecar not running — start with: docker compose up -d",
)
_skip_local = pytest.mark.skipif(
    BACKEND == "local" and not _ollama_reachable(),
    reason="Ollama not running — needed for local mode",
)


# ── Test 1: Ingestion (run once) ──────────────────────────────────────────────

@_skip_supabase
@_skip_local
def test_ingest():
    """Ingest the email corpus. Run once before iterating on test_recall."""
    raw = EMAILS_FILE.read_text(encoding="utf-8")
    header = raw.split("\n---\n")[0].strip()
    emails = parse_emails(raw)

    print(f"\n[{BACKEND}] Ingesting header + {len(emails)} emails into {PROJECT!r}...")
    observe_times: list[float] = []

    if BACKEND == "supabase":
        if header:
            t0 = perf_counter()
            _observe_http("Company reference document", header)
            observe_times.append(perf_counter() - t0)

        for i, email in enumerate(emails, 1):
            t0 = perf_counter()
            _observe_http(
                user_input=f"Email from {email['from']} to {email['to']}: {email['subject']}",
                agent_response=email["body"],
            )
            observe_times.append(perf_counter() - t0)
            print(f"  [{i:02d}/{len(emails)}] {email['subject'][:60]}")

        print("[supabase] Flushing...")
        _flush_http()
        stats = _stats_http()
        node_count = stats.get("node_count", 0)

    else:
        from memory_agent import MemoryAgent, LLMConfig
        claude_key = _load_claude_key()
        llm_cfg = LLMConfig(
            base_url=OLLAMA_BASE, model=MODEL,
            answer_model=CLAUDE_MODEL if claude_key else None,
            answer_base_url=CLAUDE_BASE if claude_key else None,
            answer_api_key=claude_key, timeout=120,
        )
        TEST_MEMORY.mkdir(exist_ok=True)
        agent = MemoryAgent(PROJECT, path=str(TEST_MEMORY), llm=llm_cfg)
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
        agent.flush()
        node_count = agent.stats()["node_count"]
        agent.close()

    assert node_count > 0, "No nodes stored after ingestion"
    print(f"\n[ingest] Done. nodes={node_count}")
    print(f"[ingest] Observe p50={percentile(observe_times, 50)*1000:.0f}ms  p99={percentile(observe_times, 99)*1000:.0f}ms")


# ── Test 2: Recall benchmark (iterate freely) ─────────────────────────────────

@_skip_supabase
@_skip_local
def test_recall():
    """Run all 60 questions against already-ingested data. Safe to re-run."""
    answer_key = parse_answer_key(ANSWER_KEY_FILE.read_text(encoding="utf-8"))

    # Verify data exists before querying
    if BACKEND == "supabase":
        stats = _stats_http()
        node_count = stats.get("node_count", 0)
        cfg = _config_http()
        ingest_model = cfg.get("ingest_model", MODEL)
        answer_model = cfg.get("answer_model", MODEL)
    else:
        from memory_agent import MemoryAgent, LLMConfig
        claude_key = _load_claude_key()
        llm_cfg = LLMConfig(
            base_url=OLLAMA_BASE, model=MODEL,
            answer_model=CLAUDE_MODEL if claude_key else None,
            answer_base_url=CLAUDE_BASE if claude_key else None,
            answer_api_key=claude_key, timeout=120,
        )
        TEST_MEMORY.mkdir(exist_ok=True)
        agent = MemoryAgent(PROJECT, path=str(TEST_MEMORY), llm=llm_cfg)
        node_count = agent.stats()["node_count"]
        ingest_model = MODEL
        answer_model = CLAUDE_MODEL if _load_claude_key() else MODEL

    assert node_count > 0, (
        f"No nodes found in project {PROJECT!r}. Run test_ingest first."
    )
    print(f"\n[{BACKEND}] {node_count} nodes — querying {len(QUESTIONS)} questions (answer model: {answer_model})...")

    results: dict[str, str] = {}
    recall_times: list[float] = []

    for q in QUESTIONS:
        t0 = perf_counter()
        if BACKEND == "supabase":
            answer = _recall_http(q["q"])
        else:
            answer = agent.recall(q["q"], mode="answer")
        elapsed = perf_counter() - t0
        recall_times.append(elapsed)
        results[q["id"]] = answer
        score = auto_score(q["id"], answer)
        print(f"  {q['id']} [{elapsed*1000:.0f}ms] {score}")

    if BACKEND == "local":
        agent.close()

    # ── Write results file ────────────────────────────────────────────────────
    lines: list[str] = []
    def w(s: str = "") -> None: lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w("=== VELORIS SUPABASE BENCHMARK ===")
    w(f"Run     : {ts}")
    w(f"Backend : {BACKEND}")
    w(f"Ingest  : {ingest_model}")
    w(f"Answer  : {answer_model}")
    w(f"Nodes   : {node_count}")
    w()

    tier_labels = {
        1: "TIER 1 — DIRECT LOOKUP (20 questions)",
        2: "TIER 2 — SINGLE HOP (20 questions)",
        3: "TIER 3 — MULTI-HOP (10 questions)",
        0: "UNANSWERABLE (10 questions — U-05 is the trap, answer is AWS)",
    }
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

    w()
    w("--- LATENCY ---")
    w(f"Recall p50 : {percentile(recall_times, 50)*1000:.0f}ms")
    w(f"Recall p99 : {percentile(recall_times, 99)*1000:.0f}ms")
    w()
    w("--- SCORE SUMMARY (fill in manually after review) ---")
    w("T1 accuracy : __/20")
    w("T2 accuracy : __/20")
    w("T3 accuracy : __/10")
    w(f"U  auto-scored : {u_auto_pass}/{u_auto_total - 1} pass (excl. trap)")
    u05_score = auto_score("U-05", results.get("U-05", ""))
    w(f"Trap (U-05)    : {u05_score}")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[benchmark] Written to {RESULTS_FILE}")
    print(f"[benchmark] U auto-scored: {u_auto_pass}/{u_auto_total - 1}  Trap: {u05_score}")
    print(f"[benchmark] p50={percentile(recall_times, 50)*1000:.0f}ms  p99={percentile(recall_times, 99)*1000:.0f}ms")
