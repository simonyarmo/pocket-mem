"""
Docker + Supabase / local integration test.

Ingests the Veloris email corpus via the chosen backend, runs a small set of
recall queries to confirm retrieval works, and prints the pocket-mem show
command to open the visual explorer manually.

Backend switch (top of file or via env var):
    "supabase"  →  HTTP requests to Docker sidecar (localhost:8000) → Supabase cloud
    "local"     →  direct Python MemoryAgent API  → test_memory/ SQLite

Run (local, no Docker needed):
    POCKET_MEM_BACKEND=local pytest tests/simulation/docker_supabase_show_test/ -v -s

Run (Supabase via Docker):
    docker compose up -d
    pytest tests/simulation/docker_supabase_show_test/ -v -s

Then visualise (local mode only):
    pocket-mem show --path tests/simulation/docker_supabase_show_test/test_memory --project veloris-docker-test
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

# ── Load .env file at repo root (before reading any env vars) ──────────────────
def _load_dotenv() -> None:
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key not in os.environ:          # don't overwrite real env vars
            os.environ[key] = val.strip()

_load_dotenv()

# ── Switch backend ─────────────────────────────────────────────────────────────
# "supabase"  → HTTP → Docker sidecar → Supabase cloud
# "local"     → direct Python API    → test_memory/ (SQLite)
BACKEND: str = os.environ.get("POCKET_MEM_BACKEND", "supabase")

# ── Paths ──────────────────────────────────────────────────────────────────────
TEST_DIR    = Path(__file__).parent
TEST_MEMORY = TEST_DIR / "test_memory"
DATA_DIR    = TEST_DIR.parent / "test_data"
EMAILS_FILE = DATA_DIR / "test_emails.txt"
RESULTS_FILE = TEST_DIR / "last_run_results.txt"

# ── Service config ─────────────────────────────────────────────────────────────
DOCKER_URL  = "http://localhost:8000"
PROJECT     = "veloris-docker-test"
MODEL       = "qwen2.5:7b"
OLLAMA_BASE = "http://localhost:11434/v1"
CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ── Sample questions (T1 subset) ───────────────────────────────────────────────
SAMPLE_QS = [
    "What is Marcus Webb's job title?",
    "How much was Veloris paying for Datadog per month?",
    "What security firm did Marcus choose for the audit?",
    "What build toolchain is Suki using for the React Native app?",
    "Which cloud provider does Veloris use?",   # trap — answer is AWS
]


# ── Pre-flight checks ──────────────────────────────────────────────────────────

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


# ── Email parsing (same as benchmark) ─────────────────────────────────────────

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


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ── HTTP helpers (supabase mode) ───────────────────────────────────────────────

def _observe_http(user_input: str, agent_response: str) -> None:
    requests.post(f"{DOCKER_URL}/observe", json={
        "project": PROJECT,
        "user_input": user_input,
        "agent_response": agent_response,
        "path": "supabase://",
    }, timeout=120).raise_for_status()


def _recall_http(query: str) -> str:
    r = requests.post(f"{DOCKER_URL}/recall", json={
        "project": PROJECT,
        "query": query,
        "mode": "answer",
        "path": "supabase://",
    }, timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def _flush_http() -> None:
    """Block until all background ingestion for this project is done."""
    requests.post(f"{DOCKER_URL}/flush", json={
        "project": PROJECT,
        "path": "supabase://",
    }, timeout=600).raise_for_status()


def _stats_http() -> dict:
    r = requests.get(f"{DOCKER_URL}/stats",
                     params={"project": PROJECT, "path": "supabase://"},
                     timeout=10)
    r.raise_for_status()
    return r.json()


# ── Test ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    BACKEND == "supabase" and not _docker_healthy(),
    reason="Docker sidecar not running — start with: docker compose up -d",
)
@pytest.mark.skipif(
    BACKEND == "local" and not _ollama_reachable(),
    reason="Ollama not running — needed for local mode ingestion",
)
def test_ingest_and_show():
    raw = EMAILS_FILE.read_text(encoding="utf-8")
    header = raw.split("\n---\n")[0].strip()
    emails = parse_emails(raw)
    print(f"\n[{BACKEND}] Ingesting header + {len(emails)} emails into {PROJECT!r}...")

    observe_times: list[float] = []

    if BACKEND == "supabase":
        # ── Supabase mode: HTTP → Docker sidecar ──────────────────────────────
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

        print("[supabase] Waiting for background ingestion to complete...")
        _flush_http()

        stats = _stats_http()
        node_count = stats.get("node_count", 0)
        print(f"[supabase] node_count={node_count}")
        assert node_count > 0, "No nodes stored after ingestion"

        # Session persistence: re-query stats (supabase is always persistent)
        stats2 = _stats_http()
        session_persistence = stats2.get("node_count", 0) > 0

        # Sample recall
        sample_answers: dict[str, str] = {}
        recall_times: list[float] = []
        for q in SAMPLE_QS:
            t0 = perf_counter()
            answer = _recall_http(q)
            recall_times.append(perf_counter() - t0)
            sample_answers[q] = answer
            print(f"  Q: {q[:50]}")
            print(f"  A: {answer[:100]}")

        show_note = (
            "Note: pocket-mem show requires local SQLite.\n"
            f"Re-run with: POCKET_MEM_BACKEND=local pytest {__file__} -v -s"
        )

    else:
        # ── Local mode: direct Python API → test_memory/ ──────────────────────
        from pocket_mem import MemoryAgent, LLMConfig

        claude_key = _load_claude_key()
        llm_cfg = LLMConfig(
            base_url=OLLAMA_BASE,
            model=MODEL,
            answer_model=CLAUDE_MODEL if claude_key else None,
            answer_base_url=CLAUDE_BASE if claude_key else None,
            answer_api_key=claude_key,
            timeout=120,
        )

        TEST_MEMORY.mkdir(exist_ok=True)
        agent = MemoryAgent(PROJECT, path=str(TEST_MEMORY), llm=llm_cfg)

        if header:
            t0 = perf_counter()
            agent.observe(user_input="Company reference document", agent_response=header)
            observe_times.append(perf_counter() - t0)

        for i, email in enumerate(emails, 1):
            t0 = perf_counter()
            agent.observe(
                user_input=f"Email from {email['from']} to {email['to']}: {email['subject']}",
                agent_response=email["body"],
            )
            observe_times.append(perf_counter() - t0)
            print(f"  [{i:02d}/{len(emails)}] {email['subject'][:60]}")

        print("[local] Flushing background ingestion...")
        agent.flush()
        stats = agent.stats()
        node_count = stats["node_count"]
        print(f"[local] node_count={node_count}")
        assert node_count > 0, "No nodes stored after ingestion"
        agent.close()

        # Session persistence: restart agent
        agent2 = MemoryAgent(PROJECT, path=str(TEST_MEMORY), llm=llm_cfg)
        session_persistence = agent2.stats()["node_count"] > 0
        print(f"[local] session persistence: {'PASS' if session_persistence else 'FAIL'}")

        # Sample recall
        sample_answers: dict[str, str] = {}
        recall_times: list[float] = []
        for q in SAMPLE_QS:
            t0 = perf_counter()
            answer = agent2.recall(q, mode="answer")
            recall_times.append(perf_counter() - t0)
            sample_answers[q] = answer
            print(f"  Q: {q[:50]}")
            print(f"  A: {answer[:100]}")
        agent2.close()

        show_note = (
            f"pocket-mem show --path {TEST_MEMORY} --project {PROJECT}"
        )

    # ── Results file ───────────────────────────────────────────────────────────
    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    w(f"=== VELORIS DOCKER/SUPABASE TEST ===")
    w(f"Run     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Backend : {BACKEND}")
    w(f"Model   : {MODEL}")
    w()
    w("--- INGESTION ---")
    w(f"Emails ingested : {len(emails)}")
    w(f"Observe p50     : {percentile(observe_times, 50)*1000:.0f}ms")
    w(f"Observe p99     : {percentile(observe_times, 99)*1000:.0f}ms")
    w(f"Nodes stored    : {node_count}")
    w()
    w("--- SESSION PERSISTENCE ---")
    w("PASS" if session_persistence else "FAIL")
    w()
    w("--- SAMPLE RECALL ---")
    for q, a in sample_answers.items():
        w(f"Q: {q}")
        w(f"A: {a.strip()}")
        w()
    w(f"Recall p50 : {percentile(recall_times, 50)*1000:.0f}ms")
    w(f"Recall p99 : {percentile(recall_times, 99)*1000:.0f}ms")
    w()
    w("--- SHOW COMMAND ---")
    w(show_note)

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[results] Written to {RESULTS_FILE}")
    print(f"\n[show] {show_note}")
