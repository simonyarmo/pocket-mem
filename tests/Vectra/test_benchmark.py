"""
Vectra benchmark — identity vs baseline comparison.

Ingests Jordan Mercer's Alexa interaction log into two separate databases in a
single test run: one with the Vectra identity active, one without. Queries both
with the same 60 questions and writes side-by-side results.

Run (both modes, fresh ingest):
    pytest tests/Vectra/ -v -s

Run (skip ingestion, use existing DBs):
    USE_EXISTING_DB=1 pytest tests/Vectra/ -v -s

Each DB is ingested independently. If USE_EXISTING_DB is set and a DB already
exists it is reused; if the DB is missing, that mode is re-ingested regardless.
"""
from __future__ import annotations

import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pytest

from pocket_mem import LLMConfig, MemoryAgent, MemoryConfig, IdentityConfig
from pocket_mem.retrieval import check_qa_cache

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VECTRA_DIR   = Path(__file__).parent
DATA_DIR     = VECTRA_DIR / "test_data"
MEMORY_DIR   = VECTRA_DIR / "memory"
RESULTS_FILE = VECTRA_DIR / "last_run_results.txt"
BENCHMARK_MD = VECTRA_DIR / "BENCHMARK.md"

CONVERSATIONS_FILE = DATA_DIR / "alexa_conversations.txt"
ANSWER_KEY_FILE    = DATA_DIR / "alexa_answer_key.txt"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL        = "qwen2.5:7b"
OLLAMA_BASE  = "http://localhost:11434/v1"
CLAUDE_BASE  = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
IDENTITY_DESCRIPTION = (
    "Personal AI assistant named Vectra managing Jordan's daily life in Austin, "
    "Texas. I listen to all voice interactions, emails, and text messages to track "
    "Jordan's schedule, reminders, appointments, shopping lists, financial tasks, "
    "contacts, and anything that needs to be remembered or acted on."
)

_USE_LLM    = os.environ.get("IDENTITY_USE_LLM", "").strip() == "1"
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or None

# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------
# Set USE_EXISTING_DB=1 to skip ingestion for any DB that already exists.
USE_EXISTING_DB = os.environ.get("USE_EXISTING_DB", "").strip() == "1"

# ---------------------------------------------------------------------------
# Questions (60 total: 20 T1 + 20 T2 + 10 T3 + 10 U)
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Tier 1 — Direct lookup
    {"id": "A-T1-01", "tier": 1, "q": "What is Jordan's current monthly rent?"},
    {"id": "A-T1-02", "tier": 1, "q": "What would the new monthly rent be if Jordan renews the lease?"},
    {"id": "A-T1-03", "tier": 1, "q": "What is the confirmation code for Jordan's Denver flight?"},
    {"id": "A-T1-04", "tier": 1, "q": "What time does Jordan's flight to Denver depart Austin?"},
    {"id": "A-T1-05", "tier": 1, "q": "What is the name of Jordan's doctor?"},
    {"id": "A-T1-06", "tier": 1, "q": "Where is Jordan's doctor's appointment?"},
    {"id": "A-T1-07", "tier": 1, "q": "What is Jordan's Chase credit card balance?"},
    {"id": "A-T1-08", "tier": 1, "q": "What item was Jordan's Amazon package?"},
    {"id": "A-T1-09", "tier": 1, "q": "What is the name of Jordan's landlord?"},
    {"id": "A-T1-10", "tier": 1, "q": "What day and time is the Q1 review meeting?"},
    {"id": "A-T1-11", "tier": 1, "q": "What is the team name for Wednesday trivia night?"},
    {"id": "A-T1-12", "tier": 1, "q": "Where is trivia night held?"},
    {"id": "A-T1-13", "tier": 1, "q": "What time is Jordan's grandmother's birthday dinner?"},
    {"id": "A-T1-14", "tier": 1, "q": "What is Jordan's gym called?"},
    {"id": "A-T1-15", "tier": 1, "q": "How much is Jordan's annual gym membership?"},
    {"id": "A-T1-16", "tier": 1, "q": "What flight number is Jordan's Denver flight?"},
    {"id": "A-T1-17", "tier": 1, "q": "What is Jordan's mother's name?"},
    {"id": "A-T1-18", "tier": 1, "q": "What is Jordan's coworker's name who sent the Henderson slides?"},
    {"id": "A-T1-19", "tier": 1, "q": "What Chase bank account is Jordan's credit card payment coming from?"},
    {"id": "A-T1-20", "tier": 1, "q": "What is the deadline to respond to the landlord about the lease?"},
    # Tier 2 — Single hop
    {"id": "A-T2-01", "tier": 2, "q": "What did Jordan's mom ask Jordan to bring to the birthday dinner, and what did Jordan add to the shopping list as a result?"},
    {"id": "A-T2-02", "tier": 2, "q": "What happened to Jordan's Chase credit card payment — did Jordan pay the minimum or the full balance, and how much was actually paid?"},
    {"id": "A-T2-03", "tier": 2, "q": "What slides is Jordan responsible for in the Henderson presentation, and what should Jordan focus on according to Ryan?"},
    {"id": "A-T2-04", "tier": 2, "q": "Why was the Q1 review meeting rescheduled, and what did Marcus ask Jordan to bring?"},
    {"id": "A-T2-05", "tier": 2, "q": "What pre-appointment instructions did Dr. Okafor's office send Jordan?"},
    {"id": "A-T2-06", "tier": 2, "q": "What does Paul Whitfield require in addition to the lease renewal decision?"},
    {"id": "A-T2-07", "tier": 2, "q": "What happened to Jordan's Saturday spin class, and when is the next available one?"},
    {"id": "A-T2-08", "tier": 2, "q": "What did Ryan text Jordan about the Henderson call on Thursday morning, and what did Jordan reply?"},
    {"id": "A-T2-09", "tier": 2, "q": "What did Jordan's grocery shopping list contain by the end of the week?"},
    {"id": "A-T2-10", "tier": 2, "q": "What does Marcus want to discuss in his Q1 review follow-up email, and when does he want to meet?"},
    {"id": "A-T2-11", "tier": 2, "q": "What did Jordan respond to Marcus's follow-up email?"},
    {"id": "A-T2-12", "tier": 2, "q": "What reminder did Jordan set related to the lease renewal, and on what date?"},
    {"id": "A-T2-13", "tier": 2, "q": "Who are the members of the trivia team on Wednesday night?"},
    {"id": "A-T2-14", "tier": 2, "q": "What was the weather forecast for Jordan's grandmother's birthday on Saturday March 8th?"},
    {"id": "A-T2-15", "tier": 2, "q": "What gym membership event is coming up on March 20th and how much will it cost?"},
    {"id": "A-T2-16", "tier": 2, "q": "What reminder did Jordan set related to the Henderson presentation?"},
    {"id": "A-T2-17", "tier": 2, "q": "What device did Jordan use for nighttime interactions vs daytime?"},
    {"id": "A-T2-18", "tier": 2, "q": "What did Dani Torres say about trivia night when she confirmed Jordan was coming?"},
    {"id": "A-T2-19", "tier": 2, "q": "What is the lease expiry date on Jordan's current apartment?"},
    {"id": "A-T2-20", "tier": 2, "q": "What was the weather forecast for Monday March 3rd when Jordan asked in the morning?"},
    # Tier 3 — Multi-hop
    {"id": "A-T3-01", "tier": 3, "q": "What is the full chain of events related to the Henderson account from Monday to Friday?"},
    {"id": "A-T3-02", "tier": 3, "q": "What is Jordan's complete schedule and commitments for the weekend of March 8-9?"},
    {"id": "A-T3-03", "tier": 3, "q": "Trace everything Jordan needs to do to prepare for the Denver trip on March 14th."},
    {"id": "A-T3-04", "tier": 3, "q": "What is the full picture of Jordan's financial obligations and transactions this week?"},
    {"id": "A-T3-05", "tier": 3, "q": "Which person appears in both Jordan's work life and social life, and what is their connection to each context?"},
    {"id": "A-T3-06", "tier": 3, "q": "What is the complete thread of communication between Jordan and their mother Carol Mercer?"},
    {"id": "A-T3-07", "tier": 3, "q": "What are all the reminders Jordan set across the week and what are they each for?"},
    {"id": "A-T3-08", "tier": 3, "q": "What medical-related tasks does Jordan have coming up and what preparation is required for each?"},
    {"id": "A-T3-09", "tier": 3, "q": "What is the full picture of Jordan's work situation heading into next week (March 10 onwards)?"},
    {"id": "A-T3-10", "tier": 3, "q": "Jordan set a reminder but then Alexa questioned the timing. What happened and what was the final outcome?"},
    # Unanswerable
    {"id": "A-U-01", "tier": 0, "q": "What is Jordan's home address?"},
    {"id": "A-U-02", "tier": 0, "q": "What is Jordan's job title?"},
    {"id": "A-U-03", "tier": 0, "q": "What airline lounge does Jordan use at Austin airport?"},
    {"id": "A-U-04", "tier": 0, "q": "What is Jordan's Chase credit card interest rate?"},
    {"id": "A-U-05", "tier": 0, "q": "Who is Jordan flying to Denver to visit or meet?"},  # TRAP — purpose not stated
    {"id": "A-U-06", "tier": 0, "q": "What is Jordan's grandmother's name?"},
    {"id": "A-U-07", "tier": 0, "q": "How long has Jordan lived in the current apartment?"},
    {"id": "A-U-08", "tier": 0, "q": "What medications is Jordan currently taking?"},
    {"id": "A-U-09", "tier": 0, "q": "Does Jordan have a car?"},
    {"id": "A-U-10", "tier": 0, "q": "What is Marcus's last name?"},
]

_UNCERTAINTY_MARKERS = [
    "not mentioned", "don't know", "no information", "cannot determine",
    "not in the", "i don't", "no data", "not available", "not provided",
    "not stated", "not specified", "unable to find", "no record", "no mention",
    "does not specify", "not documented",
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


def _load_claude_key() -> str | None:
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLAUDE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("CLAUDE_API_KEY")


def parse_interactions(text: str) -> list[dict]:
    """Parse INTERACTION NNN blocks from the conversations file."""
    blocks = re.split(r"\n---\n", text)
    interactions = []
    for block in blocks:
        block = block.strip()
        if not re.match(r"^INTERACTION \d+", block):
            continue
        lines = block.split("\n")
        h: dict = {}
        user_lines: list[str] = []
        alexa_lines: list[str] = []
        current_speaker = None
        for line in lines[1:]:
            if line.startswith("Date:"):
                h["date"] = line[5:].strip()
            elif line.startswith("Device:"):
                h["device"] = line[7:].strip()
            elif line.startswith("User:"):
                current_speaker = "user"
                user_lines.append(line[5:].strip())
            elif line.startswith("Alexa:"):
                current_speaker = "alexa"
                alexa_lines.append(line[6:].strip())
            elif line and current_speaker == "user":
                user_lines.append(line.strip())
            elif line and current_speaker == "alexa":
                alexa_lines.append(line.strip())
        h["user"] = " ".join(user_lines).strip()
        h["alexa"] = " ".join(alexa_lines).strip()
        if h.get("user") and h.get("alexa"):
            interactions.append(h)
    return interactions


def parse_answer_key(text: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    current_id: str | None = None
    for line in text.split("\n"):
        m = re.match(r"^((?:[A-Z]+-)?T[123]-\d+|(?:[A-Z]+-)?U-\d+): ", line)
        if m:
            current_id = m.group(1)
        elif line.startswith("ANSWER: ") and current_id:
            answers[current_id] = line[8:].strip()
            current_id = None
    return answers


def auto_score(q_id: str, answer: str) -> str:
    a = answer.lower()
    if q_id == "A-U-05":
        # Trap: Denver trip purpose is never mentioned — correct answer is "not mentioned"
        not_mentioned = any(m in a for m in _UNCERTAINTY_MARKERS)
        return "PASS" if not_mentioned else "FAIL — missed trap (fabricated trip purpose)"
    if q_id.startswith("A-U-"):
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
# BENCHMARK.md logging
# ---------------------------------------------------------------------------

def _next_run_number() -> int:
    if not BENCHMARK_MD.exists():
        return 1
    text = BENCHMARK_MD.read_text(encoding="utf-8")
    runs = re.findall(r"^### Run (\d+)", text, re.MULTILINE)
    return max((int(n) for n in runs), default=0) + 1


def _append_benchmark_run(
    ts: str,
    identity_derived: dict | None,
    nodes_identity: int,
    nodes_baseline: int,
    ingest_tokens_identity: dict,
    ingest_tokens_baseline: dict,
    total_tokens_identity: dict,
    total_tokens_baseline: dict,
    recall_times_identity: list[float],
    recall_times_baseline: list[float],
    u_pass_identity: int,
    u_pass_baseline: int,
    u_total: int,
    u05_identity: str,
    u05_baseline: str,
    identity_skipped: bool,
    baseline_skipped: bool,
) -> None:
    run_num = _next_run_number()
    date = ts.split(" ")[0]
    haiku_cost_id = (total_tokens_identity["tokens_in"] * 0.00000025
                     + total_tokens_identity["tokens_out"] * 0.00000125)
    haiku_cost_bl = (total_tokens_baseline["tokens_in"] * 0.00000025
                     + total_tokens_baseline["tokens_out"] * 0.00000125)

    id_mode = "existing DB" if identity_skipped else "fresh ingest"
    bl_mode = "existing DB" if baseline_skipped else "fresh ingest"
    id_role = identity_derived.get("role", "unknown") if identity_derived else "derivation FAILED"

    entry = f"""
### Run {run_num} — {date} · {MODEL} ingest + {CLAUDE_MODEL} answer

| Metric | Identity ({id_role}) | Baseline (no identity) |
|--------|----------------------|------------------------|
| Mode | {id_mode} | {bl_mode} |
| T1 accuracy | __/20 | __/20 |
| T2 accuracy | __/20 | __/20 |
| T3 accuracy | __/10 | __/10 |
| Overall | __/50 | __/50 |
| Trap (A-U-05) | {u05_identity} | {u05_baseline} |
| U auto-scored | {u_pass_identity}/{u_total - 1} pass (excl. trap) | {u_pass_baseline}/{u_total - 1} pass (excl. trap) |
| Nodes stored | {nodes_identity} | {nodes_baseline} |
| Recall p50 | {percentile(recall_times_identity, 50)*1000:.0f}ms | {percentile(recall_times_baseline, 50)*1000:.0f}ms |
| Recall p99 | {percentile(recall_times_identity, 99)*1000:.0f}ms | {percentile(recall_times_baseline, 99)*1000:.0f}ms |
| Tokens in (recall) | {total_tokens_identity['tokens_in']} | {total_tokens_baseline['tokens_in']} |
| Est. cost | ${haiku_cost_id:.4f} | ${haiku_cost_bl:.4f} |

**Identity ingest: in={ingest_tokens_identity['tokens_in']} out={ingest_tokens_identity['tokens_out']}**
**Baseline ingest: in={ingest_tokens_baseline['tokens_in']} out={ingest_tokens_baseline['tokens_out']}**

---
"""
    with open(BENCHMARK_MD, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[benchmark] Appended Run {run_num} to {BENCHMARK_MD}")


# ---------------------------------------------------------------------------
# Ingestion helper
# ---------------------------------------------------------------------------

def _run_ingest(
    project: str,
    db_path: Path,
    interactions: list[dict],
    header: str,
    llm_cfg: LLMConfig,
    mem_cfg: MemoryConfig,
    label: str,
) -> tuple[MemoryAgent, dict, int]:
    """Ingest interactions into a fresh DB. Returns (agent, token_stats, node_count)."""
    if db_path.exists():
        db_path.unlink()
        print(f"[{label}] Cleared existing DB")
    MEMORY_DIR.mkdir(exist_ok=True)
    agent = MemoryAgent(project, path=str(MEMORY_DIR), llm=llm_cfg, config=mem_cfg)

    if header:
        agent.observe(user_input="User profile and device setup", agent_response=header)

    for i, ia in enumerate(interactions, 1):
        agent.observe(
            user_input=f"[{ia.get('date', '')} — {ia.get('device', '')}] {ia['user']}",
            agent_response=ia["alexa"],
        )
        print(f"  [{label}] [{i:02d}/{len(interactions)}] {ia['user'][:70]}")

    print(f"[{label}] Flushing...")
    agent.flush()
    tokens = agent.token_stats()
    stats = agent.stats()
    agent.close()
    print(f"[{label}] nodes={stats['node_count']}  tokens={tokens}")
    return tokens, stats["node_count"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_vectra_benchmark():
    claude_key = _load_claude_key()
    llm_cfg = LLMConfig(
        base_url=OLLAMA_BASE,
        model=MODEL,
        answer_model=CLAUDE_MODEL if claude_key else None,
        answer_base_url=CLAUDE_BASE if claude_key else None,
        answer_api_key=claude_key,
        timeout=120,
    )
    identity_cfg = MemoryConfig(
        identity=IdentityConfig(
            description=IDENTITY_DESCRIPTION,
            skip_prebuilt=_USE_LLM,
            derivation_api_key=_GEMINI_KEY if _USE_LLM else None,
        )
    )
    baseline_cfg = MemoryConfig()  # no identity

    raw = CONVERSATIONS_FILE.read_text(encoding="utf-8")
    header = raw.split("\n---\n")[0].strip()
    interactions = parse_interactions(raw)
    answer_key = parse_answer_key(ANSWER_KEY_FILE.read_text(encoding="utf-8"))

    project_id = "vectra-identity"
    project_bl = "vectra-baseline"
    db_id = MEMORY_DIR / f"{project_id}.db"
    db_bl = MEMORY_DIR / f"{project_bl}.db"

    # ------------------------------------------------------------------
    # Phase 1: Ingestion (skipped per DB if USE_EXISTING_DB and DB exists)
    # ------------------------------------------------------------------
    MEMORY_DIR.mkdir(exist_ok=True)

    identity_skipped = USE_EXISTING_DB and db_id.exists()
    baseline_skipped = USE_EXISTING_DB and db_bl.exists()

    print(f"\n[benchmark] Interactions: {len(interactions)}")
    print(f"[benchmark] Identity DB : {'EXISTING' if identity_skipped else 'FRESH INGEST'}")
    print(f"[benchmark] Baseline DB : {'EXISTING' if baseline_skipped else 'FRESH INGEST'}")

    if identity_skipped:
        agent_tmp = MemoryAgent(project_id, path=str(MEMORY_DIR), llm=llm_cfg, config=identity_cfg)
        ingest_tokens_id = {"tokens_in": 0, "tokens_out": 0}
        nodes_id = agent_tmp.stats()["node_count"]
        print(f"[identity] Loaded existing DB — nodes={nodes_id}")
        agent_tmp.close()
    else:
        ingest_tokens_id, nodes_id = _run_ingest(
            project_id, db_id, interactions, header, llm_cfg, identity_cfg, "identity"
        )

    if baseline_skipped:
        agent_tmp = MemoryAgent(project_bl, path=str(MEMORY_DIR), llm=llm_cfg, config=baseline_cfg)
        ingest_tokens_bl = {"tokens_in": 0, "tokens_out": 0}
        nodes_bl = agent_tmp.stats()["node_count"]
        print(f"[baseline] Loaded existing DB — nodes={nodes_bl}")
        agent_tmp.close()
    else:
        ingest_tokens_bl, nodes_bl = _run_ingest(
            project_bl, db_bl, interactions, header, llm_cfg, baseline_cfg, "baseline"
        )

    # ------------------------------------------------------------------
    # Phase 2: Restart both agents (session persistence check)
    # ------------------------------------------------------------------
    agent_id = MemoryAgent(project_id, path=str(MEMORY_DIR), llm=llm_cfg, config=identity_cfg)
    agent_bl = MemoryAgent(project_bl, path=str(MEMORY_DIR), llm=llm_cfg, config=baseline_cfg)

    identity_derived = None
    if agent_id._config.identity:
        identity_derived = agent_id._config.identity.derived
    if identity_derived:
        print(f"[identity] Role: {identity_derived.get('role')}  "
              f"Seed topics: {identity_derived.get('seed_topics', [])}")
    else:
        print("[identity] WARNING: Identity derivation failed — running in generic mode")

    session_ok_id = agent_id.stats()["node_count"] > 0
    session_ok_bl = agent_bl.stats()["node_count"] > 0
    print(f"[benchmark] Session persistence — identity: {'PASS' if session_ok_id else 'FAIL'}, "
          f"baseline: {'PASS' if session_ok_bl else 'FAIL'}")

    # ------------------------------------------------------------------
    # Phase 3: Recall — both agents, same questions
    # ------------------------------------------------------------------
    print(f"[benchmark] Querying {len(QUESTIONS)} questions on both agents...")

    results_id: dict[str, str] = {}
    results_bl: dict[str, str] = {}
    times_id: dict[str, float] = {}
    times_bl: dict[str, float] = {}
    cache_hits_id: dict[str, bool] = {}
    cache_hits_bl: dict[str, bool] = {}

    for q in QUESTIONS:
        cache_hits_id[q["id"]] = check_qa_cache(q["q"], agent_id._store) is not None
        cache_hits_bl[q["id"]] = check_qa_cache(q["q"], agent_bl._store) is not None

        t0 = perf_counter()
        results_id[q["id"]] = agent_id.recall(q["q"], mode="answer")
        times_id[q["id"]] = perf_counter() - t0

        t0 = perf_counter()
        results_bl[q["id"]] = agent_bl.recall(q["q"], mode="answer")
        times_bl[q["id"]] = perf_counter() - t0

        score_id = auto_score(q["id"], results_id[q["id"]])
        score_bl = auto_score(q["id"], results_bl[q["id"]])
        src_id = "CACHE" if cache_hits_id[q["id"]] else "retrieval"
        src_bl = "CACHE" if cache_hits_bl[q["id"]] else "retrieval"
        print(f"  {q['id']} | identity [{times_id[q['id']]*1000:.0f}ms {src_id}] {score_id}"
              f"  |  baseline [{times_bl[q['id']]*1000:.0f}ms {src_bl}] {score_bl}")

    total_tokens_id = agent_id.token_stats()
    total_tokens_bl = agent_bl.token_stats()
    agent_id.close()
    agent_bl.close()

    # ------------------------------------------------------------------
    # Phase 4: Write results
    # ------------------------------------------------------------------
    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    derive_mode = (f"LLM ({('gemini-2.5-flash' if _GEMINI_KEY else MODEL)})"
                   if _USE_LLM else "prebuilt")

    w(f"=== VECTRA BENCHMARK — identity vs baseline ===")
    w(f"Run: {ts}   Ingest: {MODEL}   Answer: {llm_cfg.answer_model or MODEL}   Identity derivation: {derive_mode}")
    w()
    w("--- IDENTITY ---")
    w(f"Description : {IDENTITY_DESCRIPTION}")
    if identity_derived:
        w(f"Derived role: {identity_derived.get('role', 'unknown')}")
        w(f"Seed topics : {', '.join(identity_derived.get('seed_topics', []))}")
        w(f"Complexity  : {identity_derived.get('question_complexity_target', 'medium')}")
    else:
        w("Identity derivation FAILED — identity agent ran in generic mode")
    w()
    w("--- INGESTION ---")
    w(f"Interactions ingested : {len(interactions)}")
    for label, skipped, tokens, nodes in [
        ("Identity", identity_skipped, ingest_tokens_id, nodes_id),
        ("Baseline", baseline_skipped, ingest_tokens_bl, nodes_bl),
    ]:
        if skipped:
            w(f"{label:<12}: SKIPPED (existing DB) — nodes={nodes}")
        else:
            w(f"{label:<12}: nodes={nodes}  in={tokens['tokens_in']}  out={tokens['tokens_out']}")
    w()
    w("--- SESSION PERSISTENCE ---")
    w(f"Identity: {'PASS' if session_ok_id else 'FAIL'}   Baseline: {'PASS' if session_ok_bl else 'FAIL'}")
    w()

    tier_labels = {1: "TIER 1 — DIRECT LOOKUP", 2: "TIER 2 — SINGLE HOP",
                   3: "TIER 3 — MULTI-HOP", 0: "UNANSWERABLE (A-U-01 to A-U-10)"}
    current_tier = None
    u_pass_id = u_pass_bl = u_total = 0

    for q in QUESTIONS:
        if q["tier"] != current_tier:
            current_tier = q["tier"]
            w()
            w(f"--- {tier_labels[current_tier]} ---")
            w()

        expected = answer_key.get(q["id"], "—")
        ans_id   = results_id[q["id"]]
        ans_bl   = results_bl[q["id"]]
        score_id = auto_score(q["id"], ans_id)
        score_bl = auto_score(q["id"], ans_bl)

        if q["tier"] == 0:
            u_total += 1
            if score_id.startswith("PASS"):
                u_pass_id += 1
            if score_bl.startswith("PASS"):
                u_pass_bl += 1

        src_id = "CACHE HIT" if cache_hits_id.get(q["id"]) else "retrieval"
        src_bl = "CACHE HIT" if cache_hits_bl.get(q["id"]) else "retrieval"

        w(f"{q['id']} | {q['q']}")
        w(f"  EXPECTED        : {expected}")
        w(f"  IDENTITY ANSWER : {ans_id.strip()}")
        w(f"  IDENTITY SCORE  : {score_id}  [{src_id}]  {times_id.get(q['id'], 0)*1000:.0f}ms")
        w(f"  BASELINE ANSWER : {ans_bl.strip()}")
        w(f"  BASELINE SCORE  : {score_bl}  [{src_bl}]  {times_bl.get(q['id'], 0)*1000:.0f}ms")
        w()

    # Cache summary
    hits_id = sum(1 for v in cache_hits_id.values() if v)
    hits_bl = sum(1 for v in cache_hits_bl.values() if v)
    w("--- CACHE ---")
    w(f"Identity cache hits : {hits_id}/{len(QUESTIONS)}")
    w(f"Baseline cache hits : {hits_bl}/{len(QUESTIONS)}")
    w()

    # Latency
    recall_times_id_list = list(times_id.values())
    recall_times_bl_list = list(times_bl.values())
    w("--- LATENCY ---")
    w(f"Identity p50 : {percentile(recall_times_id_list, 50)*1000:.0f}ms   p99: {percentile(recall_times_id_list, 99)*1000:.0f}ms")
    w(f"Baseline p50 : {percentile(recall_times_bl_list, 50)*1000:.0f}ms   p99: {percentile(recall_times_bl_list, 99)*1000:.0f}ms")
    w()

    # Tokens
    w("--- TOKEN TOTALS ---")
    haiku_cost_id = (total_tokens_id["tokens_in"] * 0.00000025
                     + total_tokens_id["tokens_out"] * 0.00000125)
    haiku_cost_bl = (total_tokens_bl["tokens_in"] * 0.00000025
                     + total_tokens_bl["tokens_out"] * 0.00000125)
    w(f"Identity  tokens_in={total_tokens_id['tokens_in']}  tokens_out={total_tokens_id['tokens_out']}  est_cost=${haiku_cost_id:.4f}")
    w(f"Baseline  tokens_in={total_tokens_bl['tokens_in']}  tokens_out={total_tokens_bl['tokens_out']}  est_cost=${haiku_cost_bl:.4f}")
    w()

    # Score summary
    u05_id = auto_score("A-U-05", results_id.get("A-U-05", ""))
    u05_bl = auto_score("A-U-05", results_bl.get("A-U-05", ""))
    w("--- SCORE SUMMARY (fill in manually) ---")
    w(f"                  {'IDENTITY':<30} {'BASELINE'}")
    w(f"T1 accuracy     : {'__/20':<30} __/20")
    w(f"T2 accuracy     : {'__/20':<30} __/20")
    w(f"T3 accuracy     : {'__/10':<30} __/10")
    w(f"Overall         : {'__/50':<30} __/50")
    w(f"Trap (A-U-05)   : {u05_id:<30} {u05_bl}")
    w(f"U auto-scored   : {u_pass_id}/{u_total - 1} pass (excl. trap){'':15} {u_pass_bl}/{u_total - 1} pass (excl. trap)")
    w(f"Nodes stored    : {nodes_id:<30} {nodes_bl}")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[benchmark] Results written to {RESULTS_FILE}")

    # Append to BENCHMARK.md
    _append_benchmark_run(
        ts=ts,
        identity_derived=identity_derived,
        nodes_identity=nodes_id,
        nodes_baseline=nodes_bl,
        ingest_tokens_identity=ingest_tokens_id,
        ingest_tokens_baseline=ingest_tokens_bl,
        total_tokens_identity=total_tokens_id,
        total_tokens_baseline=total_tokens_bl,
        recall_times_identity=recall_times_id_list,
        recall_times_baseline=recall_times_bl_list,
        u_pass_identity=u_pass_id,
        u_pass_baseline=u_pass_bl,
        u_total=u_total,
        u05_identity=u05_id,
        u05_baseline=u05_bl,
        identity_skipped=identity_skipped,
        baseline_skipped=baseline_skipped,
    )

    assert nodes_id > 0, "Identity agent stored no nodes"
    assert nodes_bl > 0, "Baseline agent stored no nodes"
