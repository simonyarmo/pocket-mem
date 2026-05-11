"""
Identity benchmark — full end-to-end simulation.

Ingests case notes / documents, restarts the agent (session persistence), queries
60 questions, auto-scores unanswerable/trap questions, writes human-readable results.

Run:  pytest tests/simulation/ -v -s
Out:  tests/simulation/identity_sim_test/last_run_results.txt
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

from pocket_mem import MemoryAgent, LLMConfig, MemoryConfig, IdentityConfig
from pocket_mem.retrieval import check_qa_cache

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR = Path(__file__).parent
_DATA_DIR = SIM_DIR.parent / "test_data"
EMAILS_FILE = _DATA_DIR / "legal_notes.txt"
ANSWER_KEY_FILE = _DATA_DIR / "legal_answer_key.txt"
RESULTS_FILE = SIM_DIR / "last_run_results.txt"
MEMORY_DIR = SIM_DIR / "memory"   # persistent SQLite store — inspect after the run

MODEL = "qwen2.5:7b"
OLLAMA_BASE = "http://localhost:11434/v1"

CLAUDE_BASE = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

IDENTITY_DESCRIPTION = (
    "Paralegal at Castellan & Briggs LLP, a litigation law firm in Chicago. "
    "My job is to read all case notes every day — tracking clients, opposing "
    "parties, attorneys, deadlines, damages, filings, and anything happening "
    "across active cases so I can brief the attorneys and answer questions "
    "about case status, key facts, and upcoming deadlines."
)

# Set IDENTITY_USE_LLM=1 to bypass the prebuilt match and derive via LLM.
# Set GEMINI_API_KEY to use Gemini 2.5 Flash for derivation (recommended).
# Without GEMINI_API_KEY, falls back to the local Ollama model.
_USE_LLM = os.environ.get("IDENTITY_USE_LLM", "").strip() == "1"
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or None

# Set USE_EXISTING_DB=1 to skip ingestion and run recall against whatever DB
# is already in MEMORY_DIR. Use this to test a DB populated with qa_cache nodes.
USE_EXISTING_DB = False  # set True to reuse an existing DB (skips ingestion)

def _load_claude_key() -> str | None:
    """Read CLAUDE_API_KEY from .env at repo root, falling back to environment."""
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
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
    {"id": "L-T1-01", "tier": 1, "q": "What city is Castellan & Briggs LLP located in?"},
    {"id": "L-T1-02", "tier": 1, "q": "What are the claimed damages in NorthBridge Capital v. Terravast Holdings?"},
    {"id": "L-T1-03", "tier": 1, "q": "Who is the client contact at NorthBridge Capital?"},
    {"id": "L-T1-04", "tier": 1, "q": "What case number was assigned to NorthBridge v. Terravast when filed?"},
    {"id": "L-T1-05", "tier": 1, "q": "What judge was assigned to NorthBridge v. Terravast?"},
    {"id": "L-T1-06", "tier": 1, "q": "What was Harlan Voss's annual salary at Kellerman & Drape?"},
    {"id": "L-T1-07", "tier": 1, "q": "On what date was Harlan Voss terminated?"},
    {"id": "L-T1-08", "tier": 1, "q": "What were Harlan Voss's performance review ratings in 2022 and 2023?"},
    {"id": "L-T1-09", "tier": 1, "q": "What was Solenne Marchand's claimed injury from the slip and fall?"},
    {"id": "L-T1-10", "tier": 1, "q": "What is the total damages amount claimed by Solenne Marchand?"},
    {"id": "L-T1-11", "tier": 1, "q": "What is the name of Brightfield Pharmaceuticals' blood pressure medication at the center of the product liability case?"},
    {"id": "L-T1-12", "tier": 1, "q": "Who is the defense expert retained in the Brightfield case, and where is he from?"},
    {"id": "L-T1-13", "tier": 1, "q": "What is Duncan Farrow's role at Castellan & Briggs?"},
    {"id": "L-T1-14", "tier": 1, "q": "What was the amount Terravast was contractually obligated to contribute by December 1, 2024?"},
    {"id": "L-T1-15", "tier": 1, "q": "What mediation service was agreed upon for the Voss case?"},
    {"id": "L-T1-16", "tier": 1, "q": "What time did the salt truck treat North Michigan Avenue on the morning of Marchand's fall?"},
    {"id": "L-T1-17", "tier": 1, "q": "What is the EEOC charge number for the Voss case?"},
    {"id": "L-T1-18", "tier": 1, "q": "What was the demand amount in the Voss demand letter?"},
    {"id": "L-T1-19", "tier": 1, "q": "Who is the plaintiff's counsel in Solenne Marchand's case?"},
    {"id": "L-T1-20", "tier": 1, "q": "What are the three plaintiffs in the consolidated Brightfield case?"},
    # Tier 2 — Single hop
    {"id": "L-T2-01", "tier": 2, "q": "What problem did Renata Osei identify in the NorthBridge contract, and how was it resolved?"},
    {"id": "L-T2-02", "tier": 2, "q": "What three causes of action did NorthBridge file against Terravast?"},
    {"id": "L-T2-03", "tier": 2, "q": "What did Terravast's asset investigation reveal about their real estate holdings?"},
    {"id": "L-T2-04", "tier": 2, "q": "What are the three counts filed in the Voss complaint, and who are the named defendants?"},
    {"id": "L-T2-05", "tier": 2, "q": "How many days passed between Harlan Voss's whistleblower complaint and his termination, and why is that significant legally?"},
    {"id": "L-T2-06", "tier": 2, "q": "What specific observations did witness Thomas Quayle make that helped the City's defense?"},
    {"id": "L-T2-07", "tier": 2, "q": "What counterclaim did Terravast file against NorthBridge, and for how much?"},
    {"id": "L-T2-08", "tier": 2, "q": "What is the expert fee structure for Dr. Belkin in the Brightfield case?"},
    {"id": "L-T2-09", "tier": 2, "q": "What is the significance of Anton Devereux's case among the three Brightfield plaintiffs?"},
    {"id": "L-T2-10", "tier": 2, "q": "What is Kellerman & Drape's counter-offer to the demand letter, and why did Voss reject it?"},
    {"id": "L-T2-11", "tier": 2, "q": "What was Leo Nakashima's settlement recommendation for the Marchand case, and under what condition would he recommend trial?"},
    {"id": "L-T2-12", "tier": 2, "q": "What is Terravast's affirmative defense related to zoning, and how does it change what Castellan & Briggs needs to investigate?"},
    {"id": "L-T2-13", "tier": 2, "q": "What Illinois law case did Imani Foster cite as precedent for the Voss whistleblower claim?"},
    {"id": "L-T2-14", "tier": 2, "q": "What was the bridge loan amount, who was it from, and at what interest rate?"},
    {"id": "L-T2-15", "tier": 2, "q": "What is the deadline for NorthBridge's response to Terravast's counterclaim, and what is the date?"},
    {"id": "L-T2-16", "tier": 2, "q": "What is the legal weakness in the Voss whistleblower case that Imani Foster identified?"},
    {"id": "L-T2-17", "tier": 2, "q": "What motion did Torres, Reid & Hatch file in the Brightfield case, and what is the issue with it?"},
    {"id": "L-T2-18", "tier": 2, "q": "What opposing expert does Brightfield face in the product liability case, and what does Dr. Belkin think of her approach?"},
    {"id": "L-T2-19", "tier": 2, "q": "What are the key dates in the NorthBridge v. Terravast discovery schedule?"},
    {"id": "L-T2-20", "tier": 2, "q": "Who were the registered agent and defense counsel for Terravast Holdings?"},
    # Tier 3 — Multi-hop
    {"id": "L-T3-01", "tier": 3, "q": "Which attorney worked on both of the defense cases (representing defendants rather than plaintiffs)?"},
    {"id": "L-T3-02", "tier": 3, "q": "What is the full chain of events from Harlan Voss's whistleblower complaint to the mediation scheduled in March 2025?"},
    {"id": "L-T3-03", "tier": 3, "q": "Two cases involve Renata Osei as paralegal. Which cases are they, and what specific work did she do on each?"},
    {"id": "L-T3-04", "tier": 3, "q": "What is the full picture of Terravast's financial situation, and how does it affect litigation strategy?"},
    {"id": "L-T3-05", "tier": 3, "q": "Which paralegal/staff member handled administrative work across multiple cases, and what did she do in each?"},
    {"id": "L-T3-06", "tier": 3, "q": "Which of the four cases has the most complex expert witness situation, and why?"},
    {"id": "L-T3-07", "tier": 3, "q": "What is the connection between the NorthBridge contract dispute and the Voss whistleblower case — specifically, is there any shared name that appears in both?"},
    {"id": "L-T3-08", "tier": 3, "q": "What is the complete damages calculation for Harlan Voss as of the demand letter, and how has the calculation evolved by March 2025?"},
    {"id": "L-T3-09", "tier": 3, "q": "All three Brightfield plaintiffs shared the same pre-existing condition. What was it, how long was each on Velantrix, and whose case is most severe?"},
    {"id": "L-T3-10", "tier": 3, "q": "What would need to happen for Castellan & Briggs to add a claim against the City of Chicago zoning office in the NorthBridge case?"},
    # Unanswerable
    {"id": "L-U-01", "tier": 0, "q": "What law school did Margaret Castellan attend?"},
    {"id": "L-U-02", "tier": 0, "q": "How many attorneys work at Castellan & Briggs total?"},
    {"id": "L-U-03", "tier": 0, "q": "What is Harlan Voss's home address?"},
    {"id": "L-U-04", "tier": 0, "q": "Has Brightfield Pharmaceuticals faced product liability suits before?"},
    {"id": "L-U-05", "tier": 0, "q": "What floor is the Castellan & Briggs office on?"},
    {"id": "L-U-06", "tier": 0, "q": "What is Derek Briggs's area of specialty?"},  # TRAP — inferable as employment law
    {"id": "L-U-07", "tier": 0, "q": "How long has NorthBridge Capital been in business?"},
    {"id": "L-U-08", "tier": 0, "q": "What is the total estimated legal fees for the NorthBridge matter?"},
    {"id": "L-U-09", "tier": 0, "q": "What happened to the Meridian Tower property after it was sold at a discount?"},
    {"id": "L-U-10", "tier": 0, "q": "Is Solenne Marchand still employed?"},
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


def parse_case_notes(text: str) -> list[dict]:
    blocks = re.split(r"\n---\n", text)
    notes = []
    for block in blocks:
        block = block.strip()
        if not re.match(r"^CASE NOTE \d+", block):
            continue
        lines = block.split("\n")
        h: dict = {}
        body_start = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.startswith("Date:"):
                h["date"] = line[5:].strip()
            elif line.startswith("Case:"):
                h["case"] = line[5:].strip()
            elif line.startswith("File No.:"):
                h["file_no"] = line[9:].strip()
            elif line.startswith("Attorney:"):
                h["attorney"] = line[9:].strip()
            elif line.startswith("Note Type:"):
                h["note_type"] = line[10:].strip()
            elif line == "" and "note_type" in h:
                body_start = i + 1
                break
        h["body"] = "\n".join(lines[body_start:]).strip()
        notes.append(h)
    return notes


def parse_answer_key(text: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    current_id: str | None = None
    for line in text.split("\n"):
        m = re.match(r"^((?:L-)?T[123]-\d+|(?:L-)?U-\d+): ", line)
        if m:
            current_id = m.group(1)
        elif line.startswith("ANSWER: ") and current_id:
            answers[current_id] = line[8:].strip()
            current_id = None
    return answers


def auto_score(q_id: str, answer: str) -> str:
    a = answer.lower()
    if q_id == "L-U-06":
        # Trap: Derek Briggs's specialty is inferable as employment law from the Voss case
        return "PASS (trap)" if "employment" in a else "FAIL — missed trap (expected employment law inference)"
    if q_id.startswith("L-U-") or q_id.startswith("U-"):
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
# Ingestion helper
# ---------------------------------------------------------------------------

def _run_ingest(
    project: str,
    db_path: Path,
    notes: list[dict],
    header: str,
    llm_cfg: LLMConfig,
    mem_cfg: MemoryConfig,
    label: str,
) -> tuple[dict, int]:
    """Ingest case notes into a fresh DB. Returns (token_stats, node_count)."""
    if db_path.exists():
        db_path.unlink()
        print(f"[{label}] Cleared existing DB")
    MEMORY_DIR.mkdir(exist_ok=True)
    agent = MemoryAgent(project, path=str(MEMORY_DIR), llm=llm_cfg, config=mem_cfg)

    if header:
        agent.observe(user_input="Law firm reference document", agent_response=header)

    for i, note in enumerate(notes, 1):
        agent.observe(
            user_input=f"Case note: {note.get('case', '')} — {note.get('note_type', '')} (Attorney: {note.get('attorney', '')}, {note.get('date', '')})",
            agent_response=note["body"],
        )
        print(f"  [{label}] [{i:02d}/{len(notes)}] {note.get('case', '')[:60]}")

    print(f"[{label}] Flushing...")
    agent.flush()
    tokens = agent.token_stats()
    stats = agent.stats()
    agent.close()
    print(f"[{label}] nodes={stats['node_count']}  tokens={tokens}")
    return tokens, stats["node_count"]


# ---------------------------------------------------------------------------
# BENCHMARK.md logging
# ---------------------------------------------------------------------------

def _next_run_number() -> int:
    bm = SIM_DIR / "BENCHMARK.md"
    if not bm.exists():
        return 1
    text = bm.read_text(encoding="utf-8")
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
    u06_identity: str,
    u06_baseline: str,
    identity_skipped: bool,
    baseline_skipped: bool,
) -> None:
    bm_file = SIM_DIR / "BENCHMARK.md"
    run_num = _next_run_number()
    date = ts.split(" ")[0]

    id_role = identity_derived.get("role", "unknown") if identity_derived else "derivation FAILED"
    id_mode = "existing DB" if identity_skipped else "fresh ingest"
    bl_mode = "existing DB" if baseline_skipped else "fresh ingest"
    haiku_cost_id = (total_tokens_identity["tokens_in"] * 0.00000025
                     + total_tokens_identity["tokens_out"] * 0.00000125)
    haiku_cost_bl = (total_tokens_baseline["tokens_in"] * 0.00000025
                     + total_tokens_baseline["tokens_out"] * 0.00000125)

    entry = f"""
### Run {run_num} — {date} · {MODEL} ingest + {CLAUDE_MODEL} answer

| Metric | Identity ({id_role}) | Baseline (no identity) |
|--------|----------------------|------------------------|
| Mode | {id_mode} | {bl_mode} |
| T1 accuracy | __/20 | __/20 |
| T2 accuracy | __/20 | __/20 |
| T3 accuracy | __/10 | __/10 |
| Overall | __/50 | __/50 |
| False positives | — | — |
| Trap (L-U-06) | {u06_identity} | {u06_baseline} |
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

    with open(bm_file, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[benchmark] Appended Run {run_num} to {bm_file}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_legal_benchmark():
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
    baseline_cfg = MemoryConfig()

    raw = EMAILS_FILE.read_text(encoding="utf-8")
    header = raw.split("\n---\n")[0].strip()
    notes = parse_case_notes(raw)
    answer_key = parse_answer_key(ANSWER_KEY_FILE.read_text(encoding="utf-8"))

    project_id = "legal-benchmark-identity"
    project_bl = "legal-benchmark-baseline"
    db_id = MEMORY_DIR / f"{project_id}.db"
    db_bl = MEMORY_DIR / f"{project_bl}.db"

    # ------------------------------------------------------------------
    # Phase 1: Ingestion (skipped per DB when USE_EXISTING_DB and DB exists)
    # ------------------------------------------------------------------
    MEMORY_DIR.mkdir(exist_ok=True)

    use_existing = os.environ.get("USE_EXISTING_DB", "").strip() == "1" or USE_EXISTING_DB
    identity_skipped = use_existing and db_id.exists()
    baseline_skipped = use_existing and db_bl.exists()

    print(f"\n[benchmark] Case notes: {len(notes)}")
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
            project_id, db_id, notes, header, llm_cfg, identity_cfg, "identity"
        )

    if baseline_skipped:
        agent_tmp = MemoryAgent(project_bl, path=str(MEMORY_DIR), llm=llm_cfg, config=baseline_cfg)
        ingest_tokens_bl = {"tokens_in": 0, "tokens_out": 0}
        nodes_bl = agent_tmp.stats()["node_count"]
        print(f"[baseline] Loaded existing DB — nodes={nodes_bl}")
        agent_tmp.close()
    else:
        ingest_tokens_bl, nodes_bl = _run_ingest(
            project_bl, db_bl, notes, header, llm_cfg, baseline_cfg, "baseline"
        )

    # ------------------------------------------------------------------
    # Phase 2: Restart both agents (session persistence)
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
    # Phase 3: Recall — both agents, same 60 questions
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
        print(f"  {q['id']} | id [{times_id[q['id']]*1000:.0f}ms {src_id}] {score_id}"
              f"  |  bl [{times_bl[q['id']]*1000:.0f}ms {src_bl}] {score_bl}")

    total_tokens_id = agent_id.token_stats()
    total_tokens_bl = agent_bl.token_stats()
    agent_id.close()
    agent_bl.close()

    # ------------------------------------------------------------------
    # Phase 4: Write results file
    # ------------------------------------------------------------------
    lines: list[str] = []
    def w(s: str = "") -> None:
        lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    derive_mode = (f"LLM ({('gemini-2.5-flash' if _GEMINI_KEY else MODEL)})"
                   if _USE_LLM else "prebuilt")

    w(f"=== LEGAL BENCHMARK (Castellan & Briggs) — identity vs baseline ===")
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
    w(f"Case notes ingested : {len(notes)}")
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
                   3: "TIER 3 — MULTI-HOP", 0: "UNANSWERABLE (L-U-01 to L-U-10)"}
    current_tier = None
    u_pass_id = u_pass_bl = u_total = 0

    for q in QUESTIONS:
        if q["tier"] != current_tier:
            current_tier = q["tier"]
            w()
            w(f"--- {tier_labels[current_tier]} ---")
            w()

        expected = answer_key.get(q["id"], "—")
        ans_id = results_id[q["id"]]
        ans_bl = results_bl[q["id"]]
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
    times_id_list = list(times_id.values())
    times_bl_list = list(times_bl.values())
    w("--- LATENCY ---")
    w(f"Identity p50 : {percentile(times_id_list, 50)*1000:.0f}ms   p99: {percentile(times_id_list, 99)*1000:.0f}ms")
    w(f"Baseline p50 : {percentile(times_bl_list, 50)*1000:.0f}ms   p99: {percentile(times_bl_list, 99)*1000:.0f}ms")
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
    u06_id = auto_score("L-U-06", results_id.get("L-U-06", ""))
    u06_bl = auto_score("L-U-06", results_bl.get("L-U-06", ""))
    w("--- SCORE SUMMARY (fill in manually) ---")
    w(f"                  {'IDENTITY':<30} {'BASELINE'}")
    w(f"T1 accuracy     : {'__/20':<30} __/20")
    w(f"T2 accuracy     : {'__/20':<30} __/20")
    w(f"T3 accuracy     : {'__/10':<30} __/10")
    w(f"Overall         : {'__/50':<30} __/50")
    w(f"Trap (L-U-06)   : {u06_id:<30} {u06_bl}")
    w(f"U auto-scored   : {u_pass_id}/{u_total - 1} pass (excl. trap){'':15} {u_pass_bl}/{u_total - 1} pass (excl. trap)")
    w(f"Nodes stored    : {nodes_id:<30} {nodes_bl}")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[benchmark] Results written to {RESULTS_FILE}")

    _append_benchmark_run(
        ts=ts,
        identity_derived=identity_derived,
        nodes_identity=nodes_id,
        nodes_baseline=nodes_bl,
        ingest_tokens_identity=ingest_tokens_id,
        ingest_tokens_baseline=ingest_tokens_bl,
        total_tokens_identity=total_tokens_id,
        total_tokens_baseline=total_tokens_bl,
        recall_times_identity=times_id_list,
        recall_times_baseline=times_bl_list,
        u_pass_identity=u_pass_id,
        u_pass_baseline=u_pass_bl,
        u_total=u_total,
        u06_identity=u06_id,
        u06_baseline=u06_bl,
        identity_skipped=identity_skipped,
        baseline_skipped=baseline_skipped,
    )

    assert nodes_id > 0, "Identity agent stored no nodes"
    assert nodes_bl > 0, "Baseline agent stored no nodes"
