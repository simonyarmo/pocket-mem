"""
Veloris Benchmark V2 — multi-dataset simulation.

Three independent datasets:
  1. Medical  (Hargrove Family Medicine)  — 20 clinical notes, 60 questions
  2. Legal    (Castellan & Briggs LLP)   — 20 case notes,     60 questions
  3. Combined                             — all 3 datasets loaded, 180 questions (60 per dataset)

Run all:      pytest tests/simulation/complex_sim_test/ -v -s
Run medical:  pytest tests/simulation/complex_sim_test/ -v -s -k medical
Run legal:    pytest tests/simulation/complex_sim_test/ -v -s -k legal
Run combined: pytest tests/simulation/complex_sim_test/ -v -s -k combined

Memory (SQLite) is written to tests/simulation/complex_sim_test/memory/ and persists
after the run so you can inspect the database with any SQLite viewer.
Delete the relevant subfolder before a run to start fresh.
"""
from __future__ import annotations

import os
import re
import socket
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pytest

from pocket_mem import MemoryAgent, LLMConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DIR   = Path(__file__).parent
_DATA_DIR = SIM_DIR.parent / "test_data"

MEDICAL_NOTES_FILE = _DATA_DIR / "medical_notes.txt"
MEDICAL_KEY_FILE   = _DATA_DIR / "medical_answer_key.txt"
LEGAL_NOTES_FILE   = _DATA_DIR / "legal_notes.txt"
LEGAL_KEY_FILE     = _DATA_DIR / "legal_answer_key.txt"
EMAILS_FILE        = _DATA_DIR / "test_emails.txt"
EMAILS_KEY_FILE    = _DATA_DIR / "answer_key.txt"

MEMORY_DIR = SIM_DIR / "memory"

MODEL        = "qwen2.5:7b"
OLLAMA_BASE  = "http://localhost:11434/v1"
CLAUDE_BASE  = "https://api.anthropic.com/v1"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

MEDICAL_QUESTIONS = [
    # Tier 1 — Direct lookup
    {"id": "M-T1-01", "q": "What is Arthur Pemberton's date of birth?"},
    {"id": "M-T1-02", "q": "What medication was Arthur Pemberton already taking for hypertension at his first visit?"},
    {"id": "M-T1-03", "q": "What was Arthur Pemberton's BNP level at initial labs?"},
    {"id": "M-T1-04", "q": "What is Cecilia Vance's occupation?"},
    {"id": "M-T1-05", "q": "How much weight did Cecilia Vance lose in the 6 months before her visit?"},
    {"id": "M-T1-06", "q": "What was Cecilia Vance's initial HbA1c result?"},
    {"id": "M-T1-07", "q": "What continuous glucose monitor was prescribed to Cecilia Vance?"},
    {"id": "M-T1-08", "q": "What was Raymond Chu's blood pressure when he arrived at the clinic on February 5th?"},
    {"id": "M-T1-09", "q": "What medication did Raymond Chu run out of and for how long?"},
    {"id": "M-T1-10", "q": "What did the CT head scan show for Raymond Chu?"},
    {"id": "M-T1-11", "q": "What is Dolores Estrada's occupation?"},
    {"id": "M-T1-12", "q": "What medication is Dolores Estrada on for osteoporosis?"},
    {"id": "M-T1-13", "q": "What was Dolores Estrada's T-score on her DEXA scan?"},
    {"id": "M-T1-14", "q": "Who is the cardiologist at Hargrove Family Medicine?"},
    {"id": "M-T1-15", "q": "What was Arthur Pemberton's ejection fraction on echocardiogram?"},
    {"id": "M-T1-16", "q": "What pharmacy did Raymond Chu use to fill his prescription?"},
    {"id": "M-T1-17", "q": "What is the name of the medical records coordinator at Hargrove Family Medicine?"},
    {"id": "M-T1-18", "q": "What were Cecilia Vance's two episodes of hypoglycemia (glucose readings)?"},
    {"id": "M-T1-19", "q": "What anticoagulant was Arthur Pemberton started on?"},
    {"id": "M-T1-20", "q": "What imaging showed moderate medial compartment osteoarthritis?"},
    # Tier 2 — Single hop
    {"id": "M-T2-01", "q": "What condition did Dr. Suresh diagnose Arthur Pemberton with, and what confirmed it?"},
    {"id": "M-T2-02", "q": "Why did Dr. Suresh hold off on increasing Arthur Pemberton's Lisinopril at the cardiology visit?"},
    {"id": "M-T2-03", "q": "What type of diabetes does Cecilia Vance have, and how is it different from Type 2?"},
    {"id": "M-T2-04", "q": "Why was Metformin specifically ruled out for Cecilia Vance?"},
    {"id": "M-T2-05", "q": "What happened to Raymond Chu's blood pressure over the 2-hour monitoring period on February 5th?"},
    {"id": "M-T2-06", "q": "What did Arthur Pemberton's echocardiogram show about his left atrium?"},
    {"id": "M-T2-07", "q": "What lab finding confirmed Cecilia Vance had autoimmune diabetes rather than Type 2?"},
    {"id": "M-T2-08", "q": "Why was Arthur Pemberton's Lisinopril finally increased at his February follow-up?"},
    {"id": "M-T2-09", "q": "What did Dolores Estrada's X-ray show, and what was Dr. Marsh's referral decision based on it?"},
    {"id": "M-T2-10", "q": "What improvements did Arthur Pemberton show at his 4-week cardiology follow-up?"},
    {"id": "M-T2-11", "q": "What did Cecilia Vance's CGM show at her 6-week follow-up compared to earlier readings?"},
    {"id": "M-T2-12", "q": "What was the concern about Dolores Estrada's living situation in relation to her knee?"},
    {"id": "M-T2-13", "q": "What two medications were given to Raymond Chu at the clinic on February 5th?"},
    {"id": "M-T2-14", "q": "What was Raymond Chu's medication regimen after his February 18 follow-up?"},
    {"id": "M-T2-15", "q": "What was Arthur Pemberton's weight change from January to February?"},
    {"id": "M-T2-16", "q": "Where was Cecilia Vance referred for her annual diabetic eye exam?"},
    {"id": "M-T2-17", "q": "What was Cecilia Vance's insulin dosage at her 6-week follow-up vs when she started?"},
    {"id": "M-T2-18", "q": "What finding on Arthur Pemberton's ECG indicated long-standing hypertension effects?"},
    {"id": "M-T2-19", "q": "What referral did Dr. Marsh make for Raymond Chu at his February 18 follow-up, and why?"},
    {"id": "M-T2-20", "q": "What minor bleeding events did Arthur Pemberton report, and what was Dr. Suresh's response?"},
    # Tier 3 — Multi-hop
    {"id": "M-T3-01", "q": "Which nurse handled lab result notifications for multiple patients, and which result required immediate physician contact?"},
    {"id": "M-T3-02", "q": "Two patients had conditions discovered while being evaluated for something else. Who and what was discovered?"},
    {"id": "M-T3-03", "q": "Which patient had the most medications at their February follow-up, and what were all of them?"},
    {"id": "M-T3-04", "q": "What is the connection between Cecilia Vance's father and her diagnosis — and what is the important distinction?"},
    {"id": "M-T3-05", "q": "Which radiologist reported on tests for the most patients, and list all the tests they reported on?"},
    {"id": "M-T3-06", "q": "What was the full chain of events that led to Cecilia Vance being diagnosed with LADA rather than Type 2?"},
    {"id": "M-T3-07", "q": "Which patient had the highest blood pressure recorded, and what was the full sequence of clinical decisions made that day?"},
    {"id": "M-T3-08", "q": "Two patients were referred to specialists. Who were the specialists and what did they each order or start?"},
    {"id": "M-T3-09", "q": "How does Dolores Estrada's osteoporosis affect her overall care plan for her knee?"},
    {"id": "M-T3-10", "q": "Which patient's condition was identified as a medical urgency (not emergency), and what is the clinical distinction?"},
    # Unanswerable
    {"id": "M-U-01", "q": "What is Dr. Eleanor Marsh's medical school?"},
    {"id": "M-U-02", "q": "What is Arthur Pemberton's home address?"},
    {"id": "M-U-03", "q": "Does Cecilia Vance have health insurance?"},
    {"id": "M-U-04", "q": "What caused Raymond Chu's small vessel disease?"},   # TRAP — hypertension-related
    {"id": "M-U-05", "q": "What is Dolores Estrada's blood type?"},
    {"id": "M-U-06", "q": "Who is Arthur Pemberton's emergency contact?"},
    {"id": "M-U-07", "q": "Has Cecilia Vance ever been hospitalized before?"},
    {"id": "M-U-08", "q": "What hospital is Hargrove Family Medicine affiliated with?"},
    {"id": "M-U-09", "q": "What is Nurse James Obi's nursing specialty certification?"},
    {"id": "M-U-10", "q": "How long has Dr. Patricia Yuen been practicing endocrinology?"},
]

LEGAL_QUESTIONS = [
    # Tier 1 — Direct lookup
    {"id": "L-T1-01", "q": "What city is Castellan & Briggs LLP located in?"},
    {"id": "L-T1-02", "q": "What are the claimed damages in NorthBridge v. Terravast?"},
    {"id": "L-T1-03", "q": "Who is the client contact at NorthBridge Capital?"},
    {"id": "L-T1-04", "q": "What case number was assigned to NorthBridge v. Terravast?"},
    {"id": "L-T1-05", "q": "What judge was assigned to NorthBridge v. Terravast?"},
    {"id": "L-T1-06", "q": "What was Harlan Voss's annual salary at Kellerman & Drape?"},
    {"id": "L-T1-07", "q": "On what date was Harlan Voss terminated?"},
    {"id": "L-T1-08", "q": "What were Harlan Voss's performance review ratings in 2022 and 2023?"},
    {"id": "L-T1-09", "q": "What was Solenne Marchand's claimed injury from the slip and fall?"},
    {"id": "L-T1-10", "q": "What is the total damages amount claimed by Solenne Marchand?"},
    {"id": "L-T1-11", "q": "What is Brightfield Pharmaceuticals' medication at the center of the product liability case?"},
    {"id": "L-T1-12", "q": "Who is the defense expert retained in the Brightfield case, and where is he from?"},
    {"id": "L-T1-13", "q": "What is Duncan Farrow's role at Castellan & Briggs?"},
    {"id": "L-T1-14", "q": "What was the amount Terravast was contractually obligated to contribute?"},
    {"id": "L-T1-15", "q": "What mediation service was agreed upon for the Voss case?"},
    {"id": "L-T1-16", "q": "What time did the salt truck treat North Michigan Avenue on the morning of Marchand's fall?"},
    {"id": "L-T1-17", "q": "What is the EEOC charge number for the Voss case?"},
    {"id": "L-T1-18", "q": "What was the demand amount in the Voss demand letter?"},
    {"id": "L-T1-19", "q": "Who is the plaintiff's counsel in Marchand's case?"},
    {"id": "L-T1-20", "q": "What are the three plaintiffs in the consolidated Brightfield case?"},
    # Tier 2 — Single hop
    {"id": "L-T2-01", "q": "What problem did Renata Osei identify in the NorthBridge contract, and how was it resolved?"},
    {"id": "L-T2-02", "q": "What three causes of action did NorthBridge file against Terravast?"},
    {"id": "L-T2-03", "q": "What did Terravast's asset investigation reveal about their real estate holdings?"},
    {"id": "L-T2-04", "q": "What are the three counts in the Voss complaint, and who are the named defendants?"},
    {"id": "L-T2-05", "q": "How many days passed between Voss's whistleblower complaint and his termination, and why is that significant?"},
    {"id": "L-T2-06", "q": "What specific observations did witness Thomas Quayle make that helped the City's defense?"},
    {"id": "L-T2-07", "q": "What counterclaim did Terravast file against NorthBridge, and for how much?"},
    {"id": "L-T2-08", "q": "What is the expert fee structure for Dr. Belkin in the Brightfield case?"},
    {"id": "L-T2-09", "q": "What is the significance of Anton Devereux's case among the three Brightfield plaintiffs?"},
    {"id": "L-T2-10", "q": "What was Kellerman & Drape's counter-offer, and why did Voss reject it?"},
    {"id": "L-T2-11", "q": "What was Leo Nakashima's settlement recommendation for Marchand, and under what condition would he recommend trial?"},
    {"id": "L-T2-12", "q": "What is Terravast's affirmative defense related to zoning, and how does it change what Castellan needs to investigate?"},
    {"id": "L-T2-13", "q": "What Illinois case did Imani Foster cite as precedent for the Voss whistleblower claim?"},
    {"id": "L-T2-14", "q": "What was the bridge loan amount, who was it from, and at what interest rate?"},
    {"id": "L-T2-15", "q": "What is the deadline for NorthBridge's response to Terravast's counterclaim?"},
    {"id": "L-T2-16", "q": "What is the legal weakness in the Voss whistleblower case that Imani Foster identified?"},
    {"id": "L-T2-17", "q": "What motion did Torres, Reid & Hatch file in the Brightfield case, and what is the issue with it?"},
    {"id": "L-T2-18", "q": "Who is Brightfield's opposing expert, and what does Dr. Belkin think of her approach?"},
    {"id": "L-T2-19", "q": "What are the key dates in the NorthBridge v. Terravast discovery schedule?"},
    {"id": "L-T2-20", "q": "Who were the registered agent and defense counsel for Terravast Holdings?"},
    # Tier 3 — Multi-hop
    {"id": "L-T3-01", "q": "Which attorney worked on both defense cases, and which cases were they?"},
    {"id": "L-T3-02", "q": "What is the full chain of events from Voss's whistleblower complaint to mediation scheduled in March 2025?"},
    {"id": "L-T3-03", "q": "Two cases involve Renata Osei. Which cases, and what did she do on each?"},
    {"id": "L-T3-04", "q": "What is the full picture of Terravast's financial situation, and how does it affect litigation strategy?"},
    {"id": "L-T3-05", "q": "What is the full picture of all Terravast affirmative defenses and how does each affect NorthBridge's case?"},
    {"id": "L-T3-06", "q": "Which case has the most complex expert witness situation, and why?"},
    {"id": "L-T3-07", "q": "What is the connection between the NorthBridge case and the Voss case regarding a shared name, and what is the risk?"},
    {"id": "L-T3-08", "q": "What is the complete damages picture for Harlan Voss as of the demand letter, and how has it evolved by March 2025?"},
    {"id": "L-T3-09", "q": "All three Brightfield plaintiffs shared the same pre-existing condition. What was it, how long was each on Velantrix, and whose case is most severe?"},
    {"id": "L-T3-10", "q": "What would need to happen to add a zoning claim against the City of Chicago in the NorthBridge case?"},
    # Unanswerable
    {"id": "L-U-01", "q": "What law school did Margaret Castellan attend?"},
    {"id": "L-U-02", "q": "How many attorneys work at Castellan & Briggs total?"},
    {"id": "L-U-03", "q": "What is Harlan Voss's home address?"},
    {"id": "L-U-04", "q": "Has Brightfield faced product liability suits before?"},
    {"id": "L-U-05", "q": "What floor is the Castellan & Briggs office on?"},
    {"id": "L-U-06", "q": "What is Derek Briggs's area of specialty?"},   # TRAP — employment law (inferable)
    {"id": "L-U-07", "q": "How long has NorthBridge Capital been in business?"},
    {"id": "L-U-08", "q": "What are the total estimated legal fees for the NorthBridge matter?"},
    {"id": "L-U-09", "q": "What happened to the Meridian Tower property after it was sold?"},
    {"id": "L-U-10", "q": "Is Solenne Marchand still employed?"},
]

# Veloris questions use V- prefix IDs; key_id maps to answer_key.txt format (T1-01 etc.)
VELORIS_QUESTIONS = [
    # Tier 1 — Direct lookup
    {"id": "V-T1-01", "key_id": "T1-01", "q": "What is Marcus Webb's job title?"},
    {"id": "V-T1-02", "key_id": "T1-02", "q": "Who is Leo Reyes's manager?"},
    {"id": "V-T1-03", "key_id": "T1-03", "q": "What is the name of Veloris's main product?"},
    {"id": "V-T1-04", "key_id": "T1-04", "q": "How much was Veloris paying for Datadog per month?"},
    {"id": "V-T1-05", "key_id": "T1-05", "q": "What city is Veloris headquartered in?"},
    {"id": "V-T1-06", "key_id": "T1-06", "q": "What message queue technology did Darnell propose for the Conduit rewrite?"},
    {"id": "V-T1-07", "key_id": "T1-07", "q": "What analytics store did Priya originally suggest for Conduit?"},
    {"id": "V-T1-08", "key_id": "T1-08", "q": "What analytics store did Darnell propose instead, and why?"},
    {"id": "V-T1-09", "key_id": "T1-09", "q": "What is the test coverage percentage of the Atlas payments module?"},
    {"id": "V-T1-10", "key_id": "T1-10", "q": "How much money does Veloris process monthly through the payments module?"},
    {"id": "V-T1-11", "key_id": "T1-11", "q": "What build toolchain is Suki using for the React Native app?"},
    {"id": "V-T1-12", "key_id": "T1-12", "q": "What is the name of the internal monitoring project?"},
    {"id": "V-T1-13", "key_id": "T1-13", "q": "What security firm did Marcus choose for the audit?"},
    {"id": "V-T1-14", "key_id": "T1-14", "q": "What was the other security firm Marcus considered?"},
    {"id": "V-T1-15", "key_id": "T1-15", "q": "When does the security audit start?"},
    {"id": "V-T1-16", "key_id": "T1-16", "q": "What crash reporting tool does the Atlas mobile app use?"},
    {"id": "V-T1-17", "key_id": "T1-17", "q": "How many enterprise customers' users signed up for the Atlas mobile beta?"},
    {"id": "V-T1-18", "key_id": "T1-18", "q": "What is the actual monthly cost of the self-hosted Nightwatch stack?"},
    {"id": "V-T1-19", "key_id": "T1-19", "q": "Who is Tobias Hunt's direct manager?"},
    {"id": "V-T1-20", "key_id": "T1-20", "q": "What training did Marcus require all engineers to complete by end of February?"},
    # Tier 2 — Single hop
    {"id": "V-T2-01", "key_id": "T2-01", "q": "Who did Marcus assign to lead the Conduit rewrite?"},
    {"id": "V-T2-02", "key_id": "T2-02", "q": "What specific part of the Conduit system is Tobias responsible for?"},
    {"id": "V-T2-03", "key_id": "T2-03", "q": "What technology did Priya explicitly ban from the Conduit rewrite, and why?"},
    {"id": "V-T2-04", "key_id": "T2-04", "q": "What was Darnell's preferred timeline for Conduit if the backfill feature was dropped?"},
    {"id": "V-T2-05", "key_id": "T2-05", "q": "Why did Priya insist on keeping the backfill feature?"},
    {"id": "V-T2-06", "key_id": "T2-06", "q": "What log aggregation tool is part of the new Nightwatch stack?"},
    {"id": "V-T2-07", "key_id": "T2-07", "q": "What is the projected annual saving from the Datadog migration?"},
    {"id": "V-T2-08", "key_id": "T2-08", "q": "What was Leo's fallback option if Loki had problems?"},
    {"id": "V-T2-09", "key_id": "T2-09", "q": "What phase of the Atlas mobile app includes offline sync?"},
    {"id": "V-T2-10", "key_id": "T2-10", "q": "Why is the Atlas backend off-limits for mobile development in Phase 1?"},
    {"id": "V-T2-11", "key_id": "T2-11", "q": "What compliance issue did Leo discover in the payment logging?"},
    {"id": "V-T2-12", "key_id": "T2-12", "q": "Who discovered the race condition in the payment refund logic?"},
    {"id": "V-T2-13", "key_id": "T2-13", "q": "Where are the Stripe test credentials stored?"},
    {"id": "V-T2-14", "key_id": "T2-14", "q": "What problem did Suki encounter with Expo related to Meridian Group's requirements?"},
    {"id": "V-T2-15", "key_id": "T2-15", "q": "What solution did Suki recommend for the Expo biometric bug?"},
    {"id": "V-T2-16", "key_id": "T2-16", "q": "What Conduit throughput did Darnell achieve in load testing, and how did it compare to the target?"},
    {"id": "V-T2-17", "key_id": "T2-17", "q": "What p99 latency did Conduit achieve in load testing?"},
    {"id": "V-T2-18", "key_id": "T2-18", "q": "Where did Leo document the LogQL learning guide?"},
    {"id": "V-T2-19", "key_id": "T2-19", "q": "What is the mobile beta launch date?"},
    {"id": "V-T2-20", "key_id": "T2-20", "q": "What known issue does Sentry's React Native SDK have?"},
    # Tier 3 — Multi-hop
    {"id": "V-T3-01", "key_id": "T3-01", "q": "Which engineer was originally assigned to write tests for the payments module but was pulled away, and what were they pulled onto?"},
    {"id": "V-T3-02", "key_id": "T3-02", "q": "Who flagged the PCI compliance issue, and what action did Marcus take as a result?"},
    {"id": "V-T3-03", "key_id": "T3-03", "q": "What technology stack replaced Datadog, and which team member needed to be involved in both the replacement AND the Conduit rewrite?"},
    {"id": "V-T3-04", "key_id": "T3-04", "q": "The mobile beta includes a customer with special authentication requirements. What is that customer, what was their requirement, and how was it solved?"},
    {"id": "V-T3-05", "key_id": "T3-05", "q": "How many dashboards did the new monitoring system have after migration, and how does that compare to before?"},
    {"id": "V-T3-06", "key_id": "T3-06", "q": "Who reported a concern to Priya about test coverage, what module was the concern about, and who was ultimately assigned to fix it?"},
    {"id": "V-T3-07", "key_id": "T3-07", "q": "Which project involves both Suki Tanaka and Camille Russo working together, and what did Camille change about Suki's original scope?"},
    {"id": "V-T3-08", "key_id": "T3-08", "q": "Tobias Hunt worked on two separate things in Q1. What were they, and on which one did his manager say he deserved a performance note?"},
    {"id": "V-T3-09", "key_id": "T3-09", "q": "What was the actual monthly cost saving achieved from the Nightwatch migration, and how did it compare to Leo's original estimate?"},
    {"id": "V-T3-10", "key_id": "T3-10", "q": "Which engineer discovered a bug while doing a different task, and what were both the task they were doing and the bug they found?"},
    # Unanswerable
    {"id": "V-U-01", "key_id": "U-01", "q": "What is Priya Nair's salary?"},
    {"id": "V-U-02", "key_id": "U-02", "q": "What programming language is the Atlas backend written in?"},
    {"id": "V-U-03", "key_id": "U-03", "q": "How many total employees does Veloris have?"},
    {"id": "V-U-04", "key_id": "U-04", "q": "What did Marcus think about Python as a language?"},
    {"id": "V-U-05", "key_id": "U-05", "q": "Which cloud provider does Veloris use?"},  # TRAP — AWS
    {"id": "V-U-06", "key_id": "U-06", "q": "What is Hana Bergström's salary?"},
    {"id": "V-U-07", "key_id": "U-07", "q": "Has Veloris ever had a data breach?"},
    {"id": "V-U-08", "key_id": "U-08", "q": "What is Tobias Hunt's favorite programming language?"},
    {"id": "V-U-09", "key_id": "U-09", "q": "Does Veloris have offices outside Austin?"},
    {"id": "V-U-10", "key_id": "U-10", "q": "What database does the Atlas backend use?"},
]

# ---------------------------------------------------------------------------
# Auto-score
# ---------------------------------------------------------------------------

_UNCERTAINTY_MARKERS = [
    "not mentioned", "don't know", "no information", "cannot determine",
    "not in the", "i don't", "no data", "not available", "not provided",
    "not stated", "not specified", "unable to find", "no record",
]

_TRAPS = {
    "M-U-04": "hypertension",   # cause of Raymond Chu's small vessel disease
    "L-U-06": "employment",     # Derek Briggs's area of specialty
    "V-U-05": "aws",            # cloud provider Veloris uses
}


def auto_score(q_id: str, answer: str) -> str:
    a = answer.lower()
    if q_id in _TRAPS:
        kw = _TRAPS[q_id]
        return "PASS (trap)" if kw in a else f"FAIL — missed trap (expected: {kw})"
    parts = q_id.split("-")
    # Unanswerable: M-U-xx / L-U-xx (3 parts, middle is U) or V-U-xx (3 parts, middle is U)
    if len(parts) == 3 and parts[1] == "U":
        uncertain = any(m in a for m in _UNCERTAINTY_MARKERS)
        return "PASS" if uncertain else "FAIL? (possible false positive — review)"
    return "REVIEW"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_clinical_notes(text: str) -> tuple[str, list[dict]]:
    blocks = re.split(r"\n---\n", text)
    header = blocks[0].strip()
    notes = []
    for block in blocks[1:]:
        block = block.strip()
        if not re.match(r"^CLINICAL NOTE \d+", block):
            continue
        h: dict = {"body": block}
        for line in block.split("\n")[1:6]:
            if line.startswith("Date:"):       h["date"] = line[5:].strip()
            elif line.startswith("Patient:"):  h["patient"] = line[8:].strip()
            elif line.startswith("Provider:"): h["provider"] = line[9:].strip()
            elif line.startswith("Note Type:"): h["note_type"] = line[10:].strip()
        notes.append(h)
    return header, notes


def parse_case_notes(text: str) -> tuple[str, list[dict]]:
    blocks = re.split(r"\n---\n", text)
    header = blocks[0].strip()
    notes = []
    for block in blocks[1:]:
        block = block.strip()
        if not re.match(r"^CASE NOTE \d+", block):
            continue
        h: dict = {"body": block}
        for line in block.split("\n")[1:7]:
            if line.startswith("Date:"):       h["date"] = line[5:].strip()
            elif line.startswith("Case:"):     h["case"] = line[5:].strip()
            elif line.startswith("Attorney:"): h["attorney"] = line[9:].strip()
            elif line.startswith("Note Type:"): h["note_type"] = line[10:].strip()
        notes.append(h)
    return header, notes


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
            if line.startswith("From:"):      h["from"] = line[5:].strip()
            elif line.startswith("To:"):      h["to"] = line[3:].strip()
            elif line.startswith("Subject:"): h["subject"] = line[8:].strip()
            elif line == "" and "subject" in h:
                body_start = i + 1
                break
        h["body"] = "\n".join(lines[body_start:]).strip()
        emails.append(h)
    return emails


def parse_answer_key(text: str) -> dict[str, str]:
    """Handles M-T1-01 (medical/legal), T1-01 (Veloris), and D-01 (disambiguation) formats."""
    answers: dict[str, str] = {}
    current_id: str | None = None
    for line in text.split("\n"):
        m = re.match(r"^([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+): ", line)
        if m:
            current_id = m.group(1)
        elif line.startswith("ANSWER: ") and current_id:
            answers[current_id] = line[8:].strip()
            current_id = None
    return answers


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ---------------------------------------------------------------------------
# Shared infra
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=2)
        s.close()
        return True
    except OSError:
        return False


def _load_claude_key() -> str | None:
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLAUDE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("CLAUDE_API_KEY")


def _llm_cfg(claude_key: str | None) -> LLMConfig:
    return LLMConfig(
        base_url=OLLAMA_BASE,
        model=MODEL,
        answer_model=CLAUDE_MODEL if claude_key else None,
        answer_base_url=CLAUDE_BASE if claude_key else None,
        answer_api_key=claude_key,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _tier_key(q_id: str) -> str:
    """Extract tier from question ID. Handles M-T1-01, L-T2-01, V-T3-01, V-U-05 formats."""
    parts = q_id.split("-")
    return parts[1]  # T1, T2, T3, or U for all current ID formats


def _ask_all(
    agent: MemoryAgent,
    questions: list[dict],
    answer_key: dict[str, str],
    section_label: str,
) -> dict:
    """Run recall on all questions and return results. Prints progress."""
    print(f"[{section_label}] Querying {len(questions)} questions...")
    results: dict[str, str] = {}
    recall_times: list[float] = []
    tokens_out: dict[str, int] = {}

    for q in questions:
        pre = agent.token_stats()
        t0 = perf_counter()
        answer = agent.recall(q["q"], mode="answer")
        elapsed = perf_counter() - t0
        post = agent.token_stats()
        recall_times.append(elapsed)
        results[q["id"]] = answer
        tokens_out[q["id"]] = post["tokens_out"] - pre["tokens_out"]
        score = auto_score(q["id"], answer)
        print(f"  {q['id']} [{elapsed*1000:.0f}ms] {score}")

    return {"results": results, "recall_times": recall_times, "tokens_out": tokens_out}


_TIER_LABELS = {
    "T1": "TIER 1 — DIRECT LOOKUP",
    "T2": "TIER 2 — SINGLE HOP",
    "T3": "TIER 3 — MULTI-HOP",
    "U":  "UNANSWERABLE",
}


def _format_question_block(
    questions: list[dict],
    answer_key: dict[str, str],
    run_data: dict,
) -> list[str]:
    """Format questions + answers into result lines including a score summary."""
    lines: list[str] = []
    w = lines.append
    results = run_data["results"]
    tokens_out = run_data["tokens_out"]

    current_tier = None
    u_pass = 0
    u_total = 0
    trap_lines: list[str] = []

    for q in questions:
        tk = _tier_key(q["id"])
        if tk != current_tier:
            current_tier = tk
            w("")
            w(f"--- {_TIER_LABELS.get(tk, tk)} ---")
            w("")

        answer = results[q["id"]]
        score = auto_score(q["id"], answer)
        key_id = q.get("key_id", q["id"])
        expected = answer_key.get(key_id, "—")

        if q["id"] in _TRAPS:
            trap_lines.append(f"{q['id']}: {score}")
        elif tk == "U":
            u_total += 1
            if score.startswith("PASS"):
                u_pass += 1

        w(f"{q['id']} | {q['q']}")
        w(f"  EXPECTED : {expected}")
        w(f"  ANSWER   : {answer.strip()}")
        w(f"  SCORE    : {score}")
        w(f"  TOKENS   : {tokens_out.get(q['id'], 0)} output tokens")
        w("")

    w("")
    w("--- SCORE SUMMARY (fill in manually) ---")
    for tier in ("T1", "T2", "T3"):
        qs = [q for q in questions if _tier_key(q["id"]) == tier]
        if qs:
            w(f"{tier} accuracy : __/{len(qs)}")
    if u_total:
        w(f"U  auto-scored : {u_pass}/{u_total} pass (excl. traps)")
    for tl in trap_lines:
        w(f"Trap: {tl}")

    return lines


def _query_and_write(
    agent: MemoryAgent,
    questions: list[dict],
    answer_key: dict[str, str],
    results_file: Path,
    label: str,
    ingest_tokens: dict,
    node_count: int,
    observe_times: list[float],
    session_persistence: bool,
    notes_count: int,
) -> None:
    """Run recall on all questions and write the full results file."""
    run_data = _ask_all(agent, questions, answer_key, label)

    lines: list[str] = []
    w = lines.append

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w(f"=== {label.upper()} BENCHMARK — pocket-mem ===")
    w(f"Run: {ts}   Ingest model: {MODEL}   Answer model: {agent._llm.config.answer_model or MODEL}")
    w("")
    w("--- INGESTION ---")
    w(f"Notes ingested  : {notes_count}")
    w(f"Tokens (ingest) : in={ingest_tokens['tokens_in']}  out={ingest_tokens['tokens_out']}")
    w(f"Nodes stored    : {node_count}")
    w(f"Observe p50     : {percentile(observe_times, 50)*1000:.0f}ms")
    w(f"Observe p99     : {percentile(observe_times, 99)*1000:.0f}ms")
    w("")
    w("--- SESSION PERSISTENCE ---")
    w("PASS" if session_persistence else "FAIL")

    lines.extend(_format_question_block(questions, answer_key, run_data))

    recall_times = run_data["recall_times"]
    total_tokens = agent.token_stats()
    w("")
    w("--- LATENCY ---")
    w(f"Recall p50 : {percentile(recall_times, 50)*1000:.0f}ms")
    w(f"Recall p99 : {percentile(recall_times, 99)*1000:.0f}ms")
    w("")
    w("--- TOKEN TOTALS ---")
    w(f"tokens_in  : {total_tokens['tokens_in']}")
    w(f"tokens_out : {total_tokens['tokens_out']}")
    cost = total_tokens["tokens_in"] * 0.00000025 + total_tokens["tokens_out"] * 0.00000125
    w(f"Est. cost (Haiku pricing) : ${cost:.4f}")

    results_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[{label}] Results written to {results_file}")


# ---------------------------------------------------------------------------
# Test 1 — Medical dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_medical_benchmark():
    claude_key = _load_claude_key()
    llm = _llm_cfg(claude_key)
    project = "hargrove-benchmark"
    mem_path = MEMORY_DIR / "medical"
    mem_path.mkdir(parents=True, exist_ok=True)
    results_file = SIM_DIR / "last_run_results_medical.txt"

    raw = MEDICAL_NOTES_FILE.read_text(encoding="utf-8")
    header, notes = parse_clinical_notes(raw)
    answer_key = parse_answer_key(MEDICAL_KEY_FILE.read_text(encoding="utf-8"))

    print(f"\n[medical] Ingesting header + {len(notes)} clinical notes...")
    print(f"[medical] Memory store: {mem_path / (project + '.db')}")
    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    observe_times: list[float] = []

    if header:
        agent.observe(user_input="Clinic reference document", agent_response=header)

    for i, note in enumerate(notes, 1):
        patient = note.get("patient", "unknown patient")
        note_type = note.get("note_type", "clinical note")
        t0 = perf_counter()
        agent.observe(
            user_input=f"Clinical note for {patient} — {note_type}",
            agent_response=note["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{i:02d}/{len(notes)}] {patient} — {note_type}")

    print("[medical] Flushing ingestion...")
    agent.flush()
    ingest_tokens = agent.token_stats()
    stats = agent.stats()
    agent.close()
    print(f"[medical] Complete. nodes={stats['node_count']} tokens={ingest_tokens}")
    assert stats["node_count"] > 0, "No nodes stored after medical ingestion"

    print("[medical] Restarting agent...")
    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    session_persistence = agent.stats()["node_count"] > 0

    _query_and_write(
        agent=agent,
        questions=MEDICAL_QUESTIONS,
        answer_key=answer_key,
        results_file=results_file,
        label="Medical",
        ingest_tokens=ingest_tokens,
        node_count=stats["node_count"],
        observe_times=observe_times,
        session_persistence=session_persistence,
        notes_count=len(notes),
    )
    agent.close()


# ---------------------------------------------------------------------------
# Test 2 — Legal dataset
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_legal_benchmark():
    claude_key = _load_claude_key()
    llm = _llm_cfg(claude_key)
    project = "castellan-benchmark"
    mem_path = MEMORY_DIR / "legal"
    mem_path.mkdir(parents=True, exist_ok=True)
    results_file = SIM_DIR / "last_run_results_legal.txt"

    raw = LEGAL_NOTES_FILE.read_text(encoding="utf-8")
    header, notes = parse_case_notes(raw)
    answer_key = parse_answer_key(LEGAL_KEY_FILE.read_text(encoding="utf-8"))

    print(f"\n[legal] Ingesting header + {len(notes)} case notes...")
    print(f"[legal] Memory store: {mem_path / (project + '.db')}")
    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    observe_times: list[float] = []

    if header:
        agent.observe(user_input="Law firm and cases reference document", agent_response=header)

    for i, note in enumerate(notes, 1):
        case = note.get("case", "unknown case")
        note_type = note.get("note_type", "case note")
        t0 = perf_counter()
        agent.observe(
            user_input=f"Case note: {case} — {note_type}",
            agent_response=note["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{i:02d}/{len(notes)}] {case[:60]} — {note_type}")

    print("[legal] Flushing ingestion...")
    agent.flush()
    ingest_tokens = agent.token_stats()
    stats = agent.stats()
    agent.close()
    print(f"[legal] Complete. nodes={stats['node_count']} tokens={ingest_tokens}")
    assert stats["node_count"] > 0, "No nodes stored after legal ingestion"

    print("[legal] Restarting agent...")
    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    session_persistence = agent.stats()["node_count"] > 0

    _query_and_write(
        agent=agent,
        questions=LEGAL_QUESTIONS,
        answer_key=answer_key,
        results_file=results_file,
        label="Legal",
        ingest_tokens=ingest_tokens,
        node_count=stats["node_count"],
        observe_times=observe_times,
        session_persistence=session_persistence,
        notes_count=len(notes),
    )
    agent.close()


# ---------------------------------------------------------------------------
# Test 3 — Combined (all 3 datasets, 180 questions)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running")
def test_combined_benchmark():
    claude_key = _load_claude_key()
    llm = _llm_cfg(claude_key)
    project = "combined-benchmark"
    mem_path = MEMORY_DIR / "combined"
    mem_path.mkdir(parents=True, exist_ok=True)
    results_file = SIM_DIR / "last_run_results_combined.txt"

    # Load all three datasets
    veloris_raw = EMAILS_FILE.read_text(encoding="utf-8")
    veloris_header = veloris_raw.split("\n---\n")[0].strip()
    emails = parse_emails(veloris_raw)

    medical_raw = MEDICAL_NOTES_FILE.read_text(encoding="utf-8")
    medical_header, clinical_notes = parse_clinical_notes(medical_raw)

    legal_raw = LEGAL_NOTES_FILE.read_text(encoding="utf-8")
    legal_header, case_notes = parse_case_notes(legal_raw)

    total_docs = len(emails) + len(clinical_notes) + len(case_notes)
    print(f"\n[combined] Ingesting {total_docs} documents across 3 datasets...")
    print(f"[combined] Memory store: {mem_path / (project + '.db')}")

    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    observe_times: list[float] = []
    doc_num = 0

    # --- Veloris emails ---
    if veloris_header:
        agent.observe(user_input="Veloris Technologies company reference document", agent_response=veloris_header)

    for email in emails:
        doc_num += 1
        t0 = perf_counter()
        agent.observe(
            user_input=f"[Veloris] Email from {email['from']} to {email['to']}: {email['subject']}",
            agent_response=email["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{doc_num:02d}/{total_docs}] [Veloris] {email['subject'][:55]}")

    # --- Medical clinical notes ---
    if medical_header:
        agent.observe(user_input="Hargrove Family Medicine clinic reference document", agent_response=medical_header)

    for note in clinical_notes:
        doc_num += 1
        patient = note.get("patient", "patient")
        note_type = note.get("note_type", "note")
        t0 = perf_counter()
        agent.observe(
            user_input=f"[Hargrove Medical] Clinical note for {patient} — {note_type}",
            agent_response=note["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{doc_num:02d}/{total_docs}] [Medical] {patient} — {note_type}")

    # --- Legal case notes ---
    if legal_header:
        agent.observe(user_input="Castellan & Briggs LLP law firm reference document", agent_response=legal_header)

    for note in case_notes:
        doc_num += 1
        case = note.get("case", "case")
        note_type = note.get("note_type", "note")
        t0 = perf_counter()
        agent.observe(
            user_input=f"[Castellan & Briggs] Case note: {case} — {note_type}",
            agent_response=note["body"],
        )
        observe_times.append(perf_counter() - t0)
        print(f"  [{doc_num:02d}/{total_docs}] [Legal] {case[:45]} — {note_type}")

    print("[combined] Flushing ingestion...")
    agent.flush()
    ingest_tokens = agent.token_stats()
    stats = agent.stats()
    agent.close()
    print(f"[combined] Complete. nodes={stats['node_count']} tokens={ingest_tokens}")
    assert stats["node_count"] > 0, "No nodes stored after combined ingestion"

    # Phase 2 — Restart
    print("[combined] Restarting agent...")
    agent = MemoryAgent(project, path=str(mem_path), llm=llm)
    session_persistence = agent.stats()["node_count"] > 0

    # Phase 3 — Load answer keys
    veloris_key  = parse_answer_key(EMAILS_KEY_FILE.read_text(encoding="utf-8"))
    medical_key  = parse_answer_key(MEDICAL_KEY_FILE.read_text(encoding="utf-8"))
    legal_key    = parse_answer_key(LEGAL_KEY_FILE.read_text(encoding="utf-8"))

    # Phase 4 — Run all 180 questions
    veloris_run = _ask_all(agent, VELORIS_QUESTIONS,  veloris_key,  "Combined/Veloris")
    medical_run = _ask_all(agent, MEDICAL_QUESTIONS,  medical_key,  "Combined/Medical")
    legal_run   = _ask_all(agent, LEGAL_QUESTIONS,    legal_key,    "Combined/Legal")

    # Phase 5 — Write combined results file
    lines: list[str] = []
    w = lines.append

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w("=== COMBINED BENCHMARK — pocket-mem ===")
    w(f"Run: {ts}   Ingest model: {MODEL}   Answer model: {agent._llm.config.answer_model or MODEL}")
    w("")
    w("--- INGESTION ---")
    w(f"Documents ingested : {total_docs} ({len(emails)} Veloris emails + {len(clinical_notes)} medical notes + {len(case_notes)} legal notes)")
    w(f"Tokens (ingest)    : in={ingest_tokens['tokens_in']}  out={ingest_tokens['tokens_out']}")
    w(f"Nodes stored       : {stats['node_count']}")
    w(f"Observe p50        : {percentile(observe_times, 50)*1000:.0f}ms")
    w(f"Observe p99        : {percentile(observe_times, 99)*1000:.0f}ms")
    w("")
    w("--- SESSION PERSISTENCE ---")
    w("PASS" if session_persistence else "FAIL")
    w("")

    # --- Veloris section ---
    w("=" * 60)
    w("=== VELORIS TECHNOLOGIES — 60 QUESTIONS ===")
    w("=" * 60)
    lines.extend(_format_question_block(VELORIS_QUESTIONS, veloris_key, veloris_run))

    # --- Medical section ---
    w("")
    w("=" * 60)
    w("=== HARGROVE FAMILY MEDICINE — 60 QUESTIONS ===")
    w("=" * 60)
    lines.extend(_format_question_block(MEDICAL_QUESTIONS, medical_key, medical_run))

    # --- Legal section ---
    w("")
    w("=" * 60)
    w("=== CASTELLAN & BRIGGS LLP — 60 QUESTIONS ===")
    w("=" * 60)
    lines.extend(_format_question_block(LEGAL_QUESTIONS, legal_key, legal_run))

    # --- Overall totals ---
    all_recall_times = (
        veloris_run["recall_times"]
        + medical_run["recall_times"]
        + legal_run["recall_times"]
    )
    total_tokens = agent.token_stats()
    cost = total_tokens["tokens_in"] * 0.00000025 + total_tokens["tokens_out"] * 0.00000125

    w("")
    w("=" * 60)
    w("=== OVERALL TOTALS ===")
    w("=" * 60)
    w(f"Total questions    : 180 (60 Veloris + 60 Medical + 60 Legal)")
    w(f"Recall p50         : {percentile(all_recall_times, 50)*1000:.0f}ms")
    w(f"Recall p99         : {percentile(all_recall_times, 99)*1000:.0f}ms")
    w(f"tokens_in          : {total_tokens['tokens_in']}")
    w(f"tokens_out         : {total_tokens['tokens_out']}")
    w(f"Est. cost (Haiku)  : ${cost:.4f}")

    results_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[combined] Results written to {results_file}")
    agent.close()
