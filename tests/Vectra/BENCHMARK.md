# Vectra benchmark

Tests the pocket-mem identity system against a personal AI assistant use case.
Jordan Mercer's Alexa interaction log (30 interactions, Monday–Friday March 3–7 2025)
is ingested into two databases in a single test run — one with the Vectra identity
active, one without — then queried with the same 60 questions so results can be
compared directly.

---

## Dataset

**Source:** `test_data/alexa_conversations.txt`
**Interactions:** 30 Alexa voice/text/email exchanges across 5 days
**Devices:** Amazon Echo (living room) + Echo Dot (bedroom)
**User:** Jordan Mercer, Austin, Texas

Key content: appointments (doctor, Q1 review, Henderson call, Denver flight),
reminders (5 set across the week), shopping list (7 items accumulated),
financial transactions (Chase payment, gym renewal, lease renewal),
contacts (mom Carol, coworker Ryan Patel, friend Dani Torres, boss Marcus,
landlord Paul Whitfield, doctor Sandra Okafor).

---

## Identity — Vectra

```
Personal AI assistant named Vectra managing Jordan's daily life in Austin, Texas.
I listen to all voice interactions, emails, and text messages to track Jordan's
schedule, reminders, appointments, shopping lists, financial tasks, contacts,
and anything that needs to be remembered or acted on.
```

Prebuilt config: `pocket_mem/identities/configs/vectra.json`
Expected role: **Personal AI Assistant**
Seed topics: Schedule and Reminders, Contacts and People, Shopping List,
Financial Tasks, Health and Appointments, Work and Meetings, Travel, Home and Household

---

## Question set

60 questions: 20 T1 (direct lookup) + 20 T2 (single hop) + 10 T3 (multi-hop) + 10 U (unanswerable)

**Trap question: A-U-05** — "Who is Jordan flying to Denver to visit or meet?"
The Denver flight details are documented (DL 447, 6:15 AM, confirmation HXMT72) but
the purpose of the trip is never stated. Correct answer: not mentioned.

---

## How to run

```bash
# Fresh ingest of both DBs + 60 questions
pytest tests/Vectra/ -v -s

# Skip ingestion, use existing DBs (both must exist)
USE_EXISTING_DB=1 pytest tests/Vectra/ -v -s

# Force identity derivation via LLM instead of prebuilt
IDENTITY_USE_LLM=1 pytest tests/Vectra/ -v -s

# Force LLM derivation via Gemini
IDENTITY_USE_LLM=1 GEMINI_API_KEY=your-key pytest tests/Vectra/ -v -s
```

Results are written to `last_run_results.txt` with identity and baseline answers
shown side-by-side for every question.

DBs:
- `memory/vectra-identity.db` — identity-active agent
- `memory/vectra-baseline.db` — no-identity agent

---

## Run history


### Run 1 — 2026-05-08 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Identity (Personal AI Assistant) | Baseline (no identity) |
|--------|----------------------------------|------------------------|
| Mode | fresh ingest | fresh ingest |
| T1 accuracy | 20/20 = **100%** | 20/20 = **100%** |
| T2 accuracy | 18.5/20 = **92.5%** | 18.5/20 = **92.5%** |
| T3 accuracy | 7.5/10 = **75%** | 6.5/10 = **65%** |
| Overall | **46/50 = 92%** | **45/50 = 90%** |
| False positives | **0/10** | **0/10** |
| Trap (A-U-05) | **PASS** | **PASS** |
| Nodes stored | 125 | 134 |
| Recall p50 | 4,296ms | 4,320ms |
| Recall p99 | 9,963ms | 8,356ms |
| Tokens in (recall) | 212,079 | 217,868 |
| Est. cost | $0.0566 | $0.0583 |

**Identity ingest: in=26,165 out=8,239**
**Baseline ingest: in=25,772 out=8,245**

**T2 wrong (both modes, 1.5 each):**
- A-T2-08 (0.5) — Partial: Ryan's Thursday text captured, Jordan's reply ("slides polished by
  Sunday night") not retrieved in identity. Baseline contradicts itself — mentions reply then
  denies it.
- A-T2-20 (0.0) — Retrieval miss: Monday morning weather forecast (58°, partly cloudy, high 74,
  20% rain) not surfaced by either mode. Interaction 001 (very first interaction) appears
  underweighted against later higher-frequency interactions.

**T3 wrong (identity 2.5, baseline 3.5):**
- A-T3-01 (0.75 both) — Henderson chain mostly correct; both miss Jordan setting Sunday reminder
  and Jordan's Marcus reply accepting March 17.
- A-T3-02 (0.75 both) — Weekend schedule correct; both miss the weekend weather forecast.
- A-T3-03 (0.5 both) — Check-in step correct; both miss arrival time Denver 8:40 AM,
  doctor/lease as contextual pre-trip items. Identity adds a fabricated "set alarm" step.
- A-T3-04 (0.75 both) — Financial picture mostly correct; both miss credit card number 4471.
- **A-T3-05 (1.0 identity / 0.0 baseline)** — THE KEY DIFFERENCE. Identity correctly
  identifies Ryan Patel as the person in both work (Henderson, Q1 review) and social (trivia
  team) contexts. Baseline retrieves Dr. Sandra Okafor and fabricates a social connection for
  her. Identity importance scoring elevated Ryan Patel's cross-context links.
- A-T3-06 (0.75 both) — Carol thread correct; both miss downstream shopping/reminder actions.
- A-T3-07 (1.0 both) — All 5 reminders correct.
- A-T3-08 (1.0 both) — Medical prep complete.
- A-T3-09 (1.0 both) — Work situation complete.
- **A-T3-10 (0.0 both)** — Shared blind spot. The Alexa-corrects-timing interaction
  (Interaction 023) wasn't retrieved by either mode. Meta-interactions (Alexa questioning
  user input rather than user initiating) aren't stored as distinct retrievable entities.

**U tier: 10/10 — zero false positives (both modes)**
Best U-tier result across all benchmarks. The personal/conversational Alexa format is less
ambiguous than legal case notes or company emails — model clearly knows what it does and
doesn't have. Denver trip trap (A-U-05) correctly declined by both.

**Identity vs Baseline summary:**
Identity beats baseline by 2pp (92% vs 90%). Entire gap is driven by T3-05 alone — one
cross-context entity identification where identity's importance scoring linked Ryan Patel across
work and social graphs, while baseline retrieved the wrong entity. Node delta is negligible
(7%): extraction fix holding across all datasets.

---
