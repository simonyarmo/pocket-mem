# pocket-mem benchmark — identity run

This directory runs benchmarks with an identity configured so extraction, topic
seeding, and importance scoring are shaped by who the agent is working for.

**Runs 1–4:** Veloris Technologies dataset (B2B SaaS company emails — same as
`first_sim_test_50_q`). Identity: Executive Assistant.

**Run 5+:** Castellan & Briggs LLP dataset (litigation law firm case notes).
Identity: Paralegal / Legal Assistant. DB: `memory/legal-benchmark.db`.

---

## Identity (Veloris runs 1–4)

```
Executive assistant at Veloris Technologies, a B2B SaaS company. My job is to
stay informed by reading all company emails every day — tracking people, projects,
decisions, costs, and anything happening across the engineering and product teams
so I can brief leadership and answer questions about what is going on.
```

Expected derived role: **Company Intelligence Assistant** (or similar — depends on LLM)

Expected seed topics: People & Org Chart, Active Projects, Decisions & Approvals,
Costs & Budgets, Risks & Issues, Deadlines, Technical Initiatives, Product Updates

---

## Run history

### Run 1 — 2026-05-05 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · identity derivation FAILED

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 20/20 = **100%** |
| T3 accuracy | 8.5/10 = **85%** |
| Overall accuracy | 48.5/50 = **97%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated 8 named employees as definitive count) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=21,040  out=12,298 |
| Tokens — total (recall phase) | in=382,642  out=4,814 |
| Estimated API cost | $0.1017 (Haiku pricing) |
| Nodes stored | 121 |
| Recall p50 | 7,509ms |
| Recall p99 | 9,203ms |

**Identity derivation: FAILED — ran in generic mode.**

qwen2.5:7b could not produce a valid JSON response with all required keys across
3 retry attempts (2s / 4s / 8s backoff). The identity system fell back to generic
v1 behaviour: no seed topics seeded, no identity-shaped extraction prompt, no
importance scoring differences. This run is effectively identical to a standard
v2 generic run.

**T3 wrong (1.0):**
- T3-03 (0.5) — "Grafana + Prometheus stack replaced Datadog"; Alertmanager and
  Loki missing. Same omission seen in most prior runs (first_sim_test Run 25 was
  the only exception where T3-03 was fully correct).
- T3-05 (0.5) — "50 total (47 migrated + 3 new)" correct but "context does not
  specify how many dashboards existed before the migration." Same structural
  inference gap as every prior run — model uses the 47 figure as migration count
  but won't apply it as comparison baseline.

**U-tier notes:**
- U-02: Auto-scored "possible false positive" but PASS on manual review. Answer
  says "The context does not specify..." which is equivalent to "not mentioned."
- U-03: FALSE POSITIVE — persistent across all runs. Enumerates 8 named employees
  from the ingested company header as a definitive total headcount.
- U-07: Auto-scored "possible false positive" but PASS on manual review. Answer
  correctly says "no mention of a data breach" while adding relevant context.

**Comparison to first_sim_test Run 25 (generic v2, same config):**

| Metric | Run 25 (first_sim_test) | Run 1 (identity, generic mode) |
|--------|------------------------|-------------------------------|
| Overall | 49.5/50 — 99% | 48.5/50 — 97% |
| T1 | 20/20 | 20/20 |
| T2 | 20/20 | 20/20 |
| T3 | 9.5/10 | 8.5/10 |
| Nodes stored | 141 | 121 |
| T3-03 (full stack) | ✅ PASS | ❌ 0.5 (Alertmanager+Loki missing) |
| T3-05 (dashboard compare) | ❌ 0.5 | ❌ 0.5 |

The 2pp gap (99% vs 97%) is within normal run-to-run variance caused by
non-deterministic extraction. The identity system not activating means this run
provides no signal about whether identity helps or hurts accuracy. A proper
comparison requires identity derivation to succeed first.

**Root cause of derivation failure:**

The `resolve_identity` chain is: prebuilt match → SQLite cache → LLM derivation →
fallback. The "executive assistant" description does not match any of the 8
prebuilt roles (similarity < 0.85 against all role labels). The cache was empty
(fresh DB). All 3 LLM derivation attempts using qwen2.5:7b failed — the model
either returned malformed JSON or a response missing required keys.

**Fix needed before re-running:** Either add a prebuilt config for
"company_assistant" / "executive_assistant" to `pocket_mem/identities/configs/`,
or use `IdentityConfig(derivation_api_key=GEMINI_API_KEY)` to route derivation
to Gemini 2.5 Flash which reliably produces valid structured JSON.

### Run 2 — 2026-05-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest

| Metric | Score |
|--------|-------|
| T1 accuracy | __/20 |
| T2 accuracy | __/20 |
| T3 accuracy | __/10 |
| Overall accuracy | __/50 |
| False positive rate | — |
| Trap question (U-05) | PASS (trap) |
| U auto-scored | 7/9 pass (excl. trap) |
| Nodes stored | 39 |
| QA cache entries | 0 |
| Cache hits: 0/60 (0%) | |
| Tokens — recall in | 276567 |
| Tokens — recall out | 3127 |
| Est. cost | $0.0731 |
| Recall p50 | 2010ms |
| Recall p99 | 9872ms |

**Identity: Executive Assistant (prebuilt)**
**Ingest tokens: in=20871 out=10049**
**No cache hits this run**

---

### Run 3 — 2026-05-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest

| Metric | Score |
|--------|-------|
| T1 accuracy | __/20 |
| T2 accuracy | __/20 |
| T3 accuracy | __/10 |
| Overall accuracy | __/50 |
| False positive rate | — |
| Trap question (U-05) | PASS (trap) |
| U auto-scored | 7/9 pass (excl. trap) |
| Nodes stored | 45 |
| QA cache entries | 0 |
| Cache hits: 0/60 (0%) | |
| Tokens — recall in | 287984 |
| Tokens — recall out | 3081 |
| Est. cost | $0.0758 |
| Recall p50 | 5474ms |
| Recall p99 | 10407ms |

**Identity: Executive Assistant (prebuilt)**
**Ingest tokens: in=20871 out=10026**
**No cache hits this run**

---

### Run 4 — 2026-05-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · existing DB (qa_cache active)

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | 8.5/10 = **85%** |
| Overall accuracy | 47.5/50 = **95%** |
| False positive rate | 1/9 ⚠️ (U-03 — persistent employee count) |
| Trap question (U-05) | **PASS** |
| U auto-scored | 7/9 pass (excl. trap) |
| Nodes stored | 43 |
| QA cache entries | 4 |
| Cache hits: 0/60 (0%) | |
| Tokens — recall in | 276,027 |
| Tokens — recall out | 3,046 |
| Est. cost | $0.0728 |
| Recall p50 | 5,463ms |
| Recall p99 | 9,891ms |

**Identity: Executive Assistant (prebuilt)**
**Mode: skipped ingestion — used pre-built DB with qa_cache nodes**
**No cache hits this run**

**T2 wrong (1):**
- T2-05 ❌ — retrieval miss: "I don't have that information. The archived records do not contain Priya's reasoning." The "Camille will never forgive us" quote is in the store but not surfaced from the 43-node DB.

**T3 wrong (1.5):**
- T3-03 (0.5) — stack listed as "Grafana + Prometheus" only; Leo's Conduit role described as "configured Stripe sandbox" rather than Kubernetes deployment.
- T3-05 (0.5) — persistent inference gap: has the 47-dashboard figure but won't apply it as comparison baseline.

**Cache experiment result:**
4 qa_cache entries were in the store (generated by test_question_loop.py from 43 entity nodes).
0/60 benchmark questions produced a cache hit above the 0.88 similarity threshold.
Root cause: the question loop generated topic-coverage questions phrased differently from
benchmark queries — e.g. "Who is responsible for the race condition?" vs the benchmark's
"Who discovered the race condition in the payment refund logic?" Close in meaning, too
different in phrasing to clear 0.88. The cache will only produce hits when cached questions
are semantically close to actual user queries.

**Comparison vs Run 3 (fresh ingest, same DB lineage):**

| | Run 3 (fresh, identity) | Run 4 (existing DB + cache) |
|--|--|--|
| Overall | 48.5/50 — 97% | 47.5/50 — 95% |
| T2-05 | ✅ | ❌ retrieval miss |
| Cache hits | — | 0/60 |
| Cost | $0.0758 | $0.0728 |

---

### Run 5 — 2026-05-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest · Castellan & Briggs legal dataset

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | 7/10 = **70%** |
| Overall accuracy | 46/50 = **92%** |
| False positive rate | 3/9 ⚠️ (U-02, U-04, U-09) |
| Trap question (L-U-06) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=22,642  out=13,203 |
| Tokens — total (recall phase) | in=422,804  out=5,761 |
| Estimated API cost | $0.1129 (Haiku pricing) |
| Nodes stored | 130 |
| Recall p50 | 8,554ms |
| Recall p99 | 9,940ms |

**Identity: derivation FAILED — ran in generic mode.**

Prebuilt matching attempted for "Paralegal at Castellan & Briggs LLP, a litigation law firm..."
Description didn't match any prebuilt config above threshold (paralegal.json was just created and
was available, but similarity scoring against "Legal Assistant" role label was insufficient).
Result: 130 nodes stored in generic mode — higher than Veloris Runs 2–4 (39–45 nodes), likely due
to the structured case note format being entity-rich compared to casual email prose.

**T2 wrong (1):**
- T2-17 ❌ — retrieval miss: Torres, Reid & Hatch motion to compel Brightfield's internal safety
  committee minutes (Case Note 019). Agent returned "I don't have that information."

**T3 wrong (3.0):**
- T3-01 ❌ (1.0) — Wrong attorney identified. Model says "Derek Briggs worked on both defense
  cases." Correct answer is Leo Nakashima (lead on Marchand v. City of Chicago + associate on
  Brightfield). Derek Briggs represents Voss who is the *plaintiff*. Reasoning error: model failed
  to distinguish plaintiff-side vs. defendant-side representation.
- T3-02 ❌ (0.5) — Partial chain. Hits major milestones but missing: Jan 16 Castellan retention,
  Jan 21 Imani Foster legal research, EEOC charge filed Feb 6, March 3 complaint filing.
- T3-04 ❌ (0.5) — Partial financial picture. Correct equity math ($3.3M) but omits: no pending
  bankruptcy + two prior lawsuits both settled — the key collectability evidence.
- T3-05 ❌ (0.5) — Missing Priscilla Thorne (secretary on Marchand case). Only identifies Renata
  Osei.
- T3-06 ❌ (0.5) — Partial reasoning. Misses the FDA safety communication and the Torres Reid
  motion to compel safety meeting minutes as complexity factors.

**U-tier false positives (3/9):**
- U-02: FALSE POSITIVE — "Castellan & Briggs has 4 attorneys total" — enumerates named staff as
  definitive count. Identical pattern to U-03 in Veloris runs.
- U-04: FALSE POSITIVE — Asked "Has Brightfield faced suits *before*?" — model describes the
  current active case rather than recognising the question asks about prior history.
- U-09: FALSE POSITIVE — Asked what happened to Meridian Tower *after* the sale. Model restates
  the damages context from the sale, not post-sale fate (genuinely unmentioned in notes).

**Identity derivation notes for next run:**

The paralegal.json config was created but the prebuilt matcher didn't fire. Likely cause: the
description contains "Paralegal" and "litigation law firm" — the cosine similarity against the
"Legal Assistant" role label may fall below the 0.50 threshold or within margin of another role.
Test with `IDENTITY_USE_LLM=1` or inspect prebuilt matching scores before the next run.

---


### Run 6 — 2026-05-08 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest

| Metric | Score |
|--------|-------|
| T1 accuracy | __/20 |
| T2 accuracy | __/20 |
| T3 accuracy | __/10 |
| Overall accuracy | __/50 |
| False positive rate | — |
| Trap question (L-U-06) | PASS (trap) |
| U auto-scored | 6/9 pass (excl. trap) |
| Nodes stored | 70 |
| QA cache entries | 0 |
| Cache hits: 0/60 (0%) | |
| Tokens — recall in | 331124 |
| Tokens — recall out | 4411 |
| Est. cost | $0.0883 |
| Recall p50 | 6600ms |
| Recall p99 | 12301ms |

**Identity: Paralegal (prebuilt)**
**Ingest tokens: in=24477 out=12351**
**No cache hits this run**

---

### Run 6 — 2026-05-08 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest · identity ACTIVE

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 18.5/20 = **92.5%** |
| T3 accuracy | 6.5/10 = **65%** |
| Overall accuracy | 45/50 = **90%** |
| False positive rate | 3/9 ⚠️ (U-02, U-04, U-09) |
| Trap question (L-U-06) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=24,477  out=12,351 |
| Tokens — total (recall phase) | in=331,124  out=4,411 |
| Estimated API cost | $0.0883 (Haiku pricing) |
| Nodes stored | 70 |
| Recall p50 | 6,600ms |
| Recall p99 | 12,301ms |

**Identity: Paralegal (prebuilt) — ACTIVE for first time on legal dataset.**
**Role fix: paralegal.json `role` changed from "Legal Assistant" → "Paralegal" to match description vocabulary.**
**Ingest tokens: in=24,477 out=12,351**
**No cache hits this run**

**T2 wrong (1.5):**
- T2-06 ❌ (1.0) — NEW retrieval miss. Thomas Quayle observations (Case Note 012) not in 70-node
  DB. Model answers "needs interview" — the pre-interview case note was stored, but the interview
  results note wasn't extracted. Direct casualty of identity node reduction (130 → 70 nodes).
- T2-17 ❌ (0.5) — Partial improvement vs Run 5. Now knows the motion to compel safety minutes
  exists, but still misses the privilege issue. Half the content was retrieved.

**T3 wrong (3.5):**
- T3-01 ❌ (1.0) — Persistent wrong answer across all runs. "Derek Briggs worked on both defense
  cases." Briggs represents plaintiff Voss, not defendants. Leo Nakashima is correct (lead on
  Marchand, associate on Brightfield). Reasoning failure — not a retrieval problem.
- T3-02 ✅ (0.75, improved from 0.5) — Now captures 6/8 events including Castellan retention
  and complaint filing. Still missing: EEOC charge (Feb 6) and Imani Foster research (Jan 21).
- T3-04 ❌ (0.5) — Correct equity math but misses "no pending bankruptcy" and "two prior lawsuits
  both settled" — Case Note 007 collectability facts.
- T3-05 ❌ (0.5) — Identifies Renata Osei correctly; misses Priscilla Thorne (Marchand).
  Also adds fabricated detail about Renata's Voss work not in the source notes.
- T3-06 ❌ (0.5) — Correct on competing experts; misses FDA safety communication and motion to
  compel safety meeting minutes as complexity factors.
- T3-08 ❌ (0.5) — NEW regression from Run 5. $300,000 settlement floor (Case Note 020) not in
  70-node graph. Run 5 answered this correctly with 130 nodes.
- T3-10 ❌ (0.0) — NEW regression from Run 5. Run 5 correctly said "not in the records."
  This run fabricates a legal theory ("establish the zoning office was responsible..."). Fewer
  nodes → less retrieved context → model fills gap with general legal reasoning. Hallucination
  on the inference trap.

**Core finding — identity is still net negative with qwen2.5:7b:**

| | Run 5 (generic, 130 nodes) | Run 6 (identity active, 70 nodes) |
|--|--|--|
| Overall | 46/50 — 92% | 45/50 — 90% |
| T2 | 19/20 — 95% | 18.5/20 — 92.5% |
| T3 | 7/10 — 70% | 6.5/10 — 65% |
| p50 latency | 8,554ms | 6,600ms |
| Cost | $0.1129 | $0.0883 |

Identity activation reduced nodes by 46% (130 → 70). The lost nodes contain real facts that
are needed for T2 and T3: Quayle's witness observations, the $300k settlement floor, the
privilege issue, the bankruptcy check. The identity extraction prompt is over-filtering —
keeping "high importance signal" entities while discarding supporting context.

Positive signals: latency −23%, cost −22%, T1 still perfect. The identity priority signals
are working — but the 46% node reduction is too steep for net accuracy gain.

**Root cause:** qwen2.5:7b treats the identity extraction hint as an exclusion filter rather
than a priority hint. A minimum node floor or relaxing the extraction focus (use identity only
for importance scoring and seed topics, not extraction selection) would likely resolve this.

---

### Run 7 — 2026-05-08 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest

| Metric | Score |
|--------|-------|
| T1 accuracy | __/20 |
| T2 accuracy | __/20 |
| T3 accuracy | __/10 |
| Overall accuracy | __/50 |
| False positive rate | — |
| Trap question (L-U-06) | PASS (trap) |
| U auto-scored | 7/9 pass (excl. trap) |
| Nodes stored | 142 |
| QA cache entries | 0 |
| Cache hits: 0/60 (0%) | |
| Tokens — recall in | 378892 |
| Tokens — recall out | 4540 |
| Est. cost | $0.1004 |
| Recall p50 | 7558ms |
| Recall p99 | 13883ms |

**Identity: Paralegal (prebuilt)**
**Ingest tokens: in=24819 out=11630**
**No cache hits this run**

---

### Run 7 — 2026-05-08 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · fresh ingest · identity ACTIVE · extraction fix applied

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | 7.75/10 = **77.5%** |
| Overall accuracy | 46.75/50 = **93.5%** |
| False positive rate | 3/9 ⚠️ (U-02, U-04, U-09) |
| Trap question (L-U-06) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=24,819  out=11,630 |
| Tokens — total (recall phase) | in=378,892  out=4,540 |
| Estimated API cost | $0.1004 (Haiku pricing) |
| Nodes stored | 142 |
| Recall p50 | 7,558ms |
| Recall p99 | 13,883ms |

**Identity: Paralegal (prebuilt) — ACTIVE.**
**Fix applied: `build_extract_prompt()` no longer injects identity extraction hint.**
**Ingest tokens: in=24,819 out=11,630**
**No cache hits this run**

**T2 wrong (1.0):**
- T2-13 ❌ (0.5) — Finds Solis v. Meridian Partners but wrongly hedges: "mentioned as a
  comparable verdict case, but does not identify it as the precedent cited." Distinction is
  incorrect — Foster explicitly cited it. Name present, framing wrong.
- T2-19 ❌ (0.5) — All discovery dates correct; missing trial date requested for February 2026.

**T3 wrong (2.25):**
- T3-02 ❌ (0.75) — 7/8 events: still missing EEOC charge filed Feb 6 and Imani Foster legal
  research (Jan 21). All other milestones including Castellan retention and complaint filing now
  present.
- T3-04 ❌ (0.5) — Equity math correct; still omits "no pending bankruptcy" and "two prior
  lawsuits both settled" from Case Note 007.
- T3-05 ❌ (0.5) — Renata Osei correct; Priscilla Thorne (Marchand) missing across all runs.
  Fabricated Voss work details ("organize evidence, create timeline") not in source notes.
- T3-10 ❌ (0.0) — Persistent inference trap hallucination. With 142 nodes the model now has
  MORE zoning context and builds a more elaborate fabricated legal theory. Inference traps are
  harder as node count grows — more plausible context = more confident hallucination.

**Key improvements vs Run 5 (generic, 130 nodes):**
- T3-01 ✅ FIXED — Leo Nakashima correctly identified as attorney on both defense cases.
  Was wrong in Runs 5 and 6. 142 nodes gave retrieval system enough coverage to link both cases.
- T3-06 ✅ IMPROVED to 1.0 — Now hits all 5 complexity factors: competing experts,
  three-plaintiff structure, FDA safety communication, motion to compel safety minutes,
  bellwether structure. First full mark on this question.
- T3-08 ✅ FIXED — $300,000 settlement floor recovered (was lost in Run 6's 70-node graph).
- T2-17 ✅ FIXED — Motion to compel + privilege grounds fully answered.
- T2-11 ✅ IMPROVED — Now includes 25% verdict risk figure.

**Architecture finding confirmed:**

| | Run 5 (generic) | Run 6 (identity, old) | Run 7 (identity, fixed) |
|--|--|--|--|
| Nodes | 130 | 70 | **142** |
| Overall | 92% | 90% | **93.5%** |
| Cost | $0.1129 | $0.0883 | **$0.1004** |
| p50 | 8,554ms | 6,600ms | **7,558ms** |

Identity is now net positive for the first time: +1.5pp over generic, −11% cost, −12% latency.
Removing extraction focus from the identity prompt restored node counts while keeping identity's
importance-scoring and seed-topic benefits active at retrieval time.

**Remaining issue — T3-10 inference trap (persistent):**
The T3-10 hallucination has appeared in Runs 6 and 7 but not Run 5 (generic). With more nodes
comes more zoning-investigation context, which causes the model to construct a legal argument
rather than recognising the question is unanswerable. This is a Haiku reasoning pattern, not a
retrieval failure. May require an explicit "do not speculate on legal strategies not in the
records" instruction in ANSWER_SYSTEM, or a dedicated inference-trap detector.

---

### Run 8 — 2026-05-11 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · dual-mode (identity vs baseline)

| Metric | Identity (Paralegal) | Baseline (no identity) |
|--------|----------------------|------------------------|
| Mode | fresh ingest | fresh ingest |
| T1 accuracy | 20/20 = **100%** | 20/20 = **100%** |
| T2 accuracy | 18/20 = **90%** | 18/20 = **90%** |
| T3 accuracy | 7.0/10 = **70%** | 5.75/10 = **57.5%** |
| Overall | **45/50 = 90%** | **43.75/50 = 87.5%** |
| False positives | **3/9** (U-02, U-04, U-09) | **3/9** (U-02, U-04, U-09) |
| Trap (L-U-06) | **PASS** | **PASS** |
| Nodes stored | 138 | 142 |
| Recall p50 | 8,149ms | 7,938ms |
| Recall p99 | 14,036ms | 13,553ms |
| Tokens in (recall) | 402,728 | 403,521 |
| Est. cost | $0.1063 | $0.1062 |

**Identity ingest: in=24,235 out=13,193**
**Baseline ingest: in=23,295 out=11,604**

**T2 wrong (identity 2.0, baseline 2.0):**
- L-T2-11 (identity 0.5 / baseline 1.0) — Identity missing 25% verdict risk figure from Nakashima's
  recommendation. Baseline surfaces it correctly.
- L-T2-13 (identity 1.0 / baseline 0.5) — Baseline hedges: "mentioned as comparable verdict
  case, not identified as Illinois precedent." Wrong — Foster explicitly cited Solis v. Meridian
  Partners (N.D. Ill. 2022). Identity correctly identifies the citation.
- L-T2-17 (0.0 both) — Retrieval miss. Torres, Reid & Hatch motion to compel safety committee
  minutes (Case Note 019) not retrieved by either DB.
- L-T2-19 (0.5 both) — Both retrieve all discovery dates; both miss trial date February 2026.

**T3 wrong (identity 3.0, baseline 4.25):**
- L-T3-01 (0.0 both) — Persistent cross-run failure. Both identify Derek Briggs as the attorney
  on both defense cases. Correct answer: Leo Nakashima (lead Marchand, associate Brightfield).
  Identity hedges ("though Leo Nakashima is listed as lead on Marchand") but still leads with
  wrong attribution. Reasoning failure, not retrieval.
- L-T3-02 (identity 0.75 / baseline 0.5) — Identity captures 7/8 events including Castellan
  retention. Both miss Jan 21 Imani Foster legal research and EEOC charge. Baseline also
  misses Jan 16 retention date.
- L-T3-04 (0.5 both) — Both retrieve equity math ($3.3M). Both miss "no pending bankruptcy"
  and "two prior lawsuits both settled" — the collectability evidence from Case Note 007.
- L-T3-05 (0.75 both) — Renata Osei (NorthBridge + Voss) correctly identified by both.
  Both miss Priscilla Thorne (secretary listed on Marchand case).
- L-T3-06 (identity 0.75 / baseline 0.5) — Identity hits FDA safety communication and
  competing expert conflict; misses motion to compel minutes and bellwether structure explicitly.
  Baseline misses FDA communication, motion to compel, and bellwether.
- L-T3-08 (0.5 both) — Both retrieve $560k demand breakdown and 12-week unemployment update.
  Both miss client's $300,000 minimum settlement floor from Case Note 020.
- L-T3-09 (identity 0.75 / baseline 1.0) — Baseline correctly states Devereux "will be the
  bellwether trial." Identity hedges: "will likely serve as the bellwether trial." Baseline wins
  this one.
- **L-T3-10 (identity 1.0 / baseline 0.0) — THE KEY DIFFERENCE.** Identity correctly declines
  the inference trap: "there is no information about conditions that would allow Castellan &
  Briggs to add a claim against the City of Chicago zoning office." Baseline constructs a
  hallucinated legal framework: "would need to establish that the City of Chicago zoning
  office's actions caused NorthBridge's damages" — fabricating a legal theory absent from
  the records. Identity's importance scoring suppressed the speculative zoning reasoning that
  misled the baseline.

**U-tier false positives (3/9 both modes — persistent pattern):**
- U-02: Both enumerate 4 named attorneys as definitive total headcount. Same pattern as Veloris
  runs (U-03 there).
- U-04: Both describe the current active Brightfield case when asked about *prior* suits.
  Question asks about history; both answer about present.
- U-09: Both restate the $1.3M discounted sale loss when asked what happened *after* the sale.
  Post-sale fate is genuinely not in the records.

**Identity vs Baseline summary:**
Identity beats baseline by 2.5pp (90% vs 87.5%). The entire gap is T3: identity 7.0/10 vs
baseline 5.75/10 (+1.25 points). T3-10 (inference trap) drives +1.0 of that gap — identity
correctly declined to speculate while baseline built a convincing but fabricated legal theory.
T3-02 and T3-06 add +0.5, partially offset by T3-09 where baseline outperformed (-0.25).
T2 is exactly tied (18/20 both). Node delta is negligible (3%): extraction fix confirmed
holding across datasets.

**Cross-dataset identity confirmation (Run 8 + Vectra Run 1 + Run 7):**
Legal dual-mode: +2.5pp (90% vs 87.5%)
Vectra: +2pp (92% vs 90%)
Legal identity-only (vs generic Run 5): +1.5pp (93.5% vs 92%)
Identity is net positive on all three datasets after the extraction focus fix.

---
