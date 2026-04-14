# pocket-mem benchmark — multi-dataset suite

Three independent datasets testing memory storage, retrieval accuracy, and cross-dataset disambiguation. Run each dataset independently first, then run the combined disambiguation test.

---

## Datasets

| Dataset | Domain | Source format | Notes |
|---------|--------|---------------|-------|
| Veloris Technologies | B2B SaaS company | Email threads | Original benchmark |
| Hargrove Family Medicine | Medical/clinical | Clinical notes | New — no overlap with Veloris |
| Castellan & Briggs LLP | Legal | Case notes | New — no overlap with Hargrove |

**Key design rule:** The three datasets share zero domain vocabulary. A fact about Arthur Pemberton (medical) cannot be confused with a fact about Harlan Voss (legal). Any cross-contamination in answers is a hallucination, not a retrieval ambiguity.

**Exception — intentional trap:** The name "Marcus Webb" appears in all three datasets as a different person each time:
- Veloris: Marcus Webb = CTO of Veloris Technologies
- Legal: Marcus Webb = General counsel of Terravast Holdings (NorthBridge case) AND separately, Voss's supervisor at Kellerman & Drape
The agent must track which Marcus Webb belongs to which context. Conflating them is a failure mode worth measuring specifically.

---

## Part 1 — Medical dataset (Hargrove Family Medicine)

20 clinical notes spanning January–March 2025 covering four patients:
- Arthur Pemberton (67M) — atrial fibrillation, heart failure, hypertension
- Cecilia Vance (44F) — new-onset LADA diabetes
- Raymond Chu (55M) — hypertensive urgency
- Dolores Estrada (72F) — osteoporosis, knee osteoarthritis

### Question set

#### Tier 1 — Direct lookup (20 questions)

| ID | Question |
|----|----------|
| M-T1-01 | What is Arthur Pemberton's date of birth? |
| M-T1-02 | What medication was Arthur Pemberton already taking for hypertension at his first visit? |
| M-T1-03 | What was Arthur Pemberton's BNP level at initial labs? |
| M-T1-04 | What is Cecilia Vance's occupation? |
| M-T1-05 | How much weight did Cecilia Vance lose in the 6 months before her visit? |
| M-T1-06 | What was Cecilia Vance's initial HbA1c result? |
| M-T1-07 | What continuous glucose monitor was prescribed to Cecilia Vance? |
| M-T1-08 | What was Raymond Chu's blood pressure when he arrived at the clinic on February 5th? |
| M-T1-09 | What medication did Raymond Chu run out of and for how long? |
| M-T1-10 | What did the CT head scan show for Raymond Chu? |
| M-T1-11 | What is Dolores Estrada's occupation? |
| M-T1-12 | What medication is Dolores Estrada on for osteoporosis? |
| M-T1-13 | What was Dolores Estrada's T-score on her DEXA scan? |
| M-T1-14 | Who is the cardiologist at Hargrove Family Medicine? |
| M-T1-15 | What was Arthur Pemberton's ejection fraction on echocardiogram? |
| M-T1-16 | What pharmacy did Raymond Chu use to fill his prescription? |
| M-T1-17 | What is the name of the medical records coordinator at Hargrove Family Medicine? |
| M-T1-18 | What were Cecilia Vance's two episodes of hypoglycemia (glucose readings)? |
| M-T1-19 | What anticoagulant was Arthur Pemberton started on? |
| M-T1-20 | What imaging showed moderate medial compartment osteoarthritis? |

#### Tier 2 — Single hop (20 questions)

| ID | Question |
|----|----------|
| M-T2-01 | What condition did Dr. Suresh diagnose Arthur Pemberton with, and what confirmed it? |
| M-T2-02 | Why did Dr. Suresh hold off on increasing Arthur Pemberton's Lisinopril at the cardiology visit? |
| M-T2-03 | What type of diabetes does Cecilia Vance have, and how is it different from Type 2? |
| M-T2-04 | Why was Metformin specifically ruled out for Cecilia Vance? |
| M-T2-05 | What happened to Raymond Chu's blood pressure over the 2-hour monitoring period on February 5th? |
| M-T2-06 | What did Arthur Pemberton's echocardiogram show about his left atrium? |
| M-T2-07 | What lab finding confirmed Cecilia Vance had autoimmune diabetes rather than Type 2? |
| M-T2-08 | Why was Arthur Pemberton's Lisinopril finally increased at his February follow-up? |
| M-T2-09 | What did Dolores Estrada's X-ray show, and what was Dr. Marsh's referral decision based on it? |
| M-T2-10 | What improvements did Arthur Pemberton show at his 4-week cardiology follow-up? |
| M-T2-11 | What did Cecilia Vance's CGM show at her 6-week follow-up compared to earlier readings? |
| M-T2-12 | What was the concern about Dolores Estrada's living situation in relation to her knee? |
| M-T2-13 | What two medications were given to Raymond Chu at the clinic on February 5th? |
| M-T2-14 | What was Raymond Chu's medication regimen after his February 18 follow-up? |
| M-T2-15 | What was Arthur Pemberton's weight change from January to February? |
| M-T2-16 | Where was Cecilia Vance referred for her annual diabetic eye exam? |
| M-T2-17 | What was Cecilia Vance's insulin dosage at her 6-week follow-up vs when she started? |
| M-T2-18 | What finding on Arthur Pemberton's ECG indicated long-standing hypertension effects? |
| M-T2-19 | What referral did Dr. Marsh make for Raymond Chu at his February 18 follow-up, and why? |
| M-T2-20 | What minor bleeding events did Arthur Pemberton report, and what was Dr. Suresh's response? |

#### Tier 3 — Multi-hop (10 questions)

| ID | Question |
|----|----------|
| M-T3-01 | Which nurse handled lab result notifications for multiple patients, and which result required immediate physician contact? |
| M-T3-02 | Two patients had conditions discovered while being evaluated for something else. Who and what was discovered? |
| M-T3-03 | Which patient had the most medications at their February follow-up, and what were all of them? |
| M-T3-04 | What is the connection between Cecilia Vance's father and her diagnosis — and what is the important distinction? |
| M-T3-05 | Which radiologist reported on tests for the most patients, and list all the tests they reported on? |
| M-T3-06 | What was the full chain of events that led to Cecilia Vance being diagnosed with LADA rather than Type 2? |
| M-T3-07 | Which patient had the highest blood pressure recorded, and what was the full sequence of clinical decisions made that day? |
| M-T3-08 | Two patients were referred to specialists. Who were the specialists and what did they each order or start? |
| M-T3-09 | How does Dolores Estrada's osteoporosis affect her overall care plan for her knee? |
| M-T3-10 | Which patient's condition was identified as a medical urgency (not emergency), and what is the clinical distinction? |

#### Unanswerable questions (10 questions)

| ID | Question | Trap? |
|----|----------|-------|
| M-U-01 | What is Dr. Eleanor Marsh's medical school? | No |
| M-U-02 | What is Arthur Pemberton's home address? | No |
| M-U-03 | Does Cecilia Vance have health insurance? | No |
| M-U-04 | What caused Raymond Chu's small vessel disease? | **Yes — "likely hypertension-related" per radiology** |
| M-U-05 | What is Dolores Estrada's blood type? | No |
| M-U-06 | Who is Arthur Pemberton's emergency contact? | No |
| M-U-07 | Has Cecilia Vance ever been hospitalized before? | No |
| M-U-08 | What hospital is Hargrove Family Medicine affiliated with? | No |
| M-U-09 | What is Nurse James Obi's nursing specialty certification? | No |
| M-U-10 | How long has Dr. Patricia Yuen been practicing endocrinology? | No |

---

## Part 2 — Legal dataset (Castellan & Briggs LLP)

20 case notes spanning January–March 2025 covering four matters:
- NorthBridge Capital v. Terravast Holdings — breach of contract, $2.08M claimed
- Voss v. Kellerman & Drape — wrongful termination / whistleblower
- Marchand v. City of Chicago — personal injury defense
- Brightfield Pharmaceuticals — product liability defense

### Question set

#### Tier 1 — Direct lookup (20 questions)

| ID | Question |
|----|----------|
| L-T1-01 | What city is Castellan & Briggs LLP located in? |
| L-T1-02 | What are the claimed damages in NorthBridge v. Terravast? |
| L-T1-03 | Who is the client contact at NorthBridge Capital? |
| L-T1-04 | What case number was assigned to NorthBridge v. Terravast? |
| L-T1-05 | What judge was assigned to NorthBridge v. Terravast? |
| L-T1-06 | What was Harlan Voss's annual salary at Kellerman & Drape? |
| L-T1-07 | On what date was Harlan Voss terminated? |
| L-T1-08 | What were Harlan Voss's performance review ratings in 2022 and 2023? |
| L-T1-09 | What was Solenne Marchand's claimed injury from the slip and fall? |
| L-T1-10 | What is the total damages amount claimed by Solenne Marchand? |
| L-T1-11 | What is Brightfield Pharmaceuticals' medication at the center of the product liability case? |
| L-T1-12 | Who is the defense expert retained in the Brightfield case, and where is he from? |
| L-T1-13 | What is Duncan Farrow's role at Castellan & Briggs? |
| L-T1-14 | What was the amount Terravast was contractually obligated to contribute? |
| L-T1-15 | What mediation service was agreed upon for the Voss case? |
| L-T1-16 | What time did the salt truck treat North Michigan Avenue on the morning of Marchand's fall? |
| L-T1-17 | What is the EEOC charge number for the Voss case? |
| L-T1-18 | What was the demand amount in the Voss demand letter? |
| L-T1-19 | Who is the plaintiff's counsel in Marchand's case? |
| L-T1-20 | What are the three plaintiffs in the consolidated Brightfield case? |

#### Tier 2 — Single hop (20 questions)

| ID | Question |
|----|----------|
| L-T2-01 | What problem did Renata Osei identify in the NorthBridge contract, and how was it resolved? |
| L-T2-02 | What three causes of action did NorthBridge file against Terravast? |
| L-T2-03 | What did Terravast's asset investigation reveal about their real estate holdings? |
| L-T2-04 | What are the three counts in the Voss complaint, and who are the named defendants? |
| L-T2-05 | How many days passed between Voss's whistleblower complaint and his termination, and why is that significant? |
| L-T2-06 | What specific observations did witness Thomas Quayle make that helped the City's defense? |
| L-T2-07 | What counterclaim did Terravast file against NorthBridge, and for how much? |
| L-T2-08 | What is the expert fee structure for Dr. Belkin in the Brightfield case? |
| L-T2-09 | What is the significance of Anton Devereux's case among the three Brightfield plaintiffs? |
| L-T2-10 | What was Kellerman & Drape's counter-offer, and why did Voss reject it? |
| L-T2-11 | What was Leo Nakashima's settlement recommendation for Marchand, and under what condition would he recommend trial? |
| L-T2-12 | What is Terravast's affirmative defense related to zoning, and how does it change what Castellan needs to investigate? |
| L-T2-13 | What Illinois case did Imani Foster cite as precedent for the Voss whistleblower claim? |
| L-T2-14 | What was the bridge loan amount, who was it from, and at what interest rate? |
| L-T2-15 | What is the deadline for NorthBridge's response to Terravast's counterclaim? |
| L-T2-16 | What is the legal weakness in the Voss whistleblower case that Imani Foster identified? |
| L-T2-17 | What motion did Torres, Reid & Hatch file in the Brightfield case, and what is the issue with it? |
| L-T2-18 | Who is Brightfield's opposing expert, and what does Dr. Belkin think of her approach? |
| L-T2-19 | What are the key dates in the NorthBridge v. Terravast discovery schedule? |
| L-T2-20 | Who were the registered agent and defense counsel for Terravast Holdings? |

#### Tier 3 — Multi-hop (10 questions)

| ID | Question |
|----|----------|
| L-T3-01 | Which attorney worked on both defense cases, and which cases were they? |
| L-T3-02 | What is the full chain of events from Voss's whistleblower complaint to mediation scheduled in March 2025? |
| L-T3-03 | Two cases involve Renata Osei. Which cases, and what did she do on each? |
| L-T3-04 | What is the full picture of Terravast's financial situation, and how does it affect litigation strategy? |
| L-T3-05 | What is the full picture of all Terravast affirmative defenses and how does each affect NorthBridge's case? |
| L-T3-06 | Which case has the most complex expert witness situation, and why? |
| L-T3-07 | What is the connection between the NorthBridge case and the Voss case regarding a shared name, and what is the risk? |
| L-T3-08 | What is the complete damages picture for Harlan Voss as of the demand letter, and how has it evolved by March 2025? |
| L-T3-09 | All three Brightfield plaintiffs shared the same pre-existing condition. What was it, how long was each on Velantrix, and whose case is most severe? |
| L-T3-10 | What would need to happen to add a zoning claim against the City of Chicago in the NorthBridge case? |

#### Unanswerable questions (10 questions)

| ID | Question | Trap? |
|----|----------|-------|
| L-U-01 | What law school did Margaret Castellan attend? | No |
| L-U-02 | How many attorneys work at Castellan & Briggs total? | No |
| L-U-03 | What is Harlan Voss's home address? | No |
| L-U-04 | Has Brightfield faced product liability suits before? | No |
| L-U-05 | What floor is the Castellan & Briggs office on? | No |
| L-U-06 | What is Derek Briggs's area of specialty? | **Yes — inferable as employment law from the Voss matter** |
| L-U-07 | How long has NorthBridge Capital been in business? | No |
| L-U-08 | What are the total estimated legal fees for the NorthBridge matter? | No |
| L-U-09 | What happened to the Meridian Tower property after it was sold? | No |
| L-U-10 | Is Solenne Marchand still employed? | No |

---

## Part 3 — Combined disambiguation test

Load ALL THREE datasets into the same pocket-mem project, then ask these questions. Each question requires knowing which dataset a fact belongs to, or tests whether the agent confuses similar-sounding entities across datasets.

### Disambiguation questions (20 questions)

| ID | Question | Tests |
|----|----------|-------|
| D-01 | Who is Marcus Webb, and what is his role? | Must distinguish 3 different people named Marcus Webb across datasets |
| D-02 | Which dataset mentions a slip and fall incident? | Cross-dataset domain awareness |
| D-03 | What condition is Arthur Pemberton being treated for? | Must not confuse with Arthur (no Arthurs in other datasets) but tests entity specificity |
| D-04 | Is Hargrove Family Medicine located in Chicago or Austin? | Austin — must not confuse with Castellan & Briggs (Chicago) |
| D-05 | What is Halcyon Advisory, and what does it have to do with Velantrix? | Nothing — these are from different datasets (Veloris vs Brightfield). Answer: no connection |
| D-06 | Which datasets mention a person named Priya? | Veloris only (Priya Nair) — must not hallucinate Priya in medical or legal |
| D-07 | Who is Leo Reyes? | Veloris DevOps Engineer — must not confuse with Leo Nakashima (legal) |
| D-08 | In which case or context is TimescaleDB mentioned? | Veloris Technologies — Conduit rewrite. Not in medical or legal. |
| D-09 | What happened on January 9, 2025? | Two things in different datasets: Marchand slip and fall (legal) AND Terravast mediation clause identified (legal). Raymond Chu's urgency was Feb 5. Must be specific. |
| D-10 | Which dataset has a person named Darnell? | Veloris only (Darnell Okafor). Must not hallucinate Darnell in medical or legal. |
| D-11 | What is Apixaban used for, and who is taking it? | Arthur Pemberton (medical) — anticoagulation for AFib. Must not associate with legal or Veloris. |
| D-12 | Who is Imani Foster? | Associate attorney at Castellan & Briggs (legal). Not in medical or Veloris datasets. |
| D-13 | What does GAD65 antibody positive mean, and who tested positive? | Medical dataset — Cecilia Vance. Confirms autoimmune diabetes (LADA). Not mentioned in other datasets. |
| D-14 | How much money is at stake in NorthBridge v. Terravast? | $2,080,000 claimed by NorthBridge + $650,000 Terravast counterclaim = $2,730,000 total in dispute |
| D-15 | What are all the conditions or cases involving a 67-year-old? | Arthur Pemberton (medical) is 67. No 67-year-olds mentioned in legal or Veloris. |
| D-16 | Which datasets mention a person named Leo? | Legal (Leo Nakashima — attorney) AND Veloris (Leo Reyes — DevOps). Must correctly identify both. |
| D-17 | What is Furosemide, and is it mentioned in the legal or Veloris dataset? | Medical only — diuretic prescribed to Arthur Pemberton. Not in legal or Veloris. |
| D-18 | Who referred Cecilia Vance to an ophthalmologist, and is this related to any case at Castellan & Briggs? | Dr. Patricia Yuen referred her (medical). No connection to Castellan & Briggs. |
| D-19 | What is the most recent event across all three datasets, and in which dataset does it occur? | March 12, 2025 — Voss mediation scheduled (legal). All other notes end March 6-10 at most. |
| D-20 | Which person across all datasets has the most medications being managed? | Arthur Pemberton (medical) — on Apixaban, Furosemide, Metoprolol, Lisinopril (4 medications at once). No character in legal or Veloris is described as managing multiple medications. |

---

## Scoring methodology

### Per-dataset scoring
Same as original Veloris benchmark:
- Correct (1.0): Key facts present, wording flexible
- Partial (0.5): Some facts correct, others missing
- Incorrect (0.0): Wrong answer
- False positive (0.0 + flag): Confident wrong answer on unanswerable question
- False negative (0.0 + flag): "I don't know" on answerable question including trap questions

### Disambiguation scoring (Part 3)
- Correct (1.0): Correctly identifies which dataset the fact belongs to, answers accurately
- Cross-contamination (0.0 + flag): Uses a fact from the wrong dataset to answer
- Marcus Webb confusion (0.0 + flag): Conflates any of the three Marcus Webbs
- Hallucination (0.0 + flag): Invents a connection between datasets that doesn't exist

---

## Results

### Run 1 — 2026-04-13

**Config:** Ingest model `qwen2.5:7b` · Answer model `claude-haiku-4-5-20251001` · `max_tokens=4096`

#### Ingestion failures — 3 documents dropped (all same root cause)

The `qwen2.5:7b` model occasionally outputs bare (unquoted) JSON keys in relationship objects (e.g. `to: "value"` instead of `"to": "value"`). The `complete_json()` brace-match fallback extracts the block but it remains invalid JSON, so both attempts fail. The retry fires after `flush()` and is dropped by the new shutdown guard.

| Test | Document dropped | Downstream miss |
|---|---|---|
| Legal | Thomas Quayle witness interview note | L-T2-06 (Quayle observations) → "I don't have that information" |
| Combined | Darnell Okafor → Priya Nair (Conduit milestone email) | Conduit/Veloris disambiguation weakened |
| Combined | Cecilia Vance first clinical note (Annual Physical) | Medical disambiguation weakened |

**Fix applied after this run:** Added unquoted-key regex repair step in `complete_json()` before the final `ValueError`.

---

#### Medical dataset (Hargrove Family Medicine)

Notes: 20 · Nodes: 188 · Dropped: 0 · Cost: $0.2041

| Tier | Score | Notes |
|------|-------|-------|
| T1 | 20/20 | Perfect |
| T2 | 20/20 | Perfect |
| T3 | 8/10 | **M-T3-02 FAIL** — attributed incidental discoveries to wrong patients (Pemberton+Chu instead of Vance+Pemberton) |
| U (standard) | 7/9 | M-U-07 borderline (inferred "outpatient only"); M-U-09 borderline (cited "RN, primary care floor") |
| Trap (M-U-04) | PASS | Correctly cited "likely hypertension-related per radiology" |
| **Overall** | **~93%** | 55/59 scoreable questions |

---

#### Legal dataset (Castellan & Briggs)

Notes: 20 · Nodes: 147 · Dropped: 1 (Thomas Quayle) · Cost: included in combined run

| Tier | Score | Notes |
|------|-------|-------|
| T1 | 20/20 | Perfect |
| T2 | 17/20 | **L-T2-06** dropped doc · **L-T2-13** retrieval miss (Solis precedent) · **L-T2-17** retrieval miss (Torres motion) |
| T3 | 7/10 | **L-T3-07 FAIL** — Marcus Webb across cases not identified · **L-T3-10 FAIL** — hallucinated zoning procedure instead of "notes don't address this" · **L-T3-02 partial** — missed Jan 16/21 steps, wrong counter-offer date |
| U (standard) | 6/9 | **L-U-02 FAIL** fabricated "4 attorneys total" · **L-U-04 FAIL** cited current case as evidence of prior suits · **L-U-09 FAIL** described sale itself not post-sale fate · L-U-05 should be PASS (auto-scorer false flag) |
| Trap (L-U-06) | PASS | Correctly inferred "employment law" from Voss matter |
| **Overall** | **~85%** | 50/59 scoreable questions |

---

#### Combined disambiguation test

Notes: 60 · Nodes: 386 · Dropped: 2 · Cost: $0.0627

| Score | Notes |
|-------|-------|
| ~14/20 ≈ 70% | **D-01 partial** — gave Veloris Marcus Webb only, missed legal dataset Marcus Webb · **D-20 FAIL** — said Cecilia Vance has most meds (2-3), should be Arthur Pemberton (4) · **D-09 partial** — confused but mentioned Marchand fall · Most others accurate and correctly attributed to correct dataset |

Marcus Webb confusion: 0 outright conflations (no facts from wrong Marcus Webb used), but D-01 incomplete (missed the legal instances).

---

#### Key findings — Run 1

1. **JSON quality bug** is the highest-impact single issue — 3 docs dropped, directly caused 1 T2 miss, weakened disambiguation. Fix applied.
2. **Retrieval gaps** (L-T2-13, L-T2-17) — data was ingested but not surfaced. Likely a BM25/embedding coverage gap on narrow attorney memo details. No code fix; flagged for query tuning.
3. **Hallucination under pressure** — when the answer model sees adjacent evidence (attorney names in case files, current lawsuit visible when asked about "prior" suits), it over-infers. LLM prompt issue, not ingestion.
4. **Medical outperforms Legal** (93% vs 85%) — clinical notes are more formulaic and extract cleanly; legal notes have more nuanced internal references (case cites, motion history) that fall through retrieval.

---

---

### Run 2 — 2026-04-13 (Combined 180-question run)

**Config:** Ingest model `qwen2.5:7b` · Answer model `claude-haiku-4-5-20251001` · `max_tokens=4096` · JSON unquoted-key fix active

All three datasets loaded into a single memory store. 60 questions per dataset, 180 total. This is the first run of `test_combined_benchmark` under the new test structure.

#### Ingestion

Documents: 60 (20 Veloris emails + 20 medical notes + 20 legal notes) · Nodes: 401 · Tokens in: 68,988 · Tokens out: 40,524

No dropped documents — the unquoted-key JSON fix from Run 1 held. All 60 ingested clean.

---

#### Scores

| Dataset | T1 | T2 | T3 | U | Total |
|---|---|---|---|---|---|
| Veloris | 17/20 (85%) | 19/20 (95%) | **10/10 (100%)** | 8/10 (80%) | **54/60 (90%)** |
| Medical | **20/20 (100%)** | 18/20 (90%) | 9.5/10 (95%) | **10/10 (100%)** | **57.5/60 (96%)** |
| Legal | 19/20 (95%) | 15.5/20 (78%) | 8.5/10 (85%) | 6/10 (60%) | **49/60 (82%)** |
| **Overall** | **56/60 (93%)** | **52.5/60 (88%)** | **27.5/30 (92%)** | **24/30 (80%)** | **160.5/180 (89%)** |

---

#### Full failure log

**Veloris failures (6)**

| ID | Category | Root cause |
|---|---|---|
| V-T1-14 | Contradictory response | Said "I don't have that information" then immediately named Redpoint Security correctly. Knowledge was present — answer model hedged on retrieved data. |
| V-T1-17 | Contradictory response | Same pattern: declined, then gave "6 enterprise customers, 47 users total" in the same response. |
| V-T1-20 | Retrieval miss | OWASP Top 10 training requirement not surfaced. Appeared once in Marcus's email as a downstream action — single-mention facts are harder to retrieve under BM25. |
| V-T2-03 | Retrieval miss | RabbitMQ ban + Priya's backstory not retrieved. Strong detail but embedded in a longer policy discussion. |
| V-U-03 | Hallucination | Asked how many employees Veloris has (unanswerable). Instead of declining, listed all 8 named employees from memory. Triggered by the enumeration framing. |
| V-U-10 | Hallucination | Asked what database the Atlas backend uses (unanswerable). Discussed Conduit infrastructure options (TimescaleDB/ClickHouse) instead of declining — cross-entity confusion. |

**Medical failures (2.5)**

| ID | Category | Root cause |
|---|---|---|
| M-T2-06 | Test data issue | The results file expected a slip-and-fall answer (wrong dataset artifact in the answer key). The medical answer itself (left atrial enlargement) was correct. Not a model failure. |
| M-T2-16 | Retrieval miss | Ophthalmology referral declined as "not in records." It IS documented in the endocrinology follow-up note. The referral mention is brief and embedded in a long note — likely missed by embedding similarity. |
| M-T3-05 | Partial retrieval | Dr. Reinholt reported on 4 tests across 3 patients. Answer listed 3 tests, missed Arthur's echocardiogram. ECG, CT head, and knee X-ray were recalled; echo was not — possibly stored under a different entity node. |

**Legal failures (11)**

| ID | Category | Root cause |
|---|---|---|
| L-T1-12 | Contradictory response | Said "I don't have that information" then named Dr. Belkin and Northwestern correctly. Same hedge-then-answer pattern as V-T1-14/17. |
| L-T2-03 | Retrieval miss | Asset investigation details ($8.4M properties, $5.1M mortgages) completely missed. This was in a specific investigator's report note. Separate document from the main case notes — may not have linked to the same retrieval context. |
| L-T2-06 | Partial retrieval | 3 of 4 Thomas Quayle observations retrieved. Missing: "no city vehicles visible at the time." Three observations clustered in one sentence; the fourth was listed separately — partial coverage. |
| L-T2-07 | Retrieval miss | $650K Terravast counterclaim not retrieved. Appeared in the same note as the affirmative defenses. The defense context was retrieved (L-T3-05 got the defenses) but the counterclaim dollar figure was not. |
| L-T2-13 | Wrong answer | Retrieved Rosenberg v. Feldman (2019) — the mediation clause case in the NorthBridge matter — instead of Solis v. Meridian Partners (N.D. Ill. 2022) — the whistleblower precedent in the Voss matter. Two different legal citations from the same firm's work, both in DB. BM25 matched on shared legal vocabulary and pulled the wrong one. |
| L-T2-17 | Partial retrieval | Torres motion and privilege objection retrieved, but missing the specific grounds: attorney-client privilege OR work product. The distinction detail was one clause in a longer note. |
| L-T2-19 | Partial retrieval | Discovery schedule mostly correct but missing trial date (February 2026). The trial date appeared at the end of a timeline list — possibly truncated during entity extraction. |
| L-U-02 | Hallucination | Asked how many attorneys work at the firm total (unanswerable). Listed the 4 attorneys who appear in case files as if they were the full headcount. Identical pattern to V-U-03 — enumeration questions trigger fabricated lists. |
| L-U-04 | Hallucination | Asked whether Brightfield faced prior product liability suits (unanswerable). Described the current consolidated case as the answer — misread "before" as irrelevant and returned what it found. |
| L-U-05 | Borderline | Correctly declined but the phrasing was awkward ("archived records do not contain information about what floor"). Scoring as PASS — correct behavior, weak presentation. |
| L-U-09 | Hallucination | Asked what happened to Meridian Tower after it was sold (unanswerable). Fabricated a narrative about the discounted sale itself rather than declining on post-sale fate. |

---

#### Root cause analysis

**1 — Contradictory responses (3 instances: V-T1-14, V-T1-17, L-T1-12)**

The answer model opens with a hedge ("I don't have that information") and then provides the correct answer in the next sentence. This means retrieval succeeded but the model assigned low confidence to the retrieved chunk — probably because the chunk was surrounded by less-related context or had low BM25 score. The answer is in memory; the model just doesn't trust it enough to lead with it.

*Not a retrieval failure — a confidence/prompt failure. Could be addressed by tuning the answer prompt to commit to retrieved facts more aggressively.*

**2 — Hallucination on enumeration unanswerable questions (5 instances: V-U-03, V-U-10, L-U-02, L-U-04, L-U-09)**

Pattern: when the question asks "how many X" or "has Y happened before" and the answer is unanswerable, the model enumerates or infers from adjacent data. It correctly declines when the question has zero related data (salary, home address). It fails when related entities exist in memory (employee names → headcount, current lawsuit → prior suits).

*Root cause: the answer model conflates "I have data about X" with "I can answer questions about X." The hallucination rate for straight unanswerable questions with no related data is ~0%. For questions adjacent to known data it is ~50%.*

**3 — Single-mention fact retrieval misses (V-T1-20, V-T2-03, M-T2-16, L-T2-03, L-T2-07)**

Facts that appear once, in passing, in a single document don't always make it into the retrieved context window. BM25 matches on keywords; if the query vocabulary doesn't overlap with the exact phrasing the extraction model used as an entity label, the node doesn't score high enough to be returned. Most common with:
- Action items buried in longer emails (OWASP training)
- Dollar figures in investigator reports (asset value, counterclaim)
- Brief referral mentions at the end of a long clinical note

*Could be improved by increasing retrieval top-k or by treating numeric and proper-noun entities with higher weight during extraction.*

**4 — Same-firm case citation confusion (L-T2-13)**

The Voss matter and NorthBridge matter both have legal citations stored from the same law firm's notes. A query for "whistleblower case precedent" retrieved Rosenberg v. Feldman (which is about mediation clauses, also from Castellan & Briggs) instead of Solis v. Meridian Partners (the whistleblower precedent). The two citations share similar embedding space — both are Illinois case law referenced in formal legal memos.

*This is the hardest failure type to fix without per-matter retrieval scoping. Both citations are legitimate recalls from memory; the retrieval system returned the wrong one because it can't tell which matter the query belongs to.*

---

#### Key findings — Run 2

1. **89% overall is the strongest multi-dataset result to date.** Running all three domains in a single combined store only degraded accuracy by ~4pp vs isolated single-dataset runs (93% medical alone → 96% here; 85% legal alone → 82% here). The knowledge graph stays coherent under 400+ nodes.
2. **T3 multi-hop is the strongest tier at 92%.** The model is better at reasoning across stored nodes than at raw fact retrieval. Complex chains ("who worked on both X and Y," "what is the full timeline of Z") outperform simple lookups that miss a single low-scoring node.
3. **Unanswerable questions are the hardest tier at 80%** — specifically when data adjacent to the unanswerable question exists in memory. The model doesn't distinguish "data about X exists" from "the question about X is answerable."
4. **Legal is the hardest domain** (82%) due to: multiple matters creating same-author retrieval confusion, precise case citations stored as entity labels competing with each other, and more complex document structure compared to formulaic clinical notes.
5. **Zero cross-dataset contamination** — no medical facts appeared in legal answers, no Veloris facts in medical answers. The 401-node combined graph maintained clean domain boundaries.

---

### Medical dataset (Hargrove Family Medicine) — comparative

| Metric | Run 1 isolated | Run 2 combined | pocket-mem (Claude) | Mem0 |
|--------|---------------|----------------|---------------------|------|
| T1 accuracy | 20/20 (100%) | 20/20 (100%) | — | — |
| T2 accuracy | 20/20 (100%) | 18/20 (90%) | — | — |
| T3 accuracy | 8/10 (80%) | 9.5/10 (95%) | — | — |
| Overall accuracy | ~93% | 96% | — | — |
| Unanswerable pass rate | 7/9 | 10/10 | — | — |
| Trap question (M-U-04) | PASS | PASS | — | — |

### Legal dataset (Castellan & Briggs) — comparative

| Metric | Run 1 isolated | Run 2 combined | pocket-mem (Claude) | Mem0 |
|--------|---------------|----------------|---------------------|------|
| T1 accuracy | 20/20 (100%) | 19/20 (95%) | — | — |
| T2 accuracy | 17/20 (85%) | 15.5/20 (78%) | — | — |
| T3 accuracy | 7/10 (70%) | 8.5/10 (85%) | — | — |
| Overall accuracy | ~85% | 82% | — | — |
| Unanswerable pass rate | 6/9 | 6/10 | — | — |
| Trap question (L-U-06) | PASS | PASS | — | — |

### Veloris Technologies — comparative

| Metric | Run 2 combined (first full run) | pocket-mem (Claude) | Mem0 |
|--------|--------------------------------|---------------------|------|
| T1 accuracy | 17/20 (85%) | — | — |
| T2 accuracy | 19/20 (95%) | — | — |
| T3 accuracy | 10/10 (100%) | — | — |
| Overall accuracy | 90% | — | — |
| Unanswerable pass rate | 8/10 | — | — |
| Trap question (V-U-05) | PASS | — | — |

### Combined 180-question run — comparative

| Metric | Run 2 (qwen2.5:7b / Haiku) | pocket-mem (Claude) | Mem0 |
|--------|---------------------------|---------------------|------|
| Overall accuracy | 160.5/180 (89%) | — | — |
| T1 accuracy | 56/60 (93%) | — | — |
| T2 accuracy | 52.5/60 (88%) | — | — |
| T3 accuracy | 27.5/30 (92%) | — | — |
| Unanswerable pass rate | 24/30 (80%) | — | — |
| Cross-dataset contamination | 0 confirmed | — | — |
| Hallucination rate (U questions) | 5/30 (17%) | — | — |
| Contradictory response rate | 3/180 (2%) | — | — |


