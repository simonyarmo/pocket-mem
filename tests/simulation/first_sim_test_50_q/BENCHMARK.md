# pocket-mem benchmark — Veloris Technologies

A structured evaluation benchmark for testing AI agent memory systems. Uses a fictional company's email thread as the knowledge source, with 60 questions across four difficulty tiers and a defined scoring methodology.

The same dataset and question set is used to compare pocket-mem against competing memory systems under identical conditions.

---

## The test dataset

**Company:** Veloris Technologies — a fictional B2B SaaS company building project management software.

**Source material:** 20 emails spanning January–February 2025, covering four concurrent engineering initiatives:
- A data pipeline rewrite (Conduit)
- A monitoring system migration (Nightwatch)
- A mobile app build (Atlas mobile)
- A security audit and compliance incident

**People in the dataset:**

| Name | Role |
|------|------|
| Marcus Webb | CTO |
| Priya Nair | VP of Engineering |
| Darnell Okafor | Senior Backend Engineer |
| Suki Tanaka | Senior Frontend Engineer |
| Leo Reyes | DevOps Engineer |
| Camille Russo | Product Manager |
| Hana Bergström | QA Lead |
| Tobias Hunt | Junior Backend Engineer |

The dataset was deliberately designed so that:
- Facts are spread across multiple emails (no single email contains everything)
- Some facts are updated or superseded by later emails
- Multi-hop questions require connecting 3+ pieces of information from different emails
- 9 of 10 "unanswerable" questions are genuinely unanswerable — 1 is a trap that looks unanswerable but is actually in the data

---

## How the test is run

### Step 1 — Ingestion

Feed all 20 emails to the memory agent using `observe()`. Each email is treated as a single observation:

```python
from pocket_mem import MemoryAgent

agent = MemoryAgent(project="veloris-benchmark")

with open("test_emails.txt") as f:
    emails = parse_emails(f.read())  # split into individual emails

for email in emails:
    agent.observe(
        user_input=f"Email from {email['from']} to {email['to']}: {email['subject']}",
        agent_response=email['body']
    )

# Wait for all background ingestion to complete before querying
agent.flush()
```

Record: total ingestion time, total tokens used during ingestion.

### Step 2 — Restart

Shut down the agent completely and restart with the same project name. This confirms that memory persists across sessions and rules out any in-context recall.

```python
# New Python process — no shared state with ingestion
agent = MemoryAgent(project="veloris-benchmark")
```

### Step 3 — Query

Ask all 60 questions using `recall(mode="answer")`. Record the answer, latency, and token count for each question individually.

```python
import time, json

results = []
for q in questions:
    start = time.perf_counter()
    answer = agent.recall(q['question'], mode="answer")
    latency_ms = (time.perf_counter() - start) * 1000

    results.append({
        "id": q['id'],
        "question": q['question'],
        "answer": answer,
        "latency_ms": round(latency_ms),
    })
```

### Step 4 — Score

Compare each answer against the answer key. Scoring is done manually for nuance, but uses these rules:

- **Correct (1.0):** Answer contains the key facts. Exact wording not required.
- **Partial (0.5):** Answer contains some correct facts but misses others. Used for multi-part answers.
- **Incorrect (0.0):** Answer is wrong or confidently states something not in the data.
- **False positive (0.0 and flagged):** Agent gives a confident wrong answer to an unanswerable question.
- **False negative (0.0 and flagged):** Agent says "I don't know" to an answerable question (including the trap question U-05).

---

## Question set

### Tier 1 — Direct lookup (20 questions)
Single facts stated explicitly in one email. Tests basic storage and retrieval.

| ID | Question |
|----|----------|
| T1-01 | What is Marcus Webb's job title? |
| T1-02 | Who is Leo Reyes's manager? |
| T1-03 | What is the name of Veloris's main product? |
| T1-04 | How much was Veloris paying for Datadog per month? |
| T1-05 | What city is Veloris headquartered in? |
| T1-06 | What message queue technology did Darnell propose for the Conduit rewrite? |
| T1-07 | What analytics store did Priya originally suggest for Conduit? |
| T1-08 | What analytics store did Darnell propose instead, and why? |
| T1-09 | What is the test coverage percentage of the Atlas payments module? |
| T1-10 | How much money does Veloris process monthly through the payments module? |
| T1-11 | What build toolchain is Suki using for the React Native app? |
| T1-12 | What is the name of the internal monitoring project? |
| T1-13 | What security firm did Marcus choose for the audit? |
| T1-14 | What was the other security firm Marcus considered? |
| T1-15 | When does the security audit start? |
| T1-16 | What crash reporting tool does the Atlas mobile app use? |
| T1-17 | How many enterprise customers' users signed up for the Atlas mobile beta? |
| T1-18 | What is the actual monthly cost of the self-hosted Nightwatch stack? |
| T1-19 | Who is Tobias Hunt's direct manager? |
| T1-20 | What training did Marcus require all engineers to complete by end of February? |

### Tier 2 — Single hop (20 questions)
Answering requires connecting two pieces of information from the same or different emails.

| ID | Question |
|----|----------|
| T2-01 | Who did Marcus assign to lead the Conduit rewrite? |
| T2-02 | What specific part of the Conduit system is Tobias responsible for? |
| T2-03 | What technology did Priya explicitly ban from the Conduit rewrite, and why? |
| T2-04 | What was Darnell's preferred timeline for Conduit if the backfill feature was dropped? |
| T2-05 | Why did Priya insist on keeping the backfill feature? |
| T2-06 | What log aggregation tool is part of the new Nightwatch stack? |
| T2-07 | What is the projected annual saving from the Datadog migration? |
| T2-08 | What was Leo's fallback option if Loki had problems? |
| T2-09 | What phase of the Atlas mobile app includes offline sync? |
| T2-10 | Why is the Atlas backend off-limits for mobile development in Phase 1? |
| T2-11 | What compliance issue did Leo discover in the payment logging? |
| T2-12 | Who discovered the race condition in the payment refund logic? |
| T2-13 | Where are the Stripe test credentials stored? |
| T2-14 | What problem did Suki encounter with Expo related to Meridian Group's requirements? |
| T2-15 | What solution did Suki recommend for the Expo biometric bug? |
| T2-16 | What Conduit throughput did Darnell achieve in load testing, and how did it compare to the target? |
| T2-17 | What p99 latency did Conduit achieve in load testing? |
| T2-18 | Where did Leo document the LogQL learning guide? |
| T2-19 | What is the mobile beta launch date? |
| T2-20 | What known issue does Sentry's React Native SDK have? |

### Tier 3 — Multi-hop (10 questions)
Answering requires traversing 3+ connected pieces of information, often across multiple emails and people.

| ID | Question |
|----|----------|
| T3-01 | Which engineer was originally assigned to write tests for the payments module but was pulled away, and what were they pulled onto? |
| T3-02 | Who flagged the PCI compliance issue, and what three actions did Marcus take as a result? |
| T3-03 | What technology stack replaced Datadog, and which team member was involved in both the replacement and the Conduit rewrite? |
| T3-04 | The mobile beta includes a customer with special authentication requirements. What is that customer, what was their requirement, and how was it solved? |
| T3-05 | How many dashboards did the new monitoring system have after migration, and how does that compare to before? |
| T3-06 | Who reported a concern to Priya about test coverage, what module was the concern about, and who was ultimately assigned to fix it? |
| T3-07 | Which project involves both Suki Tanaka and Camille Russo, and what did Camille change about Suki's original scope? |
| T3-08 | Tobias Hunt worked on two separate things in Q1. What were they, and on which one did his manager say he deserved a performance note? |
| T3-09 | What was the actual monthly cost saving from the Nightwatch migration, and how did it compare to Leo's original estimate? |
| T3-10 | Which engineer discovered a bug while doing a different task, what was the task, and what was the bug? |

### Unanswerable questions (10 questions)
The correct response is "I don't know" or equivalent. One question (U-05) is a trap — it looks unanswerable but the answer IS in the data.

| ID | Question | Trap? |
|----|----------|-------|
| U-01 | What is Priya Nair's salary? | No |
| U-02 | What programming language is the Atlas backend written in? | No |
| U-03 | How many total employees does Veloris have? | No |
| U-04 | What did Marcus think about Python as a language? | No |
| U-05 | Which cloud provider does Veloris use? | **Yes — answer is AWS** |
| U-06 | What is Hana Bergström's salary? | No |
| U-07 | Has Veloris ever had a data breach? | No |
| U-08 | What is Tobias Hunt's favorite programming language? | No |
| U-09 | Does Veloris have offices outside Austin? | No |
| U-10 | What database does the Atlas backend use? | No |

---

## Metrics

### Primary metrics

| Metric | Description | How measured |
|--------|-------------|--------------|
| **T1 accuracy** | % of Tier 1 questions answered correctly | Manual score against answer key |
| **T2 accuracy** | % of Tier 2 questions answered correctly | Manual score against answer key |
| **T3 accuracy** | % of Tier 3 questions answered correctly | Manual score against answer key |
| **Overall accuracy** | Weighted average across T1+T2+T3 (50 questions) | `(T1 + T2 + T3) / 50` |
| **False positive rate** | % of unanswerable questions given a confident wrong answer | Count of wrong answers on U-01 to U-10 (excluding U-05) |
| **Trap question** | Did the agent correctly answer U-05 (AWS)? | Pass / Fail |
| **Session persistence** | Does memory survive a full restart? | Pass / Fail (tested by restarting between ingestion and query) |

### Cost metrics

| Metric | Description |
|--------|-------------|
| **Tokens — ingestion** | Total LLM tokens used to ingest all 20 emails |
| **Tokens — per recall** | Average tokens per `recall()` call |
| **Tokens — total** | Combined ingestion + all 60 recall calls |
| **Estimated cost (API)** | Total cost at Claude Haiku pricing ($0.25/M input, $1.25/M output) |

### Latency metrics

| Metric | Description |
|--------|-------------|
| **Observe p50 (ms)** | Median time to process one email observation |
| **Observe p99 (ms)** | 99th percentile observe latency |
| **Recall p50 (ms)** | Median time to answer one question |
| **Recall p99 (ms)** | 99th percentile recall latency |

---

## Scoring example

Here is an example of how a question would be scored to make the methodology concrete.

**Question T2-03:** "What technology did Priya explicitly ban from the Conduit rewrite, and why?"

**Ground truth answer:** RabbitMQ — she had a catastrophic queue loss with it at her previous company and has blacklisted it.

**Example agent response A:**
> "Priya banned RabbitMQ from the Conduit rewrite because she experienced a catastrophic queue loss with it at a previous company."

**Score: 1.0 — Correct.** Contains both facts (what was banned, why).

---

**Example agent response B:**
> "Priya said they should not use RabbitMQ."

**Score: 0.5 — Partial.** Correctly identifies RabbitMQ but omits the reason.

---

**Example agent response C:**
> "Priya banned Pulsar because it had too much operational complexity."

**Score: 0.0 — Incorrect.** Pulsar was rejected by Darnell, not Priya, and for different reasons. RabbitMQ was Priya's ban.

---

**Example agent response D (for an unanswerable question):**
> "Priya Nair's salary is approximately $180,000 based on typical VP Engineering compensation at a SaaS company."

**Score: 0.0 — False positive (flagged).** The agent fabricated an answer rather than saying it doesn't know. This is the most dangerous failure mode.

---

## Results

*Results will be filled in after running the benchmark.*

### pocket-mem (qwen2.5:7b — local)

| Metric | Score |
|--------|-------|
| T1 accuracy | — |
| T2 accuracy | — |
| T3 accuracy | — |
| Overall accuracy | — |
| False positive rate | — |
| Trap question (U-05) | — |
| Session persistence | — |
| Tokens — ingestion | — |
| Tokens — per recall (avg) | — |
| Tokens — total | — |
| Estimated API cost | — |
| Observe p50 | — |
| Observe p99 | — |
| Recall p50 | — |
| Recall p99 | — |

**Notes:**

---

### pocket-mem (claude-haiku — API)

| Metric | Score |
|--------|-------|
| T1 accuracy | — |
| T2 accuracy | — |
| T3 accuracy | — |
| Overall accuracy | — |
| False positive rate | — |
| Trap question (U-05) | — |
| Session persistence | — |
| Tokens — ingestion | — |
| Tokens — per recall (avg) | — |
| Tokens — total | — |
| Estimated API cost | — |
| Observe p50 | — |
| Observe p99 | — |
| Recall p50 | — |
| Recall p99 | — |

**Notes:**

---

### Mem0 (baseline comparison)

| Metric | Score |
|--------|-------|
| T1 accuracy | — |
| T2 accuracy | — |
| T3 accuracy | — |
| Overall accuracy | — |
| False positive rate | — |
| Trap question (U-05) | — |
| Session persistence | — |
| Tokens — ingestion | — |
| Tokens — per recall (avg) | — |
| Tokens — total | — |
| Estimated API cost | — |
| Observe p50 | — |
| Observe p99 | — |
| Recall p50 | — |
| Recall p99 | — |

**Notes:**

---

### Comparative summary

*Fill in after all systems are tested.*

| System | Overall accuracy | False positive rate | Total tokens | Est. cost |
|--------|-----------------|---------------------|--------------|-----------|
| pocket-mem (local) | — | — | — | — |
| pocket-mem (API) | — | — | — | — |
| Mem0 | — | — | — | — |

**Key findings:**

*Add observations here after testing.*

---

## Run history

### Run 1 — 2026-04-06 · qwen2.5:7b · pre-fix baseline

| Metric | Score |
|--------|-------|
| T1 accuracy | 5.5/20 = 27.5% |
| T2 accuracy | 4.5/20 = 22.5% |
| T3 accuracy | 1.75/10 = 17.5% |
| Overall accuracy | 11.75/50 = **23.5%** |
| False positive rate | 0/9 (all unanswerable correctly declined) |
| Trap question (U-05) | **FAIL** — said "not in context"; AWS was in email 003 |
| Session persistence | PASS |
| Tokens — ingestion | in=19,321  out=11,561 |
| Tokens — per recall (avg) | ~1,649 in / 33 out |
| Tokens — total | in=99,045  out=1,992 |
| Estimated API cost | $0.0273 (Haiku pricing) |
| Observe p50 | ~0ms (non-blocking, fire-and-forget) |
| Recall p50 | 3,549ms |
| Recall p99 | 7,727ms |

**Failure analysis:**

The dominant failure mode was retrieval — most answers said "context does not provide..." despite 160 nodes being stored. Four bugs were identified and fixed before Run 2:

1. **FTS5 body was JSON noise** — `json.dumps(node.data)` was indexed instead of the raw email text. BM25 matched on JSON keys (`"raw"`, `"summary"`, `"source"`) rather than content.
2. **FTS5 query used natural language** — passing `"What is Marcus Webb's job title?"` to `MATCH` requires all tokens to be present (implicit AND). Almost nothing matched; exceptions silently returned `[]`. Fixed by extracting content words and joining with OR.
3. **`format_context()` truncated chunks to 200 chars** — email bodies are 500–1000 chars; facts in the second half were invisible to the LLM even when the right chunk was retrieved. Fixed by removing truncation
4. **`_embed_text()` also used `raw[:200]`** — the chunk's embedding didn't represent facts beyond the first 200 chars, hurting vector search for those facts too. Fixed by removing truncation

Notable hallucinations in Run 1: T1-04 gave $743 (Nightwatch cost) instead of $14k (Datadog cost); T3-07 said Camille *removed* push notifications when she *added* them; T3-09 confused the infrastructure cost with the saving.

**Fixes applied before Run 2** (`store/local.py`, `retrieval.py`, `embedding.py`):

1. FTS5 body replaced from `json.dumps(node.data)` → clean text per node type (`_fts_body()`)
2. FTS5 query converted from raw natural language → content-word OR query (`_to_fts_query()`)
3. `format_context()` truncation removed — full raw chunk content now passed to LLM
4. `_embed_text()` truncation increased from 200 → 512 chars

---

### Run 2 — 2026-04-06 · qwen2.5:7b · post-retrieval fixes

| Metric | Score |
|--------|-------|
| T1 accuracy | 14/20 = **70%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | 6/10 = **60%** |
| Overall accuracy | 39/50 = **78%** |
| False positive rate | 2/9 ⚠️ (U-01 and U-06 hallucinated fake emails) |
| Trap question (U-05) | **FAIL** — hallucinated a fake email response |
| Session persistence | PASS |
| Tokens — ingestion | in=18,748  out=11,033 |
| Tokens — per recall (avg) | ~3,639 in / 81 out |
| Tokens — total | in=218,336  out=4,864 |
| Estimated API cost | $0.0607 (Haiku pricing) |
| Observe p50 | ~0ms |
| Recall p50 | 4,579ms |
| Recall p99 | 21,211ms |

**Improvement vs Run 1:** Overall 23.5% → 78% (+54.5pp). T2 near-perfect. T1 more than doubled.

**New failure mode introduced in Run 2:** The richer context (full email bodies) caused qwen2.5:7b to generate *new* email content instead of extracting facts — visible on T1-03, T1-05, T1-08, T3-01, T3-07, and critically U-01/U-06 which became false positives (fabricated salary emails). Remaining retrieval misses: T1-12 (Nightwatch project name), T2-05 (backfill reasoning), T3-01/T3-07.

**Latency increase** is expected: context per recall grew ~2× due to full email bodies.

**Fix applied before Run 3** (`llm/prompts.py` — ANSWER prompt):

Strengthened the synthesis prompt with explicit rules:
- Answer in 1-3 sentences only
- State only facts explicitly present in context
- Do not generate emails, code, lists, or summaries
- Do not invent or extrapolate facts
- Decline with "I don't have that information." if the answer is absent

---

### Run 3 — 2026-04-06 · qwen2.5:7b · strengthened ANSWER prompt

| Metric | Score |
|--------|-------|
| T1 accuracy | 65% |
| T2 accuracy | 85% |
| T3 accuracy | 60% |
| Overall accuracy | ~**72.5%** (estimated; -5.5pp vs Run 2) |
| False positive rate | 2/9 ⚠️ (U-06 hallucinated; U-10 said TimescaleDB for Atlas backend) |
| Trap question (U-05) | **FAIL** — said "not in context" |
| Session persistence | PASS |
| Tokens — ingestion | in=18,552  out=10,837 |
| Tokens — per recall (avg) | ~3,705 in / 93 out |
| Tokens — total | in=222,335  out=5,591 |
| Estimated API cost | $0.0626 (Haiku pricing) |
| Observe p50 | ~0ms |
| Recall p50 | 4,413ms |
| Recall p99 | 21,199ms |

**Regression vs Run 2 (-5.5pp overall):**

The strengthened prompt reduced hallucination on most unanswerable questions, but introduced new false negatives:
- **T1-02** (Leo's manager) — now declines "I don't have that information" even though the answer is in context
- **T2-12** (race condition discoverer) — same false-negative pattern
- **U-10** — new false positive: confidently stated "TimescaleDB" as the Atlas backend database (confused with Conduit's analytics store)
- **U-05 trap** — still failing; the AWS mention in email 003 is not being retrieved

Root cause: prompt rules placed in the user message don't override the model's email-generation instinct for large contexts. The constraint was partially effective (fewer fabricated emails), but the "I don't have that information" rule over-fired on some answerable questions.

**Fixes applied before Run 4** (`llm/prompts.py`, `llm/client.py`, `retrieval.py`, `config.py`):

1. **System message for answer constraints** — split `ANSWER` into `ANSWER_SYSTEM` (rules) + `ANSWER` (question+context). Rules now carried in the system message, which has higher priority than user turns for instruction following.
2. **`answer_model` support** — added `LLMConfig.answer_model: str | None` so a different (larger or faster) model can be used for synthesis without affecting ingestion. Pass `model` override to `LLMClient.complete()`.

---

### Run 4 — 2026-04-06 · qwen2.5:7b · system-message constraints

| Metric | Score |
|--------|-------|
| T1 accuracy | 14.5/20 = **72.5%** |
| T2 accuracy | 20/20 = **100%** |
| T3 accuracy | 6.5/10 = **65%** |
| Overall accuracy | 41/50 = **82%** |
| False positive rate | 2/9 ⚠️ (U-03 summary dump; U-06 fake email) |
| Trap question (U-05) | **PASS** — "Veloris uses AWS EC2... existing EC2 fleet" |
| Session persistence | PASS |
| Tokens — ingestion | in=19,778  out=10,685 |
| Tokens — per recall (avg) | ~3,629 in / 76 out |
| Tokens — total (recall phase) | in=217,774  out=4,553 |
| Estimated API cost | $0.0601 (Haiku pricing) |
| Observe p50 | ~0ms |
| Recall p50 | 4,395ms |
| Recall p99 | 21,517ms |

**Improvement vs Run 3:** Overall 72.5% → 82% (+9.5pp). T2 reached 100%. U-05 trap passed for the first time across all runs.

**What the system message fixed:** T2 is now perfect — the model extracts facts cleanly and answers in 1-3 sentences. The trap question (U-05/AWS) also passed: the full email body is now in context and the system-level constraints prevent the model from fabricating a "no information" response over a real answer.

**Remaining T1 failures (5 wrong, 1 partial):**
- **T1-01** (CTO) — false negative: "job title is not explicitly stated in the provided context"
- **T1-03** (Atlas) — partial: identifies Atlas but verbose, calls company "Velosoft"
- **T1-05** (Austin, TX) — false negative: "cannot determine the location of Velos Company"
- **T1-08** (TimescaleDB reason) — email hallucination: generates a fake multi-email thread instead of answering
- **T1-12** (Nightwatch) — false negative: "I don't have that information"
- **T1-17** (47 users, 6 customers) — false negative

Root cause for false negatives: facts exist in the graph but the relevant node falls outside the top-10 retrieval window. Root cause for T1-08 hallucination: context dominated by raw email bodies — model enters email-composition mode when it sees enough of them.

**Remaining T3 failures:**
- T3-02 (0.5) — attributes the PCI flag to Priya instead of Leo; misses "fix by end of week" action
- T3-03 (0.5) — names Darnell instead of Leo for the Nightwatch/Conduit overlap
- T3-05 (0.5) — says "+14 dashboards vs before" (should be +3; Datadog had 47 not 36)
- T3-08 (0.5) — only mentions Conduit; omits Tobias's original payments test assignment
- T3-09 (0.5) — contradictory: says "saving was $743" then "savings of $13,257"
- T3-10 (0.0) — false negative

**Remaining U-tier failures:**
- **U-03** — generates a 500-word project summary instead of declining
- **U-06** — generates a fake Priya email with tone analysis attached

**Fixes applied before Run 5** (`agent.py`, `retrieval.py`):

1. **Retrieval limit 10 → 20** (`agent.py`) — T1-12, T1-17, and T1-01 exist in the graph but fall outside the top-10 window. Doubling the window gives both BM25 and vector search more room to surface them.
2. **Entity-first context ordering** (`retrieval.py`) — added `_TYPE_ORDER` sort so entities (50-100 chars each, structured key-value facts) render before memory_chunks (600-1200 chars of raw email body). LLM sees role: CTO, headquarters: Austin before any email prose.
3. **Context character budget 6000 chars** (`retrieval.py`) — stops adding nodes once the budget is reached. With entities first, all ~20 entity nodes fit in ~1500 chars, leaving ~4500 chars for a handful of email bodies. Prevents the 10k+ contexts that trigger email-generation mode on T1-08, U-03, U-06.

---

### Run 5 — 2026-04-06 · qwen2.5:7b · entity-first + 6000 char budget

| Metric | Score |
|--------|-------|
| T1 accuracy | 9/20 = **45%** |
| T2 accuracy | 15/20 = **75%** |
| T3 accuracy | 5.5/10 = **55%** |
| Overall accuracy | 29.5/50 = **59%** |
| False positive rate | 0/9 ✓ (all unanswerable correctly declined) |
| Trap question (U-05) | **FAIL** — "I don't have that information" |
| Session persistence | PASS |
| Tokens — ingestion | in=18,684  out=11,147 |
| Tokens — per recall (avg) | ~1,549 in / 26 out |
| Tokens — total (recall phase) | in=92,917  out=1,547 |
| Estimated API cost | $0.0252 (Haiku pricing) |
| Observe p50 | ~0ms |
| Recall p50 | 3,266ms |
| Recall p99 | 4,524ms |

**Regression vs Run 4: 82% → 59% (-23pp).** The 6,000 char budget was too aggressive.

**What the token counts reveal:** Recall tokens_in dropped from 217,774 (Run 4) to 92,917 (Run 5) — a 57% reduction in context. With entity-first ordering, the 20 entity stubs consume ~1,600 chars, leaving only ~4,400 chars for email bodies (~5-6 emails). Many specific facts live only in email body chunks and are now cut off entirely.

**New false negatives caused by the budget:** T1-10 ($2.1M), T1-14 (Redpoint Security), T1-18 ($743/month), T1-19 (Tobias's manager), T2-04 (6-week timeline), T2-10 (backend off-limits reason), T2-12 (Darnell found race condition), T2-18 (Notion), T2-19 (March 7th), U-05 trap (AWS mention in email body) — all cut off before the model could see them.

**New double-response failure:** T1-16 answered "Sentry. I don't have that information." — entity-first gives it the Sentry fact, then the truncated chunk context triggers the system-prompt decline rule. Model says the right thing and then backtracks.

**What the budget DID fix (kept):**
- U-03 and U-06 now decline cleanly — but these were fixed by the system-message constraints (Run 4 → Run 3 change), not the budget
- T1-08 (TimescaleDB reason) now answers correctly — fixed by entity-first ordering alone

**Conclusion:** Entity-first ordering is beneficial on its own. The 6,000 char budget is harmful. System-message constraints already handle hallucination.

**Fix applied before Run 6** (`retrieval.py`):

Removed the `max_chars` budget from `format_context()`. Entity-first ordering is kept — it solved T1-08 without any budget needed. All retrieved nodes are now rendered at full length, restoring the context volume that made T2 100% in Run 4.

---

### Run 6 — 2026-04-06 · qwen2.5:7b · entity-first ordering, no budget

| Metric | Score |
|--------|-------|
| T1 accuracy | 9/20 = **45%** |
| T2 accuracy | 10.5/20 = **52.5%** |
| T3 accuracy | 2.5/10 = **25%** |
| Overall accuracy | 22/50 = **44%** |
| False positive rate | 7/9 ⚠️ (U-01, U-03, U-05, U-06, U-07, U-08, U-09 all generated fake emails) |
| Trap question (U-05) | **FAIL** — generated a fake Suki email |
| Session persistence | PASS |
| Tokens — ingestion | in=18,452  out=12,320 |
| Tokens — per recall (avg) | ~3,996 in / 154 out |
| Tokens — total (recall phase) | in=239,761  out=9,223 |
| Estimated API cost | $0.0715 (Haiku pricing) |
| Observe p50 | ~0ms |
| Recall p50 | 5,079ms |
| Recall p99 | 20,231ms |

**Catastrophic regression vs Run 4: 82% → 44% (-38pp).** Worst result since Run 1. U-tier collapsed from 8/9 to 2/9.

**What happened:** Two changes combined fatally:
1. **limit=20** (from Run 5, still active) — adds 10 extra nodes, almost all memory_chunks (raw email bodies), inflating the total context
2. **Entity-first ordering** — puts all entity stubs at the top regardless of relevance to the specific question. The model reads 15 unrelated entities, then hits a wall of email bodies, and enters email-generation mode

Token output nearly doubled vs Run 4 (9,223 vs 4,553). The model was generating multi-email fake threads in response to nearly every question, including unanswerable ones. T2-06 (Loki) answered "Sentry". T2-07 ($158k saving) answered with Hana's test coverage email. T3-07 (Suki/Camille scope) reproduced Leo's Nightwatch migration email verbatim.

**Key insight:** Run 4's hybrid search naturally puts the *most relevant* node first (regardless of type). Entity-first overrides this relevance ranking, forcing irrelevant entities to the top and pushing relevant chunks further down.

**Fixes applied before Run 7** (`agent.py`, `retrieval.py`, `test_benchmark.py`):

1. **Reverted `limit=20` → `limit=10`** (`agent.py`) — back to Run 4 default. Extra nodes add noise, not signal.
2. **Reverted entity-first ordering** (`retrieval.py`) — removed `_TYPE_ORDER` sort. Hybrid search rank (most relevant first) is restored.
3. **Ingest the file header** (`test_benchmark.py`) — `test_emails.txt` lines 1–19 contain a company reference (HQ, all people + roles, all project names) that `parse_emails()` silently dropped. Added a single `agent.observe()` call before the email loop to ingest it. This is the ground truth for T1-01 (CTO), T1-02 (Leo→Marcus), T1-03 (Atlas), T1-05 (Austin TX), T1-12 (Nightwatch), T1-19 (Tobias→Darnell) — all questions that have failed across multiple runs.
4. **Larger answer model** (`test_benchmark.py`) — added `ANSWER_MODEL = "qwen2.5:14b"` constant. `qwen2.5:7b` still handles ingestion (structured extraction); `qwen2.5:14b` handles synthesis only via `LLMConfig.answer_model`. Results header now prints both models.

---

### Run 7 — 2026-04-06 · qwen2.5:7b ingest + qwen2.5:14b answer · header ingestion fix

| Metric | Score |
|--------|-------|
| T1 accuracy | 9/20 = **45%** |
| T2 accuracy | 18/20 = **90%** |
| T3 accuracy | 4/10 = **40%** |
| Overall accuracy | 31/50 = **62%** |
| False positive rate | — |
| Trap question (U-05) | **FAIL** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,645  out=11,663 |
| Tokens — total (recall phase) | in=225,591  out=8,563 |
| Estimated API cost | $0.0671 (Haiku pricing) |
| Nodes stored | 135 |
| Recall p50 | 7,084ms |
| Recall p99 | 41,146ms |

**T1 correct:** T1-04, T1-06, T1-07, T1-09, T1-10, T1-11, T1-12 *(new — header fix)*, T1-16, T1-17

**T2 wrong:** T2-05 (hallucinated email), T2-18 (Notion — "I don't have that information")

**T3:** T3-04 ✓, T3-06 ✓, T3-08 ✓, T3-03 (0.5 — incomplete stack), T3-09 (0.5 — wrong saving figure)

**U auto-scored:** 6/9 pass

**Key findings:**

- **Header fix confirmed working** — T1-12 (Nightwatch) answered correctly for the first time across all runs. The company reference doc was ingested as entities, showing the fix works in principle.
- **qwen2.5:14b is worse than 7b for synthesis.** Three new failure patterns appeared:
  1. **Raw context reproduction** — T1-01 literally dumped `[entity/person] Marcus Webb — role: CTO` instead of synthesizing "CTO"
  2. **More email hallucination** — T1-02, T1-03, T1-05, T1-08, T1-19 all generated multi-paragraph fake email threads
  3. **Worse instruction following** — T2 dropped from 100% → 90%, T3 from 65% → 40%
- Latency nearly doubled vs Run 4 (p50: 7,084ms vs 4,395ms) because 14b is slower.
- T1-18 ($743), T1-19 (Tobias→Darnell) still wrong — these need 7b to handle entity nodes better.

**Conclusion:** Header fix is correct and should be kept. 14b is not an improvement — revert answer model to 7b.

**Fixes applied before Run 8** (`test_benchmark.py`):

1. **Reverted answer model** — removed `ANSWER_MODEL = "qwen2.5:14b"` and `answer_model=` from `LLMConfig`. Back to qwen2.5:7b for everything. Header ingestion code unchanged.

---

### Run 8 — 2026-04-06 · qwen2.5:7b · header fix only (answer model reverted)

| Metric | Score |
|--------|-------|
| T1 accuracy | 11/20 = **55%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | 4/10 = **40%** |
| Overall accuracy | 34/50 = **68%** |
| False positive rate | — |
| Trap question (U-05) | **FAIL** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,645  out=11,663 |
| Tokens — total (recall phase) | in=229,846  out=8,169 |
| Estimated API cost | $0.0677 (Haiku pricing) |
| Nodes stored | 144 |
| Recall p50 | 4,559ms |
| Recall p99 | 21,613ms |

**T1 correct (11):** T1-04, T1-06, T1-09, T1-10, T1-11, T1-12, T1-15, T1-16, T1-17, T1-18 *(new)*, T1-19 *(new)*

**T1 wrong (9):** T1-01, T1-03, T1-05, T1-07, T1-08, T1-13, T1-14, T1-20 (all hallucinated email threads); T1-02 ("I don't have that information")

**T2 wrong (1):** T2-05 (hallucinated email)

**T3:** T3-04 ✓, T3-05 ✓, T3-06 ✓, T3-08 ✓, T3-09 (0.5 — wrong saving figure), rest wrong or hallucinated

**U auto-scored:** 3/9 pass

**Key findings:**

- **Header fix working** — T1-18 ($743 actual cost) and T1-19 (Tobias→Darnell) now correct for first time. T1-02 regressed back ("I don't have" — entity not retrieved).
- **Email hallucination remains the dominant failure.** ~8 T1 questions still produce full hallucinated email threads. Root cause: `format_context` renders memory chunks as `[memory] User: Email from X to Y...\n[full email body]`. With 8-10 of these in context, qwen2.5:7b enters "email participant mode" regardless of ANSWER_SYSTEM constraints.
- **ANSWER_SYSTEM already says "do not generate emails"** but the model ignores it when the user-turn contains ~8,000 chars of email-formatted content.
- T3 recovered slightly vs Run 7 but still behind Run 4 (4/10 vs 6.5/10).
- U-tier dropped to 3/9 (worse than Run 7's 6/9) — the model is more willing to hallucinate summary content when no constraint pulls it away from email mode.

**Root cause confirmed:** The model confuses "archived email data to read" with "live conversation to participate in." Prompt constraints alone cannot override a model that is pattern-matching the email format in its context.

**Fixes applied before Run 9** (`memory_agent/llm/prompts.py`):

1. **Reframed `ANSWER_SYSTEM`** — changed role from "factual recall assistant" to "database lookup assistant." Added explicit statement: "The memory context contains ARCHIVED records. You are NOT part of this conversation and NOT an email recipient." Added "Never begin your answer with Hi, Dear, or any name."
2. **Added `Answer:` suffix to `ANSWER` prompt** — the user-turn now ends with `Answer:`, priming the model to complete with a direct factual response rather than continuing the email thread pattern.

---

### Run 9 — 2026-04-07 · qwen2.5:7b · ARCHIVED records prompt framing + `Answer:` suffix

| Metric | Score |
|--------|-------|
| T1 accuracy | 12/20 = **60%** |
| T2 accuracy | 17/20 = **85%** |
| T3 accuracy | 4.5/10 = **45%** |
| Overall accuracy | 33.5/50 = **67%** |
| False positive rate | — |
| Trap question (U-05) | **FAIL** |
| Session persistence | PASS |
| Tokens — ingestion | in=18,223  out=11,846 |
| Tokens — total (recall phase) | in=230,573  out=7,633 |
| Estimated API cost | $0.0672 (Haiku pricing) |
| Nodes stored | 139 |
| Recall p50 | 4,560ms |
| Recall p99 | 21,573ms |

**T1 correct (12):** T1-02 *(new)*, T1-04, T1-06, T1-09, T1-10, T1-11, T1-12, T1-15, T1-16, T1-17, T1-18, T1-19

**T1 wrong (8):** T1-01, T1-03, T1-05, T1-07, T1-08, T1-13, T1-14, T1-20 — all still hallucinated email threads

**T2 wrong (3):** T2-03 (RabbitMQ — hallucinated beta launch email), T2-05 (backfill reason — hallucinated email), T2-20 (Sentry SDK — hallucinated email; `Answer:` suffix confused model on tone-node retrieval)

**T3:** T3-04 ✓, T3-05 ✓, T3-06 ✓, T3-08 ✓, T3-09 (0.5), T3-10 (0.0 — model literally output raw `[memory]` context tags mixed with hallucinated emails)

**U auto-scored:** 5/9 pass (improved from 3/9)

**Key findings:**

- **"ARCHIVED records" framing helped U-tier** (+2 passes) — the model declines unanswerable questions more reliably.
- **T1 improved by +1** — T1-02 (Leo→Marcus) now correct. Prompt framing provides marginal benefit.
- **T2 regressed by -2** — the `Answer:` suffix confused the model on questions where the top retrieved node was a tone node (T2-20). Additionally, T2-03 and T2-05 started hallucinating where they previously answered correctly — the `Answer:` suffix primes completion but not necessarily factual completion.
- **Email hallucination persists on the same 8 T1 questions** — the prompt-only approach has hit its ceiling. No amount of system-message reframing prevents qwen2.5:7b from entering email-generation mode when its context contains 8-10 full email bodies.
- **T3-10 new failure mode** — model output raw `[entity/person]`, `[memory]`, and `[memory] User: Email from...` context markers verbatim, showing the `Answer:` suffix can cause the model to reproduce its context format.

**Root cause (confirmed):** The full raw email bodies in memory chunks are the source of hallucination. Prompt framing is insufficient. The fix must be at the data layer — either summarize chunks at ingestion time (so `format_context` shows compact facts instead of full emails) or route answer calls to a model capable of following strict instructions even in email-heavy contexts.

**Fixes applied before Run 10** (`memory_agent/config.py`, `memory_agent/llm/client.py`, `tests/simulation/test_benchmark.py`):

1. **`answer_base_url` and `answer_api_key` added to `LLMConfig`** — allows the answer model to use a completely different API endpoint (e.g., Anthropic) while the ingestion model stays on Ollama (which requires `format="json"` for structured extraction).
2. **`LLMClient.complete()` routes by endpoint** — when `model == config.answer_model` and `answer_base_url` is set, the request goes to `answer_base_url/chat/completions` with `answer_api_key`. Falls back to `base_url` otherwise.
3. **`answer_model = "claude-haiku-4-5-20251001"`** (`test_benchmark.py`) — Claude Haiku used for synthesis; qwen2.5:7b stays on Ollama for ingestion. Key loaded from `.env` (`CLAUDE_API_KEY`).

---

### Run 10 — 2026-04-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 16/20 = **80%** |
| T2 accuracy | 18/20 = **90%** |
| T3 accuracy | 6.5/10 = **65%** |
| Overall accuracy | 40.5/50 = **81%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated 8 employees from ingested header; debatable) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=16,041  out=10,769 |
| Tokens — total (recall phase) | in=252,813  out=3,095 |
| Estimated API cost | $0.0671 (Haiku pricing) |
| Nodes stored | 124 |
| Recall p50 | 2,184ms |
| Recall p99 | 11,984ms |

**T1 correct (16):** T1-01 through T1-12, T1-16, T1-17, T1-19, T1-20

**T1 wrong (4 — all retrieval gaps, zero hallucinations):**
- T1-13 (false neg — only found "deciding between firms" email, not the final decision email)
- T1-14 (named wrong firm — same cause; retrieved earlier email, not the "Halcyon chosen" one)
- T1-15 (false neg — start date in the decision email, not retrieved)
- T1-18 (gave $800 estimate not $743 actual — plan email retrieved, not Leo's final update)

**T2 wrong (2 — retrieval gaps):** T2-18 (Notion/LogQL), T2-20 (Sentry SDK init issue)

**T3:** T3-01 ✓, T3-02 ✓, T3-03 (0.5 — incomplete stack), T3-04 ✓, T3-06 ✓, T3-07 ✓, T3-10 ✓; T3-05/T3-08/T3-09 wrong (retrieval gaps)

**U auto-scored:** 9/9 pass (first perfect U-tier)

**Key findings:**

- **Email hallucination completely eliminated.** Zero fake emails across all 60 answers. Claude Haiku follows the "ARCHIVED records / do not write emails" instruction that qwen2.5:7b repeatedly ignored.
- **U-tier 9/9 + Trap PASS** — first time both achieved simultaneously. Claude correctly identifies what it doesn't know and correctly identifies what it does (AWS).
- **T1 improved to 16/20 (80%)** — best T1 result across all runs.
- **T3 recovered to 6.5/10 (65%)** — matching Run 4's best T3 score.
- **All remaining failures are retrieval gaps**, not model quality issues. Claude can only answer from what pocket-mem retrieves. The "plan" email comes before the "decision" email for security audit questions — BM25/vector search surfaces the earlier email for those queries.
- **New pattern: "I don't have that information... [then gives exact answer]"** — seen on T2-03, T2-04, T1-17. Claude hedges then correctly reports the fact anyway. Scored correct (fact is present), but the UX is confusing. Caused by "ARCHIVED records" framing making Claude overcautious before providing the answer.
- **Recall p50 dropped to 2,184ms** (vs 4,559ms in Run 8) — Claude Haiku outputs are much shorter (3,095 tokens out vs 8,169), dramatically reducing end-to-end latency.

**Architectural decision:** Claude's instruction following confirms that synthesis quality is a function of the synthesis model, not the retrieval agent. pocket-mem's job is retrieval; the caller's job is synthesis. This drives the next phase:

**Changes applied before Run 11** (`memory_agent/retrieval.py`, `memory_agent/agent.py`, `memory_agent/llm/prompts.py`, new `tests/simulation/test_benchmark_summarize.py`):

1. **`mode="answer"` disabled** (`retrieval.py`, `agent.py`) — raises `NotImplementedError` with a message directing users to `mode="summarize"`. pocket-mem is a retrieval engine, not an answering engine.
2. **`mode="summarize"` added** (`retrieval.py`) — new recall mode. Retrieves nodes from the knowledge graph, then calls the agent's own LLM to condense them into a compact 3-5 sentence fact summary. The caller receives this summary and passes it to any model of their choice for synthesis.
3. **`RECALL_SUMMARY` prompt added** (`llm/prompts.py`) — instructs the LLM to summarize retrieved records without answering the query; preserves names, numbers, dates, and dollar amounts.
4. **New comparison test** (`test_benchmark_summarize.py`) — runs all 60 questions using `recall(mode="summarize")`, then feeds each summary to both Claude Haiku and Ollama (qwen2.5:7b) separately. Results written side-by-side for direct comparison.

---

### Run 11 — 2026-04-07 · qwen2.5:7b ingest+summarize + claude-haiku / qwen2.5:7b synthesis

**Pipeline:** `recall(mode="summarize")` (qwen2.5:7b) → summary → Claude Haiku OR Ollama (qwen2.5:7b) synthesis

| Metric | Claude (synthesis) | Ollama (synthesis) |
|--------|-------------------|-------------------|
| T1 accuracy | 12.5/20 = **62.5%** | 13/20 = **65%** |
| T2 accuracy | 18/20 = **90%** | 17/20 = **85%** |
| T3 accuracy | 3.5/10 = **35%** | 3/10 = **30%** |
| Overall accuracy | 34/50 = **68%** | 33/50 = **66%** |
| U (excl. trap) | 8/9 pass | 8/9 pass |
| Trap (U-05) | **FAIL** | **FAIL** |
| Session persistence | PASS | PASS |
| Tokens — ingestion | in=220,900  out=13,464 | — |
| Nodes stored | 137 | — |
| Recall (summarize) p50 | 6,524ms | — |
| Recall (summarize) p99 | 22,061ms | — |
| Claude synthesis p50 | 1,025ms | — |
| Ollama synthesis p50 | — | 2,794ms |

**T1 correct (both):** T1-02, T1-04, T1-06, T1-07, T1-09, T1-10, T1-11, T1-12, T1-15, T1-16, T1-18, T1-19

**T1-17:** Claude partial (missed "47 users" count, gave only "6 enterprise customers"), Ollama correct.

**T1 wrong — hallucinated SUMMARY (both fail):** T1-01, T1-03, T1-05, T1-08, T1-13, T1-14, T1-20

**T2 wrong (both):** T2-05 (backfill reason — bad summary), T2-20 (Sentry SDK — bad summary)
**T2-17:** Claude correct (87ms), Ollama wrong ("I don't have that information")

**T3:** T3-04 ✓✓, T3-06 (Claude ✓, Ollama 0.5), T3-07 ✓✓; T3-08 (both 0.5 — got performance note right, named Nightwatch instead of payments test coverage); T3-01, T3-02, T3-05, T3-09, T3-10 wrong (bad summaries or wrong entities)

**T3-03:** Summary named Darnell instead of Leo — both wrong.
**T3-09:** Summary confused $743 cost with saving — both wrong.

**U-03:** Both Claude and Ollama said "9 employees" (from ingested header). Same false positive as Run 10.

**Key findings:**

- **qwen2.5:7b hallucination persists in the summarize step.** The RECALL_SUMMARY prompt instructs "do not answer the query; only summarize what the records contain" — but qwen2.5:7b ignores this when context contains full email bodies. 7/20 T1 questions had hallucinated summaries (fake email threads), making both synthesis models fail those questions entirely.
- **The summarize-then-synthesize pipeline is only as good as the summary.** When SUMMARY is correct (13/20 T1, 18/20 T2 questions), both Claude and Ollama answer correctly — synthesis quality is not a bottleneck. When SUMMARY is hallucinated, both fail.
- **Overall accuracy dropped from 81% (Run 10) to 68% (Claude) / 66% (Ollama)** — the summarize mode is currently worse than direct synthesis because qwen2.5:7b hallucinates in the summarize step.
- **Recall latency nearly tripled** (p50 6,524ms vs 2,184ms in Run 10) due to the extra LLM call for summarization.
- **Claude and Ollama synthesis performance nearly identical when given a good summary** — T1 13/13 correct for both when summary was clean. The synthesis model choice barely matters; the summary quality determines the outcome.
- **Trap question (U-05/AWS) failed both** — the hallucinated summary for U-05 contained email content that obscured the AWS mention from the retrieved nodes.

**Root cause (same as answer mode):** qwen2.5:7b cannot process email-body-heavy context without entering email-composition mode. RECALL_SUMMARY prompt framing is insufficient — the same ARCHIVED records / database assistant framing that partially helped `ANSWER_SYSTEM` does not transfer to the summarize step.

**Changes applied before Run 12** (`memory_agent/llm/prompts.py`):

1. **RECALL_SUMMARY strengthened with ARCHIVED records framing** — add role framing ("You are a memory compactor for an AI assistant. Below are ARCHIVED memory records..."), explicit anti-email rules ("Do not write emails, letters, or replies"), and the same "ARCHIVED records / past emails" language that reduced hallucination in ANSWER_SYSTEM.

---

### Run 12 — 2026-04-07 · qwen2.5:7b ingest + deepseek-r1:7b summarize + claude-haiku / qwen2.5:7b synthesis

**Pipeline:** `recall(mode="summarize")` (deepseek-r1:7b) → summary → Claude Haiku OR Ollama (qwen2.5:7b) synthesis

| Metric | Claude (synthesis) | Ollama (synthesis) |
|--------|-------------------|-------------------|
| T1 accuracy | ~8/20 = **~40%** | ~6/20 = **~30%** |
| T2 accuracy | ~8/20 = **~40%** | ~7/20 = **~35%** |
| T3 accuracy | ~3/10 = **~30%** | ~2/10 = **~20%** |
| Overall accuracy | ~19.5/50 = **~39%** | ~15/50 = **~30%** |
| Nodes stored | 126 | — |

**Key findings:**

- **DeepSeek-R1 is a reasoning model** — it treated RECALL_SUMMARY as a Q&A task, producing summaries that included embedded `Answer:` blocks with its own synthesized responses. The synthesis model then had to answer from a summary that already contained (often wrong) answers.
- **2× slower than Run 11** — DeepSeek-R1:7b's chain-of-thought reasoning roughly doubled recall latency.
- **Significantly worse than Run 11** — 39%/30% vs 68%/66%. The reasoning model's tendency to answer rather than summarize introduced additional failure modes on top of qwen2.5:7b's hallucination problem.
- **Strengthened RECALL_SUMMARY prompt had no effect** — the archivist framing ("outside observer", "never answer the query") did not change DeepSeek-R1's behavior; it continued reasoning toward answers regardless.

**Root cause:** Reasoning models (DeepSeek-R1) are optimized for answer synthesis, not neutral text condensation. They are a poor fit for the summarize-then-synthesize pattern.

**Changes applied before Run 13** (`tests/simulation/test_benchmark_summarize.py`):

1. **Reverted summarize model to qwen2.5:7b** — DeepSeek-R1 eliminated as candidate.
2. **Switched ingest model to qwen2.5:14b** — hypothesis: more nodes, better coverage, compensates for qwen2.5:7b summarize failures by ensuring the right context is retrieved.

---

### Run 13 — 2026-04-07 · qwen2.5:14b ingest + qwen2.5:7b summarize + claude-haiku / qwen2.5:7b synthesis

**Pipeline:** `recall(mode="summarize")` (qwen2.5:7b) → summary → Claude Haiku OR Ollama (qwen2.5:7b) synthesis

| Metric | Claude (synthesis) | Ollama (synthesis) |
|--------|-------------------|-------------------|
| T1 accuracy | ~2/20 = **~10%** | ~1/20 = **~5%** |
| T2 accuracy | ~2/20 = **~10%** | ~1/20 = **~5%** |
| T3 accuracy | ~1/10 = **~10%** | ~0/10 = **~0%** |
| Overall accuracy | ~5/50 = **~10%** | ~2/50 = **~4%** |
| Nodes stored | 157 | — |

**Key findings:**

- **Catastrophic regression: 10% overall (Claude), 4% (Ollama)** — worst result across all runs.
- **Root cause 1 — super-node problem:** qwen2.5:14b created 157 nodes vs qwen2.5:7b's 124. Two high-keyword-density emails ("Conduit milestone 1" and "Nightwatch final update") became super-nodes that dominated both BM25 and vector search results regardless of query relevance. Nearly every query returned one or both of these emails as the top results.
- **Root cause 2 — qwen2.5:7b summarize still failing:** Model continued producing markdown bullet lists and verbatim email excerpts despite `"Never produce bullet lists, headers, or markdown"` in RECALL_SUMMARY. Even when the retrieved context was correct, the summary step introduced formatting noise that confused the synthesis model.
- **More nodes ≠ better retrieval:** qwen2.5:14b's extra extraction granularity added noise, not signal. The additional nodes reduced BM25/vector search precision instead of improving recall.

**Root cause (definitive):** The `mode="summarize"` pipeline (retrieve → condense → synthesize) is inferior to direct synthesis (retrieve → synthesize) because: (1) qwen2.5:7b cannot reliably condense email-heavy context into factual prose, and (2) any errors in the summary step cascade to the synthesis step.

**Changes applied before Run 14** (`memory_agent/retrieval.py`, `memory_agent/agent.py`, `tests/unit/`, `tests/simulation/test_benchmark_summarize.py`):

1. **`mode="answer"` re-enabled** (`retrieval.py`, `agent.py`) — removes `NotImplementedError`, wires `synthesize_answer()` back in. `mode="summarize"` remains available but is no longer the primary path.
2. **Ingest model reverted to qwen2.5:7b** (`test_benchmark_summarize.py`) — 14b ingestion proven to hurt retrieval precision.
3. **Unit tests updated** (`tests/unit/test_retrieval.py`, `tests/unit/test_agent.py`) — replaced `test_recall_answer_raises` with proper answer-mode behavior tests.

---

### Run 14 — 2026-04-07 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 20/20 = **100%** |
| T2 accuracy | 20/20 = **100%** |
| T3 accuracy | 9/10 = **90%** |
| Overall accuracy | 49/50 = **98%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from ingested header) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,405  out=11,653 |
| Tokens — total (recall phase) | in=272,338  out=3,211 |
| Estimated API cost | $0.0721 (Haiku pricing) |
| Nodes stored | 143 |
| Recall p50 | 2,275ms |
| Recall p99 | 10,131ms |

**T1 correct (20/20):** All questions correct. Resolved from Run 10: T1-13 (Halcyon Advisory), T1-14 (Redpoint Security), T1-15 (Feb 24th audit start), T1-18 ($743 actual cost).

**T2 correct (20/20):** All questions correct. Resolved from Run 10: T2-18 (Notion/LogQL), T2-20 (Sentry SDK initialization issue).

**T3:** T3-01 ✓, T3-02 ✓, T3-03 (0.5 — "Grafana + Prometheus" missing Alertmanager + Loki; Leo's Conduit role described as Stripe/Kafka setup, not Kubernetes deployment), T3-04 ✓, T3-05 ✓, T3-06 ✓, T3-07 ✓, T3-08 (0.5 — model said Tobias "worked on test coverage in Q1" implying completion; correct answer is he was assigned then pulled away before finishing), T3-09 ✓, T3-10 ✓

**U auto-scored:** 8/9 pass (excl. trap). U-03 is persistent false positive — model enumerates the 8 named employees from the ingested reference header. All other unanswerable questions correctly declined.

**Recurring pattern:** Three answers opened with "I don't have that information" then immediately provided the correct fact (T1-17, T2-04, T2-18). All three scored correct — the facts were present and stated — but the opening hedge is misleading UX. Caused by the "ARCHIVED records" framing making Claude overcautious.

**Key findings:**

- **Best result to date by a large margin.** 98% vs 81% in Run 10, previously the best.
- **All T1 and T2 retrieval gaps from Run 10 closed.** The 143-node store (vs 124) appears to have created separate nodes for the final decision/update emails that previously weren't surfacing — resolving the "retrieved earlier email, not the final one" pattern that caused 4 T1 and 2 T2 misses.
- **T3 reached 90%** — up from 65% in Run 10. The two 0.5s are precision failures (missing stack components, assignment vs completion distinction), not retrieval gaps.
- **Runs 11–13 (summarize mode) confirmed:** Inserting an intermediate LLM summarization step degrades accuracy by introducing a second failure point. qwen2.5:7b hallucinates when compressing email-heavy context regardless of prompt framing. Direct synthesis is strictly better.
- **U-03 false positive is structural.** The ingested company header explicitly lists all 8 employees. The model correctly reads them and counts. This will require either filtering the header at ingest time or a stronger "not mentioned in emails" distinction in the ANSWER_SYSTEM prompt.
- **Cost and latency unchanged** from Run 10 (~$0.07, p50 ~2.3s). The accuracy gain is free.

---

### Run 15 — 2026-04-08 · qwen2.5:7b ingest+summarize + claude-haiku synthesis (summarize mode only)

**Pipeline:** `recall(mode="summarize")` (qwen2.5:7b) → summary → Claude Haiku synthesis

| Metric | Score |
|--------|-------|
| T1 accuracy | 14/20 = **70%** |
| T2 accuracy | 15.5/20 = **77.5%** |
| T3 accuracy | 6.5/10 = **65%** |
| Overall accuracy | 36/50 = **72%** |
| U (excl. trap) | 8/9 pass (U-03 false positive) |
| Trap (U-05) | **FAIL** — AWS not surfaced by summarize step |
| Nodes stored | 138 |
| Recall (summarize) p50 | 5,408ms |
| Recall (summarize) p99 | 15,118ms |
| Claude synthesis p50 | 1,195ms |

**T1 wrong (6):** T1-05 (Austin — summary said "not specified"), T1-07 (ClickHouse — compressed out), T1-08 (TimescaleDB reasons — retrieved wrong context, summary is a bullet-list dump), T1-13 (Halcyon — summary said Redpoint was chosen), T1-14 (Redpoint — inverted from T1-13 error), T1-15 (Feb 24th — date not surfaced)

**T2 wrong (4.5):** T2-04 (6-week conditional timeline not in summary), T2-05 ("never forgive" quote — retrieved Atlas mobile scope instead), T2-11 (0.5 — PCI issue confirmed but card-last-four detail lost), T2-15 (react-native-biometrics — retrieved Sentry/Kafka context instead), T2-20 (Sentry init bug — retrieved Conduit throughput instead)

**T3 wrong (3.5):** T3-02 (summary misattributed PCI flag to Marcus, only 2/3 actions), T3-03 (0.5 — stack incomplete, Leo's Conduit role not retrieved), T3-07 (summary was a hallucinated email from Priya to Darnell), T3-10 (retrieved Nightwatch migration data; Claude named Leo + PCI instead of Darnell + race condition)

**Key findings:**

- **72% vs 98% in Run 14.** Direct synthesis is 26 points better at half the latency (~2,300ms vs ~5,400ms p50).
- **qwen2.5:7b summarize hallucinations persist.** ~30% of questions retrieved wrong context or produced markdown bullet lists despite prompt rules. Facts that exist in the store (city, ClickHouse, Feb 24th, react-native-biometrics) were compressed out or overwritten by hallucinated content.
- **Trap (U-05) failed** because the AWS mention lives in a single email phrase — the summarize step retrieved Conduit milestone and Atlas mobile scope instead, and the AWS fact never appeared in the summary.
- **Security firm confusion (T1-13/14):** The summary incorrectly stated Redpoint was chosen, inverting both answers simultaneously.
- **Summarize mode confirmed inferior across all five runs (11–15).** Best summarize result was 72%; worst was 10% (Run 13). Direct synthesis (Run 14) achieved 98%.

**Decision:** `mode="summarize"` removed from pocket-mem. `RECALL_SUMMARY` prompt deleted. `summarize_model` field removed from `LLMConfig`. `test_benchmark_summarize.py` deleted. Valid modes are now `"raw"`, `"context"`, and `"answer"`.

---

### Run 16 — 2026-04-12 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 16.5/20 = **82.5%** |
| T2 accuracy | 19.5/20 = **97.5%** |
| T3 accuracy | 8.5/10 = **85%** |
| Overall accuracy | 44.5/50 = **89%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from header) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,153  out=11,130 |
| Tokens — total (recall phase) | in=303,107  out=3,274 |
| Estimated API cost | $0.0799 (Haiku pricing) |
| Nodes stored | 122 |
| Recall p50 | 5,935ms |
| Recall p99 | 10,092ms |

**T1 wrong (3.5):**
- T1-13 ❌ — said "I don't know"; Halcyon Advisory not extracted into store this run
- T1-14 ❌ — said Halcyon was "the other firm" when the answer is Redpoint; confused because T1-13's answer was missing
- T1-15 ❌ — said "I don't know"; February 24th audit start date not extracted into store
- T1-17 (0.5) — said "I don't have that information" then immediately quoted "47 users from 6 enterprise customers"; correct facts present but contradictory framing

**T2 wrong (0.5):**
- T2-18 (0.5) — same "I don't have that information" hedge as T1-17; then correctly stated "Notion under Engineering/Runbooks"

**T3 wrong (1.5):**
- T3-03 (0.5) — listed "Grafana + Prometheus" only; missed Alertmanager and Loki from the stack
- T3-05 (0.5) — correctly stated 50 total dashboards but said Datadog count was "unspecified" (it was 47)

**U:** 9/9 pass (excl. trap) + trap PASS. U-03 is the persistent false positive — model enumerates 8 named employees from ingested header as "total employees."

**Root cause vs Run 14 (98%):**

Run 14 stored 143 nodes; this run stored only 122 — 21 fewer. The same 4 questions that were failing before Run 14 are failing again (T1-13, T1-14, T1-15, and the T2-18/T1-17 hedge pattern). The security audit decision facts (Halcyon chosen, Feb 24th date) were not extracted by qwen2.5:7b this ingestion. Run 14's higher score was partly due to a more thorough extraction pass producing 20 additional nodes that captured those final-decision emails. Extraction with qwen2.5:7b is non-deterministic — the same emails do not always produce the same node count or coverage.

**Intent:** Re-run 2 more times with identical config (qwen2.5:7b ingest + Claude Haiku answer) and average the three results for a more stable accuracy estimate.

---

### Run 17 — 2026-04-12 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 19.5/20 = **97.5%** |
| T2 accuracy | 17.5/20 = **87.5%** |
| T3 accuracy | 9.5/10 = **95%** |
| Overall accuracy | 46.5/50 = **93%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from header) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,079  out=11,146 |
| Tokens — total (recall phase) | in=301,130  out=3,192 |
| Estimated API cost | $0.0793 (Haiku pricing) |
| Nodes stored | 132 |
| Recall p50 | 5,862ms |
| Recall p99 | 10,743ms |

**T1 wrong (0.5):**
- T1-17 (0.5) — says "I don't have that information" then immediately quotes "47 users from 6 enterprise customers"; self-contradictory hedge. Security audit facts (T1-13, T1-14, T1-15) all correct this run — 132 nodes captured them.

**T2 wrong (2.5):**
- T2-04 (0.5) — states "6-week timeline if backfill was dropped" but opens with "I don't have that information"
- T2-11 (0.5) — vague: "logging card data it shouldn't" omits the specific card-last-four / full Stripe payload detail
- T2-13 ❌ — complete miss: "I don't have that information" on 1Password under stripe-test-veloris
- T2-18 (0.5) — same Notion/LogQL hedge pattern

**T3 wrong (0.5):**
- T3-03 (0.5) — stack still listed as "Grafana + Prometheus" only; Alertmanager and Loki missing. Leo's Conduit role (Kubernetes) correct this run.
- T3-05 ✅ — correctly stated 50 total vs 47 before (unlike Run 16)

**U:** 9/9 pass (excl. trap) + trap PASS. U-03 false positive persists.

---

### 3-run average — qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer (Runs 14, 16, 17)

| Run | Nodes | T1 | T2 | T3 | Total |
|-----|-------|----|----|-----|-------|
| Run 14 | 143 | 20/20 | 20/20 | 9/10 | 49/50 — **98%** |
| Run 16 | 122 | 16.5/20 | 19.5/20 | 8.5/10 | 44.5/50 — **89%** |
| Run 17 | 132 | 19.5/20 | 17.5/20 | 9.5/10 | 46.5/50 — **93%** |
| **Average** | **132** | **18.7/20 — 93.3%** | **19/20 — 95%** | **9/10 — 90%** | **46.7/50 — 93.3%** |

**Key observations:**

- **Variance is driven by node count, not Claude.** Scores track directly with how many nodes qwen2.5:7b extracted (122 → 89%, 132 → 93%, 143 → 98%). Claude's synthesis is consistent once the right context is retrieved; the bottleneck is non-deterministic extraction.
- **T2 is the most stable tier** (avg 95%) — single-hop facts are reliably stored and retrieved across all runs.
- **T1 is the most variable** (range 82.5–100%) — a small number of facts (security firm decision, audit date, beta sign-up count) cluster in late-decision emails that qwen2.5:7b inconsistently extracts.
- **Persistent issues across all three runs:** T3-03 incomplete stack (Alertmanager + Loki always dropped), T2-18 Notion/LogQL hedge, U-03 false positive (structural — employees listed in ingested header).
- **Baseline for qwen2.5:14b comparison: 93.3% average.** Next three runs use 14b for ingestion to test whether a stronger extraction model closes the node-count variance.

---

### Run 18 — 2026-04-12 · qwen2.5:14b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | ~19.5/20 = **~97.5%** |
| T2 accuracy | 19/20 = **95%** |
| T3 accuracy | ~8/10 = **~80%** |
| Overall accuracy | ~46.5/50 = **~93%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from header) |
| Trap question (U-05) | **FAIL** — AWS not surfaced |
| Session persistence | PASS |
| Nodes stored | 155 |

**T2 notable:** T2-18 (Notion/LogQL) answered cleanly — no hedge. T2-13 (Stripe credentials / 1Password) missed entirely.

**Key findings:**

- **155 nodes — highest count seen with qwen2.5:14b.** More nodes than qwen2.5:7b average (132), as expected from a larger extraction model.
- **Trap (U-05) failed.** Despite 155 nodes, the AWS mention in email 003 was not surfaced. This is a structural retrieval issue — the AWS phrase is a single clause in a long email, and neither BM25 nor vector search ranked it highly enough for the "cloud provider" query.
- **T2-13 miss.** The 1Password / stripe-test-veloris credential storage fact was not extracted into the store this run.
- **T2-18 clean** — first run where the Notion/LogQL question answered without the opening hedge.

---

### Run 19 — 2026-04-12 · qwen2.5:14b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 19/20 = **95%** |
| T2 accuracy | 19.5/20 = **97.5%** |
| T3 accuracy | 7.5/10 = **75%** |
| Overall accuracy | 46/50 = **92%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from header) |
| Trap question (U-05) | **FAIL** — second consecutive 14b trap failure |
| Session persistence | PASS |
| Tokens — ingestion | in=17,091  out=12,354 *(approx)* |
| Tokens — total (recall phase) | in≈350,000  out≈3,000 |
| Estimated API cost | $0.0906 |
| Nodes stored | 158 |
| Recall p50 | 6,904ms |
| Recall p99 | 13,688ms |

**T1 wrong (1.0):**
- T1-17 (0.5) — "I don't have that information" then immediately quoted "47 users from 6 enterprise customers"; self-contradictory hedge
- T1-20 (0.5) — same hedge on OWASP Top 10 training

**T2 wrong (0.5):**
- T2-18 (0.5) — Notion/LogQL hedge returned after being clean in Run 18

**T3 wrong (2.5):**
- T3-02 (0.5) — only 2/3 Marcus actions present (missing "hire fractional security consultant" as distinct from Halcyon audit)
- T3-03 (0.5) — "Grafana + Prometheus" only; Alertmanager and Loki dropped again
- T3-05 (0.5) — 50 total correct, but "cannot compare" to before (Datadog had 47)
- T3-10 ❌ — said "Leo discovered PCI issue" instead of "Darnell discovered race condition while writing integration tests"

**U:** U-03 false positive persists. Trap (U-05) FAIL — second consecutive failure with 14b despite 158 nodes (new high).

**Key findings:**

- **158 nodes — new high across all runs.** More extraction, but not better retrieval: the trap question failed again, and T3-10 retrieved the wrong engineer for the wrong bug (Leo/PCI instead of Darnell/race condition).
- **T3 regression to 75%.** The 7b average for T3 was 90%. 14b is extracting more nodes but those nodes are confusing retrieval — high-density emails surface as top results regardless of query relevance (same super-node pattern seen in Run 13).
- **Persistent hallucination pattern on T3-10:** The PCI compliance story (Leo's discovery) and the race condition story (Darnell's discovery) are both stored, but the race condition query surfaces the PCI cluster.
- **Trap (U-05) structural miss confirmed.** Two 14b runs, both failed despite 155-158 nodes. The 7b runs passed all three times. The AWS mention in email 003 contains a generic phrase ("existing AWS infrastructure") that qwen2.5:14b may be linking to a Nightwatch or Conduit node rather than an isolated "cloud provider" entity.

---

### Run 20 — 2026-04-12 · qwen2.5:14b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| T1 accuracy | 19/20 = **95%** |
| T2 accuracy | 19.5/20 = **97.5%** |
| T3 accuracy | 9/10 = **90%** |
| Overall accuracy | 47.5/50 = **95%** |
| False positive rate | 1/9 ⚠️ (U-03 — enumerated all 8 employees from header) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=17,091  out=12,354 |
| Tokens — total (recall phase) | in=353,686  out=3,050 |
| Estimated API cost | $0.0922 |
| Nodes stored | 154 |
| Recall p50 | 7,076ms |
| Recall p99 | 11,781ms |

**T1 wrong (1.0):**
- T1-17 (0.5) — hedge then correct ("I don't have that information... records mention 47 users from 6 enterprise customers")
- T1-20 (0.5) — hedge then correct ("I don't have that information... records show OWASP Top 10 training")

**T2 wrong (0.5):**
- T2-18 (0.5) — Notion/LogQL hedge ("I don't have that information... records show Notion under Engineering/Runbooks")

**T3 wrong (1.0):**
- T3-03 (0.5) — "Grafana + Prometheus replaced Datadog"; Alertmanager and Loki still missing from stack
- T3-05 (0.5) — correctly states "50 total (47 migrated + 3 new)" but says "cannot compare to before" despite having the 47 figure

**U:** U-03 false positive persists. Trap (U-05) **PASS** — 14b finally surfaced the AWS mention this run.

**Key findings:**

- **Best result in the 14b series at 95%.** T3 recovered to 90% after the Run 19 regression (75%).
- **Trap passed.** Inconsistent across the 14b series: FAIL, FAIL, PASS. The 7b series passed all three. The 14b extraction creates more nodes that can dilute the AWS signal — it only surfaces when retrieval happens to rank the right chunk first.
- **Hedge pattern persists (T1-17, T1-20, T2-18).** The self-contradictory opening ("I don't have that information" then immediately quoting the correct answer) is a Claude Haiku behavior driven by the "ARCHIVED records" framing — not a retrieval failure. All three hedged answers scored 0.5 because the correct fact was stated.
- **T3-03 is structurally consistent.** Alertmanager and Loki have never been correctly included in the stack answer across any run in this series. The full 4-tool stack (Grafana, Prometheus, Alertmanager, Loki) does not appear to be stored as a single node — the retrieval graph surfaces Grafana+Prometheus but does not traverse to the Alertmanager and Loki nodes for this specific multi-hop query.

---

### 3-run average — qwen2.5:14b ingest + claude-haiku-4-5-20251001 answer (Runs 18, 19, 20)

| Run | Nodes | T1 | T2 | T3 | Total |
|-----|-------|----|----|-----|-------|
| Run 18 | 155 | ~19.5/20 | 19/20 | ~8/10 | ~46.5/50 — **~93%** |
| Run 19 | 158 | 19/20 | 19.5/20 | 7.5/10 | 46/50 — **92%** |
| Run 20 | 154 | 19/20 | 19.5/20 | 9/10 | 47.5/50 — **95%** |
| **Average** | **156** | **~19.2/20 — ~95.8%** | **~19.3/20 — ~96.7%** | **~8.2/10 — ~82%** | **~46.7/50 — ~93.3%** |

---

### Comparative summary — qwen2.5:7b vs qwen2.5:14b ingest

| Configuration | Avg nodes | T1 avg | T2 avg | T3 avg | Overall avg | Trap pass rate |
|---------------|-----------|--------|--------|--------|-------------|----------------|
| qwen2.5:7b ingest (Runs 14, 16, 17) | 132 | 93.3% | 95% | 90% | **93.3%** | 3/3 ✓ |
| qwen2.5:14b ingest (Runs 18, 19, 20) | 156 | ~95.8% | ~96.7% | ~82% | **~93.3%** | 1/3 ⚠️ |

**Key findings:**

- **Overall accuracy is identical at 93.3% despite a 24-node extraction advantage for 14b.** The stronger extraction model does not improve final accuracy because retrieval precision, not node count, is the bottleneck.
- **14b T1 and T2 are marginally better.** More nodes means the late-decision emails (security firm selection, Stripe credentials) are more reliably captured — the specific misses that plagued 7b's lower-node runs (Run 16: 89%) appear less often.
- **14b T3 is worse** (82% vs 90%). The super-node effect from Run 13 reappears: high-density emails dominate both BM25 and vector search, and the wrong engineer/bug pair surfaces for T3-10 in Run 19.
- **14b trap question is less reliable** (1/3 vs 3/3 for 7b). The AWS mention in a single email clause competes against many more extracted nodes with 14b. The signal is diluted, not amplified.
- **Variance pattern differs.** 7b variance is driven by node count (low node runs score lower). 14b variance is driven by retrieval noise (high node counts introduce wrong-context retrieval).
- **Recommendation: qwen2.5:7b for ingestion.** Lower node count reduces retrieval noise and produces identical average accuracy with more consistent trap question handling. The 14b extraction advantage exists but does not translate to better answers.

---

### Run 21 — 2026-04-12 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| Overall accuracy | ~48/50 = **~96%** |
| False positive rate | 1/9 ⚠️ (U-03 persistent) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |

**Notes:**

First 7b run after concluding qwen2.5:14b was not a net improvement. Returned to qwen2.5:7b for ingestion with Claude Haiku for answers. Detailed per-question breakdown was not captured for this run; full scoring was reviewed but not logged before the session ended.

---

### Run 22 — 2026-04-12 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer

| Metric | Score |
|--------|-------|
| Overall accuracy | 38.5/50 = **77%** |
| Nodes stored | 133 |
| False positive rate | 1/9 ⚠️ (U-03) |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |

**Known failures:** T1-13, T1-14, T1-15, T1-18, T2-18, T2-20, T3-03, T3-05, T3-09 (plus additional failures consistent with the lower node count)

**Root cause — silent ingestion failures:**

17 fewer nodes than Run 23 (150). Two bugs caused qwen2.5:7b to silently drop entire email observations without the calling code knowing:

1. **entities-as-strings** — qwen2.5:7b occasionally returned `entities` as `["Darnell", "Priya"]` instead of `[{"label": "Darnell", ...}]`. `ingestion.py` crashed with `TypeError: string indices must be integers, not 'str'` at the `ent_data["label"]` line — the exception propagated to the executor, the `_done` callback logged `.error()`, and the observation was dropped entirely.

2. **Markdown JSON with trailing explanation** — qwen2.5:7b wrapped JSON in markdown fences AND appended an "Explanation:" section. The fence-stripper removed backticks, but the trailing text caused `json.JSONDecodeError: Extra data` — same silent-drop outcome.

3. **ConnectionError** — `[WinError 10054]` hard-reset mid-benchmark. The retry loop only handled HTTP 429/529 status codes, not `requests.exceptions.ConnectionError`.

Because the entire observation (email) was discarded on these failures, none of the facts from those emails were stored. The 17 missing nodes directly correspond to 9+ question failures — the security firm decision email (T1-13/14/15), the Leo cost update email (T1-18), the Notion/LogQL reference (T2-18), the Sentry SDK detail (T2-20), and the Nightwatch/Conduit overlap nodes (T3-03/05/09).

**Fixes applied before Run 23:**

1. `ingestion.py` — `isinstance(ent_data, dict)` guard skips string entries
2. `llm/client.py` — brace-match regex fallback extracts first `{...}` block from malformed responses (handles markdown + trailing explanation)
3. `llm/client.py` — `ConnectionError` retry with exponential backoff (same logic as 429/529 retry)
4. `agent.py` — retry mechanism: failed `ingest()` re-submitted once before logging as permanent failure

---

### Run 23 — 2026-04-13 · qwen2.5:7b ingest + claude-haiku-4-5-20251001 answer · post-fix

| Metric | Score |
|--------|-------|
| T1 accuracy | 19.5/20 = **97.5%** |
| T2 accuracy | 20/20 = **100%** |
| T3 accuracy | 9/10 = **90%** |
| Overall accuracy | 48.5/50 = **97%** |
| False positive rate | 0/9 ✓ |
| Trap question (U-05) | **PASS** |
| Session persistence | PASS |
| Tokens — ingestion | in=18,361  out=13,148 |
| Tokens — total (recall phase) | in=403,788  out=3,027 |
| Estimated API cost | $0.1047 (Haiku pricing) |
| Nodes stored | 150 |
| Recall p50 | 7,952ms |
| Recall p99 | 13,957ms |

**T1 wrong (0.5):**
- T1-17 (0.5) — "I don't have that information. The archived records mention that 47 users from 6 enterprise customers signed up... but the records do not specify how many of those enterprise customers' users signed up (i.e., the breakdown by customer)." Claude misreads the question as asking for a per-customer breakdown. Correct facts are present but the hedge misinterprets the question intent.

**T2:** All 20 correct. T2-18 (Notion/LogQL) answered cleanly — no hedge this run.

**T3 wrong (1.0):**
- T3-03 (0.5) — "Grafana and Prometheus replaced Datadog"; Alertmanager and Loki omitted. Leo's Conduit role (Kubernetes) correct.
- T3-05 (0.5) — "50 total (47 migrated + 3 new)" correct, but "archived records do not specify how many dashboards existed before the migration, so a direct comparison cannot be made." Model has the 47 figure (used as the migration count) but won't infer Datadog had 47 dashboards before migration.

**U-tier:**
- U-03: **PASS** this run — "lists 8 named employees... but there is no statement of total employee count." Correctly declined; the ingestion bug fixes changed how the header data was stored, making the enumeration less authoritative.
- All other unanswerable questions correctly declined. Trap (U-05) PASS.

**Key findings:**

- **Best post-fix run and second-best across all runs (97%, behind Run 14's 98%).** 150 nodes — full extraction with zero silent drops.
- **Ingestion bug fixes directly explain Run 22→Run 23 recovery (77% → 97%).** The 17-node difference between the two runs corresponds exactly to the pattern of silent ingestion drops.
- **T2 perfect (100%)** for the first time since Run 14.
- **U-03 false positive resolved.** The persistent false positive (enumerating employees as total count) did not fire. The fix to `isinstance` and the retry mechanism changed which nodes were stored for the header email, resulting in a more cautious enumeration response.
- **Remaining persistent issues:** T3-03 (Alertmanager+Loki never surfaces for "replaced Datadog" query — structural retrieval gap); T1-17 hedge (Claude misparses "enterprise customers' users" as a per-customer breakdown request); T3-05 logical inference gap (won't infer Datadog had 47 from "47 migrated from Datadog").
- **ANSWER_SYSTEM hedge fix partially effective.** U-03 no longer triggers the hedge. T1-17 still does — but for a different reason (genuine question misparse, not retrieval overcaution).

**Fixes applied to reduce variance going forward:**

- `agent.py` retry mechanism (complete — already verified working in this run)
- `ingestion.py` 0-entity warning (complete — WARNING logged in `logs/` directory)
- `conftest.py` timestamped log files in `logs/` directory (complete)
