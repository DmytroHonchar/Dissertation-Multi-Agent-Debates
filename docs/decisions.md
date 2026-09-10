# Project Decisions

This document is the source of truth for the current dissertation experiment.
The old three-model `0.x` pipeline is obsolete and may be consulted only as a
technical reference. A change to any fixed decision below must be dated and
recorded before it is used in an experiment.

## D001 — Research comparison

- **Status:** Fixed
- **Decision:** Evaluate whether a heterogeneous multi-agent debate changes or
  improves answer accuracy compared with the agents' independent first-round
  responses on MMLU-Pro.
- **Recorded:** 2026-08-26

The core implementation uses five heterogeneous agents. There is no homogeneous
model condition and no judge model.

## D002 — Selected models and API provider

- **Status:** Fixed for development; freeze again after the pilot
- **Decision:** Use one OpenRouter API key to access the following five exact
  model IDs.
- **Recorded:** 2026-08-26

| Agent | Developer | OpenRouter model ID | Context limit | Input price per 1M tokens | Output price per 1M tokens |
|---|---|---|---:|---:|---:|
| `agent_llama` | Meta | `meta-llama/llama-4-maverick` | 1,048,576 | $0.20 | $0.80 |
| `agent_qwen` | Alibaba | `qwen/qwen3.8-27b` | 1,000,000 | $0.425 | $2.55 |
| `agent_mistral` | Mistral AI | `mistralai/mistral-large-2512` | 262,144 | $0.50 | $1.50 |
| `agent_deepseek` | DeepSeek | `deepseek/deepseek-v4-pro-0813` | 1,048,576 | $1.32 | $3.96 |
| `agent_gemma` | Google | `google/gemma-4-31b-it` | 262,144 | $0.09 | $0.34 |

Context limits and prices were confirmed from OpenRouter's live model registry
on 2026-08-26. They are a dated snapshot and must be checked again immediately
before the main experiment.

### Generation settings — updated 2026-08-28

All five agents use **temperature `0`** and **top-p `1.0`** in both rounds. The
experiment measures how much of any accuracy change comes from aggregating
answers and how much from cross-model communication. Sampling randomness would
add a third source of variation and make that comparison less controlled.
Answer diversity in this design comes from using five different model families,
not from sampling.

Maximum completion length is `1,024` tokens, still provisional. It is not frozen
until the pilot confirms that every model, especially reasoning models, reaches a
parseable final answer within it. The 2026-08-28 smoke test showed Qwen spending
37 and DeepSeek 34 completion tokens to answer a one-word prompt, so reasoning
tokens are billed and consume the budget before any visible answer appears.

Provider pinning is addressed in D015.

## D003 — Core debate protocol

- **Status:** Fixed
- **Decision:** Use two rounds with five agents.
- **Recorded:** 2026-08-26
- **Clarified:** 2026-09-03 — Round 2 restores each agent's own valid Round 1
  response as conversation history, separately from the anonymous peer set.

### Round 1

Each of the five agents receives the same question and answer options. Responses
are generated independently: an agent must not see another agent's response
before producing its Round 1 response. The five parsed answers are used to
calculate the Round 1 group vote.

### Round 2

Each agent receives the original question, its own valid Round 1 response as its
previous conversation turn, and the anonymised valid Round 1 responses of the
other four agents. Its own response is never inserted into the anonymous peer
set, and peer model identities are never disclosed. Restoring the agent's own
reasoning makes Round 2 a deliberate reconsideration of its established answer,
not a fresh response to a larger prompt. If its Round 1 attempt produced no
usable response, that absence is stated without fabricating one. Each agent
answers again after considering the available peer responses. Round 2 answers
are then parsed and voted on separately.

The `runs` table has one run-level `prompt_version` field. A run containing both
rounds records `round1_v1+round2_v1` there. Each response row records only the
prompt version that produced it: `round1_v1` or `round2_v1`.

An additional debate round is desirable future work but is not part of the core
implementation, pilot, or main experiment.

## D004 — Voting and failed responses

- **Status:** Fixed
- **Decision:** A group answer requires at least three matching votes out of the
  five configured agents.
- **Recorded:** 2026-08-26

A failed, missing, empty, or unparseable response contributes no vote. The
threshold remains three even when one or more agents fail; it is never reduced
to a majority of successful responses. If no answer receives three votes, the
group outcome is recorded as no consensus. No judge model or tie-break call is
used.

The database must preserve both the raw response and its parse/failure status so
that failures can be audited rather than silently converted into incorrect
answers.

## D005 — Dataset and separation of pilot data

- **Status:** Frozen
- **Decision:** Use the prepared MMLU-Pro test-split artifacts in
  `data/frozen/mmlu_pro_v1`.
- **Recorded:** 2026-08-04; reaffirmed 2026-08-26

The main experiment contains 300 questions and the pilot contains 20 separate,
non-overlapping questions. Sampling is deterministic, category-stratified, and
uses seed 42. Model inputs and answer keys remain separate; answer keys must not
be loaded into model prompts or debate orchestration.

The 20 pilot questions are used to validate prompts, parsing, storage, retries,
context use, latency, and cost. After fixes from the pilot, prompts, model
settings, parser behaviour, retry rules, and experiment configuration must be
frozen before any of the 300 main questions are run.

## D006 — Call count and budget

- **Status:** Fixed budget for the core two-round design
- **Decision:** Set the maximum API budget to **£15** and the OpenRouter key
  spending limit to approximately **$20**.
- **Recorded:** 2026-08-26

The core pilot and main experiment require:

```text
320 questions × 5 agents × 2 rounds = 3,200 calls
```

At the prices recorded in D002, the expected token scenario costs about £4.67
for the core calls and £6.08 with a 30% testing/retry reserve. A conservative
scenario in which both responses reach the current 1,024-token limit costs about
£10.04, or £13.06 with the same reserve. Repeating the full pilot or increasing
completion limits substantially requires a revised estimate before proceeding.

## D007 — Reproducibility and auditability

- **Status:** Fixed principle; implementation incomplete
- **Decision:** Every model call and derived vote must be reproducible or
  auditable from stored metadata.
- **Recorded:** 2026-08-26

At minimum, stored call records must include the run ID, question stable ID,
round, agent ID, requested model ID, served model/provider, anonymised peer-input
references, raw response, parsed answer, status, token usage, cost, latency,
attempt count, prompt/config version, and timestamps. Cached API responses must
be stored separately from derived experiment results. Evaluation must operate on
the stored database and answer key, not make new model calls.

The outer script owns the database run lifecycle. It starts one run before any
questions are processed and finishes it only after every required question and
round succeeds. `finish_run()` must not run from a `finally` block: if execution
crashes, `ended_at` deliberately remains `NULL`, which marks that run as
incomplete. Clients, caches and database connections are still closed normally;
only the experimental run remains unfinished.

## D008 — Implementation milestones

- **Status:** Fixed development order
- **Recorded:** 2026-08-26

1. **Milestone 1:** Complete and inspect one real pilot question through Round 1,
   including five calls, parsing, storage, and three-of-five voting.
2. **Milestone 2:** Complete and inspect one real pilot question through both
   rounds, with exactly four anonymised peer responses supplied to each agent.
3. Test the complete pipeline with deterministic fake responses and limited real
   calls.
4. Build and validate evaluation before running the full pilot.
5. Run and inspect the 20-question pilot, then freeze all experimental settings.
6. Run the 300-question main experiment without changing frozen settings.

## D009 — Outcome measures

- **Status:** Fixed
- **Decision:** Report three separate accuracy measures on the same 300
  questions, never collapsed into one figure.
- **Recorded:** 2026-08-28 (from the CA1 proposal, Sections 3 and 8.3)

1. Per-agent Round 1 accuracy, calculated over that agent's valid parsed
   answers only.
2. Round 1 group majority-vote accuracy, computed before any communication.
3. Round 2 group majority-vote accuracy, computed after one round of
   communication.

Measure 2 minus measure 1 isolates the gain from aggregation. Measure 3 minus
measure 2 isolates the effect of debate and is the answer to the research
question. This separation is the project's contribution: prior work reports only
the final group figure, which conflates individual model ability, aggregation
and communication. D001 is unchanged; this entry makes its comparison explicit.

Also report per question: whether the answer changed between rounds, consensus
state, token usage, estimated cost and latency, for each round separately.

## D010 — Scoring of no-consensus and failed responses

- **Status:** Fixed
- **Decision:** A no-consensus outcome counts as incorrect in group accuracy and
  is additionally reported as its own category.
- **Recorded:** 2026-08-28 (from the CA1 proposal, Section 8.3)

Group accuracy therefore has a fixed denominator of all 300 questions, so Round 1
and Round 2 group accuracy remain directly comparable even when the two rounds
produce different numbers of decided questions. Unanimous, consensus,
no-consensus and insufficient-answer rates are reported separately alongside it.

Individual agent accuracy is calculated over that agent's valid parsed answers
only. Failure rates are reported next to it, broken down by `REFUSAL`,
`TRUNCATED`, `PARSE_FAIL` and `API_ERROR` and by agent. A failure is never
recorded as a wrong answer. Reporting either figure without the other is
misleading, because an agent that answers one question correctly and fails the
rest would otherwise appear perfectly accurate.

## D011 — Statistical analysis

- **Status:** Fixed
- **Decision:** Compare the paired Round 1 and Round 2 group outcomes on the same
  questions using a confidence interval and McNemar's exact test.
- **Recorded:** 2026-08-28 (from the CA1 proposal, Section 8.3)

Report the difference in group accuracy in percentage points with a 95%
confidence interval for paired proportions. Apply McNemar's exact test to the
discordant pairs to check whether significantly more questions moved from
incorrect to correct than from correct to incorrect.

Classify every question's transition between rounds as stayed correct, became
correct, became incorrect, or stayed incorrect, and report the counts. The same
four-way classification is reported for individual agents over their valid
answers.

### Resolved method — 2026-08-28

- **Primary outcome:** Round 2 group accuracy minus Round 1 group accuracy, in
  percentage points, on the same 300 questions.
- **Confidence interval:** a paired question-level bootstrap. Resample the 300
  questions with replacement 10,000 times; for each resample recompute both
  round accuracies and their difference; report the 2.5th and 97.5th percentiles
  as a **95% percentile confidence interval**. Resampling at question level keeps
  each question's Round 1 and Round 2 outcomes paired, which a two-sample
  interval would break.
- **Significance test:** McNemar's exact test on the paired correct/incorrect
  results. If there are no discordant pairs, report **p = 1**.
- **Transitions:** report all four counts — correct→correct, correct→incorrect,
  incorrect→correct, incorrect→incorrect.

The bootstrap seed must be fixed and recorded so the interval is reproducible.
This method is fixed now, before any results exist, and must not be changed after
seeing them.

## D012 — Retry policy

- **Status:** Fixed and implemented (2026-08-28)
- **Decision:** A temporary API error is retried **once**, and the retry is
  recorded.
- **Recorded:** 2026-08-28 (from the CA1 proposal, Section 13)

`REFUSAL`, `TRUNCATED` and `PARSE_FAIL` are never retried. They are real model
outcomes, not technical faults, and an identical retry stops in the same place.

`api_client.DEFAULT_MAX_ATTEMPTS` was lowered from 5 to **2** on 2026-08-28, so
one initial attempt plus one retry. `tests/test_api_client.py` asserts the value,
and the 2026-08-28 smoke test confirmed it live: the rate-limited Mistral call
reported `failed after 2 attempts`. The documentation was not relaxed to permit
five attempts; the code was corrected to match the proposal.

## D013 — Replay interface

- **Status:** Fixed scope
- **Decision:** A Streamlit page replays one completed debate, read-only.
- **Recorded:** 2026-08-28 (from the CA1 proposal, Sections 3.3 and 9)

It reads the results database and never calls a model, never writes, and never
changes a stored result. It displays: a question selector with subject; the
question and all its options; the five Round 1 responses with each agent's
answer and reasoning; the first majority vote and how it was reached; the five
Round 2 responses each marked changed or unchanged; the final majority vote; the
outcome; and per-question tokens, estimated cost and response time.

The interface must be working by CA2 on 2026-11-06, because the demonstration
uses a real stored debate.

## D014 — Desirable extensions

- **Status:** Desirable only; not part of the core implementation
- **Recorded:** 2026-08-28 (from the CA1 proposal, Section 3.3)

1. One further debate round offered only to questions that still have no
   majority after Round 2, giving those agents a final chance to reach
   consensus. This is narrower than a general third round: it applies to
   undecided questions only. It must not be added before the core two-round
   experiment is complete and reported.
2. Results broken down by MMLU-Pro subject category.

Neither may change the frozen core configuration or the headline comparison in
D009.

## D015 — Provider routing

- **Status:** Pre-pilot pins selected; awaiting Milestone 2 and pilot validation
- **Decision:** Use the exact provider pins in `agents_v5`, with fallbacks off.
  Retry the same endpoint once on a temporary failure; if it remains
  unavailable, store `API_ERROR` and give that agent no vote. Never silently
  change provider during an experimental run.
- **Recorded:** 2026-08-28
- **Resolved for the pilot candidate:** 2026-09-03

Automatic routing was acceptable while building connectivity, but it changed
providers repeatedly in the stored live runs. The pilot candidate therefore
sends `provider.only` with one exact endpoint tag per agent,
`allow_fallbacks: false`, and `require_parameters: true`.

| Agent | Frozen pilot-candidate endpoint | Selection reason |
|---|---|---|
| Llama | `digitalocean` | Three successful stored calls; one endpoint; 99.90% one-day uptime in the selection snapshot |
| Qwen | `parasail/fp8` | Five stored calls supplied the controlled comparisons; exact fp8 endpoint; 99.80% one-day uptime |
| Mistral | `mistral/eu` | Mistral is the only provider; the EU variant had the best one-day uptime of its three variants (96.00%) |
| DeepSeek | `digitalocean` | Two successful stored calls; avoids the first-party endpoint rejected by the account's data policy; 99.57% one-day uptime |
| Gemma | `deepinfra/fp8` | Three successful stored calls; exact fp8 tag prevents a move to DeepInfra's fp4/turbo variant |

Selection used stored reachability first, then the free endpoint metadata
snapshot taken on 2026-09-03: active status, support for temperature and top-p,
one-day availability, explicit endpoint tag/quantisation where published, and
price. The metadata check made no completion calls and spent no tokens. Endpoint
availability changes over time; that is why the dated choice is frozen instead
of dynamically following whichever endpoint later looks best.

The reason for pinning is experimental control, not the belief that one host is
always more accurate. OpenRouter exposed different quantisations and endpoint
variants for the same model, and automatic routing changed provider between
runs. An unpinned 300-question run could therefore change its serving stack
partway through the experiment.

Two limitations remain explicit. First, the first-party DeepSeek endpoint was
rejected by the account's data-policy restrictions during the earlier smoke
test, so the proven DigitalOcean route is used instead. Second, Mistral is the
only company serving `mistralai/mistral-large-2512`: its EU pin fixes the
endpoint variant but cannot provide a different host if Mistral is unavailable.
That availability risk remains under D016.

## D016 — Mistral upstream availability

- **Status:** Open risk; not blocking the current milestone
- **Recorded:** 2026-08-28

`mistralai/mistral-large-2512` intermittently returns HTTP 429, "temporarily
rate-limited upstream", from OpenRouter's shared provider pool, sometimes
followed by HTTP 504. On 2026-08-28 it failed both attempts in two batch smoke
runs, then succeeded on a standalone call twenty seconds later returning
`'Yes.'`. The model is reachable; the shared pool is intermittently saturated,
and back-to-back calls in a batch are more likely to hit it.

Mistral is the only provider serving this model, so there is no alternative to
route to. With one retry, a 429 during the main run costs `agent_mistral` its
vote for that question. Under D004 the three-of-five threshold does not drop, so
lost votes make no-consensus outcomes more likely rather than changing the
majority rule.

To settle before the pilot: measure the real 429 rate over the 20 pilot
questions, and decide whether pacing between calls, a longer backoff, or an
accepted and reported failure rate is the answer. Do not change the model.

Separately noted for the parser (P6): Mistral answered `'Yes.'` to a prompt
asking for the single word `ready`, as it did on 2026-08-26. Weak
instruction-format compliance is a parser concern, not a connection fault.

## D017 — Parser tolerance rules

- **Status:** Implemented, open for review before the pilot freeze
- **Decision:** `parser_v1.py` accepts a bare letter on a `FINAL ANSWER:` line
  and rejects everything else.
- **Recorded:** 2026-08-31

P6 in `docs/pipeline.md` fixed the rules the parser must follow. Two cases it did
not name came up while implementing it, and both are settled here.

**Markdown asterisks are ignored.** Asterisks are stripped from a copy of the
reply before matching, so `**FINAL ANSWER:** B`, `**FINAL ANSWER: B**` and
`FINAL ANSWER: **B**` all parse as `B`. These models emit markdown constantly,
and treating bold text as a parse failure would discard valid answers and
inflate the failure rate. The raw text is stored unmodified; only the copy used
for matching is stripped. Brackets and parentheses are still rejected, as P6
requires: `FINAL ANSWER: [B]` is not an answer.

**Trailing punctuation after the letter is allowed, word characters are not.**
`FINAL ANSWER: B.` and `FINAL ANSWER: B, because ...` parse as `B`. The letter
must be bare, so `FINAL ANSWER: Berlin`, `FINAL ANSWER: BC`, `FINAL ANSWER: B2`
and `FINAL ANSWER: B_x` are all rejected. The lookahead is `(?!\w)`, which
covers letters, digits and underscore; an earlier `(?![A-Za-z])` accepted `B2`
as `B` and was wrong.

**Refusal detection is deliberately conservative, and text evidence is ranked
below a provider signal.** A reply is classified `REFUSAL` on an explicit
provider signal (`content_filter`, or a refusal flag passed in), or on one of
eleven fixed phrases of the form "I cannot answer". Words like "cannot" and
"unable" occur constantly inside legitimate reasoning, and a loose list would
silently convert correct answers into failures.

The phrase search is not sentence-aware, so a phrase can appear in a reply that
does answer: `I cannot answer A, so FINAL ANSWER: B`. This **narrows P6's rule
that refusal takes priority over a present letter.** A provider signal still
takes priority over any letter. A refusal *phrase in the text* is only trusted
when no valid letter was extracted. Text is weak evidence, and discarding a real
answer because of a substring is worse than missing a refusal — a missed refusal
becomes `PARSE_FAIL`, which is still not scored as a wrong answer.

**Truncation guessed from text requires a long reply.** Where the provider
returns a finish reason, that decides it, and the note records the reason
actually returned (`length` or `max_tokens`). Where no finish reason comes back,
truncation is guessed only when the reply is at least 200 characters *and* does
not end in sentence punctuation. Punctuation alone is weak evidence: it made
`Yes` truncated and `Yes.` a parse failure, which is noise, not a measurement.
Short replies are `PARSE_FAIL`. Since failure categories are reported
separately, a wrong guess puts a failure in the wrong column of the table.

The phrase list and the 200-character floor must both be checked against the
pilot transcripts before the freeze.


## D018 — Token limits after the Milestone 1 truncations

- **Status:** Pre-pilot values selected; awaiting Milestone 2 and pilot validation
- **Decision:** The `agents_v5` candidate uses Llama 1024, Qwen 3072, Mistral
  1024, DeepSeek 2048 and Gemma 1024 completion tokens. Qwen also requests a
  2048 reasoning-token maximum, recorded as best-effort because Parasail did
  not enforce it exactly. Selection uses completion, truncation and cost—not
  answer correctness.
- **Recorded:** 2026-09-01

The live Milestone 1 run (`milestone1_20260901T161852Z`, settings `agents_v1`)
found Qwen and DeepSeek spending the entire 1024-token output budget on internal
reasoning: `finish_reason=length`, `content=null`, no visible answer, no vote.
Those two calls cost $0.0072 of the run's $0.0082 — 88% of the spend bought
nothing. Truncation is therefore the most expensive possible outcome, and the
ceiling exists to prevent runaway, not to ration ability.

**How the final limits are chosen.** Probe 2–3 pilot questions per candidate.
If any response ends with `finish_reason=length`, the ceiling is a lower bound,
not a measurement — move up one step (2048 → 3072 → 4096) under a new settings
version. Once everything finishes, take the largest completed token count, add
roughly 25% margin, round up to the next 256. The chosen limits are used
unchanged in both rounds and frozen with the rest of the configuration.
Selection uses truncation, valid-answer rate and cost only. Pilot accuracy is
20 questions — one question is five percentage points — so choosing settings by
it would be fitting noise, and tuning the configuration on the outcome measure
is exactly what this project criticises elsewhere.

**Why not `reasoning.effort: low`.** Lowering a model's reasoning effort
changes the model being studied: the experiment's diversity comes from five
model families as they actually behave, and results would describe
"Qwen forced to think less", not Qwen. Deferred unless larger ceilings prove
pathological (a model burning 4k+ tokens by default) or unaffordable. If it is
ever used, it becomes a new settings version and a recorded decision, and a
per-model `reasoning.max_tokens` request is preferred over effort levels. It is
intended to leave space for the visible answer, but endpoint metadata and an
accepted request do not guarantee exact enforcement; the total `max_tokens`
ceiling remains the hard termination and cost control.

**Verified 2026-09-01** via OpenRouter's free `/api/v1/models` endpoint (no
completion called): both `qwen/qwen3.8-27b` and `deepseek/deepseek-v4-pro-0813`
list `reasoning` and `reasoning_effort` in `supported_parameters`. This shows
the routing layer accepts the parameters; whether the specific serving provider
honours `reasoning.max_tokens` is **not** proven by this and ties into provider
pinning (D015). Do not assume it works until a pinned provider demonstrates it.

`agents_v1` stays byte-for-byte untouched: run `milestone1_20260901T161852Z`
identifies itself by that name and its provenance must stay true. The first
probe candidate is `configs/models/agents_v2.yaml`: identical except Qwen and
DeepSeek at `max_tokens: 2048`, clearly marked provisional.


## D018 addendum — the controlled reasoning-cap test (2026-09-02)

- **Status of D018:** corrected on 2026-09-03; pre-pilot values chosen
- **Test:** run `round1_agents_v4_20260902T224109Z`, maths question
  `mmlu_pro_v1:test:8844`, settings `agents_v4`: Qwen pinned to Parasail (the
  provider that served both prior failures), total 3072 unchanged from
  `agents_v3`, hidden reasoning capped at 2048 via `reasoning: {max_tokens}`.
  One deliberate change from `agents_v3`; question and provider held fixed.

**Initial result:** Qwen answered with `finish_reason=stop`, visible
`FINAL ANSWER: D`, parsed `OK`, and $0.0028 cost—against the $0.0100 uncapped
3072 truncation on the identical question and provider. It used only 578
reasoning tokens of the requested 2048 maximum. This showed that the capped
request could complete, but it did **not** prove enforcement because the model
never approached the requested boundary.

Run `round1_agents_v4_20260902T225035Z` then disproved the enforcement claim.
On malformed physics question `mmlu_pro_v1:test:9622`, Parasail reported 2740
Qwen reasoning tokens despite the requested 2048 maximum. Qwen still completed
at 2996 of 3072 total tokens, but `reasoning.max_tokens` cannot be treated as a
hard partition on this endpoint. The earlier short run may reflect an influence
from the declared budget or ordinary hosted-model variation; one run cannot
identify the mechanism.

The physics item is not used to inflate ceilings. Independent unit conversion
gives `3.8×10^7` dynes, absent from all ten options, and the stored Mistral and
Gemma replies visibly repeat the calculation while trying to force an available
choice. DeepSeek spent all 2048 tokens in hidden reasoning. These are valid
recorded failures on a malformed input, not evidence that every normal
completion ceiling should rise.

On the clearly well-formed chemistry question `mmlu_pro_v1:test:3932`, run
`round1_agents_v4_20260902T231300Z` produced five valid answers and unanimous D.
Completion use was Llama 275/1024, Qwen 537/3072 (268 reasoning), Mistral
291/1024, DeepSeek 436/2048 (262 reasoning), and Gemma 499/1024. Together with
the earlier clean philosophy run, this gives at least twofold headroom for
every agent on ordinary completed questions. The 20-question pilot, not more
one-question tuning, now measures the real truncation rate.

Consequences: ceiling escalation stops. `agents_v5` keeps Qwen's best-effort
2048 reasoning request because the identical maths request completed once with
it and the request adds no tokens or cost by itself; the documented 3072 total
ceiling is the control that can be relied upon. Provider pins and reasoning
settings are part of the cache key, so incompatible requests cannot share a
cached response.

## D019 — What counts as a valid Round 1 response in Round 2

- **Status:** Fixed and implemented (2026-09-07)
- **Decision:** Valid means `status == "OK"` and nothing else.
- **Recorded:** 2026-09-07

Only a valid Round 1 response may be shown to another agent as a peer, and only
a valid one is restored as an agent's own preceding turn. `REFUSAL`,
`TRUNCATED`, `PARSE_FAIL` and `API_ERROR` are all invalid.

The reason is the same one behind D010: a failure is an absence, not a position.
Showing four models a truncated half-argument, or a refusal, as though it were a
peer's reasoning would put text in front of them that the parser has already
judged unusable, and any answer change it caused would be unattributable.

Two consequences follow, and neither is treated as an error:

- An agent whose own Round 1 response was invalid still takes part in Round 2.
  It receives no assistant turn, and the prompt says its previous attempt
  produced no usable response rather than inventing one.
- An agent whose four peers all failed still takes part, is told no valid peer
  responses are available, and is stored with `peer_count = 0`.

Dropping either agent would change which questions the two rounds are compared
on, which is exactly the comparison D009 depends on. `peer_count` is stored per
response because the number of peers actually seen is a confound the results
chapter must report.

## D020 — Two-round run labels and peer ordering

- **Status:** Fixed and implemented (2026-09-07)
- **Decision:** A run containing both rounds is named `debate_config_v1`, and
  peer order is registry order with the agent itself removed.
- **Recorded:** 2026-09-07

A run that performs both rounds is not a Round 1 run, so it is not labelled
`round1_config_v2`. Its `runs` row records `config_name = "debate_config_v1"`
and `prompt_version = "round1_v1+round2_v1"`. Each response row still records
only the prompt that produced it, `round1_v1` or `round2_v1`. Both rounds refuse
to process a question whose run row does not carry these exact labels, so a run
can no longer be labelled with one recipe and executed with another.

Peer order is the configured registry order with the agent itself removed. It is
deterministic, which is what allows a stored run to be reconstructed exactly.

No identity is disclosed by it: a model sees only `PEER RESPONSE 1` to
`PEER RESPONSE 4` and is told the order carries no meaning. Positions are not
fixed per agent, because removing the answering agent shifts everyone after it
up by one - `agent_qwen` is peer 1 for `agent_llama` and peer 2 for
`agent_mistral`. A fixed order may still carry a small positional effect on how
the peers are read. That is accepted as a known property of the design and
recorded here, not claimed to be harmless. The pilot is where any sign of it
would first show.

## D021 — The three evaluation choices left open by D011

- **Status:** Fixed and implemented (2026-09-10)
- **Decision:** Bootstrap seed `20260828`; exact McNemar from the standard
  library, not `scipy`; the aggregation gain reported against both the mean and
  the best agent.
- **Recorded:** 2026-09-10

**Seed.** `20260828`, the date D011 was recorded. Fixed and recorded before the
formal pilot and the main experiment, which is what D011's rule protects. Ten
development runs already exist in `storage/results.sqlite`, and none of them
influenced this choice: the seed is a date, not a value picked from a result.

**No statistics dependency.** P12 said the exact McNemar test needs `scipy` or
`statsmodels`. That is not true for this test. McNemar's exact test is a
two-sided binomial test on the discordant pairs with p = 0.5, and a symmetric
binomial is exact from `math.comb`:

    p = min(1, 2 * sum(C(n, i) for i in 0..min(b, c)) / 2^n)

This matches `scipy.stats.binomtest(b, b + c, 0.5)` on the same input. Adding
scipy would put a large compiled dependency into a project whose entire test
suite runs offline, to compute a sum of binomial coefficients. P12 is corrected.

**The aggregation gain has two baselines, and both are reported.** D009 measure
1 is per-agent accuracy - five numbers - and measure 2 is one group number, so
"measure 2 minus measure 1" was ambiguous. It is now:

- **versus the mean agent** - the average gain over picking one agent blindly.
- **versus the best agent** - whether the group beats its strongest member.

Reporting only the first flatters aggregation, because a group almost always
beats its weakest members. Li et al. found mixed groups usually failed to beat
the strongest single model, so the second is the comparison that can actually
falsify the aggregation claim. All five individual accuracies and failure rates
are reported alongside both, so neither figure can be read on its own.

**Answer-key pairing.** `question_set_version` is `mmlu_pro_v1` for both the
pilot and the experimental set, so it cannot tell them apart. Evaluation instead
requires every question in the run to appear in the supplied answer key, which
refuses a pilot run paired with the experimental key and the reverse, since the
two sets do not overlap.
