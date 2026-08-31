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

### Round 1

Each of the five agents receives the same question and answer options. Responses
are generated independently: an agent must not see another agent's response
before producing its Round 1 response. The five parsed answers are used to
calculate the Round 1 group vote.

### Round 2

Each agent receives the original question and the anonymised Round 1 responses
of the other four agents. It must not receive its own Round 1 response as a peer
response, and peer model identities must not be disclosed. Each agent answers
the question again after considering the four peer responses. Round 2 answers
are then parsed and voted on separately.

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

- **Status:** Automatic routing for now; **pinning deferred until before the pilot**
- **Decision:** Let OpenRouter select an available upstream provider. Record the
  served model and provider on every response, but never reject a response
  because a different provider served it.
- **Recorded:** 2026-08-28

The current milestone is only to establish a working connection to the five
model IDs. Requests send `provider: {allow_fallbacks: true,
require_parameters: true}`. `require_parameters` is kept so a provider that
cannot apply `temperature 0` is not selected — that is a settings guarantee, not
a pinning rule. No `provider.only` is sent, and the registry does not require a
provider name.

### Why pinning will still be needed before the pilot

Evidence gathered on 2026-08-28 from `GET /api/v1/models/<slug>/endpoints` and
from repeated live smoke runs:

- The same model is served at **different quantisations** by different providers.
  `google/gemma-4-31b-it` alone is offered as bf16, fp8, fp4 and fp16 across
  seventeen endpoints.
- Automatic routing **changes provider between runs**. Across three consecutive
  smoke runs, `agent_qwen` was served by Alibaba, then Chutes, then Reka;
  `agent_gemma` by OpenInference, then CoreWeave, then DeepInfra; `agent_llama`
  by DeepInfra, then DigitalOcean; `agent_deepseek` by Alibaba, then Novita.

For a connectivity check this is harmless. For the experiment it is not: an
unpinned run could change serving stack and numeric precision partway through
the 300 questions, which would confound the comparison the project exists to
make. Pinning must therefore be resolved and recorded before the settings are
frozen, along with the treatment of an unavailable pinned provider.

Two facts already known, to be reused when pinning is revisited:

- The first-party `DeepSeek` endpoint returns HTTP 404, "No endpoints available
  matching your guardrail restrictions and data policy" — an account privacy
  setting. Automatic routing avoids it. Alibaba, StreamLake, DigitalOcean and
  Novita were each observed serving this model successfully.
- `mistralai/mistral-large-2512` is served by Mistral alone, so it has no
  alternative provider and no benefit from pinning. Its availability is a
  separate risk, recorded in D016.

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
