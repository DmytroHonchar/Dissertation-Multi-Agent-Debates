# Build Pipeline

How each stage of the experiment is built. This document is the **technical
reference**: it says *how* to implement a stage.

Precedence, so there is exactly one answer to any question:

| Document | Answers | Authority |
|---|---|---|
| `docs/decisions.md` | *What is decided and frozen* | Highest. If this file disagrees with it, this file is wrong. |
| `docs/pipeline.md` (here) | *How to build each stage* | Implementation detail only. |
| `docs/checklist.md` | *What order, and what is done* | Progress tracking only. |
| `docs/development_log.md` | *What actually happened* | Append-only history. |

This pipeline replaces the obsolete three-model `0.x` version, which assumed
GPT/Claude/Gemini through three separate SDKs and a two-of-three vote. Its
stage-by-stage engineering detail was kept; every count, provider and threshold
was updated to the fixed five-agent design in `decisions.md`.

Stages marked **DONE** are built and tested. Stages marked **OPEN** contain a
decision that must be recorded in `decisions.md` before the pilot is frozen.

---

## P1 — Question set — DONE

Implemented in `src/mad/benchmark.py`, frozen in `data/frozen/mmlu_pro_v1`.

MMLU-Pro test split, revision pinned in `configs/benchmark/mmlu_pro_v1.yaml`.
Every question carries a stable ID assigned before filtering. Questions were
validated (non-empty text, non-empty options, no duplicate options, correct
letter maps to a real option); one invalid question was removed and the removal
is recorded in `metadata/validation_report.json`.

Seed 42, category-stratified: 300 experimental questions and 20 non-overlapping
pilot questions. Model inputs and answer keys are stored in separate files with
SHA-256 checksums. Prompt-building and calling code receives question ID, text
and options only — never the answer. Guarded by `tests/test_no_answer_leakage.py`.

**Do not regenerate.** The frozen set is final.

## P2 — Round 1 prompt

New file `src/mad/prompts_v1.py`, carrying `PROMPT_VERSION = "round1_v1"`.

The version label covers the system instructions, the prompt wording, and the
question-formatting function together. Change any one of them and the version
changes.

The prompt instructs the model to reason, then to end with `FINAL ANSWER: X` on
its own line, where `X` is exactly one of the letters offered by that question,
chosen even when the model is uncertain.

One formatting function renders the stored question as its text followed by
every option in stored order. MMLU-Pro questions have up to ten options, so the
option count must be read from the question — never hardcoded to four.

The same wording goes to all five agents. Round 1 must not mention other
agents, a later round, review, or the correct answer.

Record `PROMPT_VERSION` on every stored response.

## P3 — Generation settings

Shared defaults live in `configs/models/agents_v1.yaml`: **temperature `0`**,
**top-p `1.0`**, `max_tokens 1024`, `allow_provider_fallbacks: true`,
`require_parameters: true`.

Temperature 0 is fixed (D002): the study measures the effect of aggregation and
of communication, and sampling noise would add a third source of variation.
Diversity comes from five different model families, not from sampling.
`require_parameters` keeps routing to providers that actually honour those
settings, so temperature 0 cannot be silently dropped.

**The `agents_v5` pre-pilot candidate pins one exact endpoint per agent with
fallbacks off (D015).** The served model and provider are still recorded on
every response. If a pin is unavailable, the existing retry policy tries the
same endpoint once and the failure is stored as `API_ERROR`; an experimental
run never silently changes host or quantisation.

Still open before the pilot freeze:

- **Provider pins** (D015) — selected in `agents_v5`; validate their reachability
  and failure rate in Milestone 2 and the pilot.
- **Mistral availability** (D016) — measure the real HTTP 429 rate during the
  pilot and decide how to handle it.
- **`max_tokens` per round.** The `agents_v5` candidate is Llama 1024, Qwen
  3072, Mistral 1024, DeepSeek 2048 and Gemma 1024. Qwen also requests a
  best-effort 2048 reasoning maximum, but a Parasail response exceeded it, so
  only the total ceiling is treated as hard. Round 2 carries four peer
  responses in its input; confirm the same output ceilings in Milestone 2 and
  the pilot rather than tuning again on individual Round 1 questions (D018).
- **Bootstrap seed** for P12, so the confidence interval is reproducible.

Do not rely on the API `seed` parameter. Support varies and the experiment does
not depend on it. Record the limitation that identical settings still do not
guarantee byte-identical outputs from a hosted API.

## P4 — Results database — DONE

`src/mad/database.py`. SQLite, no server. Four tables.

`runs` — one row per run: run ID, configuration name, question-set version,
prompt version, settings version, parser version, start time, end time.

`model_responses` — one row per `(run, question, round, agent)`: run ID, question
ID, round, `agent_id`, requested slug, served slug, provider, generation ID, raw
response text, extracted letter, extraction method, status, attempt count,
selected attempt, finish reason, prompt version, temperature, `top_p`,
`max_tokens`, prompt tokens, completion tokens, cost, latency, cache-hit flag,
timestamp. For Round 2, also the anonymised references to the four peer
responses supplied to that agent (D007).

`response_attempts` — one row per API attempt, linked to its parent
`model_responses` row: attempt number, raw response, extracted letter, status,
finish reason, tokens, latency, cache-hit flag.

`question_outcomes` — one row per `(run, question, round)`: consensus state,
consensus answer, whether decided, valid-answer count, total tokens, total
latency. Round 1 and Round 2 outcomes are separate rows, because comparing them
is the research question.

Rules: store the full raw response before parsing, never only the letter. Never
overwrite a row — a repeat means a new run ID. The correct answer never enters
this path; only evaluation reads the answer key.

Built 2026-08-31. `ResultsDatabase(path)` creates `storage/results.sqlite` and
its parent directory on first open, turns foreign keys on, and stamps
`PRAGMA user_version` with `SCHEMA_VERSION`; a file written by a different
schema version is refused rather than migrated. Rules that could be enforced
structurally are: `UNIQUE (run_id, question_id, round, agent_id)` and the
`question_outcomes` primary key make a duplicate insert fail instead of
overwrite; a `CHECK` permits `extracted_letter` only on an `OK` row, so a
failure cannot carry a vote; `round = 2 OR peer_count = 0` keeps Round 1 free of
peers; and no table has a column that could hold a correct answer, which a test
asserts by name. `record_response()` writes a response and all of its attempts
in one transaction, so a half-written pair cannot understate what the run cost.
`finish_run()` is the only `UPDATE` in the module — `ended_at`, once — and a
test parses the module's SQL to keep it that way.

## P5 — Response cache — DONE

New file `src/mad/cache.py`. Separate store from the results database. It exists
only to avoid paying twice; it is not part of the experimental record.

Every call passes through it. The key hashes: model slug, the full message list,
temperature, `top_p`, `max_tokens`, and `agent_id`. `agent_id` is included even
though it is not sent to the API, so two agents issuing an identical request
cannot share one response. Round and peer content need no separate key field —
they are already inside the message list. The run ID is excluded, so responses
are reusable across runs.

Store the complete raw API body, not just the text, and store the original
latency — return that on a hit, not the lookup time.

Cache genuine model outcomes, including refusals and unparseable replies. Never
cache transport, rate-limit or provider errors; caching those would make a
temporary failure permanent.

Provide a `--no-cache` flag for deliberately measuring non-determinism.

Built 2026-09-01 as `src/mad/cache.py`. One table keyed by a SHA-256 of
agent_id, slug, messages, temperature, top_p and max_tokens; the complete raw
body is stored and a hit rebuilds the reply with original tokens and latency at
zero cost, making no API call. First write wins. The cache refuses fixture
replies and results without a raw body. The runner does lookup → call on miss →
parse → results database → cache, in that order: the paid call reaches the
audit record before the cache, so a crash between the two can leave an
unrecorded row in neither place, never a cached reply whose spend vanished from
the record. A cache-hit row stores no attempt rows — that run made no API
attempt; its `attempt_count` of 1 names the original call behind the cached
body and `cache_hit=1` marks the difference. The runner records `cache_hit` per
response and refuses a config whose `cache_enabled` disagrees with reality.
Config version `round1_config_v2` labels the cache-capable runner; `v1` was the
cacheless Milestone 1 recipe and stays on that run. The CLI enables the cache for
live runs only and honours `--no-cache`.

## P6 — Parser and failure statuses

New file `src/mad/parser_v1.py`, one parser for all five agents.

Input: raw text, the valid letters for that question, the provider finish
reason, and any refusal signal.

Search the whole response for `FINAL ANSWER: X`, case-insensitively, tolerating
extra spaces. On multiple matches use the last, and record the extraction method
as `LAST_FINAL_ANSWER_MATCH`. Accept only a bare letter belonging to that
question's real options; uppercase it. `FINAL ANSWER: B` is accepted,
`FINAL ANSWER: [B]` is not. Text after the answer line is allowed. Never infer
an answer from the reasoning.

Statuses: `OK`, `REFUSAL`, `TRUNCATED`, `PARSE_FAIL`, `API_ERROR`.

An empty response is `PARSE_FAIL`. A provider signal — `content_filter` for
refusal, `length` or `max_tokens` for truncation — takes priority even when a
letter is present. Evidence read out of the text itself does not: a refusal
phrase or a mid-sentence ending only counts when no valid letter was extracted.
See D017, which narrowed this after `I cannot answer A, so FINAL ANSWER: B` was
being classified as a refusal. Identify truncation from the provider finish
reason first; inspect text only when no finish reason is returned, and only when
the reply is long enough for the guess to mean anything.

**Retry (D012, settled).** One initial attempt plus exactly one retry — two in
total. This lives in `OpenRouterClient._post_with_retries`, where
`DEFAULT_MAX_ATTEMPTS = 2`. Do not add a second retry loop on top of it; the
parser and orchestrator add none. `REFUSAL`, `TRUNCATED` and `PARSE_FAIL` are
never retried: they are real model outcomes, and an identical retry stops in the
same place. An upstream provider error that survives both attempts is recorded as
`API_ERROR`, never as a wrong answer.

Summary selection for `model_responses`: the first `OK` attempt; if none, the
last failed attempt. Record which attempt was selected and how many were made.

Count tokens and cost from every live attempt, not only the selected one —
failures cost money. Cache hits keep their stored token counts but add no spend.

Never count a failure as a wrong answer. Always report accuracy among
successfully answered questions *together with* the failure rate broken down by
status and by agent.

Test before the pilot against: valid answers, lowercase, extra spaces, bracketed
letters, multiple answer lines, trailing text, letters outside the option set,
missing answers, empty responses, refusals, truncation, and API errors.

## P7 — Shared calling layer — DONE

`src/mad/api_client.py`. One OpenRouter key reaches all five models, so the old
three-SDK normalisation problem no longer exists.

`OpenRouterClient.complete()` returns a `CompletionResult` carrying text,
`agent_id`, requested slug, served slug, provider, generation ID, finish reason,
prompt and completion tokens, cost, latency and attempt count — the provenance
D007 requires. It also carries `raw_response`, OpenRouter's complete reply body
for P5 to cache, and `attempt_log`, an `AttemptRecord` per attempt so the tokens
and cost of a failed first attempt are not lost. A call that never succeeds
raises `ApiRequestError` with the same `attempt_log` attached — including for a
malformed HTTP 200 body, whose final attempt is relabelled `malformed_body` —
so a dead call is still costable. On a stored response row, tokens and cost sum
over every attempt (P6); the per-attempt split lives in `response_attempts`. `load_model_registry()` builds a `ModelSpec` per agent from
`configs/models/agents_v1.yaml`. Verified against all five models by
`scripts/check_models.py` on 2026-08-26.

Still to add when the orchestrator is built: fixed call order — cache lookup,
API call on miss, parse, store — five parallel calls per question per round, and
per-call exception isolation so one agent's failure cannot end the run.

## P8 — Round 1 configuration — DONE

Versioned config, currently `CONFIG_VERSION = "round1_config_v2"`, recording the
question-set version, the five model slugs and their `agent_id`s, prompt
version, settings version, parser version, cache on/off, timeout, retry policy,
and whether calls run in parallel.

The agent-to-model mapping is fixed for the whole experiment.

Round 1 has no communication between agents: same question and options to all
five, answered independently, no agent sees another's answer, no agent is told a
second round follows.

API keys stay in the ignored `.env` only — never in configs, the database, git,
prompts or results.

Built 2026-09-01 as `src/mad/round1.py`; its command is now
`scripts/run_round1.py`.
`Round1Config` records the question set, Round 1 prompt, selected model settings,
parser, cache state, timeout, two-attempt retry budget and sequential calling.
`run_round1_question()` requires an existing unfinished run and processes one
question only: up to five calls, parse, store with attempts, tally with
`expected_agents=registry.keys()`, and store the outcome. It neither starts nor
finishes the run. The outer script starts one run before processing questions
and finishes it only after every requested stage succeeds, allowing one run to
contain 1, 20 or 300 questions. A crash deliberately leaves `ended_at` as
`NULL`. One agent failing is stored as `API_ERROR` and the other four continue.
The Round 1 command defaults to a free dry run on labelled fixture replies in
a throwaway database; live mode needs both `--live` and
`--yes-spend-real-money`, accepts only an ID from the 20-question pilot file,
refuses experimental IDs by name, and a dry run is refused
`storage/results.sqlite` so fixture rows can never sit next to real results.

## P9 — Voting — DONE

Fixed rule (D004): a group answer requires **at least three matching votes out
of the five configured agents**. A failed, missing, empty or unparseable
response contributes no vote, and the threshold stays at three — it is never
reduced to a majority of the successful responses. No judge model, no tie-break.

Recorded consensus states, all derived from that single rule:

| State | Condition |
|---|---|
| `UNANIMOUS` | five valid answers, all identical |
| `CONSENSUS` | some answer holds three or four votes |
| `NO_CONSENSUS` | three or more valid answers, none reaching three votes |
| `INSUFFICIENT_ANSWERS` | fewer than three valid answers, so three votes is unreachable |

`INSUFFICIENT_ANSWERS` is a reporting label, not a second rule — it separates
"the agents disagreed" from "too many agents failed", which matter differently
in the results chapter. Both are undecided.

Store the state, the consensus letter where one exists, whether the question was
decided, and the valid-answer count — separately for Round 1 and Round 2.

Built 2026-08-31 as `src/mad/voting.py`. `tally()` takes a mapping of
`agent_id` to `ParsedResponse` and returns a frozen `VoteOutcome` carrying the
state, the consensus letter or `None`, whether it was decided, the valid-answer
count, the per-letter vote counts for inspection and replay, and which agents
failed. It refuses any group that is not exactly five identified agents, so the
threshold can never be quietly rebased on however many responses arrived.
Passing `expected_agents` additionally checks that the five *are* the configured
five; without it, only the count and the identifiers are guaranteed. The runner
should pass `load_model_registry()`'s keys — voting does not read the registry,
because it has no business knowing which models the agents are. Only a
response the parser marked `OK` votes. No judge, no tie-break: two letters
cannot both reach three of five, so a tie is already below the threshold. The
module imports the parser and nothing else — it cannot read a key, score an
answer or write a row, and a test asserts that by parsing its imports.

**Scoring (D010).** An undecided question counts as **incorrect** in group
accuracy, and is also reported as its own category. Group accuracy therefore
always has a denominator of all 300 questions, which is what keeps Round 1 and
Round 2 comparable when the two rounds decide different numbers of questions.
Do not silently drop undecided questions from the denominator — that would
inflate whichever round failed more.

## P10 — Round 2 — DONE

New file `src/mad/debate.py`. This stage did not exist in the old pipeline.

`src/mad/prompts_v1.py` carries `ROUND2_PROMPT_VERSION = "round2_v1"` and
`build_round2_messages()`. Each agent receives the original question and options,
its own valid Round 1 response as the preceding assistant turn, and the
anonymised valid Round 1 responses of the **other four** agents. Its own response
never enters the peer set, and peer model identities are never disclosed. If the
agent had no usable Round 1 response, the prompt states that without inventing
one. Peer ordering must be deterministic given the run and question, so the run
can be reconstructed.

The run-level prompt version is `round1_v1+round2_v1`, because one two-round run
uses both prompts. Each stored response keeps its own exact round-specific prompt
version.

Each agent answers again in the same `FINAL ANSWER: X` format. Round 2 answers
are parsed and voted on separately under the same three-of-five rule.

A Round 1 failure means that agent contributes no peer response, so some agents
may see fewer than four. Record how many peer responses each agent actually
received — it is a confound the results chapter must report.

A third round is desirable future work, not part of this implementation. If it
is ever built, D014 restricts it: it is offered only to questions still without a
majority after Round 2, not to every question, and it must not alter the frozen
core configuration or the D009 headline comparison.

Built 2026-09-07 as `src/mad/debate.py`, alongside a new `src/mad/runner.py`
holding what both rounds share: the run guard, the per-agent call-parse-store-
cache step, the labelled fixture client and the report shape. Writing that step
twice would let the rounds drift apart in how they retry, store or cache, and
the results would then measure that difference instead of the debate.

`Round2Config` records `debate_config_v1`, the question set, `round2_v1`, the
selected model settings, the parser, cache state, timeout, retry budget and
sequential calling. `run_round2_question()` requires an existing unfinished run
labelled for a debate and a complete stored Round 1 for that question; it
refuses before any API call if either is missing. Per agent it reads the stored
Round 1 rows, restores that agent's own valid response as an assistant turn,
supplies the other agents' valid responses anonymously in registry order, calls,
parses, and stores a `round = 2` row carrying `peer_response_ids` for exactly the
rows shown, in the order shown. Validity is `status == "OK"` only (D019). One
agent failing is stored as `API_ERROR` and the other four continue. The vote is
the same three-of-five rule, recorded as a separate `round = 2` outcome row.

`run_debate_question()` sequences Round 1 and Round 2 for one question and
checks both stage configurations before the first call. The command
`scripts/run_debate.py` starts one database run, executes that function, and
finishes the run only after both rounds succeed. It defaults to free fixture
replies and `agents_v5`; live mode requires both `--live` and
`--yes-spend-real-money` because one uncached question can make ten paid calls.

## P11 — Pilot

The 20 frozen pilot questions, both rounds, full pipeline.

Confirm by hand: all five agents received identical Round 1 input; no prompt
ever contained the correct answer; each Round 2 conversation restored that
agent's own valid response separately and contained the available responses of
the other four agents anonymously; letters were extracted correctly; raw
responses were preserved; served model, provider,
tokens, cost, latency and finish reason were stored; cache hits do not reach the
API; an induced API error does not end the run; five parallel calls do not
interfere; every row links to the right run, question, round and agent.

Check `TRUNCATED` rates specifically — the smoke test already warned about
reasoning models. If truncation is common, raise `max_tokens`, bump the settings
version and rerun the pilot. Resolve this before the main run.

Re-estimate cost from real pilot token usage against the £15 budget (D006).

Then **freeze** prompts, model settings, parser rules, retry policy and
configuration, and record every final version name in `decisions.md`.

Pilot results are development data. They never appear in the final results.

## P12 — Main run and evaluation — EVALUATION DONE

New file `src/mad/evaluation.py`, built and validated on pilot data *before* the
main run. It reads the stored database and the separate answer key. It never
makes a model call, and the answer key never returns to prompt-building or
calling code. It is the only module permitted to open an answer key.

The main run processes all 300 frozen questions through both rounds under the
frozen configuration, with a fresh run ID. Nothing changes mid-run. On
completion, record the end time, mark the run complete, and back up the database.

### The three headline measures (D009)

Computed on the same 300 questions and never collapsed into one figure:

1. **Per-agent Round 1 accuracy**, over that agent's valid parsed answers only.
2. **Round 1 group vote accuracy** — aggregation, no communication.
3. **Round 2 group vote accuracy** — after one round of communication.

(2) − (1) isolates aggregation. (3) − (2) isolates debate and answers the
research question.

### Statistics (D011)

Fixed in D011 on 2026-08-28, before any results exist.

**Primary outcome:** (3) − (2), in percentage points, on the same 300 questions.

**Confidence interval — paired question-level bootstrap.** Resample the 300
questions with replacement 10,000 times. For each resample recompute both round
accuracies and their difference. Report the 2.5th and 97.5th percentiles as the
95% interval. Resample whole *questions*, never rounds independently: each
question's Round 1 and Round 2 outcomes must stay paired. Fix and record the
bootstrap seed so the interval is reproducible.

**Significance:** McNemar's exact test on the paired correct/incorrect results.
If there are no discordant pairs, report **p = 1** rather than an error.

Report the four-way transition counts between rounds:

| | Round 2 correct | Round 2 incorrect |
|---|---|---|
| **Round 1 correct** | stayed correct | became incorrect |
| **Round 1 incorrect** | became correct | stayed incorrect |

The off-diagonal cells are the substance of the dissertation: how often debate
repaired a wrong group answer, and how often it broke a right one. Produce the
same table for individual agents over their valid answers.

### Also report, per round

Per-agent failure rate split by `REFUSAL`, `TRUNCATED`, `PARSE_FAIL` and
`API_ERROR`; unanimous / consensus / no-consensus / insufficient-answer rates;
token usage; actual cost against the £15 budget; wall-clock time and per-call
latency.

**Desirable (D014):** the same breakdown per MMLU-Pro subject category. Build it
only after the core results exist.

Dependencies: none. The earlier note here said the exact McNemar test needs
`scipy` or `statsmodels`. It does not: at p = 0.5 the test is a symmetric
two-sided binomial, exact from `math.comb`, and the bootstrap needs only
`random`. D021 records the correction, and the project stays dependency-free
with every test offline.

Built 2026-09-10 as `src/mad/evaluation.py`. `evaluate_run()` takes a database,
a run ID and an answer key and returns one `EvaluationReport` holding every
number: per-agent accuracy and failures for both rounds, group accuracy and
consensus states for both rounds, the group transition table and one per agent,
the aggregation gain against the mean and the best agent, the debate effect, the
bootstrap interval, the McNemar result, per-round usage and spend against the
budget. It refuses before computing anything if the run is unknown, unfinished,
missing a round, scoring different questions in each round, short of a response
from any agent on any question, or carrying a question the supplied key does not
cover - the last check is what stops a pilot
run being scored against the experimental key, since the two frozen sets are
disjoint. Seed `20260828`, 10,000 resamples (D021). No printing lives here, so
a later command and the replay page cannot disagree about a number.

Two reporting rules that are easy to get wrong. A cache hit keeps
`attempt_count = 1` from the original paid call, so it is excluded from
`api_attempts` and counted as a cache hit instead; the debate run that served
all five Round 1 responses from cache made zero calls, not five. And the budget
reports two numbers - this run's cost, and every run recorded in the database -
because subtracting one run from the whole £15 would report a balance the key
does not have. The cumulative figure is a floor, not a bank balance: a run in
another database, or a call whose row never landed, is spending no row can see.

Pass `expected_questions` for a formal run, 20 for the pilot and 300 for the
main experiment. A run that lost questions is then refused rather than scored
over a smaller denominator. The bootstrap itself needs nothing beyond
the standard library.

## P13 — Replay interface

New file `app/viewer.py`. Streamlit. **Read-only**: it opens the results
database, never calls a model, never writes, never changes a stored result.

Required by CA2 on 2026-11-06 — the demonstration uses a real stored debate, so
this cannot be left to the end.

Layout, per the CA1 mockup:

- Question selector, showing the question's subject.
- The question text and every option (up to ten).
- **Round 1** — the five agents side by side, each with its answer letter and
  reasoning. Then the first majority vote and how it was reached
  ("B — majority reached with three of five votes").
- **Round 2** — the same five agents, each marked **changed** or **unchanged**
  against its Round 1 answer, with the revised reasoning. Then the final vote and
  the outcome (correct / incorrect / no consensus).
- Per-question totals: tokens, estimated cost, response time.

Agents are shown as `agent_1`…`agent_5` or by model name, but the *stored* Round
2 peer inputs stay anonymised — the viewer may reveal identities after the fact,
the debate never did.

Show failures honestly. An agent that returned `TRUNCATED` or `PARSE_FAIL` is
displayed as such, not as a blank or a guess.

Dependencies: `streamlit`. Add it when this module is written, not before.
