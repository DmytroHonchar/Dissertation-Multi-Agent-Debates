# Development Log

This log records what was built, why it was built, how it was tested, and any
problems discovered. Add a short dated entry after every meaningful component
or experiment change.

## 2026-08-04 — MMLU-Pro dataset preparation

Built the benchmark loading, validation, stable-ID, deterministic sampling, and
freezing code in `src/mad/benchmark.py`. The MMLU-Pro test split was validated,
one invalid test question with duplicate options was removed, and seed 42 was
used to create a category-stratified set of 300 experimental questions plus a
separate 20-question pilot. Model-visible questions and answer keys were stored
separately to prevent answer leakage, with manifests and SHA-256 checksums for
reproducibility. Dataset and sampling tests passed.

## 2026-08-26 — Five-model OpenRouter foundation

Replaced the planned obsolete three-model design with five heterogeneous agents:
Llama, Qwen, Mistral, DeepSeek, and Gemma. Added the model registry in
`configs/models/agents_v1.yaml`, the shared OpenRouter client in
`src/mad/api_client.py`, and the connectivity checker in
`scripts/check_models.py`. A single OpenRouter key is loaded from the ignored
`.env` file and is used for every model.

The live registry confirmed all five exact model IDs, advertised context limits,
and prices. A smoke test successfully contacted all five models. Llama and Gemma
returned `ready`; Mistral returned `Yes.`; Qwen and DeepSeek returned empty
visible content when restricted to 16 completion tokens. This showed that API
access works but also revealed that the checker must reject empty responses and
that reasoning models require realistic completion limits during the pilot.

## 2026-08-26 — Repository and protocol audit

Audited the repository against the fixed dissertation design. Dataset
preparation and basic model access exist, but Round 1 prompts, response parsing,
failure statuses, caching, database storage, three-of-five voting, Round 2 peer
sharing, evaluation, experiment runners, and the replay interface are not yet
implemented. The old three-model `0.x` pipeline is not present in this Git
history and must not be reconstructed as the active design.

The fixed two-round, five-agent protocol and current model/cost snapshot were
recorded in `docs/decisions.md`. The next implementation target is Milestone 1:
run one pilot question through five independent Round 1 calls, parse the five
responses, compute the group vote, and store an inspectable result.

## 2026-08-28 — Documentation consolidation and scaffold removal

- **Built:** Deleted 17 empty placeholder files and 19 empty directories,
  including the orphaned `medqa` trees left from the abandoned second benchmark.
  Only `src/mad/__init__.py` was kept empty, because the package needs it.
  Modules are now created when their first real line is written.
  Moved the master checklist and the stage-by-stage build pipeline into the
  repository as `docs/checklist.md` and `docs/pipeline.md`, rewriting the
  pipeline from the obsolete three-model version to the fixed five-agent design:
  one OpenRouter key instead of three SDKs, three-of-five voting instead of
  two-of-three, MMLU-Pro's variable option count instead of four, and a new
  Round 2 stage that the old pipeline never specified. Added `CLAUDE.md` so AI
  tooling loads the fixed design instead of being re-briefed each session.
  Recorded D009–D014 in `docs/decisions.md` from the CA1 proposal: outcome
  measures, scoring of no-consensus, statistical analysis, retry policy, replay
  interface scope, and desirable extensions.
- **Why:** Work was spread across Notion, a browser chat history, an obsolete
  pipeline document and the repository, with no single authority, and the empty
  scaffold made the project look implemented when it was not. Each document now
  has one job: `decisions.md` for what is decided, `pipeline.md` for how to build
  it, `checklist.md` for order and progress, this log for what happened.
- **Tested:** `pytest` — 16 passed. Confirmed nothing imported the deleted
  modules before removing them.
- **Problems:** Three conflicts surfaced between the proposal, the decisions and
  the code. (1) Temperature is `0.7` in `agents_v1.yaml` and D002 but `0` in the
  old pipeline, and remains unrecorded. (2) The proposal states that a temporary
  API error is "retried once", but `OpenRouterClient._post_with_retries` retries
  up to five times with exponential backoff, so the stated behaviour and the cost
  reserve are both wrong until one side changes. (3) The proposal's Preliminary
  Work section describes OpenRouter access and model configuration as empty
  scaffolds, but both are implemented and were verified against all five live
  models on 2026-08-26. Also noted that `streamlit` and a statistics library are
  required by the design but absent from `pyproject.toml`.
- **Next:** Commit this work, resolve the temperature and retry decisions, then
  build P2 prompts and P6 parser toward Milestone 1.

## 2026-08-28 — Settings fixed, retry corrected, five-model connection confirmed

- **Built:** Set temperature `0` and top-p `1.0` for all five agents in both
  rounds (D002). Lowered `api_client.DEFAULT_MAX_ATTEMPTS` from 5 to 2 so the
  code matches the proposal's retry-once policy (D012). Rewrote the smoke checker
  so empty visible content counts as a failure, raised its completion budget to
  256 tokens, and made it log requested model, served model, provider, finish
  reason, token usage, cost and attempt count. Restored `.env.example` with
  placeholders only. Added `tests/test_api_client.py` and
  `tests/test_model_registry.py`. Resolved D011 as a paired question-level
  bootstrap with 10,000 resamples and a 95% percentile interval, plus McNemar's
  exact test.
- **Why:** Temperature 0 keeps the comparison between voting and communication
  controlled, since diversity already comes from five model families. The retry
  count contradicted the proposal. The statistical method had to be fixed before
  any results exist, not chosen after seeing them. The current milestone is only
  a working connection to the five model IDs, so requests use OpenRouter's
  automatic provider routing and `require_parameters: true`; pinning each agent
  to one provider is deferred to before the pilot and recorded in D015.
- **Tested:** `pytest` — 29 passed, up from 16. Offline registry check reports
  5/5 agents. Live smoke calls confirmed all five model IDs reachable, returning
  non-empty parseable content with usage recorded: Llama, Qwen, DeepSeek and
  Gemma passed in a single batch run; Mistral failed the batch twice with HTTP
  429 and 504 but succeeded on a standalone call twenty seconds later. All
  probing across the day cost well under $0.01.
- **Problems:** Mistral is intermittently rate-limited on OpenRouter's shared
  upstream pool and is the only provider serving that model, so a 429 during the
  main run costs `agent_mistral` its vote for that question. Recorded as D016 to
  be measured during the pilot. Automatic routing was observed selecting a
  different provider on every run — Qwen came from Alibaba, then Chutes, then
  Reka; Gemma from OpenInference, then CoreWeave, then DeepInfra — which is
  harmless for a connectivity check but would confound the main run, so D015
  keeps pinning as required before the settings freeze. The first-party DeepSeek
  endpoint returns HTTP 404 under the account's data policy; automatic routing
  avoids it and the model is reachable through other providers. Reasoning models
  again spent completion tokens before emitting visible text, so the provisional
  1,024-token limit still needs pilot confirmation. Mistral answered `'Yes.'` to
  a prompt demanding the single word `ready`, a parser concern for P6.
- **Next:** Build P2 prompts and P6 parser toward Milestone 1.

## 2026-08-30 — Round 1 prompt, and a readability pass over the whole codebase

- **Built:** `src/mad/prompts_v1.py`, which turns one frozen question into the
  two messages sent to a model. The system prompt gives a five-step reasoning
  procedure and demands `FINAL ANSWER: X` on its own line. `FINAL_ANSWER_MARKER`
  is exported as a constant so the parser can import it rather than repeat the
  string. `build_round1_messages` refuses any question carrying an answer field.
  31 tests in `tests/test_prompts.py`. Then a comments-and-layout pass over every
  file: module docstrings everywhere, numbered sections in the four longest
  files, and one-line docstrings on all 17 previously undocumented helpers in
  `benchmark.py`.
- **Why:** The prompt is the experimental instrument and gets frozen after the
  pilot, so it needed writing before anything else in the pipeline. The
  readability pass was necessary because the code had become hard for its own
  author to read, which matters for a project that has to be defended in a viva.
  `benchmark.py` was the worst case: 730 lines, no module docstring, no comments.
- **Tested:** 60 tests pass, up from 29. Confirmed the prompt text is
  byte-identical before and after the formatting pass, and that every one of the
  320 frozen questions renders without error.
- **Problems:** None. Note that the prompt asks for reasoning capped at 200
  words; whether that fits inside the provisional 1,024-token limit is still an
  open question for the pilot, since reasoning models spend tokens before
  emitting visible text.
- **Next:** `parser_v1.py` — read the answer letter out of a reply and classify
  failures as OK, REFUSAL, TRUNCATED, PARSE_FAIL or API_ERROR.


### 2026-08-31 — Response parser (P6)

- **Built:** `src/mad/parser_v1.py` and `tests/test_parser.py` (53 tests). The
  parser takes a reply, the letters that question actually offers, and the
  provider finish reason, and returns a `ParsedResponse` carrying a status, a
  letter, the extraction method and a short note. Statuses are `OK`, `REFUSAL`,
  `TRUNCATED`, `PARSE_FAIL` and `API_ERROR`. The answer-line regex is built from
  `prompts_v1.FINAL_ANSWER_MARKER` so the prompt and the parser cannot drift
  apart.
- **Why:** The vote has nothing to count until the letter is separated from the
  reasoning, and the three headline numbers depend on failures being kept apart
  from wrong answers. One parser serves all five agents, so the results measure
  the models rather than how forgiving the reading was.
- **Tested:** 113 tests pass, up from 60. Covers every letter of a ten-option
  question, case and spacing variants, multiple answer lines, trailing text,
  letters outside the option set, bracketed letters, empty replies, refusals,
  truncation by finish reason and by heuristic, and API errors. One test asserts
  the reasoning is never used to infer an answer: a reply that argues for C and
  writes `FINAL ANSWER: A` parses as A. A final parametrised test asserts that no
  failure of any kind ever contributes a vote.
- **Problems:** Markdown broke the first version — `**FINAL ANSWER:** B` failed
  to parse, which would have turned valid answers into failures. Asterisks are
  now stripped from the matching copy only. Refusal detection had the opposite
  risk: a loose phrase list would classify ordinary reasoning containing
  "cannot" as a refusal, so the list is short and must be rechecked against the
  pilot transcripts. Both choices are recorded as D017. Mistral's `'Yes.'` from
  the 2026-08-28 smoke test is in the test set and parses as `PARSE_FAIL`.
  Four further defects were found by review on the same day and fixed before
  anything was committed: `FINAL ANSWER: B2` parsed as `B` because the lookahead
  only excluded letters, not digits or underscore; the refusal phrase search
  matched anywhere in the reply, so `I cannot answer A, so FINAL ANSWER: B` was
  classified `REFUSAL` despite answering; a `max_tokens` finish reason was
  recorded in the note as `length`; and the truncation heuristic rested on
  punctuation alone, making `Yes` truncated and `Yes.` a parse failure. Text
  evidence is now ranked below provider signals, and a text-guessed truncation
  needs at least 200 characters. D017 and P6 were both updated, since the
  refusal change narrows a rule P6 had stated absolutely.
- **Next:** `database.py` — the SQLite results store, so a response and its
  parsed outcome can be written down.

### 2026-08-31 — Review fixes to `prompts_v1.py` and `api_client.py`

- **Built:** A review of the two existing modules found six defects, all fixed.
  In `prompts_v1.py`, `format_question` coerced its input with `str()`, so a
  question whose text was `None` or `123` would have been sent to a model as
  `"None"` or `"123"`; it now rejects anything that is not a non-empty string.
  In `api_client.py`: `AttemptRecord` was added and `CompletionResult` gained
  `raw_response` and `attempt_log`, because the previous version threw away the
  first attempt's tokens, cost and timing when the retry succeeded, and never
  exposed the complete API body that P5's cache is required to store;
  `ApiRequestError` now carries the same `attempt_log`, so a call that failed
  both attempts is still costable; an HTTP 200 whose body is not valid JSON is
  now treated as a retryable failure instead of raising a raw `requests` error;
  and the comment claiming a `Session` "reuses the same connection for every
  call" was corrected, since it reuses when it can and reopens when it must.
- **Why:** Two of these were silent-corruption risks rather than crashes. The
  `str()` coercion would have produced a plausible-looking prompt with no
  question in it. Discarding failed attempts contradicts P5 and P6, which
  require the complete raw body to be cached and the cost of every live attempt
  to be counted against the £15 budget.
- **Tested:** 135 tests pass, up from 121. Nine new tests drive the real retry
  loop through a fake session rather than stubbing it out: temporary failure
  then success, two temporary failures, an immediate permanent failure that must
  not be retried, a transport exception, an HTTP 200 carrying an `error` key,
  and a body that is not JSON. Two more assert the raw body is kept whole and
  that the cost of a failed attempt is recorded alongside the successful one.
  Five new tests in `tests/test_prompts.py` cover the non-string question text.
- **Problems:** The retry branches had been almost untested — the only retry
  test asserted `DEFAULT_MAX_ATTEMPTS == 2`, which proves a constant, not
  behaviour. Three existing tests stubbed `_post_with_retries` and had to be
  updated for its new return shape; that stubbing was what hid the gap.
  `docs/checklist.md` still listed the Round 1 prompt as unbuilt and is now
  corrected.
- **Next:** unchanged — `database.py`.

### 2026-08-31 — Results database (P4)

- **Built:** `src/mad/database.py` and `tests/test_database.py` (53 tests).
  SQLite through the standard library, so there is no server, no driver and no
  credential to manage. `ResultsDatabase(path)` defaults to
  `storage/results.sqlite` resolved from the repository root, creates the
  directory and file on first open, enables foreign keys, and stamps
  `PRAGMA user_version`. The four P4 tables are created as specified: `runs`,
  `model_responses`, `response_attempts`, `question_outcomes`. Writing goes
  through `start_run`, `record_response`, `record_outcome` and `finish_run`;
  reading through `read_runs`, `read_run`, `read_responses`, `read_attempts` and
  `read_outcomes`. `response_from_completion`, `response_from_failed_call` and
  `attempt_rows` join what `api_client` and `parser_v1` return, so neither of
  those modules learns about the database and the database does no parsing.
- **Why:** Nothing downstream can be built without a place to put results, and
  several project rules are better enforced by the schema than by discipline. A
  `CHECK` allows `extracted_letter` only on an `OK` row, so a failure cannot
  become a vote at the storage layer. `UNIQUE (run_id, question_id, round,
  agent_id)` makes a repeat insert fail rather than overwrite, which is the
  "never overwrite a run's rows" rule made structural. Round 1 and Round 2
  outcomes are separate rows because comparing them is the research question.
  No table has a column capable of holding a correct answer.
- **Tested:** 188 tests pass, up from 135. Coverage includes schema and
  directory creation, the schema-version guard, foreign-key enforcement, a full
  raw-text round trip, duplicate rejection for runs, responses, attempts and
  outcomes, rollback of a response when one of its attempts is rejected, all
  four failure statuses stored without a letter, a failure carrying a letter
  being refused, retry attempts kept with their HTTP status and raw body, peer
  references including the fewer-than-four case, the four consensus states,
  Round 1 and Round 2 outcomes staying separate, `finish_run` working exactly
  once, and a test that parses the module's own SQL to prove `finish_run`'s
  `ended_at` stamp is the only `UPDATE` and that no `DELETE` or
  `INSERT OR REPLACE` exists. Every test builds its database under `tmp_path`;
  no real `storage/results.sqlite` is created, no API call is made and
  `data/frozen/` is not read.
- **Problems:** Checking the integration boundary found that `AttemptRecord` had
  no raw body and no finish reason, and that its `error` field was truncated to
  400 characters — so a failed attempt could not satisfy `response_attempts`'s
  raw-response column, and P4 would have stored a summary where the record
  requires the reply. `AttemptRecord` gained `raw_response` and `finish_reason`,
  populated from the untruncated `response.text`; `error` stays short as a log
  summary. A transport error still stores an empty raw body, because no response
  arrived and inventing one would be fabrication. Three tests cover this.
  Separately, the first version of the SQL-scanning test failed on the module's
  own comment containing the words it was banning; it now parses the AST and
  inspects only strings that begin with an SQL keyword.
- **Next:** `cache.py` (P5), so a repeated call is not paid for twice, then
  `debate.py` (P9-P10).

### 2026-08-31 — Three-of-five voting (P9)

- **Built:** `src/mad/voting.py` and `tests/test_voting.py` (48 tests).
  `tally()` takes a mapping of `agent_id` to `ParsedResponse` and returns a
  frozen `VoteOutcome` holding the state, the consensus letter or `None`,
  whether the question was decided, the valid-answer count, the per-letter vote
  counts, and the agents that failed. `EXPECTED_AGENT_COUNT = 5` and
  `CONSENSUS_THRESHOLD = 3` are constants, and a group that is not exactly five
  identified agents is refused rather than accommodated.
- **Why:** D004 fixes the threshold at three of the five *configured* agents,
  not three of however many succeeded. Rebasing it on the survivors would make a
  round with more failures look more decisive, which is the exact flaw the
  project exists to measure, so the group size is validated instead of inferred.
  Only an `OK` response votes: a failure is an absence, not a wrong answer.
  `INSUFFICIENT_ANSWERS` and `NO_CONSENSUS` are kept apart because "the agents
  disagreed" and "too many agents failed" mean different things in the results
  chapter, even though D010 scores both as incorrect.
- **Tested:** 236 tests pass, up from 188. All four states; every failure status
  proved to cast no vote; the exactly-three boundary tested from both sides; two
  agreeing survivors and one lone survivor both refused a group answer; mixed
  disagreements including 2-2-1 and 2-1 with failures; five failures; every
  option letter A to J, since MMLU-Pro runs to ten options; group sizes 0, 1, 3,
  4, 6 and 7 refused; a duplicated `agent_id` collapsing to four entries refused;
  responses proved unmodified; and the four state names asserted equal to the
  database's `CHECK` constraint, so voting and storage cannot drift apart. A test
  parses the module's imports to prove it pulls in the parser and nothing that
  could store or score.
- **Problems:** None in the logic. Two test-quality corrections: an import test
  that string-matched the module source was replaced with one that parses the
  AST, and no judge or tie-break rule was needed once it was clear that two
  letters cannot both reach three of five, so a tie is already below the
  threshold.
  **Correction to the previous entry:** it names `cache.py` as next. That is
  wrong. `docs/checklist.md` puts P9 voting before the cache, and voting is what
  Milestone 1 needs. The order is: voting (done), then a small runner for
  Milestone 1, then `cache.py`.
- **Next:** Milestone 1 — one real pilot question through Round 1: five calls,
  parsed, stored, voted and inspected by hand. It needs a small runner to
  sequence call, parse, store and vote, and it is the first real spend on the
  pipeline. `cache.py` (P5) follows.

### 2026-08-31 — Review fixes to voting (P9)

- **Built:** Two corrections found by review, before any commit. `VoteOutcome`
  was declared frozen but held a plain dict in `vote_counts`, so
  `outcome.vote_counts["A"] = 0` silently rewrote a tally after the vote; it is
  now a `MappingProxyType` and read-only. `tally()` gained an optional
  `expected_agents` argument that checks the five are the configured five and
  names what is missing or unexpected.
- **Why:** The first was a real hole in an immutability guarantee the file
  claims in its own docstring. The second corrects an overstatement rather than
  a bug: the code validated the count and the identifiers, not the names, so
  five unrelated IDs could produce `UNANIMOUS`. Voting still does not read
  `agents_v1.yaml` — it has no business knowing which models the agents are, so
  the caller supplies the expected set and the runner will pass
  `load_model_registry()`'s keys.
- **Tested:** 242 tests pass, up from 236. Six new tests: `vote_counts` refuses
  assignment and deletion, outcome fields refuse reassignment, five unrelated
  IDs still vote when no expectation is given but are refused when it is, a
  single swapped agent is named in the error, the configured five pass, and an
  expected set that is not five agents is refused.
- **Problems:** None outstanding. Noted for the runner: passing
  `expected_agents` is optional, so Milestone 1 must actually pass it or the
  identity check does nothing.
- **Next:** unchanged — Milestone 1, then `cache.py` (P5).

### 2026-09-01 — Round 1 runner and P8 configuration

- **Built:** `src/mad/round1.py`, `scripts/run_milestone1.py` and
  `tests/test_round1.py` (23 tests). `Round1Config` freezes the P8 recipe:
  `round1_config_v1`, question set `mmlu_pro_v1`, prompt `round1_v1`, settings
  `agents_v1`, parser `parser_v1`, cache off, 120s timeout, two attempts,
  sequential calls. `run_round1_question()` takes one question through the full
  sequence — five calls, parse, store each response with its attempts, tally
  with `expected_agents=registry.keys()`, store the outcome, finish the run —
  and returns a report for hand inspection. `FixtureClient` supplies labelled
  deterministic replies (provider `fixture` on every row): three agents agree,
  one dissents, one refuses.
- **Why:** Milestone 1 needs something that sequences the five existing modules,
  and it is the first thing in the project that can spend money, so the
  protections are structural rather than habits: the default is a free dry run
  into a throwaway database; live mode needs both `--live` and
  `--yes-spend-real-money`; only IDs from the 20-question pilot file are
  accepted and experimental IDs are refused by name; a dry run is refused
  `storage/results.sqlite`; a registry that does not hold exactly five agents
  is refused before any call; and one agent failing is stored as `API_ERROR`
  while the other four continue.
- **Tested:** 265 tests pass, up from 242. The Round 1 tests disable the
  network for every test, so any real call fails the suite. Covered: a full
  fixture run storing five responses, their attempts and one outcome; every
  version string read back from the `runs` row; one failed agent leaving a
  five-row run with a `NO_CONSENSUS` vote; all five failing still completing;
  a four-agent registry refused with nothing stored; a spy asserting the
  registry keys actually reach `tally(expected_agents=...)`, since that check
  is optional and silently skippable; experimental, unknown, empty and multiple
  question IDs refused; both spend-guard directions; the production-path
  refusal; and a question carrying an answer key stopping the run before any
  row. CLI guards also exercised by hand: `--live` alone, a non-pilot ID and a
  dry run aimed at the real database were all refused, and no `storage/`
  directory exists.
- **Problems:** None new. The account now holds $5 of credit, but the key still
  has no spending limit — that stays a blocker for the live run (D015's pinning
  question also remains open, deferred to before the pilot). `max_tokens=1024`
  is provisional; the report prints a warning naming any truncated agent, which
  is the signal to watch on the live run.
- **Next:** set the key's spending limit in the OpenRouter dashboard, then run
  Milestone 1 live: `scripts/run_milestone1.py --question <pilot-id> --live
  --yes-spend-real-money`, inspect the five stored replies by hand, and record
  what it cost. Then `cache.py` (P5).

### 2026-09-01 — Review fixes to the Round 1 runner

- **Built:** Four corrections from review, before any live call. First, retry
  money: a call that failed once and then succeeded stored only the successful
  attempt's tokens and cost on its response row, so the outcome totals and the
  report understated spend; `response_from_completion` now sums every attempt
  from the log, with the per-attempt split still visible in
  `response_attempts`. Second, a malformed HTTP 200 body (no choices/message
  structure) raised `ApiRequestError` with an empty attempt log, losing the raw
  body and the cost of a paid reply; the log is now attached, with the last
  attempt relabelled `malformed_body` instead of the `ok` it was logged as, and
  the runner stores it as `API_ERROR` while the other four agents continue.
  Third, `Round1Config` declared `timeout_seconds` and `max_attempts` that the
  CLI ignored when building `OpenRouterClient`; one config now drives the
  client, the runner and the stored labels, and a config claiming a cache or
  parallel calls is refused, since it would describe a run that never happened.
  Fourth, the CLI: the live default database resolves to the repository's
  `storage/results.sqlite` regardless of the launch directory, every refusal
  check runs before a client exists, and the client is closed even when the run
  raises.
- **Why:** The first two were honesty bugs in the money and audit trail — the
  budget is £15 and D016 expects Mistral retries, so undercounted retry spend
  and vanished malformed replies would have surfaced during the pilot as
  unexplained account drain. The config fix closes a gap where the stored
  version labels could disagree with the run that actually happened.
- **Tested:** 278 tests pass, up from 265. New: a $0.00002 failed attempt plus
  a $0.00010 success stored as a $0.00012 response, outcome and report, with
  the per-attempt split intact; a malformed 200 body through the real client
  keeping attempt number, full raw body, cost and a truthful outcome, then
  through the runner as `API_ERROR` with the other four continuing; cache and
  parallel configs refused; and seven CLI-level tests driving `main(argv)`
  offline — dry run completes, chosen `--db` kept, `--live` alone refused, the
  spend flag alone refused, experimental IDs refused, the production path
  refused before any file exists, and the client closed even when the run
  raises. The dry run was re-run and `git diff --check` is clean; no `storage/`
  directory exists.
- **Problems:** None outstanding from the review. The key's missing spending
  limit remains the blocker for the live run.
- **Next:** unchanged — spending limit, live Milestone 1, `cache.py`, provider
  pinning before the pilot, then Round 2.


## Entry template

### YYYY-MM-DD — Component or activity

- **Built:** What changed.
- **Why:** The research or engineering reason.
- **Tested:** Commands, fixtures, real calls, or manual checks used.
- **Problems:** Failures, limitations, or unexpected behaviour.
- **Next:** The immediate follow-up.
