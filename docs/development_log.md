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



## Entry template

### YYYY-MM-DD — Component or activity

- **Built:** What changed.
- **Why:** The research or engineering reason.
- **Tested:** Commands, fixtures, real calls, or manual checks used.
- **Problems:** Failures, limitations, or unexpected behaviour.
- **Next:** The immediate follow-up.
