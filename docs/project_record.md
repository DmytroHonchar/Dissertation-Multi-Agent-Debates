# Project record — what was built, why, and what it is for

This document is the readable map of the completed experiment. It explains the
system by component rather than by date. `docs/decisions.md` remains the formal
source of truth and contains the full evidence behind decisions D001–D029.

## 1. Research design

**What was decided.** Five different open-weight model families answer the same
MMLU-Pro questions. Round 1 is independent. Round 2 lets each agent reconsider
its answer after seeing its own valid Round 1 response and the valid,
identity-free responses of the other agents. The group is voted after each
round.

**Why.** The research question is not simply whether five models are accurate.
It separates three things:

1. how well each model answers independently;
2. what is gained by combining five independent answers through voting; and
3. what changes after the agents receive one another's reasoning.

**What it is for.** This produces a matched before/after comparison on the same
300 questions: Round 1 majority vote versus Round 2 majority vote.

**Boundaries.** There is no homogeneous-model condition, judge model or third
round. Round 2 is not a pure causal test of communication because it also gives
the models a second inference. That limitation is reported, not hidden.

Authority: D001, D003, D009, D014 and D029.

## 2. Frozen benchmark data

**What was built.** The project stores one fixed MMLU-Pro snapshot under
`data/frozen/mmlu_pro_v1/`. It contains 300 experimental questions and a
separate, non-overlapping set of 20 pilot questions. Sampling is deterministic,
category-stratified and uses seed 42. Each question has a stable ID.

Model-input files contain the question and options but not the correct answer.
Answer keys live in separate files. Validation rejects malformed records and
the freeze process refuses accidental overwriting.

**Why.** Every model and both rounds must face the same questions. Separating
the key prevents accidental answer leakage and stops live generation code from
scoring or adapting itself while it calls models.

**What it is for.** The pilot tests engineering choices without contaminating
the 300 final questions. The final answer key is used only by offline
evaluation.

Authority: D005. Main code: `src/mad/benchmark.py` and
`configs/benchmark/mmlu_pro_v1.yaml`.

## 3. Models, providers and configuration versions

**What was built.** YAML registries under `configs/models/` name every agent,
exact OpenRouter model slug, provider pin, temperature, top-p, token ceiling
and routing rule. Each change created a new immutable version rather than
editing the settings attached to an old run.

The final registry is `agents_v7`. It keeps five model families: Llama, Qwen,
Mistral, DeepSeek and Gemma. Temperature is 0, top-p is 1.0, provider fallback
is off and required parameters must be honoured by the route. A free preflight
checks the selected endpoint and checks that each model has at least three
independent compatible hosts before a live run.

**Why.** Automatic OpenRouter routing previously served the same model through
different companies and quantisations. Pinning makes the serving setup stable
enough to describe and reproduce. Versioned YAML keeps historical runs honest:
a run labelled `agents_v5` continues to mean the settings it actually used.

**What changed and why.** The original Mistral Large route became unavailable.
It was not silently replaced inside an existing configuration. `agents_v6`
introduced Mistral Small 3.2 on Parasail, but the full pilot found persistent
shared-pool rate limits. Bounded host probes sent the same failed prompts to
DeepInfra and Venice; both completed. The predeclared availability, latency,
throughput and price rules selected DeepInfra, producing `agents_v7`. The full
`agents_v7` pilot then passed and the registry was frozen.

**Rejected alternatives.** Automatic fallback would mix providers inside the
experiment. The asynchronous Batch API would change execution and latency.
Removing Mistral would change the five-family design. A direct second vendor
account would add another billing and API path. None was adopted after the
pilot.

Authority: D002, D015, D016 and D024–D026.

## 4. Prompt construction and the two rounds

**What was built.** `src/mad/prompts_v1.py` contains versioned prompt builders:
`round1_v1` and `round2_v1`.

- Round 1 asks for an explanation followed by one explicit `FINAL ANSWER`
  letter.
- Round 2 contains the original question, the agent's own valid previous reply
  as conversation history, and anonymous peer responses labelled only by
  position.
- The Round 1 group vote is stored but is never shown to a Round 2 agent.
- Only `OK` responses become an agent's own history or a peer response.

Peer order is deterministic registry order with the answering agent removed.
The order is reproducible but may carry a small positional effect, which is a
documented limitation.

**Why.** Restoring the agent's own answer makes Round 2 a reconsideration, not a
fresh answer to unrelated text. Hiding identities prevents reputation or model
name from directly influencing the reply. Hiding the first vote avoids telling
models which option already has social support.

**What it is for.** The prompt creates the controlled change between rounds:
independent answering first, then one round of information sharing.

Authority: D003, D019 and D020.

## 5. API client, retries and provider evidence

**What was built.** `src/mad/api_client.py` owns the OpenRouter HTTP session,
headers, model-list request, completion request, response parsing and attempt
records. It requests token and cost information and stores the requested model,
served model, provider, finish reason, raw body, latency and every attempt.

Temporary transport failures and retryable HTTP responses receive one retry
after a short exponential delay plus jitter: two attempts maximum. Permanent
errors fail immediately. Model refusals, truncations and parser failures are
not retried.

**Why.** One retry can recover a short provider problem without concealing an
unreliable route behind many attempts. Recording both attempts exposes the real
failure rate and cost. Retrying identical truncated text would usually spend
again without changing the condition.

**What it is for.** This layer turns a provider response into auditable data
instead of returning only a text string.

Authority: D007 and D012.

## 6. Token ceilings and hidden reasoning

**What was done.** Early live probes showed Qwen and DeepSeek consuming an
entire 1,024-token output allowance on hidden reasoning and returning no visible
answer. New immutable registries raised only the affected ceilings. The final
limits are Llama 1,024, Qwen 3,072, Mistral 1,024, DeepSeek 2,048 and Gemma
1,024. Qwen also receives a best-effort 2,048 reasoning-token request.

**Why.** A ceiling that is too small pays for reasoning but loses the answer; a
ceiling that is too large permits unnecessary cost. Limits were selected from
completion, truncation and cost—not answer correctness—so the settings were not
tuned to make pilot accuracy look better.

**Important finding.** Parasail did not enforce Qwen's reasoning request as a
hard partition. The total 3,072 ceiling is the dependable limit. Pathological
questions caused most pilot truncations, while ordinary completed questions had
substantial headroom, so escalation stopped before the formal experiment.

**What it is for.** The limits bound spending and give models enough space to
produce a parseable visible answer. A truncation remains an observed failure,
not something silently repaired.

Authority: D018 and its controlled-test addendum.

## 7. Response parser and failure meanings

**What was built.** `src/mad/parser_v1.py` turns a raw completion into one of:
`OK`, `REFUSAL`, `TRUNCATED`, `PARSE_FAIL` or `API_ERROR`. An `OK` row contains
one answer letter; all other states contain no vote. The parser accepts common
Markdown bolding and punctuation around the fixed `FINAL ANSWER` format while
rejecting words or multiple letters.

Provider refusal and finish signals take priority over weak textual guesses.
Raw text is never changed; only a temporary copy is normalised for matching.

**Why.** Models vary slightly in formatting. A parser that is too strict throws
away valid answers; one that is too loose invents votes from ambiguous text.
Explicit statuses keep format failures separate from wrong answers.

**What it is for.** Voting, evaluation and the replay interface all use the
same stored interpretation of a reply.

Authority: D010 and D017.

## 8. Fixed three-of-five voting

**What was built.** `src/mad/voting.py` accepts exactly the five expected agents
and counts only `OK` letters. Five matching votes are `UNANIMOUS`; three or four
matching votes are `CONSENSUS`; at least three valid votes without a three-vote
winner are `NO_CONSENSUS`; fewer than three valid answers are
`INSUFFICIENT_ANSWERS`.

**Why.** The threshold is always three of the five configured agents, not a
majority of whichever agents happened to succeed. Lowering the threshold after
failures would make the definition of a group answer change from question to
question.

**What it is for.** The same deterministic rule produces comparable Round 1
and Round 2 group outcomes. There is no judge or tie-break model.

Authority: D004 and D010.

## 9. Cache

**What was built.** `src/mad/cache.py` uses a separate SQLite file. Its key
includes agent/model identity, the full message list, sampling settings, output
limit, reasoning settings and provider pin. Successful completions—including
valid refusals or truncations—can be reused; transport/API errors and fixture
responses are not cached. First write wins.

A cache hit costs zero, makes zero new API attempts and retains the original
response provenance. The result row explicitly records `cache_hit`.

**Why.** Repeating an identical development command should not buy the same
answer again. Including every request-shaping field prevents a response from an
old prompt, provider or ceiling from being reused under new settings.

**What it is for.** It reduces development cost and enables exact free replay.
It is not used to erase failures from the formal run.

Authority: D006 and D007.

## 10. Results database and audit trail

**What was built.** `src/mad/database.py` uses `storage/results.sqlite`. The
four core tables store runs, model responses, individual API attempts and group
outcomes. Foreign keys, uniqueness constraints and status checks prevent
orphaned, duplicated or contradictory rows. Correct answers have no column in
the experimental database.

Runs and responses carry question-set, prompt, settings, parser and
configuration versions. Raw response text is stored before derived parsing.
Round 2 rows store which peer response IDs were shown. Records are append-only
except that `finish_run()` adds the successful end time.

**Why.** A dissertation result must be traceable back to the exact input,
settings, response and provider attempt that produced it. Keeping raw text
allows parser behaviour to be audited without paying to call the model again.

**What it is for.** It is the permanent experimental evidence used by offline
evaluation and the future viewer.

Authority: D007, D019 and D020.

## 11. Orchestration and run lifecycle

**What was built.** The execution flow is split by responsibility:

- `src/mad/round1.py` runs one question independently through five agents,
  parses, stores and votes;
- `src/mad/debate.py` constructs and runs Round 2 from stored Round 1 results;
- `src/mad/runner.py` supplies common question/configuration utilities;
- `scripts/run_round1.py` is a one-round diagnostic;
- `scripts/run_debate.py` runs one complete two-round question;
- `scripts/run_pilot.py` runs the 20 pilot questions; and
- `scripts/run_experiment.py` is the protected 300-question command.

The outer script starts one database run, processes its questions and finishes
the run only after completeness checks pass. A crash leaves `ended_at` empty,
so an incomplete run cannot look complete. The formal runner can resume the
same unfinished run after complete question boundaries but refuses partial
questions.

**Why.** Run lifecycle belongs to the whole experiment, not to one question.
This permits 20 or 300 questions under one reproducible run record and makes a
crash visible.

**What it is for.** It connects configuration, cache, API calls, parsing,
storage and voting without putting all behaviour into one file.

Authority: D007, D008 and D020.

## 12. Pilot, diagnostics and configuration freeze

**What was done.** Development progressed from offline fixtures to one Round 1
question, one complete debate, the 20-question pilot, bounded diagnostics, two
Mistral replacement pilots and finally the accepted `agents_v7` pilot.

The accepted pilot was `pilot_agents_v7_20260922T181727Z`: 20 questions, 200
responses, no terminal API errors, Mistral 40/40 valid, `$0.100227`, 29.1
minutes. Remaining Qwen/Gemma truncations and one DeepSeek parse failure were
accepted as measured model behaviour. All semantic settings were then frozen.

**Why.** The pilot finds engineering failures before exposing all 300 final
questions. Diagnostics were bounded, separately stored and selected by
completion/reliability rather than answer correctness.

**What it is for.** It justified that the full pipeline could run and fixed the
configuration before the formal result existed.

Authority: D008, D022–D026. Evidence:
`docs/pilot_review_20260910.md`, `docs/diagnostic_review_20260922.md` and
`docs/provider_repair_and_v7_pilot_20260922.md`.

## 13. Protected main experiment

**What was done.** One formal run,
`experiment_agents_v7_20260923T114934Z`, processed all 300 questions. It used
commit `d4109d4`, took 10.62 hours, cost `$2.920649`, and stored 3,000 model
responses plus 600 group outcomes. There were no cache hits. The results and
cache were backed up with SQLite integrity checks and SHA-256 hashes.

**Why only once.** Provider failures, truncations and disagreement are part of
the system being evaluated. Repeating until failures disappear would replace
the observed experiment with a selectively cleaned one.

**What it is for.** This is the final dataset for the dissertation. No new
model generation is required.

Authority: D027. Full evidence: `docs/main_experiment_20260923.md`.

## 14. Evaluation and statistics

**What was built.** `src/mad/evaluation.py` checks the run is finished and
complete, joins the separate answer key, and calculates:

- per-agent accuracy and failures in both rounds;
- Round 1 and Round 2 group accuracy over all questions;
- consensus states and correct/incorrect transitions;
- aggregation relative to the mean and best individual agent;
- the Round 2 minus Round 1 difference;
- a paired question-level bootstrap interval using 10,000 resamples and seed
  `20260828`;
- McNemar's exact test;
- tokens, attempts, cache hits, latency and cost; and
- a supplementary complete-case comparison.

No-consensus and insufficient-answer outcomes count as incorrect for group
accuracy, keeping the denominator at 300 in both rounds. Individual accuracy
uses valid answers and is always shown with coverage/failure counts.

**Why the paired bootstrap.** The 300 observed question differences give one
exact result, `+3.33` points. Resampling whole question pairs estimates how much
that result could vary if a comparable set of questions had been sampled. It
does not rerun models or improve their answers. McNemar tests the imbalance
between questions that improved and questions that worsened.

**What it is for.** It turns stored experimental rows into the measures needed
to answer the research question without making another API call.

Authority: D009–D011, D021, D022 and D029.

## 15. Final export and reproducibility

**What was built.** `scripts/evaluate_experiment.py` is fixed to the accepted
run and requires all 300 questions. It generates:

- `evaluation_report.md` for reading;
- `evaluation_summary.json` for complete structured provenance; and
- seven CSV tables for dissertation tables, plots and question inspection.

It has no API client, verifies `agents_v7`, protects existing exports, and
proves the results database hash is unchanged after reading.

**Why.** One exporter prevents manually copied numbers in the dissertation,
slides and viewer from drifting apart.

**What it is for.** The accepted outputs under
`reports/main_experiment_20260923/` are the working source for writing and
visualisation.

Authority: D028.

## 16. Final result and its meaning

The fixed system moved from `245/300` correct group answers in Round 1
(`81.7%`) to `255/300` in Round 2 (`85.0%`): `+3.33` percentage points. Twelve
questions became correct and two became incorrect. The paired 95% bootstrap
interval is `[+1.00, +5.67]`; exact McNemar gives `p = 0.01294`.

The important qualification is where the gain occurred. Ten of the 12
improvements began without a Round 1 majority. On the 218 questions where all
responses were valid and Round 1 already had a majority, two improved and two
worsened: zero net change. The project therefore found that this form of debate
mainly resolved initial disagreement; it did not reliably correct an existing
majority. Unanimity increased more than accuracy, so consensus alone is not a
truth signal.

Mistral's DeepInfra route produced 77 terminal upstream HTTP 429 errors during
the formal run. They are a limitation and part of the result, not evidence of a
local code error. Only five of 22 initially undecided questions had a failed
Mistral Round 1 response, so most missing Round 1 majorities reflected model
disagreement rather than simply missing Mistral votes.

Authority: D029 and `docs/main_experiment_20260923.md`.

## 17. Work deliberately not done yet

The model experiment and numerical evaluation are complete. Remaining work is
offline:

1. build `app/viewer.py`, a read-only Streamlit replay interface for CA2;
2. create dissertation figures and tables from the accepted exports;
3. write methods, results, discussion, limitations and conclusion;
4. create and practise the CA2 demonstration; and
5. copy the ignored database backups to separate off-machine storage.

The representative-question inspection is complete in
`docs/qualitative_review_20260925.md`. It is descriptive evidence for the
discussion chapter, not a new quantitative outcome.

No third debate round, subject-level extension, new model configuration or
repeat main run is authorised as part of the completed core experiment.

## 18. Decision index

| Decision | Subject | Purpose |
|---|---|---|
| D001 | Research comparison | Define the question the system answers |
| D002 | Initial models/API | Establish five heterogeneous agents through OpenRouter |
| D003 | Two-round protocol | Separate independence from information sharing |
| D004 | Voting | Keep one fixed three-of-five group rule |
| D005 | Dataset | Freeze questions and isolate pilot/key data |
| D006 | Budget | Bound financial exposure |
| D007 | Auditability | Preserve every input, response, attempt and version |
| D008 | Development order | Move from fixtures to pilot to one formal run |
| D009 | Outcome measures | Keep individual, aggregation and Round 2 results separate |
| D010 | Failure scoring | Avoid treating absence as a wrong model answer |
| D011 | Statistics | Analyse paired question-level change |
| D012 | Retry policy | Permit one recovery without hiding instability |
| D013 | Replay interface | Demonstrate stored debates without new calls |
| D014 | Extensions | Keep third round/category work outside the core study |
| D015 | Provider routing | Prevent provider/quantisation drift |
| D016 | Early Mistral risk | Record shared-pool availability risk |
| D017 | Parser | Extract letters consistently without inventing votes |
| D018 | Token ceilings | Balance completed answers against cost |
| D019 | Valid Round 1 input | Show only usable responses in Round 2 |
| D020 | Labels/peer order | Make two-round runs reconstructable |
| D021 | Evaluation details | Fix seed, exact test and aggregation baselines |
| D022 | Complete-case supplement | Describe results with failures held constant |
| D023 | Bounded diagnostics | Investigate provider/token issues without open-ended spend |
| D024 | Mistral replacement | Preserve model family after route withdrawal |
| D025 | Mistral host repair | Replace the rate-limited Parasail pin using fixed criteria |
| D026 | Final freeze | Lock `agents_v7` and every semantic setting |
| D027 | Formal run acceptance | Keep one complete, uncleaned main experiment |
| D028 | Final export | Generate all result artifacts reproducibly and read-only |
| D029 | Interpretation | State what the result supports and what it does not |
