# Main experiment record — 23 September 2026

This is the permanent audit record for the only formal 300-question
experiment. The run is complete, backed up and accepted as the final
experimental dataset. It must not be repeated to remove inconvenient provider
failures or change the observed result.

## Provenance

- Run ID: `experiment_agents_v7_20260923T114934Z`
- Started: `2026-09-23T11:49:34.189843+00:00`
- Finished: `2026-09-23T22:26:40.659202+00:00`
- Wall-clock duration: 10.62 hours
- Code commit used: `d4109d4` (`Build protected main experiment runner`)
- Settings: frozen `agents_v7`
- `agents_v7.yaml` SHA-256:
  `a4daeca0dc6789a0d981c1f117c730a97ec79bdc63c9ee0e1e8517d6fae8d6a9`
- Questions: 300 frozen experimental questions
- Stored data: 3,000 responses and 600 group outcomes
- Execution: sequential, both rounds, five fixed agents, no cache hits
- Completion and both SQLite integrity checks: passed

The command did not read an answer key. Accuracy was calculated afterwards by
`evaluation_v2` against the separate frozen experimental answer key with
`expected_questions=300`.

## Primary result

| Measure | Correct | Accuracy |
|---|---:|---:|
| Round 1 group vote | 245/300 | 81.7% |
| Round 2 group vote | 255/300 | 85.0% |
| Change after debate | +10 net | +3.33 percentage points |

The paired transition table was:

| Round 1 to Round 2 | Questions |
|---|---:|
| Stayed correct | 243 |
| Became correct | 12 |
| Became incorrect | 2 |
| Stayed incorrect | 43 |

The paired question-level bootstrap estimate was +3.33 percentage points with
a 95% interval of `[+1.00, +5.67]`, using the fixed seed `20260828` and 10,000
resamples. McNemar's exact test gave `p = 0.01294` from 12 improvements and two
regressions.

Ten of the 12 improvements occurred on questions that had no Round 1 majority.
On the 218 questions where all ten agent responses were valid and Round 1
already had a majority, two became correct and two became incorrect: zero net
change. The main interpretation is therefore conditional: debate improved the
overall result mainly by resolving initial disagreement, not by reliably
correcting an existing majority.

## Individual Round 1 results

Individual accuracy uses only that agent's valid parsed answers. It therefore
does not share the fixed 300-question denominator used by the group result.

| Agent | Correct / valid | Accuracy | Failed responses |
|---|---:|---:|---:|
| Qwen | 241/284 | 84.9% | 16 |
| Gemma | 246/291 | 84.5% | 9 |
| DeepSeek | 251/299 | 83.9% | 1 |
| Llama | 225/299 | 75.3% | 1 |
| Mistral | 163/262 | 62.2% | 38 |

Round 1 aggregation was 3.51 points above the mean valid-answer accuracy and
3.19 points below Qwen's valid-answer accuracy. The second comparison must not
be described as the group solving fewer questions than Qwen: Qwen produced 241
correct answers across the complete set, while the group produced 245.

## Consensus and the supplementary complete-case result

| State | Round 1 | Round 2 |
|---|---:|---:|
| Unanimous | 140 | 216 |
| Consensus | 138 | 77 |
| No consensus | 19 | 7 |
| Insufficient answers | 3 | 0 |

The number of decided questions increased from 278 to 293, while unanimity
increased much more sharply than accuracy. Agreement is therefore not treated
as evidence of correctness by itself.

The supplementary complete-case subset contained 230 questions for which all
five agents returned valid answers in both rounds. It moved from 192/230
(83.5%) to 195/230 (84.8%), a descriptive increase of 1.30 points: five became
correct and two became incorrect. This selected subset does not replace the
primary all-question analysis and is not a pure causal estimate of
communication.

## Failures and the Mistral limitation

| Agent | Round 1 failures | Round 2 failures | Total |
|---|---:|---:|---:|
| Llama | 1 truncated | 0 | 1 |
| Qwen | 16 truncated | 10 truncated | 26 |
| Mistral | 36 API errors, 2 truncated | 41 API errors | 79 |
| DeepSeek | 1 API error | 0 | 1 |
| Gemma | 2 API errors, 7 truncated | 1 truncated | 10 |

There were no refusals or parse failures. In total, 117 of 3,000 stored
responses were failures: 80 API errors and 37 truncations. They remained
auditable absences, cast no vote and never reduced the fixed three-of-five
threshold.

All 77 terminal Mistral API errors were HTTP 429 responses from DeepInfra's
upstream shared pool. They appeared throughout the run rather than in one short
outage. Fourteen other Mistral requests succeeded on the second attempt, but 77
failed both permitted attempts. This was provider throttling, not a missing
model route, parser error or local code failure.

Mistral returned 521 valid responses out of 600 (86.8%). Its provider failures
are a prominent limitation, but they did not cause most Round 1 undecided
outcomes: only five of the 22 questions without a Round 1 majority also had a
failed Mistral Round 1 response. The formal run is not repeated to select a more
favourable provider state; the failure rate is part of the observed system.

## Usage and cost

| Round | Prompt tokens | Completion tokens | Cost |
|---|---:|---:|---:|
| Round 1 | 614,105 | 609,163 | $1.221097 |
| Round 2 | 2,576,773 | 409,889 | $1.699553 |
| Total | 3,190,878 | 1,019,052 | $2.920649 |

The run made 3,100 paid attempts for 3,000 responses. One hundred responses
were retried; 20 recovered and 80 ended as API errors. Round 2 added $1.70 and
substantially longer prompts for the net gain of ten correct group answers.

## Backups

SQLite-safe post-run backups were created immediately after completion:

- `storage/backups/results_after_main_run_20260924T001055Z.sqlite`
  - SHA-256:
    `1aa7ad77c398a5d7140457198afbcb04bcc3b25eb451be46f654b239e9f39604`
- `storage/backups/cache_after_main_run_20260924T001055Z.sqlite`
  - SHA-256:
    `ab78c6ef7eb390699c913d1d56219c0b82b4cab91b1e6ba25dd448739e2bd854`

Both backups returned `ok` from SQLite's integrity check. They remain ignored
by Git because they contain raw model responses. An off-machine copy is still
required for disaster recovery.

## Accepted conclusion and next work

Under this fixed five-model setup, Round 2 produced a small overall improvement
beyond Round 1 majority voting. The improvement was concentrated in questions
without an initial majority, while clean questions with an existing majority
showed no net change. Debate increased agreement much more than accuracy and
required additional tokens, money and time.

No further model calls or repeat main experiment are planned. Remaining work is
offline. The formal outputs were generated by
`scripts/evaluate_experiment.py` under
`reports/main_experiment_20260923/`: a readable Markdown report, a complete
JSON record and seven CSV tables. A balanced descriptive inspection of
successful corrections, regressions, wrong convergence and persistent
disagreement is recorded in `docs/qualitative_review_20260925.md`. Remaining
work is to build the read-only replay interface and write the dissertation and
CA2 materials.
