# Pilot review — corrected 22 September 2026

Source: stored run `pilot_agents_v5_20260910T161210Z`, scored offline with
`evaluation_v2`, `pilot_answers.jsonl` and `expected_questions=20`.
No calls, configuration changes or experimental-row changes were made.

## Results

| Comparison | Round 1 | Round 2 | Difference |
|---|---:|---:|---:|
| Primary: all 20 questions | 12/20 (60%) | 17/20 (85%) | +25 points |
| Supplement: all ten responses OK | 9/13 (69.23%) | 11/13 (84.62%) | +15.38 points |

Primary transitions: 12 stayed correct, 5 became correct, none became incorrect,
3 stayed incorrect. The unchanged paired bootstrap interval is +10 to +45
points (10,000 resamples, seed 20260828); exact McNemar p=0.0625.
Subset transitions: 9 stayed correct, 2 became correct, none became incorrect,
2 stayed incorrect. Seven questions are excluded from the subset, not the main
score. Cached OK responses count as valid, including questions 3932 and 8844.

## Every question

IDs below have prefix `mmlu_pro_v1:test:`. `—` means no group answer and is
scored incorrect. C/W mean correct/wrong against the frozen key, not a manual
judgment about ambiguous wording. T = TRUNCATED; E = API_ERROR.

| ID | R1 answer / score | R2 answer / score | R1 failures | R2 failures |
|---|---|---|---|---|
| 199 | J / C | J / C | none | DeepSeek E |
| 391 | D / C | D / C | none | none |
| 1689 | F / W | A / C | none | none |
| 1992 | D / C | D / C | none | none |
| 3244 | B / C | B / C | none | none |
| 3382 | F / C | F / C | none | none |
| 3839 | — / W | C / C | DeepSeek E | DeepSeek E |
| 3932 | D / C | D / C | none | none |
| 4961 | H / W | H / W | DeepSeek E | DeepSeek E |
| 5549 | J / C | J / C | none | none |
| 6002 | — / W | C / C | none | none |
| 6939 | H / C | H / C | none | DeepSeek E |
| 7296 | B / W | B / W | none | none |
| 8844 | — / W | D / W | none | none |
| 9622 | — / W | F / C | Gemma T, Mistral T, Qwen T | none |
| 10391 | J / C | J / C | none | none |
| 10560 | F / C | F / C | DeepSeek E | DeepSeek E |
| 10925 | F / C | F / C | none | none |
| 11408 | D / C | D / C | none | none |
| 11875 | — / W | I / C | Gemma T, Qwen T | Gemma T, Qwen T |

## Corrections to the first interpretation

- Four improvements resolved missing group answers; one changed a decided wrong
  answer (1689, F to A). The fifth missing group answer (8844) became a wrong
  consensus, not a correct answer.
- It was wrong to call three changes a "retry lottery". At 11875 and 3839 the
  same agents failed in both rounds, yet the remaining agents reached a correct
  majority. Only 9622 among these cases had the three failed responses become
  valid. This table does not establish why responses changed.
- "Aggregation lost to every single agent" was false: 60% exceeds Llama's
  55%. Other agents' valid-only accuracy has a different denominator from the
  group score; report failures alongside those comparisons.
- There are seven truncations: Qwen three, Gemma three, Mistral one, on two
  questions. The old log's "Qwen twice" was an error. Neither concentration
  nor one pilot proves the ceilings are sufficient for every future question.

## What this supports, and what it does not

The improvement does not disappear when questions with failures are excluded.
However, that subset is selected after observing completion and can favour
easier questions. It does not prove a pure effect of communication: there is
no second independent-answer control. Keep both comparisons and their sample
sizes, and do not present pilot numbers as final experimental results.

Provider reliability, any bounded token test and the long main-run schedule
remain separate decisions. No outcome here authorizes a paid rerun or changing
the frozen main-question set or answer key.
