# Main experiment evaluation

- Run: `experiment_agents_v7_20260923T114934Z`
- Settings: `agents_v7`
- Evaluation: `evaluation_v2`
- Questions: 300
- Started: `2026-09-23T11:49:34.189843+00:00`
- Finished: `2026-09-23T22:26:40.659202+00:00`
- Wall clock: 10.62 hours
- Database SHA-256 at evaluation: `1aa7ad77c398a5d7140457198afbcb04bcc3b25eb451be46f654b239e9f39604`

## Headline results

| Measure | Correct | Accuracy |
|---|---:|---:|
| Round 1 group vote | 245/300 | 81.7% |
| Round 2 group vote | 255/300 | 85.0% |
| Change after debate | +10 net | +3.33 points |

## Paired statistical result

- 95% paired-bootstrap interval: `[+1.00, +5.67]` points
- Bootstrap resamples: 10,000; seed: `20260828`
- McNemar exact p-value: `0.012939`
- Became correct: 12; became incorrect: 2

## Group transitions

| Transition | Questions |
|---|---:|
| Stayed Correct | 243 |
| Became Incorrect | 2 |
| Became Correct | 12 |
| Stayed Incorrect | 43 |

## Per-agent accuracy and failures

Accuracy is calculated over each agent's valid answers only. It does not share the group's fixed denominator.

| Round | Agent | Correct / valid | Accuracy | Refusal | Truncated | Parse fail | API error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | agent_deepseek | 251/299 | 83.9% | 0 | 0 | 0 | 1 |
| 1 | agent_gemma | 246/291 | 84.5% | 0 | 7 | 0 | 2 |
| 1 | agent_llama | 225/299 | 75.3% | 0 | 1 | 0 | 0 |
| 1 | agent_mistral | 163/262 | 62.2% | 0 | 2 | 0 | 36 |
| 1 | agent_qwen | 241/284 | 84.9% | 0 | 16 | 0 | 0 |
| 2 | agent_deepseek | 257/300 | 85.7% | 0 | 0 | 0 | 0 |
| 2 | agent_gemma | 258/299 | 86.3% | 0 | 1 | 0 | 0 |
| 2 | agent_llama | 254/300 | 84.7% | 0 | 0 | 0 | 0 |
| 2 | agent_mistral | 214/259 | 82.6% | 0 | 0 | 0 | 41 |
| 2 | agent_qwen | 248/290 | 85.5% | 0 | 10 | 0 | 0 |

## Aggregation comparison

- Round 1 group accuracy: 81.7%
- Mean agent valid-answer accuracy: 78.2%
- Group versus mean: +3.51 points
- Best valid-answer accuracy: 84.9% (agent_qwen)
- Group versus best valid-answer accuracy: -3.19 points

The group and individual figures use different denominators; this comparison does not mean that the named best agent solved more of the complete 300-question set.

## Consensus states

| State | Round 1 | Round 2 |
|---|---:|---:|
| UNANIMOUS | 140 | 216 |
| CONSENSUS | 138 | 77 |
| NO_CONSENSUS | 19 | 7 |
| INSUFFICIENT_ANSWERS | 3 | 0 |

## Complete-case supplement

- Questions with all five agents valid in both rounds: 230
- Round 1: 83.5%
- Round 2: 84.8%
- Change: +1.30 points
- Transitions: 190 stayed correct, 5 became correct, 2 became incorrect, 33 stayed incorrect.

This selected subset supplements rather than replaces the primary 300-question result.

## Usage

| Round | Prompt tokens | Completion tokens | Cost | API attempts | Retried responses |
|---:|---:|---:|---:|---:|---:|
| 1 | 614,105 | 609,163 | $1.221097 | 1549 | 49 |
| 2 | 2,576,773 | 409,889 | $1.699553 | 1551 | 51 |

Total run cost: **$2.920649**.

## Interpretation boundary

The measured Round 2 minus Round 1 change is the observed effect of the full implemented system. It is not automatically a pure causal effect of communication: Round 2 adds another inference pass and failures differ between rounds. Provider failures remain reported evidence and are not repaired by rerunning the experiment.
