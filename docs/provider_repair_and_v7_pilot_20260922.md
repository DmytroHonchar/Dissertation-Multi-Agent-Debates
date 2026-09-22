# Provider repair and accepted `agents_v7` pilot — 22 September 2026

This document records the full provider problem, the investigation, the fixes,
the rejected alternatives and the final pilot evidence. It is an engineering
and research audit, not a main-experiment result.

## Short answer

The original Mistral Large route stopped being callable through OpenRouter.
Mistral Small 3.2 was selected as a multi-host replacement. Its first pinned
host, Parasail, was then rejected because sustained shared-pool rate limiting
caused four terminal failures in the `agents_v6` pilot. A controlled comparison
showed that both DeepInfra and Venice could serve the exact failed requests.
DeepInfra won the pre-existing provider-selection criteria and became the
Mistral pin in `agents_v7`.

The full `agents_v7` pilot passed: all 40 Mistral responses were valid, no
response from any agent ended as `API_ERROR`, and two temporary Mistral 429s
were recovered by the existing retry-once rule. The core experimental settings
are therefore frozen as `agents_v7`. The main experiment has not yet run.

## What happened

1. `agents_v5` used Mistral Large 3 on `mistral/eu`. After the first formal
   pilot, OpenRouter still displayed the model's catalogue page but returned no
   callable endpoints, and a real request returned HTTP 404.
2. The project did not edit `agents_v5`. It created `agents_v6`, replacing only
   that unavailable model with Mistral Small 3.2 24B on Parasail BF16.
3. The `agents_v6` pilot showed that the model worked, but the selected host was
   unreliable under sustained use. Of 38 paid Mistral responses, 16 needed a
   retry. Twenty attempts received HTTP 429; twelve recovered and eight formed
   four terminal `API_ERROR` responses.
4. The four exact failed prompts were reconstructed and sent once to DeepInfra
   and once to Venice. Cache and retries were disabled, the answer key was not
   loaded, and both hosts returned 4/4 valid responses.
5. DeepInfra was selected using the criteria fixed before the comparison:
   availability, latency, throughput and price. `agents_v7` changes only the
   Mistral provider pin from Parasail BF16 to DeepInfra FP8.
6. The complete 20-question, two-round pilot was repeated under `agents_v7`.
   It passed the infrastructure check and became the accepted pilot.

## What was actually wrong

There were several different faults. Treating them as one problem would have
led to the wrong repair.

| Observation | Actual issue | Was it our code? |
|---|---|---|
| Mistral Large returned 404 and had no endpoints | The pinned OpenRouter route was unavailable | No |
| Mistral Small on Parasail returned many 429s | Parasail's shared upstream pool was rate-limited during the run | No |
| Two Mistral calls in `agents_v7` first returned 429 | Temporary provider throttling, recovered by retry | No; the retry behaved correctly |
| Qwen and Gemma sometimes reached their output ceilings | Model/question-dependent long generation | No transport fault |
| One DeepSeek response could not be parsed | The returned text did not satisfy the required final-answer format | Not an API failure |

The existing retry already waited roughly two to three seconds before its
second attempt. The `agents_v6` failures were spread across about twenty
minutes, so adding a slightly longer immediate delay would not have repaired
that provider-capacity problem.

## What was fixed

### 1. Versioned configuration instead of rewriting history

`agents_v5`, `agents_v6` and `agents_v7` remain separate files. Stored runs
keep the exact settings-version name used to produce them. Old responses are
not overwritten or relabelled.

### 2. A synchronous, multi-host Mistral replacement

Mistral Small 3.2 24B preserves the Mistral family, open-weight requirement and
ordinary synchronous chat-completion flow. At selection time it had at least
three independent healthy OpenRouter hosts.

### 3. Live preflight before spending

Every live runner now checks the free OpenRouter endpoint metadata before
creating a run or making a paid completion. The exact pinned endpoint must be
healthy and compatible, and every configured model must have at least three
independent healthy compatible hosts. Multiple endpoints owned by one company
count as one host.

This check proves that a pin currently exists and that replacement routes exist.
It cannot guarantee that a provider's shared rate-limit pool will remain free
for the next thirty minutes.

### 4. Controlled host diagnosis

The diagnostic replayed only the four failed Mistral requests, once per
candidate host. It used no answer key, no correctness-based selection, no cache
and no retry. That isolated the provider from the model, prompt and parser.

### 5. Better evaluation of failure effects

The primary evaluation still scores all questions. A supplementary complete-
case result also reports the subset where all five agents were valid in both
rounds. This reveals how much the headline comparison may be influenced by
different failure patterns, but it does not replace the primary result and is
not claimed as a pure causal measure of communication.

## Why this repair was chosen

- It keeps all five intended model families.
- It remains inside one OpenRouter-based calling and accounting system.
- It preserves synchronous calls, stored latency and the existing pipeline.
- It pins one exact provider, so a run does not silently mix hosts or
  quantisation formats.
- It uses provider reliability and cost rather than answer correctness to
  select infrastructure.
- It keeps failed attempts visible instead of rerunning until a favourable
  result appears.

DeepInfra and Venice both passed 4/4 diagnostic calls. DeepInfra was chosen
because its dated endpoint snapshot showed 99.74% one-day availability versus
98.97% for Venice, together with lower median latency, higher throughput and a
lower price. Both alternatives were FP8, so this tie-break did not choose
between different precisions.

## Why the alternatives were not used

- **Do not edit `agents_v5` or `agents_v6`:** real runs already identify those
  versions. Editing them would make stored provenance false.
- **Do not enable automatic fallback:** different responses within one run
  could come from different providers or quantisations, weakening experimental
  control and making provider effects harder to interpret.
- **Do not keep Parasail only because it is BF16:** greater numerical precision
  is not useful if the endpoint repeatedly rejects requests. The full pilot
  provided stronger reliability evidence than the initial smoke call.
- **Do not use Mistral Small 4:** at the time checked, it did not satisfy the
  project's multi-host replacement rule.
- **Do not use Mistral Nemo:** it was smaller and had fewer suitable healthy
  independent routes at the dated check.
- **Do not switch to the Mistral Batch API:** it is asynchronous and would
  change the execution and latency semantics of the experiment.
- **Do not move to a separate direct Mistral account:** that would add separate
  authentication, billing and calling code and would leave the single-
  OpenRouter experimental design.
- **Do not remove the Mistral agent:** that would change the fixed five-family
  research design.
- **Do not raise token ceilings to repair HTTP 429 or 404:** token ceilings
  cannot fix endpoint availability or rate limiting. A bounded Qwen test at
  4096 also showed that a larger ceiling did not reliably eliminate its
  truncations.
- **Do not repeatedly rerun the main experiment until every request succeeds:**
  that would hide the observed provider failure rate. Failures are data and
  remain stored with no vote.

## Frozen `agents_v7` settings

All agents use temperature `0`, top-p `1.0`, required-parameter routing,
provider fallback disabled and at most two attempts per paid response.

| Agent | Model | Pinned endpoint | Output ceiling |
|---|---|---|---:|
| Llama | `meta-llama/llama-4-maverick` | `digitalocean` | 1024 |
| Qwen | `qwen/qwen3.8-27b` | `parasail/fp8` | 3072 |
| Mistral | `mistralai/mistral-small-3.2-24b-instruct` | `deepinfra/fp8` | 1024 |
| DeepSeek | `deepseek/deepseek-v4-pro-0813` | `digitalocean` | 2048 |
| Gemma | `google/gemma-4-31b-it` | `deepinfra/fp8` | 1024 |

Qwen also requests a best-effort reasoning maximum of 2048 tokens. The serving
provider has previously exceeded that requested reasoning cap, so the total
3072 output ceiling is the enforceable protection.

## Accepted pilot evidence

Run: `pilot_agents_v7_20260922T181727Z`

- 20 questions, two rounds, 200 stored responses.
- 80 responses came from cache; 120 were paid.
- 122 paid attempts: two first-attempt Mistral 429s recovered on retry.
- Zero final `API_ERROR` responses.
- Mistral: 40/40 valid.
- Remaining non-API failures: Qwen 3 truncations, Gemma 2 truncations and
  DeepSeek 1 parse failure.
- Round 1 group accuracy: 11/20, or 55%.
- Round 2 group accuracy: 16/20, or 80%.
- Transitions: 11 stayed correct, 5 became correct, 0 became incorrect and 4
  stayed incorrect.
- Four improvements resolved a Round 1 no-consensus result; one changed a
  decided but wrong Round 1 answer (`A`) to the correct Round 2 answer (`C`).
- Complete-case subset: 11/17 (64.7%) to 13/17 (76.5%).
- Primary paired bootstrap interval for the +25-point pilot change: +10 to +45
  points; exact McNemar p = 0.0625. The pilot is too small for a final research
  conclusion.
- Cost: $0.100227. Wall time: 29.1 minutes.
- Projection from paid responses: about $2.51 for 300 uncached questions under
  the same sequential design. This is an estimate, not a spending guarantee.

The pilot passed because the pipeline completed, the provider repair worked,
all failures were stored and interpretable, and the remaining truncation/parse
rates were accepted as measurable model-output behaviour rather than hidden by
further tuning. Pilot accuracy does not enter the final dissertation results.

## Limitations that must be reported

- Mistral Large was replaced by the smaller Mistral Small 3.2 after the first
  pilot. The `agents_v7` Mistral achieved 60% Round 1 valid-answer accuracy on
  20 pilot questions, versus 73.7% for the earlier Large model. This small,
  hosted and partly non-identical comparison suggests a possible capability
  cost; it does not prove one.
- Mistral now uses FP8 rather than the rejected Parasail BF16 route.
- DeepInfra serves both Mistral and Gemma, so one provider-wide outage could
  affect two agents even though model identities remain different.
- The preflight is a point-in-time check, not a future uptime guarantee.
- Hosted inference can remain nondeterministic at temperature zero.
- Five truncations and one parse failure occurred. They are preserved and cast
  no vote; the three-of-five threshold never falls.
- The 20-question pilot estimates reliability and cost poorly compared with the
  300-question main run.
- Round 2 adds both peer information and another inference attempt. The change
  between rounds is therefore not a pure causal estimate of communication.

## Audit trail

- Provider/token diagnostic: `docs/diagnostic_review_20260922.md`
- Original pilot review: `docs/pilot_review_20260910.md`
- Decisions: D022–D026 in `docs/decisions.md`
- Raw diagnostic databases remain under ignored `storage/`; they are not
  committed because they contain full raw model responses.
- `agents_v7.yaml` SHA-256 at freeze:
  `a4daeca0dc6789a0d981c1f117c730a97ec79bdc63c9ee0e1e8517d6fae8d6a9`

## What remains

The model, provider, prompt, parser, voting, cache and retry semantics are
frozen for the main experiment. Before the 300-question run:

1. finish the main-run command and its offline tests;
2. choose sequential/overnight execution or implement tested bounded
   parallelism without changing experimental semantics;
3. set a finite OpenRouter key limit above the projected cost;
4. back up the frozen data, results database and cache;
5. run the free live preflight immediately before spending.

Any semantic settings change now requires `agents_v8`, a documented decision
and another 20-question pilot. The existing `agents_v5`, `agents_v6` and
`agents_v7` run records must remain untouched.
