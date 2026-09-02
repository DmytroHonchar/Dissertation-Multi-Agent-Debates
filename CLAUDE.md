# Multi-Agent Debate Frameworks for Enhanced Truthfulness

MSc dissertation, COMP702, University of Liverpool. Dmytro Honchar (201951718,
`sgdhonch`). Supervisor: Meng Fang. Second marker: Katie Atkinson.

## Why this project exists

Prior multi-agent debate (MAD) work reports final group accuracy but rarely
separates three different effects that are bundled inside it:

1. how good each individual model already is,
2. how much is gained simply by **aggregating** independent answers (majority
   voting), and
3. how much is added or lost by the agents actually **communicating**.

Results conflict as a result. ReConcile reported gains from mixed-model debate;
Smit et al. found debate did not consistently beat simpler aggregation; Choi et
al. found most of the gain came from voting and that further debate rounds *hurt*
(Qwen2.5-7B: 99% by vote, 76% after two debate rounds); M3MAD-Bench found debate
inconsistent across thirteen datasets.

**Research question: does cross-model debate improve answer accuracy beyond what
is already achieved through majority voting?**

"Truthfulness" here means correctness against the fixed MMLU-Pro answer labels.
Nothing subtler is claimed.

## The three headline numbers

Everything in the design exists to produce these, on the same 300 questions:

1. **Per-agent Round 1 accuracy** — how good each model is alone.
2. **Round 1 group vote accuracy** — aggregation without any communication.
3. **Round 2 group vote accuracy** — after one round of communication.

(2) − (1) isolates aggregation. (3) − (2) isolates debate, and is the answer to
the research question. Reporting only (3) would reproduce the exact flaw the
project is criticising, so **never collapse these into one number.**

## Fixed design — do not redesign

Five heterogeneous agents, one OpenRouter key. Exact slugs in
`configs/models/agents_v1.yaml`:

| `agent_id` | Model |
|---|---|
| `agent_llama` | `meta-llama/llama-4-maverick` |
| `agent_qwen` | `qwen/qwen3.8-27b` |
| `agent_mistral` | `mistralai/mistral-large-2512` |
| `agent_deepseek` | `deepseek/deepseek-v4-pro-0813` |
| `agent_gemma` | `google/gemma-4-31b-it` |

The agent-to-model mapping is fixed for the whole experiment. Agents keep their
model identity throughout — this is deliberate, and is one of the things Zhang et
al. did not control.

**Round 1** — all five agents get the same question and options and answer
independently, returning reasoning plus one letter. No agent sees another's
answer. No agent is told a second round follows. Then the first majority vote.

**Round 2** — each agent receives the anonymised Round 1 responses of the **other
four**. Never its own. Never with model identities attached. It answers again.
Then the second majority vote.

**Settings** — temperature `0`, top-p `1.0`, for every agent in both rounds.
The pre-pilot candidate is `agents_v5` (D015/D018): Llama 1024, Qwen 3072,
Mistral 1024, DeepSeek 2048 and Gemma 1024 completion tokens. Qwen requests a
2048 reasoning-token maximum, but a live Parasail response reported 2740, so
this is best-effort; the total 3072 ceiling is the hard guard. Every agent is
pinned to one exact endpoint with provider fallbacks off. A temporary failure
retries the same endpoint once, then becomes `API_ERROR` and no vote. Earlier
registry versions are immutable because real stored runs identify themselves by
name. `agents_v5` is used unchanged for Milestone 2 and the 20-question pilot;
it becomes the final frozen configuration only if the pilot passes. Any repair
creates `agents_v6` rather than editing a used version.

**Voting** — a group answer needs at least three matching votes out of five. A
failed or unparseable response contributes no vote, and the threshold stays at
three; it is never reduced to a majority of the successful responses. No judge
model, no tie-break, no confidence weighting, no homogeneous-model condition.

**Dataset** — MMLU-Pro test split, frozen at `data/frozen/mmlu_pro_v1`: 300
experimental and 20 non-overlapping pilot questions, seed 42, stratified across
all 14 subjects. Up to ten options per question — never hardcode four.

There was an earlier three-model GPT/Claude/Gemini design, and an earlier plan to
also use MedQA. Both are obsolete and absent from this git history. Do not
reconstruct either.

## Read these before proposing anything

| File | Answers | Authority |
|---|---|---|
| `docs/decisions.md` | What is decided and frozen | **Highest.** Never contradict it. |
| `docs/pipeline.md` | How each build stage works (P1–P13) | Implementation detail |
| `docs/checklist.md` | Work order and progress | Tracking |
| `docs/development_log.md` | Dated record of what was built and what broke | History |

A decision reached in chat does not exist until it is written into
`docs/decisions.md` with a date. The dissertation is written from
`docs/development_log.md`, so an undocumented change is lost work — after any
meaningful change, append **Built / Why / Tested / Problems / Next**.

## Hard rules

- **Answer keys never reach prompt-building or model-calling code.** Model inputs
  and answer keys are separate files; `src/mad/evaluation.py` is the only module
  permitted to read a key. `tests/test_no_answer_leakage.py` guards this — never
  weaken it.
- `data/frozen/mmlu_pro_v1` is frozen. Never regenerate or overwrite it.
- Store the full raw response before parsing. Never store only the parsed letter.
- Never overwrite a run's rows. Repeating an experiment means a new run ID.
- **A failure is never scored as a wrong answer.** Always report accuracy among
  valid parsed answers *together with* the failure rate by status and by agent.
- The API key lives in the ignored `.env` only — never in configs, the database,
  git, prompts, or results.
- **Real calls cost money against a £15 budget** (~$20 OpenRouter key limit).
  3,200 calls are planned. Do not call models casually, and never run the
  300-question set unless explicitly asked.
- Pilot results are development data and never enter the final results.
- Once the pilot passes, prompts, settings, parser and retry rules are **frozen**.
  Changing one afterwards means a new configuration version and a new run — not
  an edit.

## Ethics

Data Category A, Participant Category 0. No human participants, no personal data,
no clinical records. The replay interface is for inspecting results only and
collects nothing. If anything would involve participants, an ethics amendment is
required first — so do not propose user studies, usability testing, or
questionnaires.

Generative AI is declared as used for planning, discussion and wording. It is
**not** used to create experimental data or results. Never fabricate, simulate or
"fill in" a model response outside an explicitly labelled deterministic test
fixture.

## Deadlines

| Date | Milestone |
|---|---|
| 11 Sept 2026 | CA1 proposal |
| 12 Sept 2026 | Travel checkpoint — everything committed, pushed, backed up |
| 6 Nov 2026 | CA2 video |
| 9–13 Nov 2026 | CA2 Q&A |
| 27 Nov 2026 | Dissertation + code/data ZIP |

The replay interface must work by CA2, because the demonstration uses a real
stored debate.

## Layout

```
configs/    benchmark + model registry + per-round settings (yaml)
data/       raw parquet; frozen question sets and answer keys
docs/       decisions, pipeline, checklist, development log
scripts/    operational entry points
src/mad/    the package (pythonpath = ["src"])
tests/      pytest
storage/    gitignored: results.sqlite, cache.sqlite, logs
```

Implemented: `benchmark.py`, `api_client.py`, `prompts_v1.py`, `parser_v1.py`,
`database.py`, `voting.py`, `round1.py`, `cache.py`. Still to write:
`debate.py` (P10), `evaluation.py` (P12) and `app/viewer.py` (P13). **Create a module when you
write its first real line — do not scaffold empty files.**

## Commands

```bash
.venv/bin/python -m pytest                 # 301 tests, all passing, all offline
.venv/bin/python scripts/check_models.py   # live OpenRouter check — real calls, costs money
.venv/bin/python scripts/run_milestone1.py --question <pilot-id>   # dry run, free; --live --yes-spend-real-money for the real thing
```
