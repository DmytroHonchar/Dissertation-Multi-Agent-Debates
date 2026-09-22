# Where we stopped — 22 September 2026

## Current resume point

The provider repair is complete and the core experiment is frozen as
`agents_v7` (D026). The accepted run is
`pilot_agents_v7_20260922T181727Z`: all 20 questions and both rounds completed,
Mistral returned 40/40 valid responses and no response ended as `API_ERROR`.
Two first-attempt Mistral 429s were recovered by the existing retry. Remaining
measured failures were three Qwen truncations, two Gemma truncations and one
DeepSeek parse failure.

The pilot scored 11/20 in Round 1 and 16/20 in Round 2. These are development
results, not main-experiment conclusions. Cost was $0.100227 and wall time was
29.1 minutes; the paid-response projection is about $2.51 for 300 uncached
questions. Read `docs/provider_repair_and_v7_pilot_20260922.md` for the full
timeline, rejected alternatives, limitations and freeze record.

Next, build and test the 300-question main-run command. Before any main spend:

1. choose sequential/overnight execution or tested bounded parallelism;
2. set a finite OpenRouter key limit above the projection;
3. back up `data/frozen/`, `storage/results.sqlite` and `storage/cache.sqlite`;
4. run the free endpoint preflight immediately before the experiment.

Do not alter a frozen setting. A semantic change now requires `agents_v8`, a
new decision and another complete pilot. Main evaluation must pass
`expected_questions=300`.

The older checkpoint below is historical, not the current instruction to run.

## Historical checkpoint — 7 September

The complete one-question debate path now exists offline:

```text
question
→ Round 1: five independent responses and a vote
→ Round 2: own response plus anonymous peer responses, then five new responses
→ Round 2 vote
→ both rounds stored in one completed database run
```

The current provider-pinned candidate is `agents_v5`. The formal 20-question
pilot and 300-question experiment have not been run.

## Commands

Both commands are free fixture runs unless both live-spending flags are passed.

```bash
.venv/bin/python scripts/run_round1.py --question <pilot-id>
.venv/bin/python scripts/run_debate.py --question <pilot-id>
.venv/bin/python -m pytest
```

`run_round1.py` is a one-round diagnostic. `run_debate.py` is the complete
two-round command and is the one to inspect next.

## Immediate next step

Done on 2026-09-07: one dry debate, then one approved live debate
(`debate_agents_v5_20260907T161401Z`, question 3932, `UNANIMOUS D` in both
rounds, Round 2 $0.0089). The stored rows were checked by hand; see the
development log for the DeepSeek repetition-loop finding.

Also done: the split question `mmlu_pro_v1:test:8844`
(`debate_agents_v5_20260907T163127Z`, $0.0137). Round 1 `NO_CONSENSUS` with two
agents on the key; Round 2 `UNANIMOUS D`, which the key says is wrong. Group
score unchanged under D010 (undecided and wrong both count incorrect); per-agent
correctness fell 2/5 to 0/5. The question is ambiguous and its literal wording
supports D - a stress test, not evidence about accuracy. Read the log entry
before writing anything about this result.

Single live questions are finished. `evaluation.py` (P12, D021) and
`scripts/run_pilot.py` (P11) are both built and tested offline.

Next: run the pilot live, once, by explicit instruction:

```bash
.venv/bin/python scripts/run_pilot.py --live --yes-spend-real-money
```

Then read the summary against P11's checklist, score the stored run, and only
then freeze. Scoring **must** pass `expected_questions=20`, and the main run
`expected_questions=300`. Without it the completeness check is inactive and a
run that lost questions would be scored over a smaller denominator.

Do not run the 300 experimental questions. `load_pilot_question()` refuses them
for the current commands.

## Still to build

1. A command that prints an `EvaluationReport` and writes the result tables.
2. The 300-question main experiment command.
3. `app/viewer.py`, the read-only Streamlit replay interface (P13, needed for CA2).

## Documentation authority

- `docs/decisions.md` — fixed research decisions.
- `docs/pipeline.md` — how each stage works.
- `docs/checklist.md` — completed and remaining work.
- `docs/development_log.md` — dated evidence for the dissertation.
