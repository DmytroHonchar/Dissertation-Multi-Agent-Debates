# Where we stopped — 24 September 2026

## Current resume point

The one formal main experiment is complete. Accepted run
`experiment_agents_v7_20260923T114934Z` used commit `d4109d4` and frozen
`agents_v7`. All 300 questions, 3,000 responses and 600 outcomes were stored;
the run cost $2.920649 and took 10.62 hours. Post-run results and cache backups
passed integrity checks.

Group accuracy moved from 245/300 (81.7%) in Round 1 to 255/300 (85.0%) in
Round 2: +3.33 percentage points, with 12 improvements and two regressions.
Ten of the 12 improvements began without a Round 1 majority. Mistral had 77
terminal upstream HTTP 429 errors from DeepInfra; they remain part of the
observed result and are not repaired by rerunning.

Read `docs/main_experiment_20260923.md` for the complete provenance, results,
failure audit, backup hashes and careful interpretation. Do not rerun the main
experiment or spend more model credit.

Read `docs/project_record.md` for the consolidated explanation of what every
part of the system does, why each decision was made, what alternatives were
rejected and how the final result may be interpreted. Detailed chronological
authority remains in `docs/decisions.md` (D001–D029).

The offline result-table/export command is complete. Its accepted outputs are
under `reports/main_experiment_20260923/`: one readable Markdown report, one
complete JSON record and seven CSV tables. The command rechecks all 300
questions and proves the results database stayed byte-for-byte unchanged.

The representative-question review is complete in
`docs/qualitative_review_20260925.md`. Next: build `app/viewer.py`, and write the
dissertation and CA2 materials. Copy the Git-ignored post-run backups to
separate storage.

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

1. `app/viewer.py`, the read-only Streamlit replay interface (P13, needed for CA2).

## Documentation authority

- `docs/decisions.md` — fixed research decisions.
- `docs/project_record.md` — readable whole-project design and rationale.
- `docs/pipeline.md` — how each stage works.
- `docs/checklist.md` — completed and remaining work.
- `docs/development_log.md` — dated evidence for the dissertation.
