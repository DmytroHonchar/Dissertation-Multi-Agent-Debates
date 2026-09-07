# Where we stopped — 7 September 2026

## Current position

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

That question was easy and all five agreed before talking. Next: one more
approved live debate on a question where Round 1 split (`mmlu_pro_v1:test:8844`
went `NO_CONSENSUS` three times), so Round 2 is seen doing its job. Then build
the 20-question pilot runner and `evaluation.py`.

Do not run the 300 experimental questions. `load_pilot_question()` refuses them
for the current commands.

## Still to build

1. The batch runner for all 20 pilot questions.
2. `src/mad/evaluation.py`, including the fixed bootstrap seed and McNemar test.
3. The 300-question main experiment command.
4. `app/viewer.py`, the read-only Streamlit replay interface.

## Documentation authority

- `docs/decisions.md` — fixed research decisions.
- `docs/pipeline.md` — how each stage works.
- `docs/checklist.md` — completed and remaining work.
- `docs/development_log.md` — dated evidence for the dissertation.
