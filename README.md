# Multi-Agent Debate Frameworks for Enhanced Truthfulness

Source code for the COMP702 MSc dissertation project (University of Liverpool,
2025/26).

The project asks one question: does cross-model debate improve answer accuracy
beyond what majority voting already achieves? Five open-weight language models
answer the same 300 MMLU-Pro multiple-choice questions in two rounds. In
Round 1 each model answers alone and the group votes. In Round 2 each model
sees the anonymised answers of the others, answers again, and the group votes
a second time. The code runs the two rounds, stores every response, and scores
the stored results afterwards.

## What you can run

Everything below is free and works offline. No API key is needed.

| What | Command | What it shows |
|---|---|---|
| Automated tests | `python -m pytest -q` | The rules of the experiment hold in the code. |
| One-question demo | `python scripts/run_debate.py --question mmlu_pro_v1:test:7296` | Both rounds and both votes for one question, using prepared test replies. |
| Replay viewer | `python -m streamlit run app/viewer.py` | The real results of the 300-question experiment and six stored debates, step by step. |

## Setup

Requires Python 3.11 or newer. The project was developed and tested with
Python 3.13 on macOS.

Run these commands from the folder that contains this README.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt pytest
```

## Running the code

### 1. Tests

```bash
python -m pytest -q
```

Expected result: `548 passed, 6 skipped` in a few seconds. The six skipped
tests need the private results database and cache, which are not part of this
archive
(see "What is not included").

### 2. One-question demo (dry run)

```bash
python scripts/run_debate.py --question mmlu_pro_v1:test:7296
```

This sends one pilot question through Round 1, the first vote, Round 2 and the
second vote. It is a dry run: no model is called. The replies are prepared test
fixtures, some of them deliberately broken, to show how failures are handled.
The results go to a temporary database and cost nothing.

Two more dry runs use the same fixtures:

```bash
python scripts/run_pilot.py        # all 20 pilot questions
python scripts/run_experiment.py   # all 300 experimental questions
```

Dry runs print no accuracy. The answer key is never loaded by code that calls
models, and fixture replies are not real model answers. The pilot summary uses
the words "paid attempts" and prints a cost projection; in a dry run these are
simulated and cost nothing.

### 3. Replay viewer

```bash
python -m streamlit run app/viewer.py
```

This opens a page in the browser. It reads
`data/public_viewer/accepted_experiment_v1.json`, a prepared copy of the real
experiment's results. It shows the overall results and a nine-step replay of
six selected debates. The viewer is read-only: it calls no model and collects
nothing from the person using it.

## The real experiment

The results in the dissertation come from one recorded run,
`experiment_agents_v7_20260923T114934Z`, made on 23 September 2026 through the
OpenRouter API. Its outputs are in `reports/main_experiment_20260923/`: a
Markdown report, a JSON summary and seven CSV tables.

**The experiment is not meant to be rerun.** A live run needs a paid
OpenRouter API key in a `.env` file (see `.env.example`), costs real money, and
would not reproduce the stored replies exactly, because hosted models can
change. The debate, pilot and main experiment runners default to dry mode.
They make paid model calls only when both `--live` and
`--yes-spend-real-money` are supplied. `scripts/check_models.py` is different:
it always contacts OpenRouter and needs a key, so it is not part of the steps
above.

## What is not included

- **The private results database and cache** (`storage/`). They hold the raw
  replies of all runs, including pilot and development runs. Because of this,
  `scripts/evaluate_experiment.py`, which re-scores the real run from its raw
  records, cannot be run from this archive. Its saved output is in `reports/`.
- **The API key** (`.env`).
- **The dissertation**, which is submitted separately.

## Folder guide

| Folder | Contents |
|---|---|
| `src/mad/` | The package: prompts, API client, parser, voting, database, the two rounds, cache and evaluation. |
| `scripts/` | Entry points: the dry-run and live runners, the evaluation exporter, and diagnostic scripts used during the pilot. |
| `app/` | The Streamlit replay viewer. |
| `tests/` | The automated tests. All run offline. |
| `configs/` | The benchmark settings and the seven versions of the agent settings. `agents_v7.yaml` is the frozen version used in the experiment. |
| `data/` | The MMLU-Pro download, the frozen question sets with their separate answer keys, and the viewer's data file. |
| `reports/` | The saved results of the real experiment. |
| `docs/` | Project records: decisions, pipeline notes and the development log. |

## Third-party material and AI use

The questions come from the MMLU-Pro benchmark (TIGER-Lab, MIT licence). The
source, revision and licence are given in `THIRD_PARTY_NOTICES.md`. The Python
libraries used are listed in `requirements.txt`.

AI assistants were used to write and debug this code. This is declared in the
dissertation's Generative AI Use Declaration.
