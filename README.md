# Multi-Agent Debate Frameworks for Enhanced Truthfulness

Source code for a COMP702 MSc dissertation project (University of Liverpool,
2025/26).

## What the project is

The project asks one question: does debate between different language models
improve answer accuracy beyond what a simple majority vote already gives?

Five open-weight models from five developers answer the same 300
multiple-choice questions from the MMLU-Pro benchmark. Each question goes
through two rounds:

1. **Round 1.** Each model answers alone. The group answer is the letter that
   at least three of the five models chose.
2. **Round 2.** Each model sees the anonymous Round 1 answers of the other
   models, answers again, and the group votes a second time.

The code runs both rounds, stores every reply, and scores the stored replies
afterwards against a separate answer key.

## What the experiment found

The group vote was correct on 245 of 300 questions after Round 1 (81.7%) and
on 255 after Round 2 (85.0%). Twelve questions became correct and two stopped
being correct. Ten of the twelve gains were on questions where Round 1 had no
group answer.

Two limits apply to these numbers:

- "Truthfulness" here only means that an answer matches the MMLU-Pro answer
  key.
- Round 2 gives every model both the other answers and a second attempt. The
  experiment has no second attempt without the other answers, so it cannot say
  how much of the gain came from the debate itself.

The full results are in `reports/main_experiment_20260923/`.

## Quick start

If you only want to see the project work, these four commands are enough on
macOS or Linux. They are explained in the sections below.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m pytest -q
```

## What you need

- Python 3.11 or newer.
- An internet connection for the install step. After that, the tests and the
  dry runs work offline. The viewer also works offline, but it then shows
  fallback fonts, because its fonts come from Google Fonts. A live run of the
  experiment always needs the internet.
- No API key, no account and no payment.

The project was developed and tested on macOS with Python 3.13. The Windows
commands below are the standard ones for Python, but they were not tested on a
Windows computer.

## Setup

Open a terminal in the folder that contains this README. All commands in this
file are run from that folder.

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
```

**Windows, PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt pytest
```

**Windows, Command Prompt**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt pytest
```

The first command makes a private Python environment in a `.venv` folder. The
second turns it on for this terminal window. The third installs the libraries.
When you open a new terminal later, run only the second command again.

## What you can run

Everything in this section is free. No model is called.

### 1. The tests

```bash
python -m pytest -q
```

Expected result: `548 passed, 6 skipped`, in a few seconds. The six skipped
tests compare the public replay data with the private databases, which are not
part of this copy of the project.

To run the tests of one file only, name it:

```bash
python -m pytest tests/test_voting.py
```

### 2. One question through Round 1

```bash
python scripts/run_round1.py --question mmlu_pro_v1:test:7296
```

This is the smallest run. Five agents answer one question alone, the replies
are read and stored, and the group votes.

### 3. One question through both rounds

```bash
python scripts/run_debate.py --question mmlu_pro_v1:test:7296
```

This runs one complete debate: Round 1, the first vote, Round 2 and the second
vote. It prints each agent's status and letter and both votes.

### 4. The 20 pilot questions and the 300 experimental questions

```bash
python scripts/run_pilot.py
python scripts/run_experiment.py
```

These run the same two rounds for many questions. The second command stores
3,000 response records in a temporary database and takes a few seconds.

### What a dry run is

Commands 2 to 4 are dry runs. A dry run does everything a real run does, except
that the replies do not come from models. They come from prepared test replies
called fixtures. Some fixtures are broken on purpose, to show how a refusal or
a failed call is handled. The results go to a database in a temporary folder
of your system, never to the real results database. The run prints the path.

Because the replies are not real, a dry run prints no accuracy. The pilot
summary uses the words "paid attempts" and prints a cost projection. In a dry
run these are simulated and cost nothing.

### 5. The replay viewer

```bash
python -m streamlit run app/viewer.py
```

This opens a page in your browser. If it does not open, go to
`http://localhost:8501`. Stop it with Ctrl+C in the terminal.

The viewer shows the real results of the 300-question experiment and replays
six stored debates step by step: the question, each model's Round 1 answer, the
first vote, what each model was shown in Round 2, the revised answers and the
second vote. It only reads a prepared data file. It calls no model and collects
nothing from the person using it. The only thing the page loads from another
site is its fonts.

## How the code fits together

The files in `scripts/` are the ones you run. The files in `src/mad/` are the
parts they use. Each Python file starts with a short text that says what the
file is for, whether you run it, and which files call it.

```text
scripts/run_round1.py ─────────────► round1.py
scripts/run_debate.py ──┐
scripts/run_pilot.py ───┼──► debate.py ──► round1.py   (Round 1 comes first)
scripts/run_experiment.py ┘

round1.py and debate.py use, for every agent:
    prompts_v1.py   builds the messages
    runner.py       gets the reply, has it read, stores it
        api_client.py   sends the request to OpenRouter (live runs only)
        cache.py        reuses a stored reply for an identical request
        parser_v1.py    reads the answer letter and gives the reply a status
        database.py     stores the reply and every attempt
    voting.py       gives the group answer after all five have replied

After a run has finished:
    scripts/evaluate_experiment.py ──► evaluation.py   (the only reader of the answer key)
    scripts/export_public_viewer_bundle.py ──► the data file for app/viewer.py
```

Two rules shape this layout. The answer key never reaches code that builds
prompts or calls models, so scoring happens in a separate step. And both rounds
share the same calling code in `runner.py`, so they cannot differ in how they
retry, read or store a reply.

## Folder guide

| Folder | What is in it |
|---|---|
| `src/mad/` | The package. One file for each job: prompts, API client, parser, voting, database, cache, the two rounds, evaluation, and the data layers of the viewer. |
| `scripts/` | The files you run: the four runners, the evaluation exporter, the viewer data exporter, and the diagnostic scripts used during the pilot. |
| `app/` | The Streamlit replay viewer and its page template and logos. |
| `tests/` | The automated tests. All of them run offline. |
| `configs/` | The benchmark settings and seven versions of the agent settings. `agents_v7.yaml` is the frozen version used in the experiment. |
| `data/` | The MMLU-Pro download, the frozen question sets with their separate answer keys, and the data file of the viewer. |
| `reports/` | The saved results of the real experiment: a report, a summary and seven tables. |
| `docs/` | The project records. See the next section. |

## Where to start reading

A good order for someone new to the code:

1. `docs/project_record.md`. It describes the finished system part by part.
2. `src/mad/round1.py`, then `src/mad/debate.py`. These are the two rounds.
3. `src/mad/runner.py`. It is the work done for each agent in both rounds.
4. `src/mad/voting.py` and `src/mad/parser_v1.py`. They are short and hold the
   two rules that matter most for the results.
5. `src/mad/evaluation.py`. It turns stored replies into the reported numbers.

Code comments refer to decisions (D001 to D030) and build stages (P1 to P13).
The decisions are in `docs/decisions.md` and the stages in `docs/pipeline.md`.

| File in `docs/` | What it is |
|---|---|
| `project_record.md` | The finished system, part by part. Start here. |
| `decisions.md` | Every fixed decision, with its date and reason. |
| `pipeline.md` | How each build stage works. |
| `development_log.md` | The dated diary of the work, the tests and the problems. |
| `main_experiment_20260923.md` | The record of the one formal run. |
| `pilot_review_20260910.md` | The review of the first pilot. |
| `provider_repair_and_v7_pilot_20260922.md` | Why the host of one model changed before the main run. |
| `diagnostic_review_20260922.md` | Six paid test calls made after the first pilot. |
| `qualitative_review_20260925.md` | A close look at eight stored debates. |
| `deployment.md` | How the public viewer gets its data. |

## The real experiment

The results come from one recorded run,
`experiment_agents_v7_20260923T114934Z`, made on 23 September 2026 through the
OpenRouter API. It cost $2.92.

The experiment is not meant to be run again. A live run needs a paid OpenRouter
API key in a `.env` file (copy `.env.example` and fill it in), it costs real
money, and it would not give the same replies, because hosted models can
change.

The debate, pilot and experiment runners are dry by default. They make paid
model calls only when both `--live` and `--yes-spend-real-money` are given.
`scripts/check_models.py` is different: it always contacts OpenRouter and needs
a key, so it is not part of the steps above.

## What is not included

- **The private results database and cache** (the `storage/` folder). They hold
  the raw replies of all runs, including the pilot and development runs.
- **The API key** (the `.env` file).
- **The dissertation**, which is a separate document.

Because the private database is missing, four scripts cannot do their full job
in this copy: `evaluate_experiment.py`, `export_public_viewer_bundle.py`,
`diagnose_pilot.py` and `probe_mistral_hosts.py`. The saved output of the first
two is in `reports/` and `data/public_viewer/`.

## Problems you may meet

- **`python3` or `py` is not found.** Python is not installed, or it is not on
  the path. Install Python 3.11 or newer from python.org.
- **PowerShell refuses to run `Activate.ps1`.** Use the Command Prompt commands
  instead, or allow scripts for the current window with
  `Set-ExecutionPolicy -Scope Process RemoteSigned`.
- **`No module named pytest` or `streamlit`.** The environment is not active.
  Run the activate command from the Setup section again.
- **The tests show 554 passed and none skipped.** You have the private
  databases. This is correct.
- **`scripts/check_public_deployment.py` fails with a git error.** That script
  asks git which files are tracked, so it works only in a git clone of the
  project and not in an unpacked ZIP file.

## Third-party material and AI use

The questions come from the MMLU-Pro benchmark (TIGER-Lab, MIT licence). The
source, revision and licence are in `THIRD_PARTY_NOTICES.md`. The Python
libraries are listed in `requirements.txt`.

AI assistants were used to write and debug this code. The dissertation has a
Generative AI Use Declaration that describes this.
