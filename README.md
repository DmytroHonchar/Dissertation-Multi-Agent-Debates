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

Expected result: `548 passed, 6 skipped`, in a few seconds.

The project has 554 tests. Six of them compare the public replay data with
the private databases, which are not part of this copy of the project, so
those six are skipped. With the private databases present, all 554 pass.

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

There is also an interactive map of the project. Open
`docs/project_map.html` in a browser. It has two views: every folder and file
with the reason it exists, and the path of one question with the file and line
of each function.

### The path of one question

The main run is `scripts/run_experiment.py`. Its `main()` function manages
every step below. Both rounds finish for one question before the next question
starts, and the agents are called one at a time.

**Preparation, done once before any run.** `benchmark.py` checked the MMLU-Pro
download and saved the question sets, the separate answer keys and their
metadata. The run scripts do not call this code again.

**Before any question**

1. `require_spend_confirmation()` refuses a paid run unless both flags are
   given.
2. `load_experimental_questions()` loads and checks the 300 frozen questions.
   They have no answers.
3. `load_model_registry()` reads `agents_v7.yaml`: the five models, their
   providers and their settings.
4. `_inspect_existing_run()` checks a resume, or refuses a second main run in
   the same database.
5. `assert_registry_routes_available()` checks, in live mode only, that each
   model's provider is available. The check is free.
6. `db.start_run()` saves the record of this run.

**Round 1, for one question.** `run_debate_question()` in `debate.py` organises
both rounds. It first calls `run_round1_question()` in `round1.py`:

1. `build_round1_messages()` makes two messages: the instructions, and the
   question with lettered options. All five agents get the same messages.
2. For each of the five agents, one by one, `run_one_agent()` in `runner.py`
   does this:
   - `cache.lookup()` looks for a stored reply to this exact request. If there
     is one, the API request is skipped.
   - If there is none, `complete()` in `api_client.py` builds the request and
     `_post_with_retries()` sends it, with one retry after a temporary failure.
     In a dry run a fixture client gives a prepared reply instead.
   - `parse_response()` in `parser_v1.py` finds "FINAL ANSWER: X" and gives a
     letter or a failure status. It checks if the reply is usable. It does not
     check if it is correct.
   - `db.record_response()` saves the response and its attempt records
     together.
   - `cache.store()` puts a new model reply in the cache.
   - If the API call fails, `run_one_agent()` records `API_ERROR` and goes on
     to the next agent.
3. `tally()` in `voting.py` counts the votes. A group answer needs three
   matching letters out of five.
4. `record_round_outcome()` saves the group result of Round 1.

**Round 2, for the same question.** `run_round2_question()` in `debate.py`:

1. `valid_round1_responses()` reads the stored Round 1 replies and keeps the
   usable ones.
2. For each agent, `peers_for()` takes the valid replies of the other agents,
   and `build_round2_messages()` builds the conversation: the question, the
   agent's own Round 1 reply, and the peer replies labelled PEER RESPONSE 1 to
   4. There are no model names, and the Round 1 vote is not shown.
3. `run_one_agent()` gets the new reply. It is the same function as in Round 1.
4. `tally()` and `record_round_outcome()` give and save the Round 2 result.

**Closing the run.** `_question_states()` checks that every required record
exists, and `db.finish_run()` saves the end time.

**Scoring, a separate command that you run later.**
`scripts/evaluate_experiment.py` checks the accepted run, loads the answer key
with `load_answer_key()`, compares the stored answers with it in
`evaluate_run()`, and writes the files in `reports/`. This is the only place
where the answers are read.

Five things are never saved: the correct answer in the experiment database,
the API key, a fixture reply in the cache, a failed call in the cache, and any
overwrite of an existing record.

## Folder guide

| Folder | What is in it | Why it exists |
|---|---|---|
| `data/` | The questions. `raw/` is the MMLU-Pro download. `frozen/mmlu_pro_v1/model_inputs/` holds the 300 + 20 questions without answers. `answer_keys/` holds the correct answers. `metadata/` holds the chosen IDs, checksums and the validation report. | All five agents must answer the same questions, and the models must never see the answers. |
| `configs/` | Settings. `benchmark/mmlu_pro_v1.yaml` names the dataset and its file hashes. `models/agents_v7.yaml` is the frozen setup of the five agents. | The code reads its settings from these files. Versions v1 to v6 stay because old runs refer to them. These files are frozen, comments included. The SHA-256 fingerprint of `agents_v7.yaml` is part of the experiment record, so any edit would break it. |
| `src/mad/` | The main code, with one job in each file. See the table below. | The scripts call this code. |
| `scripts/` | The commands you run. Four runners in growing size (1 question and 1 round, 1 question and 2 rounds, 20 questions, 300 questions), the scoring command, three check scripts and two scripts for the public viewer. | The project was built in small steps, and each step kept its own command. |
| `app/` | The Streamlit viewer: `viewer.py`, the page template `debate_stage.html`, and the model logos. | To look at the results. |
| `tests/` | The automated tests. `test_no_answer_leakage.py` is the most important one. | They check the code without the internet. |
| `reports/` | The final result tables: a Markdown report, a JSON summary and seven CSV files. | The scoring command writes the scores here. |
| `docs/` | The written record of the project. See the table further down. | To explain the decisions and the history. |
| `storage/` | The private results database, the cache and their backups. This folder is not in the repository. | A run saves its data here. |

The files in `src/mad/`:

| File | Its job |
|---|---|
| `benchmark.py` | Picks and freezes the 300 + 20 questions. This was done once. |
| `prompts_v1.py` | Builds the messages sent to the models. It stops if a question carries an answer field. |
| `api_client.py` | Sends requests to OpenRouter and retries once. The real network call happens here. |
| `cache.py` | Saves replies, so the same request is not paid for twice. |
| `parser_v1.py` | Reads the answer letter and gives a status: `OK`, `REFUSAL`, `TRUNCATED`, `PARSE_FAIL` or `API_ERROR`. |
| `database.py` | Saves everything in the results database. |
| `voting.py` | The three-out-of-five vote. |
| `runner.py` | The code that both rounds share, with `run_one_agent()`. |
| `round1.py` | Runs Round 1 for one question. |
| `debate.py` | Runs Round 2, and runs both rounds together. |
| `evaluation.py` | The only file that reads the answer key. It computes accuracy and the statistics. |
| `viewer_data.py`, `public_viewer_data.py`, `debate_stage.py` | Make the data for the viewer. |
| `diagnostics.py` | Labels why an API call failed. |

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
