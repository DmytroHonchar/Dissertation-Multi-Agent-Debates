# Where we stopped — 30 August 2026

Read this first tomorrow, then delete it or let it be overwritten.

## What just happened

Everything is committed. `git log` shows six commits from this session; the last
two are `Add Round 1 prompt` and `Make the code readable`. 60 tests pass.

Two things got done today:

1. **`src/mad/prompts_v1.py` was written.** It turns one frozen question into the
   two messages sent to a model.
2. **Every file got a readability pass.** Plain-English docstrings, numbered
   sections, short comments. No code was changed — comments and layout only.

## What you wanted to do next

**Re-read `src/mad/api_client.py`.** You got tired partway through the
walkthrough and asked to go over it again. Start there.

The style we agreed on, so you can hold me to it:

- One line per function, saying what it does
- A `#` comment only where the code looks arbitrary without one
- Section headings like `# 1. Settings`, no dashed lines, no ASCII boxes
- No `(D003)` / `(P2)` references inside function bodies

## The files, most important first

| File | Lines | What it is |
|---|---|---|
| `src/mad/prompts_v1.py` | 163 | The exact words sent to the models |
| `src/mad/api_client.py` | 323 | Every paid API call **← you are here** |
| `tests/test_no_answer_leakage.py` | 36 | Proves answers never reach a model |
| `src/mad/benchmark.py` | 779 | Built the frozen data. Already run, never runs again |
| `tests/test_prompts.py` | 165 | Guards the prompt rules |
| `tests/test_api_client.py` | 110 | Guards retry = 2, temperature 0 |
| `tests/test_benchmark.py` | 144 | Guards dataset validation |
| `tests/test_sampling.py` | 138 | Guards seed-42 reproducibility |
| `tests/test_model_registry.py` | 77 | Guards the five model IDs |
| `scripts/check_models.py` | 124 | Connection checker |

## Still to build — six files, none started

| File | What it does |
|---|---|
| `parser_v1.py` | **Next.** Read the letter out of a reply, classify failures |
| `database.py` | Store every response |
| `debate.py` | Round 2 and the three-of-five vote |
| `cache.py` | Don't pay twice for the same call |
| `evaluation.py` | Accuracy, McNemar, the tables |
| `app/viewer.py` | Streamlit replay screen |

Then: pilot on 20 questions → freeze everything → run the 300.

## Two open things not about code

- **CA1 is 11 September.** The proposal needs its title-page date changed, and
  Section 5 rewritten — it still says the OpenRouter work is an empty scaffold,
  which stopped being true a week ago. Full list in `docs/checklist.md` under
  "Pending proposal corrections".
- **Mistral hits rate limits** during batch calls. Known, harmless for now,
  measured during the pilot. See D016.

## Where things are written down

- `docs/decisions.md` — what is decided (highest authority)
- `docs/pipeline.md` — how to build each stage, P1–P13
- `docs/checklist.md` — what is done, what is next
- `docs/development_log.md` — dated history. **The dissertation is written from this.**
