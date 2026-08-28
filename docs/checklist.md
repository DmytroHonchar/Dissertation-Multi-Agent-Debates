# Master Checklist

Order of work and current progress. This file tracks *what is done*; it is not
the authority on what is decided (`docs/decisions.md`) or how to build a stage
(`docs/pipeline.md`).

Keep one copy. If a checklist also lives in Notion, replace it there with a link
to this file — two copies means neither is trusted.

Last reconciled against the repository: 2026-08-28.

The core experiment is exactly two rounds. A third round is a desirable future
extension only (D014) and appears nowhere in this checklist.

## Fixed design

Five heterogeneous agents — Llama, Qwen, Mistral, DeepSeek, Gemma — through one
OpenRouter key.

Round 1: five independent responses, then a group vote. Round 2: each agent sees
the anonymised responses of the other four and answers again.

A group answer needs three matching votes out of five. A failed response gets no
vote and the threshold stays three. No judge model, no homogeneous condition. A
third round is desirable only.

Dataset is frozen: 300 experimental and 20 separate pilot questions from the
MMLU-Pro test split.

The three-model `0.x` design is obsolete. Its engineering detail now lives,
updated, in `docs/pipeline.md`.

## Done

- [x] Send the proposal to Meng
- [x] Set up OpenRouter and confirm the five model IDs, context limits and prices
- [x] Estimate 3,200 calls plus retries; budget set to £15 / ~$20 key limit
- [x] Create `docs/development_log.md`
- [x] Record models and decisions in `docs/decisions.md`
- [x] Freeze the question set — `data/frozen/mmlu_pro_v1` (P1)
- [x] Build the shared OpenRouter calling layer and call all five models — `src/mad/api_client.py`, `scripts/check_models.py` (P7)
- [x] Create the five-model registry — `configs/models/agents_v1.yaml`

## Now — before anything else

- [ ] Commit the five-model work, the scaffold deletion and the documentation consolidation
- [ ] Build P2 prompts and P6 parser toward Milestone 1

Settled on 2026-08-28: temperature `0` and top-p `1.0` (D002), retry once /
two attempts total (D012), paired bootstrap plus McNemar (D011). Connectivity to
all five model IDs confirmed with live calls.

## Before the pilot — deferred, not forgotten

- [ ] **Provider pinning (D015).** Automatic routing is fine for building, but it changed provider between consecutive runs and providers serve the same model at different quantisations. Pin each agent and record what happens when a pinned provider is unavailable, before freezing settings.
- [ ] **Mistral availability (D016).** Intermittent HTTP 429 from the shared upstream pool. Measure the real rate over the 20 pilot questions and decide how to handle it. Do not change the model.
- [ ] Confirm `max_tokens` per round — 1024 is provisional and reasoning models spend tokens before answering (D002)
- [ ] Fix and record the bootstrap seed for the D011 confidence interval
- [ ] Note for P6: Mistral answered `'Yes.'` to a prompt demanding the single word `ready`, twice. Test the parser against loose instruction-format compliance.

## Pending proposal corrections — do before submitting CA1 on 11 September

The proposal is not yet submitted. These are documentation-only edits, made in
the proposal source, not in this repository.

- [ ] Change the title-page submission date from 28 August to **11 September 2026**
- [ ] Rewrite Section 5 (Preliminary Work). It currently describes model configuration and OpenRouter access as empty scaffolds. Both are implemented and were verified with live calls to all five models. State accurately that: dataset preparation, validation, frozen sampling and tests are working; the five-model configuration and OpenRouter foundation are working; all five model IDs have been checked with live calls; and debate execution, voting, caching, evaluation and the replay interface remain unfinished.
- [ ] Record **temperature 0** in the proposal's model settings
- [ ] Record the **retry-once** policy consistently with D012
- [ ] Record the resolved statistical method from D011: paired question-level bootstrap, 10,000 resamples, 95% percentile interval, McNemar's exact test, four transition counts
- [ ] Mention provider pinning and its two unresolved cases (D015) in the risk table

## Build

- [ ] Round 1 prompt and formatting function — `prompts_v1.py` (P2)
- [ ] Parser and failure statuses, with the full test set — `parser_v1.py` (P6)
- [ ] Results database and schema — `database.py` (P4)
- [ ] Three-of-five voting and consensus states (P9)
- [ ] **Milestone 1** — one real pilot question through Round 1: five calls, parsed, stored, voted, inspected by hand
- [ ] Response cache — `cache.py` (P5)
- [ ] Round 1 configuration and version labels (P8)
- [ ] Round 2 with four anonymised peer responses per agent — `debate.py` (P10)
- [ ] **Milestone 2** — one real pilot question through both rounds, peer inputs verified by hand
- [ ] Whole pipeline on deterministic fake responses, then limited real calls
- [ ] Evaluation script, built before the pilot runs — `evaluation.py` (P12). Add `scipy` or `statsmodels` to `pyproject.toml` when starting it.

## Travel checkpoint — 12 September

- [ ] Commit and push everything
- [ ] Back up `data/frozen/` and the results database off-machine
- [ ] Write the exact resume point into `docs/development_log.md`

## Pilot and main experiment

- [ ] Run the 20-question pilot through both rounds (P11)
- [ ] Inspect transcripts, parsing, failures, truncation, context use, cost, latency
- [ ] Validate the evaluation script on pilot results
- [ ] Fix problems and repeat the pilot if needed
- [ ] Freeze prompts, settings, parser, retry rules and configuration; record every version name in `decisions.md`
- [ ] Run the 300-question main experiment with nothing changed (P12)
- [ ] Verify the run completed and back up the results database
- [ ] Final evaluation: the three D009 measures, McNemar, transition table, cost and latency
- [ ] Streamlit replay interface — **needed working by CA2, 6 November** (P13). Add `streamlit` to `pyproject.toml` when starting it.

## Writing alongside development

- [ ] After each component, add a dated entry to `docs/development_log.md`: built, why, tested, problems, next
- [ ] Once a week, turn log entries into dissertation paragraphs
- [ ] Save diagrams, screenshots and example transcripts as you go

## CA1 — 11 September

- [ ] Apply only the necessary feedback from Meng
- [ ] Compile and visually inspect the final PDF
- [ ] Submit and save the submitted version

## CA2 — 6 November (Q&A 9–13 November)

- [ ] Check the current format and length on Canvas
- [ ] Slides: aim, architecture, completed implementation, demonstration, results, next steps
- [ ] Working demonstration from a real stored debate
- [ ] Practise and record
- [ ] Review and submit the recording
- [ ] Prepare for and attend the Q&A

## CA3 — 27 November

- [ ] Check the current page limit and submission instructions on Canvas
- [ ] Complete the dissertation from CA1 and the development log
- [ ] Results, discussion, limitations, conclusion, self-reflection
- [ ] Abstract, ethics statement, GenAI declaration, references, appendices
- [ ] Check every figure, table, citation and cross-reference
- [ ] Compile and inspect the final PDF
- [ ] Code-and-data ZIP with no API keys or temporary files
- [ ] Confirm both files open correctly
- [ ] Submit both and save the confirmations
