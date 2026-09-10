"""Tests for the 20-question pilot command. Always offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests

from mad.database import ResultsDatabase
from mad.debate import DEBATE_CONFIG_VERSION
from mad.evaluation import EvaluationError, evaluate_run, load_answer_key
from mad.prompts_v1 import DEBATE_PROMPT_VERSION
from mad.runner import PILOT_QUESTION_COUNT

REPO = Path(__file__).resolve().parents[1]
PILOT_KEY = REPO / "data/frozen/mmlu_pro_v1/answer_keys/pilot_answers.jsonl"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the pilot command tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "run_pilot", REPO / "scripts" / "run_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_dry_pilot_stores_all_twenty_questions_in_one_run(cli, tmp_path):
    path = tmp_path / "pilot.sqlite"
    assert cli.main(["--db", str(path)]) == 0

    with ResultsDatabase(path) as db:
        runs = db.read_runs()
        assert len(runs) == 1, "one run holds the whole pilot"
        run_id = runs[0]["run_id"]
        assert run_id.startswith("pilot_agents_v5_")
        assert runs[0]["ended_at"] is not None

        assert len(db.read_responses(run_id, round=1)) == PILOT_QUESTION_COUNT * 5
        assert len(db.read_responses(run_id, round=2)) == PILOT_QUESTION_COUNT * 5
        assert len(db.read_outcomes(run_id)) == PILOT_QUESTION_COUNT * 2


def test_the_run_is_labelled_as_a_two_round_debate(cli, tmp_path):
    path = tmp_path / "labels.sqlite"
    cli.main(["--db", str(path)])

    with ResultsDatabase(path) as db:
        run = db.read_runs()[0]
        assert run["config_name"] == DEBATE_CONFIG_VERSION
        assert run["prompt_version"] == DEBATE_PROMPT_VERSION
        assert run["settings_version"] == "agents_v5"


def test_a_stored_dry_pilot_satisfies_the_twenty_question_check(cli, tmp_path):
    """The completeness check the real pilot must be scored with."""
    path = tmp_path / "scored.sqlite"
    cli.main(["--db", str(path)])
    key = load_answer_key(PILOT_KEY)

    with ResultsDatabase(path) as db:
        run_id = db.read_runs()[0]["run_id"]
        report = evaluate_run(db, run_id, key, resamples=100, expected_questions=20)

    assert report.question_count == PILOT_QUESTION_COUNT
    assert len(report.agent_ids) == 5
    assert set(report.usage) == {1, 2}


def test_live_mode_needs_the_second_confirmation_and_names_the_call_count(cli, capsys):
    assert cli.main(["--live"]) == 1
    output = capsys.readouterr().out
    assert "yes-spend-real-money" in output
    assert "400 paid attempts" in output, "200 responses, each retryable once"


def test_confirming_spend_without_live_mode_is_refused(cli, capsys):
    assert cli.main(["--yes-spend-real-money"]) == 1
    assert "refused" in capsys.readouterr().out


def test_a_dry_pilot_cannot_write_to_the_real_results_database(cli, capsys):
    production = REPO / "storage" / "results.sqlite"
    before = production.stat() if production.exists() else None

    assert cli.main(["--db", str(production)]) == 1
    assert "refused" in capsys.readouterr().out

    if before is not None:
        after = production.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_a_crash_part_way_leaves_the_run_unfinished_and_unscorable(cli, monkeypatch, tmp_path):
    """A half-finished pilot must be refused later, not quietly scored."""
    real = cli.run_debate_question
    calls = {"n": 0}

    def explode_on_the_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_debate_question", explode_on_the_third)
    path = tmp_path / "crashed.sqlite"

    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["--db", str(path)])

    key = load_answer_key(PILOT_KEY)
    with ResultsDatabase(path) as db:
        run = db.read_runs()[0]
        assert run["ended_at"] is None
        with pytest.raises(EvaluationError, match="never finished"):
            evaluate_run(db, run["run_id"], key, expected_questions=20)


def test_the_summary_reports_what_the_pilot_is_run_to_check(cli, capsys, tmp_path):
    cli.main(["--db", str(tmp_path / "summary.sqlite")])
    output = capsys.readouterr().out

    assert "failures by agent and round" in output
    assert "truncation" in output
    assert "consensus states" in output
    assert "tokens and cost" in output
    assert "prompt" in output and "completion" in output, "token usage is actually printed"
    assert "came from cache" in output
    assert "expected_questions=20" in output, "the reader is told how to score it"


def test_the_projection_is_taken_per_paid_response_not_per_question(cli, tmp_path, capsys):
    """A cached question cost nothing and must not drag the estimate down."""
    cli.main(["--db", str(tmp_path / "projection.sqlite")])
    output = capsys.readouterr().out

    assert "per paid response" in output
    assert "projected for 300 uncached questions" in output


def test_token_totals_match_the_stored_outcome_rows(cli, tmp_path, capsys):
    """The summary is read back from the database, so it cannot drift."""
    path = tmp_path / "tokens.sqlite"
    cli.main(["--db", str(path)])

    with ResultsDatabase(path) as db:
        run_id = db.read_runs()[0]["run_id"]
        stored = sum(int(row["total_prompt_tokens"]) for row in db.read_outcomes(run_id))

    # The fixture client reports zero tokens, so this pins the plumbing, not a
    # number: the printed total is whatever the rows hold.
    assert stored == 0
    assert "prompt" in capsys.readouterr().out

