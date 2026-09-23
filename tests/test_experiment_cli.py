"""Tests for the protected 300-question main-experiment command. Offline only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import requests

from mad.api_client import ApiConfigurationError, load_model_registry
from mad.database import ResultsDatabase
from mad.evaluation import evaluate_run, load_answer_key
from mad.round1 import Round1Config, run_round1_question
from mad.runner import (
    EXPERIMENTAL_QUESTION_COUNT,
    FixtureClient,
    RunnerError,
    load_experimental_questions,
)


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTAL_KEY = (
    REPO / "data/frozen/mmlu_pro_v1/answer_keys/experimental_answers.jsonl"
)
EXPERIMENTAL_IDS = (
    REPO / "data/frozen/mmlu_pro_v1/metadata/experimental_ids.json"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the experiment command tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_returns_exactly_the_frozen_experimental_order():
    questions = load_experimental_questions()
    expected_ids = json.loads(EXPERIMENTAL_IDS.read_text())
    assert len(questions) == EXPERIMENTAL_QUESTION_COUNT == 300
    assert [question["stable_id"] for question in questions] == expected_ids


def test_dry_main_run_stores_and_scores_all_300_once(cli, tmp_path, capsys):
    path = tmp_path / "experiment.sqlite"
    assert cli.main(["--db", str(path)]) == 0
    output = capsys.readouterr().out

    with ResultsDatabase(path) as db:
        runs = db.read_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["run_id"].startswith("experiment_agents_v7_")
        assert run["settings_version"] == "agents_v7"
        assert run["ended_at"] is not None
        assert len(db.read_responses(run["run_id"], round=1)) == 1_500
        assert len(db.read_responses(run["run_id"], round=2)) == 1_500
        assert len(db.read_outcomes(run["run_id"])) == 600
        report = evaluate_run(
            db,
            run["run_id"],
            load_answer_key(EXPERIMENTAL_KEY),
            expected_questions=300,
            resamples=100,
        )
    assert report.question_count == 300
    assert "agents_v7 (frozen; no settings override)" in output
    assert "300 experimental questions" in output
    assert "6000 paid attempts" in output
    assert "expected_questions=300" in output
    assert "answer key never entered this process" in output

    # A formal database cannot quietly acquire a second main run.
    assert cli.main(["--db", str(path)]) == 1
    assert "main-experiment run already exists" in capsys.readouterr().out


def test_live_mode_needs_confirmation_and_names_worst_case(cli, capsys):
    assert cli.main(["--live"]) == 1
    output = capsys.readouterr().out
    assert "yes-spend-real-money" in output
    assert "6000 paid attempts" in output


def test_confirmation_without_live_is_refused(cli, capsys):
    assert cli.main(["--yes-spend-real-money"]) == 1
    assert "refused" in capsys.readouterr().out


def test_dry_main_cannot_write_to_production_database(cli, capsys):
    production = REPO / "storage" / "results.sqlite"
    before = production.stat() if production.exists() else None
    assert cli.main(["--db", str(production)]) == 1
    assert "refused" in capsys.readouterr().out
    if before is not None:
        after = production.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_main_command_has_no_model_or_cache_override(cli):
    with pytest.raises(SystemExit):
        cli.main(["--agents", "agents_v6"])
    with pytest.raises(SystemExit):
        cli.main(["--no-cache"])


def test_resume_skips_complete_questions_in_the_same_run(
    cli, monkeypatch, tmp_path, capsys
):
    questions = load_experimental_questions()[:3]
    monkeypatch.setattr(cli, "load_experimental_questions", lambda: questions)
    real_run = cli.run_debate_question
    calls = {"count": 0}

    def stop_before_third(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("local interruption")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(cli, "run_debate_question", stop_before_third)
    path = tmp_path / "resume.sqlite"
    with pytest.raises(RuntimeError, match="local interruption"):
        cli.main(["--db", str(path)])

    with ResultsDatabase(path) as db:
        run = db.read_runs()[0]
        run_id = run["run_id"]
        assert run["ended_at"] is None
        assert len(db.read_outcomes(run_id)) == 4

    monkeypatch.setattr(cli, "run_debate_question", real_run)
    assert cli.main(["--db", str(path), "--resume", run_id]) == 0
    output = capsys.readouterr().out
    assert output.count("already complete — skipped") == 2

    with ResultsDatabase(path) as db:
        assert db.read_run(run_id)["ended_at"] is not None
        assert len(db.read_responses(run_id)) == 30
        assert len(db.read_outcomes(run_id)) == 6


def test_partial_question_is_refused_instead_of_overwritten(cli, tmp_path):
    path = tmp_path / "partial.sqlite"
    question = load_experimental_questions()[0]
    registry = load_model_registry(REPO / "configs/models/agents_v7.yaml")
    config = Round1Config(
        config_version=cli.DEBATE_CONFIG_VERSION,
        settings_version="agents_v7",
        cache_enabled=False,
    )
    run_id = "experiment_agents_v7_partial"
    with ResultsDatabase(path) as db:
        db.start_run(
            run_id,
            config_name=cli.DEBATE_CONFIG_VERSION,
            question_set_version=config.question_set_version,
            prompt_version=cli.DEBATE_PROMPT_VERSION,
            settings_version="agents_v7",
            parser_version=config.parser_version,
        )
        run_round1_question(
            question,
            registry=registry,
            client=FixtureClient(question),
            db=db,
            run_id=run_id,
            config=config,
        )
        with pytest.raises(RunnerError, match="partial audit rows"):
            cli._question_states(
                db,
                run_id,
                question_ids=[question["stable_id"]],
                agent_ids=list(registry),
            )


def test_failed_preflight_creates_no_run_or_paid_call(cli, monkeypatch, tmp_path, capsys):
    path = tmp_path / "preflight.sqlite"
    calls = {"closed": False}

    class RefusingClient:
        def __init__(self, **kwargs):
            pass

        def assert_registry_routes_available(self, registry):
            raise ApiConfigurationError("pin unavailable")

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(cli, "OpenRouterClient", RefusingClient)
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    assert cli.main(
        ["--live", "--yes-spend-real-money", "--db", str(path)]
    ) == 1
    assert "pin unavailable" in capsys.readouterr().out
    assert calls["closed"]
    assert not path.exists(), "preflight must finish before a run/database is created"
