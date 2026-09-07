"""Tests for the complete one-question debate command. Always offline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import requests

from mad.database import ResultsDatabase
from mad.debate import DEBATE_CONFIG_VERSION
from mad.prompts_v1 import DEBATE_PROMPT_VERSION

REPO = Path(__file__).resolve().parents[1]
PILOT_ID = "mmlu_pro_v1:test:7296"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the debate CLI tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "run_debate", REPO / "scripts" / "run_debate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_prints_both_rounds_and_never_uses_production_storage(cli, capsys):
    assert cli.main(["--question", PILOT_ID]) == 0
    output = capsys.readouterr().out

    assert "dry run" in output
    assert "Round 1" in output
    assert "Round 2" in output
    assert output.count("vote: CONSENSUS -> A") == 2
    assert "storage/results.sqlite" not in output


def test_dry_run_stores_one_complete_debate_with_truthful_labels(cli, tmp_path):
    path = tmp_path / "debate.sqlite"
    assert cli.main(["--question", PILOT_ID, "--db", str(path)]) == 0

    with ResultsDatabase(path) as db:
        runs = db.read_runs()
        assert len(runs) == 1
        assert runs[0]["config_name"] == DEBATE_CONFIG_VERSION
        assert runs[0]["prompt_version"] == DEBATE_PROMPT_VERSION
        assert runs[0]["settings_version"] == "agents_v5"
        assert runs[0]["ended_at"] is not None
        assert len(db.read_responses(runs[0]["run_id"], round=1)) == 5
        assert len(db.read_responses(runs[0]["run_id"], round=2)) == 5
        assert len(db.read_outcomes(runs[0]["run_id"])) == 2


def test_live_mode_requires_the_second_spending_confirmation(cli, capsys):
    assert cli.main(["--question", PILOT_ID, "--live"]) == 1
    output = capsys.readouterr().out
    assert "yes-spend-real-money" in output
    assert "10 calls" in output


def test_spending_confirmation_without_live_mode_is_refused(cli, capsys):
    assert cli.main(["--question", PILOT_ID, "--yes-spend-real-money"]) == 1
    assert "refused" in capsys.readouterr().out


def test_experimental_questions_are_refused(cli, capsys):
    experimental_id = json.loads(
        (REPO / "data/frozen/mmlu_pro_v1/metadata/experimental_ids.json").read_text()
    )[0]
    assert cli.main(["--question", experimental_id]) == 1
    assert "experimental" in capsys.readouterr().out


def test_dry_run_cannot_write_to_the_real_results_database(cli, capsys):
    production = REPO / "storage" / "results.sqlite"
    before = production.stat() if production.exists() else None

    assert cli.main(["--question", PILOT_ID, "--db", str(production)]) == 1
    assert "refused" in capsys.readouterr().out

    if before is not None:
        after = production.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)


def test_a_crash_closes_the_client_and_leaves_the_run_unfinished(
    cli, monkeypatch, tmp_path
):
    closed = []

    class TrackingFixture(cli.FixtureClient):
        def close(self):
            closed.append(True)
            super().close()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "FixtureClient", TrackingFixture)
    monkeypatch.setattr(cli, "run_debate_question", explode)
    path = tmp_path / "crashed.sqlite"

    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["--question", PILOT_ID, "--db", str(path)])

    assert closed == [True]
    with ResultsDatabase(path) as db:
        run = db.read_runs()[0]
        assert run["ended_at"] is None
