"""Tests for the Milestone 1 command line. Offline: the network is switched off."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests

REPO = Path(__file__).resolve().parents[1]
PILOT_ID = "mmlu_pro_v1:test:7296"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the CLI tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)


@pytest.fixture(scope="module")
def cli():
    """The script, imported as a module so main(argv) can be called directly."""
    spec = importlib.util.spec_from_file_location(
        "run_milestone1", REPO / "scripts" / "run_milestone1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_dry_run_completes_and_uses_a_throwaway_database(cli, capsys):
    assert cli.main(["--question", PILOT_ID]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "CONSENSUS" in out
    assert "storage/results.sqlite" not in out, "a dry run never touches the real database"


def test_a_dry_run_can_keep_its_database_at_a_chosen_path(cli, tmp_path, capsys):
    db_path = tmp_path / "keepme.sqlite"
    assert cli.main(["--question", PILOT_ID, "--db", str(db_path)]) == 0
    assert db_path.is_file()


def test_live_without_the_spend_flag_is_refused_at_the_command_line(cli, capsys):
    assert cli.main(["--question", PILOT_ID, "--live"]) == 1
    assert "yes-spend-real-money" in capsys.readouterr().out


def test_the_spend_flag_alone_is_refused_at_the_command_line(cli, capsys):
    assert cli.main(["--question", PILOT_ID, "--yes-spend-real-money"]) == 1
    assert "refused" in capsys.readouterr().out


def test_an_experimental_question_is_refused_at_the_command_line(cli, capsys):
    import json

    experimental_id = json.loads(
        (REPO / "data/frozen/mmlu_pro_v1/metadata/experimental_ids.json").read_text()
    )[0]
    assert cli.main(["--question", experimental_id]) == 1
    assert "experimental" in capsys.readouterr().out


def test_a_dry_run_aimed_at_the_real_database_is_refused_at_the_command_line(cli, capsys):
    production = REPO / "storage" / "results.sqlite"
    assert cli.main(["--question", PILOT_ID, "--db", str(production)]) == 1
    assert "refused" in capsys.readouterr().out
    assert not (REPO / "storage").exists(), "the refusal happened before any file was made"


def test_the_client_is_closed_even_when_the_run_blows_up(cli, monkeypatch, tmp_path):
    closed = []

    class TrackingFixture(cli.FixtureClient):
        def close(self):
            closed.append(True)
            super().close()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "FixtureClient", TrackingFixture)
    monkeypatch.setattr(cli, "run_round1_question", explode)

    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["--question", PILOT_ID, "--db", str(tmp_path / "x.sqlite")])
    assert closed == [True]
