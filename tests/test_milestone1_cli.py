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
    # The real Milestone 1 run lives in this file now. Its bytes must not move.
    before = production.stat() if production.exists() else None

    assert cli.main(["--question", PILOT_ID, "--db", str(production)]) == 1
    assert "refused" in capsys.readouterr().out

    if before is not None:
        after = production.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size), (
            "the real results database was touched by a dry run"
        )


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


def test_the_selected_registry_becomes_the_stored_settings_version(cli, tmp_path):
    """A run on agents_v2 must never be labelled agents_v1."""
    from mad.database import ResultsDatabase

    db_path = tmp_path / "v2.sqlite"
    assert cli.main(["--question", PILOT_ID, "--agents", "agents_v2", "--db", str(db_path)]) == 0

    with ResultsDatabase(db_path) as db:
        run = db.read_runs()[0]
        assert run["settings_version"] == "agents_v2"
        assert run["run_id"].startswith("round1_agents_v2_")


def test_agents_v2_actually_raises_the_two_ceilings(cli, tmp_path, capsys):
    assert cli.main(["--question", PILOT_ID, "--agents", "agents_v2",
                     "--db", str(tmp_path / "x.sqlite")]) == 0
    out = capsys.readouterr().out
    assert "agent_qwen=2048" in out and "agent_deepseek=2048" in out
    assert "agent_llama=1024" in out


def test_agents_v3_raises_only_qwen_again(cli, tmp_path, capsys):
    assert cli.main(["--question", PILOT_ID, "--agents", "agents_v3",
                     "--db", str(tmp_path / "v3.sqlite")]) == 0
    out = capsys.readouterr().out
    assert "agent_qwen=3072" in out
    assert "agent_deepseek=2048" in out
    assert "agent_llama=1024" in out


def test_a_dry_run_never_creates_or_touches_the_real_cache_file(cli, tmp_path):
    real_cache = REPO / "storage" / "cache.sqlite"
    before = real_cache.stat() if real_cache.exists() else None

    assert cli.main(["--question", PILOT_ID, "--db", str(tmp_path / "y.sqlite")]) == 0

    if before is None:
        assert not real_cache.exists(), "a dry run created the real cache"
    else:
        after = real_cache.stat()
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
