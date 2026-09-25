"""Offline tests for the accepted-main-run evaluation/export command."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mad.database import OutcomeRecord, ResponseRecord, ResultsDatabase
from mad.parser_v1 import STATUS_API_ERROR, STATUS_OK


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "experiment_agents_v7_report_test"
AGENTS = ("agent_a", "agent_b", "agent_c", "agent_d", "agent_e")


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "evaluate_experiment", REPO_ROOT / "scripts" / "evaluate_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _store_response(
    database: ResultsDatabase,
    *,
    question_id: str,
    round_number: int,
    agent_id: str,
    letter: str | None,
    status: str = STATUS_OK,
) -> None:
    database.record_response(
        ResponseRecord(
            run_id=RUN_ID,
            question_id=question_id,
            round=round_number,
            agent_id=agent_id,
            requested_slug=f"test/{agent_id}",
            served_slug=f"test/{agent_id}",
            provider="fixture",
            generation_id=f"{question_id}-{round_number}-{agent_id}",
            raw_response=(
                f"private fixture reasoning FINAL ANSWER: {letter}"
                if letter
                else "private fixture transport failure"
            ),
            status=status,
            extracted_letter=letter if status == STATUS_OK else None,
            extraction_method="LAST_FINAL_ANSWER_MATCH" if status == STATUS_OK else None,
            finish_reason="stop" if status == STATUS_OK else "",
            attempt_count=1,
            selected_attempt=1,
            prompt_version="round1_v1" if round_number == 1 else "round2_v1",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            latency_seconds=1.0,
            cache_hit=False,
        )
    )


def _store_outcome(
    database: ResultsDatabase,
    *,
    question_id: str,
    round_number: int,
    state: str,
    answer: str | None,
    valid_answers: int,
) -> None:
    database.record_outcome(
        OutcomeRecord(
            run_id=RUN_ID,
            question_id=question_id,
            round=round_number,
            consensus_state=state,
            consensus_answer=answer,
            decided=answer is not None,
            valid_answer_count=valid_answers,
            total_prompt_tokens=50,
            total_completion_tokens=100,
            total_cost_usd=0.005,
            total_latency_seconds=5.0,
        )
    )


def _build_two_question_run(path: Path) -> Path:
    with ResultsDatabase(path) as database:
        database.start_run(
            RUN_ID,
            config_name="debate_config_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version="round1_v1+round2_v1",
            settings_version="agents_v7",
            parser_version="parser_v1",
            started_at="2026-09-23T10:00:00+00:00",
        )

        # q1 stays correct and unanimous.
        for round_number in (1, 2):
            _store_outcome(
                database,
                question_id="q1",
                round_number=round_number,
                state="UNANIMOUS",
                answer="A",
                valid_answers=5,
            )
            for agent_id in AGENTS:
                _store_response(
                    database,
                    question_id="q1",
                    round_number=round_number,
                    agent_id=agent_id,
                    letter="A",
                )

        # q2 has no Round 1 majority and one API absence, then becomes correct.
        _store_outcome(
            database,
            question_id="q2",
            round_number=1,
            state="NO_CONSENSUS",
            answer=None,
            valid_answers=4,
        )
        for agent_id, letter in zip(AGENTS[:4], ("A", "C", "D", "E")):
            _store_response(
                database,
                question_id="q2",
                round_number=1,
                agent_id=agent_id,
                letter=letter,
            )
        _store_response(
            database,
            question_id="q2",
            round_number=1,
            agent_id=AGENTS[4],
            letter=None,
            status=STATUS_API_ERROR,
        )

        _store_outcome(
            database,
            question_id="q2",
            round_number=2,
            state="UNANIMOUS",
            answer="B",
            valid_answers=5,
        )
        for agent_id in AGENTS:
            _store_response(
                database,
                question_id="q2",
                round_number=2,
                agent_id=agent_id,
                letter="B",
            )

        database.finish_run(RUN_ID, ended_at="2026-09-23T12:00:00+00:00")

    key = path.with_name("experimental_answers.jsonl")
    key.write_text(
        json.dumps({"stable_id": "q1", "correct_answer": "A"})
        + "\n"
        + json.dumps({"stable_id": "q2", "correct_answer": "B"})
        + "\n",
        encoding="utf-8",
    )
    return key


def test_export_is_offline_complete_and_does_not_change_database(
    cli, monkeypatch, tmp_path, capsys
):
    database_path = tmp_path / "results.sqlite"
    answer_key = _build_two_question_run(database_path)
    output = tmp_path / "reports"
    monkeypatch.setattr(cli, "MAIN_RUN_ID", RUN_ID)
    monkeypatch.setattr(cli, "EXPECTED_QUESTIONS", 2)
    before = _sha256(database_path)

    assert cli.main(
        [
            "--db",
            str(database_path),
            "--answer-key",
            str(answer_key),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert _sha256(database_path) == before
    assert {path.name for path in output.iterdir()} == set(cli.OUTPUT_FILES)

    printed = capsys.readouterr().out
    assert "no API calls and no database changes" in printed
    assert "R1 1/2 (50.0%) -> R2 2/2 (100.0%)" in printed

    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["provenance"]["expected_questions"] == 2
    assert summary["provenance"]["wall_clock_hours"] == 2.0
    assert summary["group_accuracy"][0]["correct"] == 1
    assert summary["group_accuracy"][1]["correct"] == 2
    assert summary["group_transitions"]["became_correct"] == 1
    assert summary["complete_cases"]["question_count"] == 1

    markdown = (output / "evaluation_report.md").read_text()
    assert "Round 1 group vote | 1/2 | 50.0%" in markdown
    assert "Round 2 group vote | 2/2 | 100.0%" in markdown
    assert "private fixture reasoning" not in markdown

    with (output / "question_comparisons.csv").open(newline="") as source:
        questions = list(csv.DictReader(source))
    assert len(questions) == 2
    assert questions[1]["round1_failures"] == "agent_e:API_ERROR"
    assert "private fixture" not in (output / "evaluation_summary.json").read_text()


def test_existing_reports_require_explicit_overwrite(cli, monkeypatch, tmp_path, capsys):
    database_path = tmp_path / "results.sqlite"
    answer_key = _build_two_question_run(database_path)
    output = tmp_path / "reports"
    monkeypatch.setattr(cli, "MAIN_RUN_ID", RUN_ID)
    monkeypatch.setattr(cli, "EXPECTED_QUESTIONS", 2)
    arguments = [
        "--db",
        str(database_path),
        "--answer-key",
        str(answer_key),
        "--output-dir",
        str(output),
    ]
    assert cli.main(arguments) == 0
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    assert cli.main(arguments) == 1
    assert "Pass --overwrite" in capsys.readouterr().out
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original

    assert cli.main([*arguments, "--overwrite"]) == 0
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_missing_database_is_refused_without_creating_output(cli, tmp_path, capsys):
    output = tmp_path / "reports"
    assert cli.main(
        ["--db", str(tmp_path / "missing.sqlite"), "--output-dir", str(output)]
    ) == 1
    assert "database does not exist" in capsys.readouterr().out
    assert not output.exists()


def test_script_has_no_model_calling_dependency():
    source = (REPO_ROOT / "scripts" / "evaluate_experiment.py").read_text()
    assert "mad.api_client" not in source
    assert "OpenRouterClient" not in source
    assert "requests." not in source
