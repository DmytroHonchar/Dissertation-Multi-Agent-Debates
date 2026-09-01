"""Tests for the results database. Offline - no API calls, no frozen data.

Every test builds its own database under tmp_path, so the real
storage/results.sqlite is never created or touched.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from mad import database
from mad.api_client import AttemptRecord, CompletionResult, ModelSpec
from mad.database import (
    AttemptRow,
    DatabaseError,
    DuplicateRecordError,
    OutcomeRecord,
    ResponseRecord,
    ResultsDatabase,
    attempt_rows,
    response_from_completion,
    response_from_failed_call,
)
from mad.parser_v1 import PARSER_VERSION, ParsedResponse, STATUS_API_ERROR, STATUS_OK
from mad.prompts_v1 import FORBIDDEN_FIELDS, PROMPT_VERSION

RUN = "run_test_001"


@pytest.fixture
def db(tmp_path: Path) -> ResultsDatabase:
    with ResultsDatabase(tmp_path / "results.sqlite") as database_:
        database_.start_run(
            RUN,
            config_name="agents_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version=PROMPT_VERSION,
            settings_version="settings_v1",
            parser_version=PARSER_VERSION,
        )
        yield database_


def _spec(**overrides) -> ModelSpec:
    base = dict(
        agent_id="agent_qwen",
        slug="qwen/qwen3.8-27b",
        display_name="Qwen",
        developer="Qwen",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        allow_provider_fallbacks=True,
        require_parameters=True,
    )
    base.update(overrides)
    return ModelSpec(**base)


def _response(**overrides) -> ResponseRecord:
    base = dict(
        run_id=RUN,
        question_id="q0001",
        round=1,
        agent_id="agent_qwen",
        requested_slug="qwen/qwen3.8-27b",
        served_slug="qwen/qwen3.8-27b",
        provider="DeepInfra",
        generation_id="gen-1",
        raw_response="REASONING: alkene.\nFINAL ANSWER: C",
        status=STATUS_OK,
        extracted_letter="C",
        extraction_method="LAST_FINAL_ANSWER_MATCH",
        finish_reason="stop",
        attempt_count=1,
        selected_attempt=1,
        prompt_version=PROMPT_VERSION,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    base.update(overrides)
    return ResponseRecord(**base)


# 1. The file itself


def test_opening_creates_the_folder_the_file_and_the_four_tables(tmp_path):
    path = tmp_path / "nested" / "deeper" / "results.sqlite"
    with ResultsDatabase(path) as db:
        tables = {
            row["name"]
            for row in db._query("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert path.is_file()
    assert {"runs", "model_responses", "response_attempts", "question_outcomes"} <= tables


def test_the_schema_version_is_stamped_on_the_file(tmp_path):
    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        assert db._query("PRAGMA user_version")[0]["user_version"] == database.SCHEMA_VERSION


def test_a_file_from_a_different_schema_version_is_refused(tmp_path):
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {database.SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()

    with pytest.raises(DatabaseError, match="schema version"):
        ResultsDatabase(path)


def test_opening_an_existing_database_changes_none_of_its_bytes(tmp_path):
    """Reading results must never modify them - not even the file header."""
    import hashlib

    path = tmp_path / "results.sqlite"
    with ResultsDatabase(path) as db:
        db.start_run(
            RUN,
            config_name="agents_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version=PROMPT_VERSION,
            settings_version="settings_v1",
            parser_version=PARSER_VERSION,
        )
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    with ResultsDatabase(path) as db:
        db.read_runs()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_reopening_an_existing_file_keeps_its_rows(tmp_path):
    path = tmp_path / "results.sqlite"
    with ResultsDatabase(path) as db:
        db.start_run(
            RUN,
            config_name="agents_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version=PROMPT_VERSION,
            settings_version="settings_v1",
            parser_version=PARSER_VERSION,
        )
    with ResultsDatabase(path) as db:
        assert db.read_run(RUN)["config_name"] == "agents_v1"


def test_foreign_keys_are_enforced(db):
    """Without PRAGMA foreign_keys = ON, SQLite would accept an orphan row."""
    with pytest.raises(DatabaseError):
        db.record_response(_response(run_id="run_that_was_never_started"))


# 2. No answer key may live here


def test_no_table_has_a_column_that_could_hold_the_answer(tmp_path):
    """The hard rule made structural: the key cannot be stored even by mistake."""
    banned = set(FORBIDDEN_FIELDS) | {
        "is_correct", "correct", "correctness", "score", "ground_truth", "expected_answer"
    }
    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        for table in ("runs", "model_responses", "response_attempts", "question_outcomes"):
            columns = {row["name"].lower() for row in db._query(f"PRAGMA table_info({table})")}
            assert not columns & banned, f"{table} exposes {columns & banned}"


def _sql_statements_in(module) -> list[str]:
    """Every SQL string the module can execute, comments and docstrings excluded."""
    tree = ast.parse(Path(module.__file__).read_text())
    keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "PRAGMA", "REPLACE")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            text = first.value if isinstance(first, ast.Constant) else ""
        else:
            continue
        if text.strip().upper().startswith(keywords):
            found.append(" ".join(text.split()).upper())
    return found


def test_the_module_never_replaces_or_deletes_a_row():
    """Only finish_run may update, and only ended_at. Nothing else may rewrite."""
    statements = _sql_statements_in(database)

    assert not [sql for sql in statements if "INSERT OR REPLACE" in sql or "REPLACE INTO" in sql]
    assert not [sql for sql in statements if sql.startswith("DELETE")]

    updates = [sql for sql in statements if sql.startswith("UPDATE")]
    assert updates == ["UPDATE RUNS SET ENDED_AT = ? WHERE RUN_ID = ? AND ENDED_AT IS NULL"], (
        "finish_run stamping ended_at is the only update this file may perform"
    )


# 3. Runs


def test_a_run_cannot_be_started_twice(db):
    with pytest.raises(DuplicateRecordError):
        db.start_run(
            RUN,
            config_name="agents_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version=PROMPT_VERSION,
            settings_version="settings_v1",
            parser_version=PARSER_VERSION,
        )


def test_a_run_records_the_versions_that_produced_it(db):
    run = db.read_run(RUN)
    assert run["prompt_version"] == PROMPT_VERSION
    assert run["parser_version"] == PARSER_VERSION
    assert run["ended_at"] is None


def test_finish_run_sets_the_end_time_once_and_only_once(db):
    db.finish_run(RUN, ended_at="2026-08-31T12:00:00+00:00")
    assert db.read_run(RUN)["ended_at"] == "2026-08-31T12:00:00+00:00"

    with pytest.raises(DatabaseError, match="already finished"):
        db.finish_run(RUN, ended_at="2026-08-31T13:00:00+00:00")
    assert db.read_run(RUN)["ended_at"] == "2026-08-31T12:00:00+00:00"


def test_finishing_an_unknown_run_is_an_error(db):
    with pytest.raises(DatabaseError):
        db.finish_run("no_such_run")


# 4. Responses


def test_the_full_raw_text_survives_the_round_trip(db):
    raw = "REASONING: " + "a long chain of reasoning, " * 60 + "\nFINAL ANSWER: C\nThanks!"
    db.record_response(_response(raw_response=raw))

    stored = db.read_responses(RUN)[0]
    assert stored["raw_response"] == raw, "the full reply is the record, not the letter"
    assert stored["extracted_letter"] == "C"


def test_the_same_agent_cannot_answer_the_same_question_twice_in_a_round(db):
    db.record_response(_response())
    with pytest.raises(DuplicateRecordError):
        db.record_response(_response(raw_response="a different reply", extracted_letter="D"))

    stored = db.read_responses(RUN)
    assert len(stored) == 1
    assert stored[0]["extracted_letter"] == "C", "the first row was not overwritten"


def test_the_same_agent_answers_both_rounds(db):
    db.record_response(_response(round=1))
    db.record_response(_response(round=2, peer_response_ids=(1,)))
    assert len(db.read_responses(RUN)) == 2


def test_provenance_is_kept_when_the_served_model_differs_from_the_requested_one(db):
    db.record_response(_response(served_slug="qwen/qwen3.8-27b:free", provider="Together"))
    stored = db.read_responses(RUN)[0]
    assert stored["requested_slug"] == "qwen/qwen3.8-27b"
    assert stored["served_slug"] == "qwen/qwen3.8-27b:free"
    assert stored["provider"] == "Together"


@pytest.mark.parametrize("status", ["REFUSAL", "TRUNCATED", "PARSE_FAIL", "API_ERROR"])
def test_a_failure_is_stored_with_no_letter_and_contributes_no_vote(db, status):
    db.record_response(
        _response(status=status, extracted_letter=None, extraction_method=None)
    )
    stored = db.read_responses(RUN)[0]
    assert stored["status"] == status
    assert stored["extracted_letter"] is None


@pytest.mark.parametrize("status", ["REFUSAL", "TRUNCATED", "PARSE_FAIL", "API_ERROR"])
def test_a_failure_carrying_a_letter_is_rejected(db, status):
    """A failure is never a wrong answer, so it may never smuggle a vote in."""
    with pytest.raises(DatabaseError, match="never an answer"):
        db.record_response(_response(status=status, extracted_letter="C"))


def test_an_ok_response_without_a_letter_is_rejected(db):
    with pytest.raises(DatabaseError):
        db.record_response(_response(status=STATUS_OK, extracted_letter=None))


def test_an_unknown_status_is_rejected(db):
    with pytest.raises(DatabaseError):
        db.record_response(_response(status="PROBABLY_FINE", extracted_letter=None))


def test_an_impossible_round_is_rejected(db):
    with pytest.raises(DatabaseError):
        db.record_response(_response(round=3))


# 5. Attempts


def test_every_attempt_is_stored_including_the_one_that_failed(db):
    """A retried call cost two attempts. Both are on the record and both are costed."""
    response_id = db.record_response(
        _response(attempt_count=2, selected_attempt=2, cost_usd=0.0001),
        [
            AttemptRow(
                attempt_number=1,
                raw_response="rate limited",
                status=STATUS_API_ERROR,
                outcome="http_error",
                http_status=429,
                error="HTTP 429",
                latency_seconds=0.4,
                cost_usd=0.0,
            ),
            AttemptRow(
                attempt_number=2,
                raw_response="FINAL ANSWER: C",
                status=STATUS_OK,
                outcome="ok",
                extracted_letter="C",
                finish_reason="stop",
                http_status=200,
                latency_seconds=2.1,
                cost_usd=0.0001,
            ),
        ],
    )

    attempts = db.read_attempts(response_id)
    assert [row["attempt_number"] for row in attempts] == [1, 2]
    assert attempts[0]["http_status"] == 429
    assert attempts[0]["raw_response"] == "rate limited"
    assert attempts[1]["extracted_letter"] == "C"


def test_an_attempt_cannot_be_recorded_twice(db):
    row = AttemptRow(attempt_number=1, raw_response="x", status=STATUS_OK, outcome="ok")
    with pytest.raises(DatabaseError):
        db.record_response(_response(), [row, row])


def test_a_rejected_attempt_rolls_back_the_whole_response(db):
    """Half a record would understate what the run cost, so neither part is kept."""
    good = AttemptRow(attempt_number=1, raw_response="x", status=STATUS_OK, outcome="ok")
    duplicate = AttemptRow(attempt_number=1, raw_response="y", status=STATUS_OK, outcome="ok")

    with pytest.raises(DatabaseError):
        db.record_response(_response(), [good, duplicate])

    assert db.read_responses(RUN) == [], "the response must not survive its attempts failing"
    assert db._query("SELECT * FROM response_attempts") == []


def test_the_selected_attempt_must_be_one_that_was_stored(db):
    with pytest.raises(DatabaseError, match="selected attempt"):
        db.record_response(
            _response(attempt_count=2, selected_attempt=2),
            [AttemptRow(attempt_number=1, raw_response="x", status=STATUS_OK, outcome="ok")],
        )


# 6. Round 2 peer references


def test_round_two_records_which_peer_responses_the_agent_was_shown(db):
    first = db.record_response(_response(agent_id="agent_llama"))
    second = db.record_response(_response(agent_id="agent_mistral"))
    third = db.record_response(_response(agent_id="agent_gemma"))
    fourth = db.record_response(_response(agent_id="agent_deepseek"))

    db.record_response(
        _response(round=2, peer_response_ids=(first, second, third, fourth))
    )
    stored = db.read_responses(RUN, round=2)[0]
    assert stored["peer_response_ids"] == [first, second, third, fourth]
    assert stored["peer_count"] == 4


def test_a_round_two_agent_can_see_fewer_than_four_peers(db):
    """P10: a Round 1 failure means that agent supplies no peer response."""
    first = db.record_response(_response(agent_id="agent_llama"))
    db.record_response(_response(round=2, peer_response_ids=(first,)))

    stored = db.read_responses(RUN, round=2)[0]
    assert stored["peer_count"] == 1, "the real peer count is a confound, not a default of 4"


def test_round_one_agents_see_no_peers(db):
    with pytest.raises(DatabaseError, match="no peer responses"):
        db.record_response(_response(round=1, peer_response_ids=(1,)))


def test_an_agent_cannot_be_shown_more_than_four_peers(db):
    with pytest.raises(DatabaseError, match="at most 4"):
        db.record_response(_response(round=2, peer_response_ids=(1, 2, 3, 4, 5)))


def test_the_same_peer_cannot_be_supplied_twice(db):
    with pytest.raises(DatabaseError, match="twice"):
        db.record_response(_response(round=2, peer_response_ids=(1, 1)))


def test_peer_references_are_stored_as_json_not_as_a_python_repr(db):
    first = db.record_response(_response(agent_id="agent_llama"))
    db.record_response(_response(round=2, peer_response_ids=(first,)))

    raw = db._query("SELECT peer_response_ids FROM model_responses WHERE round = 2")[0]
    assert json.loads(raw["peer_response_ids"]) == [first]


# 7. Question outcomes


def _outcome(**overrides) -> OutcomeRecord:
    base = dict(
        run_id=RUN,
        question_id="q0001",
        round=1,
        consensus_state="CONSENSUS",
        consensus_answer="C",
        decided=True,
        valid_answer_count=5,
    )
    base.update(overrides)
    return OutcomeRecord(**base)


def test_round_one_and_round_two_outcomes_are_separate_rows(db):
    """Comparing these two rows is the research question. They never merge."""
    db.record_outcome(_outcome(round=1, consensus_state="NO_CONSENSUS",
                               consensus_answer=None, decided=False, valid_answer_count=5))
    db.record_outcome(_outcome(round=2, consensus_state="CONSENSUS",
                               consensus_answer="C", decided=True, valid_answer_count=5))

    outcomes = db.read_outcomes(RUN)
    assert len(outcomes) == 2
    assert [row["round"] for row in outcomes] == [1, 2]
    assert db.read_outcomes(RUN, round=1)[0]["decided"] == 0
    assert db.read_outcomes(RUN, round=2)[0]["consensus_answer"] == "C"


def test_an_outcome_cannot_be_recorded_twice(db):
    db.record_outcome(_outcome())
    with pytest.raises(DuplicateRecordError):
        db.record_outcome(_outcome(consensus_answer="D"))
    assert db.read_outcomes(RUN)[0]["consensus_answer"] == "C"


@pytest.mark.parametrize("state", ["UNANIMOUS", "CONSENSUS", "NO_CONSENSUS", "INSUFFICIENT_ANSWERS"])
def test_the_four_consensus_states_are_accepted(db, state):
    decided = state in {"UNANIMOUS", "CONSENSUS"}
    db.record_outcome(
        _outcome(
            consensus_state=state,
            consensus_answer="C" if decided else None,
            decided=decided,
        )
    )
    assert db.read_outcomes(RUN)[0]["consensus_state"] == state


def test_an_unknown_consensus_state_is_rejected(db):
    with pytest.raises(DatabaseError, match="consensus state"):
        db.record_outcome(_outcome(consensus_state="MOSTLY_AGREED"))


def test_an_undecided_question_cannot_carry_an_answer(db):
    with pytest.raises(DatabaseError):
        db.record_outcome(
            _outcome(consensus_state="NO_CONSENSUS", consensus_answer="C", decided=False)
        )


def test_a_decided_question_must_carry_an_answer(db):
    with pytest.raises(DatabaseError):
        db.record_outcome(_outcome(decided=True, consensus_answer=None))


# 8. Building records from what the client and parser return


def _completion(**overrides) -> CompletionResult:
    base = dict(
        text="REASONING: alkene.\nFINAL ANSWER: C",
        agent_id="agent_qwen",
        requested_slug="qwen/qwen3.8-27b",
        served_slug="qwen/qwen3.8-27b",
        provider="DeepInfra",
        generation_id="gen-1",
        finish_reason="stop",
        prompt_tokens=120,
        completion_tokens=64,
        cost_usd=0.0002,
        latency_seconds=2.5,
        attempts=1,
    )
    base.update(overrides)
    return CompletionResult(**base)


def test_a_successful_call_and_its_parse_become_one_storable_row(db):
    parsed = ParsedResponse(STATUS_OK, letter="C", extraction_method="LAST_FINAL_ANSWER_MATCH")
    record = response_from_completion(
        _completion(), parsed,
        run_id=RUN, question_id="q0001", round=1,
        spec=_spec(), prompt_version=PROMPT_VERSION,
    )
    db.record_response(record)

    stored = db.read_responses(RUN)[0]
    assert stored["raw_response"] == _completion().text
    assert stored["extracted_letter"] == "C"
    assert stored["temperature"] == 0.0 and stored["top_p"] == 1.0
    assert stored["cost_usd"] == pytest.approx(0.0002)


def test_a_call_that_failed_every_attempt_is_still_recorded_and_costed(db):
    """ApiRequestError.attempt_log is what makes a dead call storable."""
    log = (
        AttemptRecord(attempt=1, outcome="http_error", latency_seconds=0.5,
                      status_code=429, raw_response="rate limited", cost_usd=0.0),
        AttemptRecord(attempt=2, outcome="http_error", latency_seconds=0.7,
                      status_code=504, raw_response="gateway timeout", cost_usd=0.0),
    )
    record = response_from_failed_call(
        log, run_id=RUN, question_id="q0001", round=1,
        spec=_spec(), prompt_version=PROMPT_VERSION,
    )
    response_id = db.record_response(record, attempt_rows(log))

    stored = db.read_responses(RUN)[0]
    assert stored["status"] == STATUS_API_ERROR
    assert stored["extracted_letter"] is None
    assert stored["attempt_count"] == 2
    assert stored["latency_seconds"] == pytest.approx(1.2)
    # Nothing is guessed about a call that never reached a provider.
    assert stored["served_slug"] == "unknown" and stored["provider"] == "unknown"
    assert len(db.read_attempts(response_id)) == 2


def test_attempt_rows_keep_the_raw_body_of_a_failed_attempt(db):
    log = (
        AttemptRecord(attempt=1, outcome="http_error", latency_seconds=0.3,
                      status_code=429, raw_response="rate limited by provider"),
        AttemptRecord(attempt=2, outcome="ok", latency_seconds=2.0, status_code=200,
                      raw_response='{"choices": []}', finish_reason="stop",
                      prompt_tokens=120, completion_tokens=64, cost_usd=0.0002),
    )
    parsed = {2: ParsedResponse(STATUS_OK, letter="C", extraction_method="LAST_FINAL_ANSWER_MATCH")}
    rows = attempt_rows(log, parsed_by_attempt=parsed)

    response_id = db.record_response(
        _response(attempt_count=2, selected_attempt=2), rows
    )
    attempts = db.read_attempts(response_id)
    assert attempts[0]["raw_response"] == "rate limited by provider"
    assert attempts[0]["status"] == STATUS_API_ERROR, "no parse exists, so nothing is invented"
    assert attempts[1]["extracted_letter"] == "C"
    assert sum(row["cost_usd"] for row in attempts) == pytest.approx(0.0002)


# 9. Reading back


def test_responses_can_be_read_by_round_question_and_agent(db):
    db.record_response(_response(agent_id="agent_qwen", question_id="q0001"))
    db.record_response(_response(agent_id="agent_llama", question_id="q0001"))
    db.record_response(_response(agent_id="agent_qwen", question_id="q0002"))

    assert len(db.read_responses(RUN)) == 3
    assert len(db.read_responses(RUN, question_id="q0001")) == 2
    assert len(db.read_responses(RUN, agent_id="agent_qwen")) == 2
    assert len(db.read_responses(RUN, round=2)) == 0


def test_reading_an_unknown_run_returns_nothing_rather_than_failing(db):
    assert db.read_run("no_such_run") is None
    assert db.read_responses("no_such_run") == []
    assert db.read_outcomes("no_such_run") == []
