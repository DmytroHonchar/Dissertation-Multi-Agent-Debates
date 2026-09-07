"""Tests for the Round 1 runner. Offline: any network attempt fails the test."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import requests

from mad.api_client import ApiRequestError, AttemptRecord, load_model_registry
from mad.database import ResultsDatabase
from mad.parser_v1 import STATUS_API_ERROR, STATUS_OK, STATUS_REFUSAL
from mad.prompts_v1 import DEBATE_PROMPT_VERSION
from mad.round1 import Round1Config, run_round1_question
from mad.runner import (
    PRODUCTION_DATABASE,
    FixtureClient,
    RunnerError,
    SpendNotConfirmedError,
    ensure_safe_database_path,
    load_pilot_question,
    production_database_path,
    require_spend_confirmation,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "models" / "agents_v1.yaml"
PILOT_ID = "mmlu_pro_v1:test:7296"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test in this file runs with the network switched off."""

    def explode(*args, **kwargs):
        raise AssertionError("a Round 1 test tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)
    monkeypatch.setattr(requests, "post", explode)
    monkeypatch.setattr(requests, "get", explode)


@pytest.fixture
def registry():
    return load_model_registry(REGISTRY_PATH)


@pytest.fixture
def question():
    return load_pilot_question(PILOT_ID)


@pytest.fixture
def db(tmp_path):
    with ResultsDatabase(tmp_path / "round1_test.sqlite") as database:
        yield database


def start_test_run(db, run_id, config=Round1Config(), *, run_prompt_version=None):
    """Tests own the lifecycle just like the real command-line callers."""
    db.start_run(
        run_id,
        config_name=config.config_version,
        question_set_version=config.question_set_version,
        prompt_version=run_prompt_version or config.prompt_version,
        settings_version=config.settings_version,
        parser_version=config.parser_version,
    )


class FailingClient(FixtureClient):
    """Fixture client where named agents fail both attempts."""

    def __init__(self, question, failing_agents):
        super().__init__(question)
        self._failing = set(failing_agents)

    def complete(self, spec, messages):
        result = super().complete(spec, messages)  # keeps the plan order aligned
        if spec.agent_id in self._failing:
            raise ApiRequestError(
                f"{spec.slug} failed after 2 attempts: HTTP 429",
                (
                    AttemptRecord(attempt=1, outcome="http_error", latency_seconds=0.4,
                                  status_code=429, raw_response="rate limited"),
                    AttemptRecord(attempt=2, outcome="http_error", latency_seconds=0.5,
                                  status_code=429, raw_response="rate limited"),
                ),
            )
        return result


class CountingClient(FixtureClient):
    """Proves lifecycle mistakes are refused before a model call."""

    def __init__(self, question):
        super().__init__(question)
        self.calls = 0

    def complete(self, spec, messages):
        self.calls += 1
        return super().complete(spec, messages)


# 1. A complete dry run


def test_a_full_fixture_round_stores_five_responses_and_one_outcome(question, registry, db):
    start_test_run(db, "run_a")
    report = run_round1_question(
        question, registry=registry, client=FixtureClient(question), db=db, run_id="run_a"
    )

    responses = db.read_responses("run_a", round=1)
    assert len(responses) == 5
    assert {row["agent_id"] for row in responses} == set(registry)

    outcomes = db.read_outcomes("run_a")
    assert len(outcomes) == 1
    assert outcomes[0]["consensus_state"] == "CONSENSUS"
    assert outcomes[0]["consensus_answer"] == "A"
    assert outcomes[0]["valid_answer_count"] == 4

    assert report.outcome.state == "CONSENSUS"
    assert db.read_run("run_a")["ended_at"] is None, "the caller has not finished the run"


def test_a_missing_run_is_refused_before_any_call(question, registry, db):
    client = CountingClient(question)

    with pytest.raises(RunnerError, match="has not been started"):
        run_round1_question(
            question, registry=registry, client=client, db=db, run_id="missing"
        )

    assert client.calls == 0


def test_a_finished_run_is_refused_before_any_call(question, registry, db):
    start_test_run(db, "finished")
    db.finish_run("finished")
    client = CountingClient(question)

    with pytest.raises(RunnerError, match="already finished"):
        run_round1_question(
            question, registry=registry, client=client, db=db, run_id="finished"
        )

    assert client.calls == 0


def test_two_questions_share_one_run_and_the_caller_finishes_it(question, registry, db):
    second_question = load_pilot_question("mmlu_pro_v1:test:10925")
    start_test_run(db, "two_questions")

    run_round1_question(
        question,
        registry=registry,
        client=FixtureClient(question),
        db=db,
        run_id="two_questions",
    )
    run_round1_question(
        second_question,
        registry=registry,
        client=FixtureClient(second_question),
        db=db,
        run_id="two_questions",
    )

    assert len(db.read_runs()) == 1
    assert len(db.read_responses("two_questions", round=1)) == 10
    assert len(db.read_outcomes("two_questions", round=1)) == 2
    assert db.read_run("two_questions")["ended_at"] is None

    db.finish_run("two_questions")
    assert db.read_run("two_questions")["ended_at"] is not None


def test_every_response_has_its_attempts_stored(question, registry, db):
    start_test_run(db, "run_b")
    run_round1_question(
        question, registry=registry, client=FixtureClient(question), db=db, run_id="run_b"
    )
    for row in db.read_responses("run_b"):
        attempts = db.read_attempts(row["response_id"])
        assert len(attempts) == 1
        assert attempts[0]["raw_response"] != "", "the raw reply is on the attempt too"


def test_the_fixture_is_labelled_as_a_fixture_in_every_row(question, registry, db):
    """Fixture data must never be mistakable for a model response."""
    start_test_run(db, "run_c")
    run_round1_question(
        question, registry=registry, client=FixtureClient(question), db=db, run_id="run_c"
    )
    for row in db.read_responses("run_c"):
        assert row["provider"] == "fixture"
        assert row["generation_id"].startswith("fixture-")


def test_the_refusing_fixture_agent_is_stored_without_a_letter(question, registry, db):
    start_test_run(db, "run_d")
    run_round1_question(
        question, registry=registry, client=FixtureClient(question), db=db, run_id="run_d"
    )
    refused = [r for r in db.read_responses("run_d") if r["status"] == STATUS_REFUSAL]
    assert len(refused) == 1
    assert refused[0]["extracted_letter"] is None


# 2. The stored versions


@pytest.mark.parametrize(
    ("config_field", "wrong_value", "reported_field"),
    [
        ("config_version", "wrong_config", "config_name"),
        ("question_set_version", "wrong_questions", "question_set_version"),
        ("prompt_version", "wrong_prompt", "prompt_version"),
        ("settings_version", "wrong_agents", "settings_version"),
        ("parser_version", "wrong_parser", "parser_version"),
    ],
)
def test_run_labels_must_match_the_config_before_any_call(
    question, registry, db, config_field, wrong_value, reported_field
):
    base = Round1Config()
    start_test_run(db, "run_e", base)
    mismatched = replace(base, **{config_field: wrong_value})
    client = CountingClient(question)

    with pytest.raises(RunnerError, match=reported_field):
        run_round1_question(
            question,
            registry=registry,
            client=client,
            db=db,
            run_id="run_e",
            config=mismatched,
        )

    assert client.calls == 0


def test_round1_prompt_is_accepted_inside_the_combined_debate_prompt_label(
    question, registry, db
):
    config = Round1Config()
    start_test_run(
        db, "combined_prompt", config, run_prompt_version=DEBATE_PROMPT_VERSION
    )

    run_round1_question(
        question,
        registry=registry,
        client=FixtureClient(question),
        db=db,
        run_id="combined_prompt",
        config=config,
    )

    assert len(db.read_responses("combined_prompt", round=1)) == 5


# 3. One agent failing must not end the run


def test_one_failed_agent_is_stored_as_api_error_and_the_rest_continue(question, registry, db):
    client = FailingClient(question, failing_agents={"agent_mistral"})
    start_test_run(db, "run_f")
    report = run_round1_question(
        question, registry=registry, client=client, db=db, run_id="run_f"
    )

    responses = {row["agent_id"]: row for row in db.read_responses("run_f")}
    assert len(responses) == 5, "the failure did not stop the other four"
    assert responses["agent_mistral"]["status"] == STATUS_API_ERROR
    assert responses["agent_mistral"]["extracted_letter"] is None
    assert responses["agent_mistral"]["attempt_count"] == 2

    # Both failed attempts are on the record, with their HTTP status.
    attempts = db.read_attempts(responses["agent_mistral"]["response_id"])
    assert [a["http_status"] for a in attempts] == [429, 429]

    # mistral's A vote is gone: A-A from llama and qwen, B from deepseek.
    assert report.outcome.state == "NO_CONSENSUS"
    assert report.outcome.valid_answer_count == 3
    assert "agent_mistral" in report.outcome.failed_agents


def test_all_agents_failing_still_completes_and_records_the_outcome(question, registry, db):
    client = FailingClient(question, failing_agents=set(registry))
    start_test_run(db, "run_g")
    report = run_round1_question(
        question, registry=registry, client=client, db=db, run_id="run_g"
    )
    assert report.outcome.state == "INSUFFICIENT_ANSWERS"
    assert report.outcome.valid_answer_count == 0
    assert len(db.read_responses("run_g")) == 5


# 4. The configured five, and only the configured five


def test_a_registry_without_five_agents_is_refused_before_any_call(question, registry, db):
    four = dict(list(registry.items())[:4])
    with pytest.raises(RunnerError, match="must hold 5 agents"):
        run_round1_question(
            question, registry=four, client=FixtureClient(question), db=db, run_id="run_h"
        )
    assert db.read_responses("run_h") == [], "nothing was stored"


def test_the_registry_names_reach_votings_identity_check(question, registry, db):
    """expected_agents is optional in tally(); the runner must actually pass it."""
    import mad.round1 as round1_module

    seen = {}
    original = round1_module.tally
    start_test_run(db, "run_i")

    def spy(responses, *, expected_agents=None):
        seen["expected_agents"] = expected_agents
        return original(responses, expected_agents=expected_agents)

    round1_module.tally = spy
    try:
        run_round1_question(
            question, registry=registry, client=FixtureClient(question), db=db, run_id="run_i"
        )
    finally:
        round1_module.tally = original

    assert set(seen["expected_agents"]) == set(registry)


# 5. Only pilot questions, one at a time


def test_a_pilot_question_loads_and_carries_no_answer(question):
    assert question["stable_id"] == PILOT_ID
    from mad.prompts_v1 import FORBIDDEN_FIELDS

    assert not FORBIDDEN_FIELDS.intersection(question)


def test_an_experimental_question_is_refused_by_name():
    import json

    experimental_id = json.loads(
        (Path("data/frozen/mmlu_pro_v1/metadata/experimental_ids.json")).read_text()
    )[0]
    with pytest.raises(RunnerError, match="experimental"):
        load_pilot_question(experimental_id)


def test_an_unknown_question_is_refused():
    with pytest.raises(RunnerError, match="not in the pilot file"):
        load_pilot_question("mmlu_pro_v1:test:999999")


@pytest.mark.parametrize("bad", ["id1,id2", "id1 id2", "", "  "])
def test_more_or_less_than_one_question_is_refused(bad):
    with pytest.raises(RunnerError, match="one question at a time"):
        load_pilot_question(bad)


# 6. Money guards


def test_live_without_the_spend_flag_is_refused():
    with pytest.raises(SpendNotConfirmedError, match="yes-spend-real-money"):
        require_spend_confirmation(live=True, spend_confirmed=False)


def test_the_spend_flag_without_live_is_refused_as_a_mistake():
    with pytest.raises(SpendNotConfirmedError):
        require_spend_confirmation(live=False, spend_confirmed=True)


def test_a_dry_run_and_a_confirmed_live_run_both_pass_the_guard():
    require_spend_confirmation(live=False, spend_confirmed=False)
    require_spend_confirmation(live=True, spend_confirmed=True)


def test_a_dry_run_may_not_write_to_the_real_results_database():
    with pytest.raises(RunnerError, match="dry run"):
        ensure_safe_database_path(PRODUCTION_DATABASE, live=False)


def test_a_live_run_may_use_the_real_results_database():
    assert ensure_safe_database_path(PRODUCTION_DATABASE, live=True) == PRODUCTION_DATABASE


def test_a_dry_run_may_use_any_other_path(tmp_path):
    path = tmp_path / "somewhere.sqlite"
    assert ensure_safe_database_path(path, live=False) == path


# 7. Nothing here reads an answer key


def test_a_question_carrying_an_answer_key_stops_the_run(registry, db, question):
    from mad.prompts_v1 import AnswerLeakageError

    poisoned = dict(question)
    poisoned["answer"] = "A"
    start_test_run(db, "run_j")
    with pytest.raises(AnswerLeakageError):
        run_round1_question(
            poisoned, registry=registry, client=FixtureClient(poisoned), db=db, run_id="run_j"
        )


# 8. Money is counted across every attempt


class RetriedClient(FixtureClient):
    """One agent pays for a failed first attempt before succeeding."""

    def __init__(self, question, retried_agent):
        super().__init__(question)
        self._retried = retried_agent

    def complete(self, spec, messages):
        result = super().complete(spec, messages)
        if spec.agent_id != self._retried:
            return result
        from dataclasses import replace

        failed = AttemptRecord(attempt=1, outcome="upstream_error", latency_seconds=0.4,
                               status_code=200, raw_response='{"error": "upstream"}',
                               error="upstream error", cost_usd=0.00002, prompt_tokens=11)
        ok = replace(result.attempt_log[0], attempt=2, cost_usd=0.00010, prompt_tokens=11,
                     completion_tokens=40)
        return replace(result, attempts=2, cost_usd=0.00010, prompt_tokens=11,
                       completion_tokens=40, attempt_log=(failed, ok))


def test_a_paid_failed_attempt_is_counted_in_the_response_outcome_and_report(question, registry, db):
    """$0.00002 lost on attempt 1 plus $0.00010 on attempt 2 is a $0.00012 call."""
    client = RetriedClient(question, retried_agent="agent_qwen")
    start_test_run(db, "run_k")
    report = run_round1_question(
        question, registry=registry, client=client, db=db, run_id="run_k"
    )

    stored = {row["agent_id"]: row for row in db.read_responses("run_k")}
    assert stored["agent_qwen"]["cost_usd"] == pytest.approx(0.00012)
    assert stored["agent_qwen"]["prompt_tokens"] == 22
    assert stored["agent_qwen"]["status"] == STATUS_OK, "the retry saved the call"

    # The per-attempt split is still visible, and sums to the same money.
    attempts = db.read_attempts(stored["agent_qwen"]["response_id"])
    assert [a["cost_usd"] for a in attempts] == [pytest.approx(0.00002), pytest.approx(0.00010)]

    assert db.read_outcomes("run_k")[0]["total_cost_usd"] == pytest.approx(0.00012)
    assert report.total_cost_usd == pytest.approx(0.00012)


def test_a_malformed_body_from_the_real_client_is_stored_and_the_run_continues(
    question, registry, db, monkeypatch
):
    """HTTP 200 with no choices: paid for, unusable, stored as API_ERROR."""
    import json
    from mad.api_client import OpenRouterClient

    malformed = {"id": "gen-9", "model": "x",
                 "usage": {"prompt_tokens": 11, "completion_tokens": 0, "cost": 0.00003}}
    malformed_text = json.dumps(malformed)

    class FakeResponse:
        status_code = 200
        text = malformed_text

        @staticmethod
        def json():
            return malformed

    real_client = OpenRouterClient(api_key="test-key-not-real")
    monkeypatch.setattr(real_client._session, "post", lambda *a, **k: FakeResponse())

    class MostlyFixtureClient(FixtureClient):
        def complete(self, spec, messages):
            result = super().complete(spec, messages)  # keeps the plan aligned
            if spec.agent_id == "agent_deepseek":
                return real_client.complete(spec, messages)
            return result

    start_test_run(db, "run_l")
    report = run_round1_question(
        question, registry=registry, client=MostlyFixtureClient(question), db=db, run_id="run_l"
    )

    stored = {row["agent_id"]: row for row in db.read_responses("run_l")}
    assert len(stored) == 5, "the malformed reply did not stop the other agents"
    assert stored["agent_deepseek"]["status"] == STATUS_API_ERROR
    assert stored["agent_deepseek"]["cost_usd"] == pytest.approx(0.00003), "the waste is on the record"

    attempts = db.read_attempts(stored["agent_deepseek"]["response_id"])
    assert attempts[0]["outcome"] == "malformed_body"
    assert json.loads(attempts[0]["raw_response"]) == malformed
    assert "agent_deepseek" in report.outcome.failed_agents


# 9. The config describes the run that actually happens


def test_a_config_claiming_a_cache_is_refused(question, registry, db):
    with pytest.raises(RunnerError, match="cache"):
        run_round1_question(
            question, registry=registry, client=FixtureClient(question), db=db,
            run_id="run_m", config=Round1Config(cache_enabled=True),
        )


def test_a_config_claiming_parallel_calls_is_refused(question, registry, db):
    with pytest.raises(RunnerError, match="sequential"):
        run_round1_question(
            question, registry=registry, client=FixtureClient(question), db=db,
            run_id="run_n", config=Round1Config(parallel_calls=True),
        )


def test_the_production_path_is_absolute_and_inside_the_repository():
    path = production_database_path()
    assert path.is_absolute()
    assert path == Path(__file__).resolve().parents[1] / "storage" / "results.sqlite"
