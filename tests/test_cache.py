"""Tests for the response cache. Offline: the network is switched off."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from mad.api_client import ApiRequestError, AttemptRecord, CompletionResult, load_model_registry
from mad.cache import CacheError, ResponseCache, cache_key
from mad.database import ResultsDatabase
from mad.round1 import Round1Config, run_round1_question
from mad.runner import FixtureClient, RunnerError, load_pilot_question

REPO = Path(__file__).resolve().parents[1]
PILOT_ID = "mmlu_pro_v1:test:7296"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a cache test tried to reach the network")

    monkeypatch.setattr(requests.Session, "post", explode)
    monkeypatch.setattr(requests.Session, "get", explode)


@pytest.fixture
def registry():
    return load_model_registry(REPO / "configs" / "models" / "agents_v1.yaml")


@pytest.fixture
def cache(tmp_path):
    with ResponseCache(tmp_path / "cache.sqlite") as cache_:
        yield cache_


MESSAGES = [{"role": "system", "content": "answer"}, {"role": "user", "content": "2+2?"}]


def start_test_run(db, run_id, config):
    db.start_run(
        run_id,
        config_name=config.config_version,
        question_set_version=config.question_set_version,
        prompt_version=config.prompt_version,
        settings_version=config.settings_version,
        parser_version=config.parser_version,
    )


def _reply(spec, text="REASONING: sums.\nFINAL ANSWER: B", cost=0.0003):
    body = {
        "id": "gen-real-1",
        "model": spec.slug,
        "provider": "DeepInfra",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "cost": cost},
    }
    return CompletionResult(
        text=text, agent_id=spec.agent_id, requested_slug=spec.slug, served_slug=spec.slug,
        provider="DeepInfra", generation_id="gen-real-1", finish_reason="stop",
        prompt_tokens=50, completion_tokens=20, cost_usd=cost, latency_seconds=2.5,
        attempts=1, raw_response=body,
        attempt_log=(AttemptRecord(attempt=1, outcome="ok", latency_seconds=2.5,
                                   status_code=200, raw_response="{}", finish_reason="stop",
                                   prompt_tokens=50, completion_tokens=20, cost_usd=cost),),
    )


# 1. Store and look up


def test_a_stored_reply_comes_back_with_tokens_and_latency_but_no_new_spend(cache, registry):
    spec = registry["agent_qwen"]
    cache.store(spec, MESSAGES, _reply(spec))

    hit = cache.lookup(spec, MESSAGES)
    assert hit is not None
    assert hit.text.endswith("FINAL ANSWER: B")
    assert hit.prompt_tokens == 50 and hit.completion_tokens == 20, "original tokens kept"
    assert hit.latency_seconds == 2.5, "original latency, not the lookup time"
    assert hit.cost_usd == 0.0, "a hit spends nothing"
    assert hit.attempt_log == (), "a hit made no API attempt"


def test_an_unknown_request_is_a_miss(cache, registry):
    assert cache.lookup(registry["agent_qwen"], MESSAGES) is None


def test_the_first_stored_reply_wins_and_is_never_overwritten(cache, registry):
    spec = registry["agent_qwen"]
    cache.store(spec, MESSAGES, _reply(spec, text="FINAL ANSWER: B"))
    cache.store(spec, MESSAGES, _reply(spec, text="FINAL ANSWER: D"))
    assert "FINAL ANSWER: B" in cache.lookup(spec, MESSAGES).text


def test_a_refusal_is_a_genuine_outcome_and_is_cached(cache, registry):
    spec = registry["agent_gemma"]
    cache.store(spec, MESSAGES, _reply(spec, text="I cannot answer this question."))
    assert cache.lookup(spec, MESSAGES).text == "I cannot answer this question."


# 2. The key separates what must stay separate


def test_two_agents_making_the_same_request_do_not_share_a_reply(cache, registry):
    qwen, gemma = registry["agent_qwen"], registry["agent_gemma"]
    cache.store(qwen, MESSAGES, _reply(qwen))
    assert cache.lookup(gemma, MESSAGES) is None


def test_changed_settings_are_a_different_request(registry):
    from dataclasses import replace

    spec = registry["agent_qwen"]
    probe = replace(spec, max_tokens=2048)
    assert cache_key(spec, MESSAGES) != cache_key(probe, MESSAGES), (
        "the agents_v2 token probe must miss the agents_v1 cache"
    )


def test_a_pin_and_a_reasoning_cap_each_change_the_request_identity(registry):
    """The controlled test must never be served an old unpinned, uncapped reply."""
    from dataclasses import replace

    spec = registry["agent_qwen"]
    pinned = replace(spec, pinned_provider="Parasail")
    capped = replace(spec, reasoning_max_tokens=2048)

    keys = {cache_key(spec, MESSAGES), cache_key(pinned, MESSAGES), cache_key(capped, MESSAGES)}
    assert len(keys) == 3, "pin and cap must each produce a distinct key"


def test_unpinned_uncapped_specs_keep_their_old_cache_keys(registry):
    """Adding the new fields must not orphan every reply cached before them."""
    import hashlib, json

    spec = registry["agent_qwen"]
    legacy_payload = {
        "agent_id": spec.agent_id, "slug": spec.slug, "messages": MESSAGES,
        "temperature": spec.temperature, "top_p": spec.top_p, "max_tokens": spec.max_tokens,
    }
    legacy = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert cache_key(spec, MESSAGES) == legacy


def test_changed_messages_are_a_different_request(registry):
    spec = registry["agent_qwen"]
    other = [{"role": "user", "content": "3+3?"}]
    assert cache_key(spec, MESSAGES) != cache_key(spec, other)


# 3. What must never go in


def test_fixture_replies_are_refused(cache, registry):
    from dataclasses import replace

    spec = registry["agent_qwen"]
    fixture = replace(_reply(spec), provider="fixture")
    with pytest.raises(CacheError, match="fixture"):
        cache.store(spec, MESSAGES, fixture)
    assert cache.lookup(spec, MESSAGES) is None


def test_a_reply_without_its_raw_body_is_refused(cache, registry):
    from dataclasses import replace

    spec = registry["agent_qwen"]
    with pytest.raises(CacheError, match="raw body"):
        cache.store(spec, MESSAGES, replace(_reply(spec), raw_response={}))


# 4. Through the runner


class CountingFixtureClient(FixtureClient):
    def __init__(self, question):
        super().__init__(question)
        self.calls = 0

    def complete(self, spec, messages):
        self.calls += 1
        result = super().complete(spec, messages)
        # Give the fixture a real-looking provider so the cache accepts it.
        # This stays inside tmp_path caches - never the real cache file.
        from dataclasses import replace

        body = {"id": f"t-{spec.agent_id}", "model": spec.slug, "provider": "TestBench",
                "choices": [{"message": {"content": result.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.0001}}
        return replace(result, provider="TestBench", raw_response=body,
                       prompt_tokens=5, completion_tokens=7, cost_usd=0.0001)


def test_a_second_run_is_served_from_the_cache_with_no_client_calls(tmp_path, registry, cache):
    question = load_pilot_question(PILOT_ID)
    config = Round1Config(cache_enabled=True)

    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        start_test_run(db, "run_1", config)
        first = CountingFixtureClient(question)
        run_round1_question(question, registry=registry, client=first, db=db,
                            run_id="run_1", config=config, cache=cache)
        assert first.calls == 5

        start_test_run(db, "run_2", config)
        second = CountingFixtureClient(question)
        report = run_round1_question(question, registry=registry, client=second, db=db,
                                     run_id="run_2", config=config, cache=cache)
        assert second.calls == 0, "every reply came from the cache"

        rows = db.read_responses("run_2")
        assert all(row["cache_hit"] == 1 for row in rows)
        assert all(row["cost_usd"] == 0.0 for row in rows), "a cached run costs nothing"
        assert all(row["completion_tokens"] == 7 for row in rows), "original tokens kept"
        assert report.total_cost_usd == 0.0

        # The first run paid and says so.
        assert all(row["cache_hit"] == 0 for row in db.read_responses("run_1"))

        # A hit made no API attempt, so it stores no attempt rows. Its
        # attempt_count of 1 names the original call behind the cached body.
        for row in rows:
            assert db.read_attempts(row["response_id"]) == []
            assert row["attempt_count"] == 1


def test_a_crash_while_storing_the_response_leaves_nothing_in_the_cache(tmp_path, registry, cache):
    """The audit record comes first. A cached reply with no results row would
    mean paid spend missing from the experimental record."""
    question = load_pilot_question(PILOT_ID)
    config = Round1Config(cache_enabled=True)

    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        start_test_run(db, "run_1", config)
        # Break the database write for exactly one agent's insert.
        original = db.record_response
        def failing_record(record, attempts=()):
            if record.agent_id == "agent_mistral":
                raise RuntimeError("disk full")
            return original(record, attempts)
        db.record_response = failing_record

        client = CountingFixtureClient(question)
        with pytest.raises(RuntimeError, match="disk full"):
            run_round1_question(question, registry=registry, client=client, db=db,
                                run_id="run_1", config=config, cache=cache)

        spec = registry["agent_mistral"]
        from mad.prompts_v1 import build_round1_messages
        assert cache.lookup(spec, build_round1_messages(question)) is None, (
            "the unrecorded paid reply must not be in the cache"
        )


def test_a_failed_call_is_not_cached_so_a_rerun_tries_again(tmp_path, registry, cache):
    question = load_pilot_question(PILOT_ID)
    config = Round1Config(cache_enabled=True)

    class FailingOnce(CountingFixtureClient):
        def complete(self, spec, messages):
            result = super().complete(spec, messages)  # keeps plan order
            if spec.agent_id == "agent_mistral" and self.calls <= 5:
                raise ApiRequestError("HTTP 429", (AttemptRecord(
                    attempt=1, outcome="http_error", latency_seconds=0.1,
                    status_code=429, raw_response="rate limited"),))
            return result

    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        start_test_run(db, "run_1", config)
        start_test_run(db, "run_2", config)
        client = FailingOnce(question)
        run_round1_question(question, registry=registry, client=client, db=db,
                            run_id="run_1", config=config, cache=cache)
        run_round1_question(question, registry=registry, client=client, db=db,
                            run_id="run_2", config=config, cache=cache)

        # Second run: four agents cached, mistral retried for real and succeeded.
        assert client.calls == 6
        second = {r["agent_id"]: r for r in db.read_responses("run_2")}
        assert second["agent_mistral"]["status"] == "OK"
        assert second["agent_mistral"]["cache_hit"] == 0
        assert sum(r["cache_hit"] for r in second.values()) == 4


def test_the_config_must_tell_the_truth_about_the_cache(tmp_path, registry, cache):
    question = load_pilot_question(PILOT_ID)
    with ResultsDatabase(tmp_path / "results.sqlite") as db:
        with pytest.raises(RunnerError, match="truth"):
            run_round1_question(question, registry=registry, client=FixtureClient(question),
                                db=db, run_id="run_x",
                                config=Round1Config(cache_enabled=False), cache=cache)
        with pytest.raises(RunnerError, match="truth"):
            run_round1_question(question, registry=registry, client=FixtureClient(question),
                                db=db, run_id="run_y",
                                config=Round1Config(cache_enabled=True), cache=None)
