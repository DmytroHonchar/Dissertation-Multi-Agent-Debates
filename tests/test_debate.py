"""Tests for the Round 2 debate runner. Offline: any network attempt fails.

Round 2 is where the project's claim lives, so most of these tests are about
what an agent is and is not shown: never its own response as a peer, never a
model identity, never a Round 1 reply the parser judged unusable.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import requests

from mad.api_client import ApiRequestError, AttemptRecord, CompletionResult, load_model_registry
from mad.cache import ResponseCache
from mad.database import ResultsDatabase
from mad.debate import (
    DEBATE_CONFIG_VERSION,
    Round2Config,
    peers_for,
    run_round2_question,
    valid_round1_responses,
)
from mad.parser_v1 import (
    PARSER_VERSION,
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)
from mad.prompts_v1 import DEBATE_PROMPT_VERSION, ROUND2_PROMPT_VERSION
from mad.round1 import Round1Config, run_round1_question
from mad.runner import (
    QUESTION_SET_VERSION,
    SETTINGS_VERSION,
    FixtureClient,
    RunnerError,
    load_pilot_question,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "models" / "agents_v1.yaml"
PILOT_ID = "mmlu_pro_v1:test:7296"
RUN = "debate_test_run"

# Registry order. Peer order follows it, so the tests can name what each agent
# should have seen.
AGENTS = ("agent_llama", "agent_qwen", "agent_mistral", "agent_deepseek", "agent_gemma")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test in this file runs with the network switched off."""

    def explode(*args, **kwargs):
        raise AssertionError("a Round 2 test tried to reach the network")

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
    with ResultsDatabase(tmp_path / "debate_test.sqlite") as database:
        yield database


# --- Clients --------------------------------------------------------------

# Deliberately free of agent names, slugs and provider names, so a test can
# assert that none of those reached a prompt.
ANSWER_TEXTS = {
    "agent_llama": "REASONING: the alpha argument.\nFINAL ANSWER: A",
    "agent_qwen": "REASONING: the bravo argument.\nFINAL ANSWER: A",
    "agent_mistral": "REASONING: the charlie argument.\nFINAL ANSWER: A",
    "agent_deepseek": "REASONING: the delta argument.\nFINAL ANSWER: B",
    "agent_gemma": "REASONING: the echo argument.\nFINAL ANSWER: B",
}

RATE_LIMITED = ApiRequestError(
    "failed after 2 attempts: HTTP 429",
    (
        AttemptRecord(attempt=1, outcome="http_error", latency_seconds=0.4,
                      status_code=429, raw_response="rate limited"),
        AttemptRecord(attempt=2, outcome="http_error", latency_seconds=0.5,
                      status_code=429, raw_response="rate limited"),
    ),
)


class ScriptedClient:
    """Replies chosen per agent, so every parser status can be produced.

    A reply is either text, a ``(text, finish_reason)`` pair, or an
    ``ApiRequestError`` to raise. Every call is recorded, so the tests can read
    back exactly what each agent was sent.
    """

    def __init__(self, replies=None, provider="fixture"):
        self.replies = dict(replies or ANSWER_TEXTS)
        # The cache refuses "fixture" by design, so cache tests pass a real one.
        self.provider = provider
        self.calls = 0
        self.sent: list[tuple[str, list[dict[str, str]]]] = []

    def complete(self, spec, messages):
        self.calls += 1
        self.sent.append((spec.agent_id, messages))
        reply = self.replies[spec.agent_id]
        if isinstance(reply, ApiRequestError):
            raise reply
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        # A realistic body, because the cache stores it whole and rebuilds the
        # reply from it on a hit.
        body = {
            "id": f"scripted-{spec.agent_id}",
            "model": spec.slug,
            "provider": self.provider,
            "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0001},
        }
        return CompletionResult(
            text=text,
            agent_id=spec.agent_id,
            requested_slug=spec.slug,
            served_slug=spec.slug,
            provider=self.provider,
            generation_id=f"scripted-{spec.agent_id}",
            finish_reason=finish_reason,
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.0001,
            latency_seconds=0.1,
            attempts=1,
            raw_response=body,
            attempt_log=(
                AttemptRecord(attempt=1, outcome="ok", latency_seconds=0.1,
                              status_code=200, raw_response=text,
                              finish_reason=finish_reason),
            ),
        )

    def messages_for(self, agent_id: str) -> list[dict[str, str]]:
        for sent_agent, messages in self.sent:
            if sent_agent == agent_id:
                return messages
        raise AssertionError(f"{agent_id} was never called")

    def close(self):
        pass


# --- Helpers --------------------------------------------------------------


def start_debate_run(db, run_id=RUN, config=Round2Config()):
    """Open a run labelled as a two-round debate, as a milestone script will."""
    db.start_run(
        run_id,
        config_name=config.config_version,
        question_set_version=config.question_set_version,
        prompt_version=DEBATE_PROMPT_VERSION,
        settings_version=config.settings_version,
        parser_version=config.parser_version,
    )


def given_round1(db, question, registry, replies=None, run_id=RUN, cache=None):
    """Store a complete Round 1 for the question, with chosen per-agent replies."""
    config = Round1Config(
        config_version=DEBATE_CONFIG_VERSION, cache_enabled=cache is not None
    )
    return run_round1_question(
        question,
        registry=registry,
        client=ScriptedClient(replies),
        db=db,
        run_id=run_id,
        config=config,
        cache=cache,
    )


def peer_texts(messages) -> str:
    """The final user turn, which is where the peer responses are."""
    return messages[-1]["content"]


# 1. A complete Round 2


def test_a_full_round2_stores_five_responses_and_one_outcome(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)

    report = run_round2_question(
        question, registry=registry, client=ScriptedClient(), db=db, run_id=RUN
    )

    responses = db.read_responses(RUN, round=2)
    assert len(responses) == 5
    assert {row["agent_id"] for row in responses} == set(AGENTS)
    assert all(row["prompt_version"] == ROUND2_PROMPT_VERSION for row in responses)

    outcomes = db.read_outcomes(RUN, round=2)
    assert len(outcomes) == 1
    assert report.round == 2
    assert report.outcome.state == "CONSENSUS"
    assert report.outcome.consensus_answer == "A"


def test_both_rounds_share_one_run_and_one_run_row(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)
    run_round2_question(
        question, registry=registry, client=ScriptedClient(), db=db, run_id=RUN
    )

    assert len(db.read_runs()) == 1
    assert len(db.read_responses(RUN, round=1)) == 5
    assert len(db.read_responses(RUN, round=2)) == 5
    assert len(db.read_outcomes(RUN)) == 2
    assert db.read_run(RUN)["ended_at"] is None, "the caller still owns the lifecycle"


# 2. What an agent is shown


def test_no_agent_is_shown_its_own_round1_response_as_a_peer(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)
    client = ScriptedClient()

    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    for agent_id in AGENTS:
        messages = client.messages_for(agent_id)
        own_text = ANSWER_TEXTS[agent_id]
        assert peer_texts(messages).count(own_text) == 0, (
            f"{agent_id} saw its own response in the peer set"
        )
        # It is there, but as its own earlier turn, not as anonymous material.
        assert messages[2] == {"role": "assistant", "content": own_text}


def test_each_agent_sees_exactly_the_other_four(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)
    client = ScriptedClient()

    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    for agent_id in AGENTS:
        shown = peer_texts(client.messages_for(agent_id))
        assert shown.count("--- PEER RESPONSE") == 4
        for other_id, text in ANSWER_TEXTS.items():
            if other_id == agent_id:
                continue
            assert text in shown, f"{agent_id} was not shown {other_id}'s response"


def test_no_model_identity_reaches_any_message(question, registry, db):
    """Agent IDs, slugs and provider names must never enter a prompt."""
    start_debate_run(db)
    given_round1(db, question, registry)
    client = ScriptedClient()

    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    identities = set(AGENTS) | {spec.slug for spec in registry.values()}
    identities |= {slug.split("/")[0] for slug in (s.slug for s in registry.values())}
    identities.add("fixture")

    for agent_id, messages in client.sent:
        whole = "\n".join(message["content"] for message in messages).lower()
        for identity in identities:
            assert identity.lower() not in whole, (
                f"{identity!r} leaked into the prompt sent to {agent_id}"
            )


def test_peer_order_is_registry_order_and_deterministic(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)
    client = ScriptedClient()

    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    shown = peer_texts(client.messages_for("agent_llama"))
    positions = [shown.index(ANSWER_TEXTS[other]) for other in AGENTS[1:]]
    assert positions == sorted(positions), "peers were not in registry order"

    # The same stored Round 1 must rebuild the same conversation every time.
    valid = valid_round1_responses(
        db, run_id=RUN, question_id=question["stable_id"], registry=registry
    )
    first = [peer.response_id for peer in peers_for("agent_llama", valid)]
    second = [peer.response_id for peer in peers_for("agent_llama", valid)]
    assert first == second == [peer.response_id for peer in peers_for("agent_llama", valid)]


# 3. Only a valid Round 1 response is debate material (D019)


INVALID_REPLIES = {
    STATUS_REFUSAL: "I cannot answer this question.",
    STATUS_TRUNCATED: ("REASONING: the delta argument is only half writ", "length"),
    STATUS_PARSE_FAIL: "REASONING: the delta argument.\nI think it is probably B.",
    STATUS_API_ERROR: RATE_LIMITED,
}


@pytest.mark.parametrize("status", sorted(INVALID_REPLIES))
def test_an_invalid_round1_response_is_never_shown_as_a_peer(
    question, registry, db, status
):
    replies = dict(ANSWER_TEXTS, agent_deepseek=INVALID_REPLIES[status])
    start_debate_run(db)
    given_round1(db, question, registry, replies)

    stored = {
        row["agent_id"]: row["status"] for row in db.read_responses(RUN, round=1)
    }
    assert stored["agent_deepseek"] == status, "the fixture did not produce that status"

    client = ScriptedClient()
    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    for agent_id in AGENTS:
        shown = peer_texts(client.messages_for(agent_id))
        assert shown.count("--- PEER RESPONSE") == (3 if agent_id != "agent_deepseek" else 4)

    peer_counts = {
        row["agent_id"]: row["peer_count"] for row in db.read_responses(RUN, round=2)
    }
    assert peer_counts["agent_deepseek"] == 4
    assert all(count == 3 for agent, count in peer_counts.items() if agent != "agent_deepseek")


def test_an_agent_whose_own_round1_failed_still_runs_without_an_assistant_turn(
    question, registry, db
):
    replies = dict(ANSWER_TEXTS, agent_qwen=RATE_LIMITED)
    start_debate_run(db)
    given_round1(db, question, registry, replies)

    client = ScriptedClient()
    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    messages = client.messages_for("agent_qwen")
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "did not produce a usable response" in peer_texts(messages)
    assert peer_texts(messages).count("--- PEER RESPONSE") == 4

    row = next(r for r in db.read_responses(RUN, round=2) if r["agent_id"] == "agent_qwen")
    assert row["status"] == STATUS_OK, "a Round 1 failure must not end that agent's debate"


def test_an_agent_with_no_surviving_peers_still_runs_and_stores_peer_count_zero(
    question, registry, db
):
    """Every peer failed Round 1. The agent must still reconsider, alone."""
    replies = {agent: RATE_LIMITED for agent in AGENTS}
    replies["agent_llama"] = ANSWER_TEXTS["agent_llama"]
    start_debate_run(db)
    given_round1(db, question, registry, replies)

    client = ScriptedClient()
    report = run_round2_question(
        question, registry=registry, client=client, db=db, run_id=RUN
    )

    messages = client.messages_for("agent_llama")
    assert "No valid peer responses are available." in peer_texts(messages)
    assert messages[2] == {"role": "assistant", "content": ANSWER_TEXTS["agent_llama"]}

    rows = {row["agent_id"]: row for row in db.read_responses(RUN, round=2)}
    assert rows["agent_llama"]["peer_count"] == 0
    assert rows["agent_llama"]["peer_response_ids"] == []
    assert all(row["peer_count"] in (0, 1) for row in rows.values())
    assert len(report.agents) == 5


def test_peer_response_ids_are_exactly_the_rows_shown_in_the_order_shown(
    question, registry, db
):
    start_debate_run(db)
    given_round1(db, question, registry)

    round1_rows = {
        row["agent_id"]: row["response_id"] for row in db.read_responses(RUN, round=1)
    }
    client = ScriptedClient()
    run_round2_question(question, registry=registry, client=client, db=db, run_id=RUN)

    for row in db.read_responses(RUN, round=2):
        agent_id = row["agent_id"]
        expected = [round1_rows[other] for other in AGENTS if other != agent_id]
        assert row["peer_response_ids"] == expected
        assert row["peer_count"] == 4
        assert round1_rows[agent_id] not in row["peer_response_ids"]

        # The stored IDs must name the texts that were actually shown, in order.
        shown = peer_texts(client.messages_for(agent_id))
        order = [shown.index(ANSWER_TEXTS[other]) for other in AGENTS if other != agent_id]
        assert order == sorted(order)


# 4. Round 2 cannot start a debate that Round 1 never had


def test_a_question_without_a_stored_round1_is_refused(question, registry, db):
    start_debate_run(db)
    client = ScriptedClient()

    with pytest.raises(RunnerError, match="incomplete"):
        run_round2_question(
            question, registry=registry, client=client, db=db, run_id=RUN
        )

    assert client.calls == 0


# 5. Lifecycle guards, shared with Round 1


def test_a_missing_run_is_refused_before_any_call(question, registry, db):
    client = ScriptedClient()

    with pytest.raises(RunnerError, match="has not been started"):
        run_round2_question(
            question, registry=registry, client=client, db=db, run_id="missing"
        )

    assert client.calls == 0


def test_a_finished_run_is_refused_before_any_call(question, registry, db):
    start_debate_run(db)
    given_round1(db, question, registry)
    db.finish_run(RUN)
    client = ScriptedClient()

    with pytest.raises(RunnerError, match="already finished"):
        run_round2_question(
            question, registry=registry, client=client, db=db, run_id=RUN
        )

    assert client.calls == 0


@pytest.mark.parametrize(
    ("config_field", "wrong_value", "reported_field"),
    [
        ("config_version", "round1_config_v2", "config_name"),
        ("question_set_version", "wrong_questions", "question_set_version"),
        ("prompt_version", "round2_v9", "prompt_version"),
        ("settings_version", "agents_v9", "settings_version"),
        ("parser_version", "parser_v9", "parser_version"),
    ],
)
def test_run_labels_must_match_the_config_before_any_call(
    question, registry, db, config_field, wrong_value, reported_field
):
    start_debate_run(db)
    given_round1(db, question, registry)
    mismatched = replace(Round2Config(), **{config_field: wrong_value})
    client = ScriptedClient()

    with pytest.raises(RunnerError, match=reported_field):
        run_round2_question(
            question, registry=registry, client=client, db=db,
            run_id=RUN, config=mismatched,
        )

    assert client.calls == 0


def test_the_round2_prompt_is_accepted_inside_the_combined_debate_label(
    question, registry, db
):
    """round2_v1 is one component of round1_v1+round2_v1, so it must pass."""
    start_debate_run(db)
    given_round1(db, question, registry)

    run_round2_question(
        question, registry=registry, client=ScriptedClient(), db=db, run_id=RUN
    )

    assert db.read_run(RUN)["prompt_version"] == DEBATE_PROMPT_VERSION
    assert len(db.read_responses(RUN, round=2)) == 5


def test_a_cache_the_config_did_not_declare_is_refused(question, registry, db, tmp_path):
    start_debate_run(db)
    given_round1(db, question, registry)
    client = ScriptedClient()

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(RunnerError, match="cache_enabled"):
            run_round2_question(
                question, registry=registry, client=client, db=db,
                run_id=RUN, cache=cache,
            )

    assert client.calls == 0


# 6. Failures and the cache


def test_one_agent_failing_round2_is_stored_as_api_error_and_the_rest_continue(
    question, registry, db
):
    start_debate_run(db)
    given_round1(db, question, registry)

    client = ScriptedClient(dict(ANSWER_TEXTS, agent_mistral=RATE_LIMITED))
    report = run_round2_question(
        question, registry=registry, client=client, db=db, run_id=RUN
    )

    rows = {row["agent_id"]: row for row in db.read_responses(RUN, round=2)}
    assert len(rows) == 5
    assert rows["agent_mistral"]["status"] == STATUS_API_ERROR
    assert rows["agent_mistral"]["extracted_letter"] is None
    # The failed agent still records which peers it was given.
    assert rows["agent_mistral"]["peer_count"] == 4
    assert report.outcome.failed_agents == ("agent_mistral",)
    # Two of the four survivors said A; the threshold is still three of five.
    assert report.outcome.valid_answer_count == 4


def test_a_cached_round2_reply_makes_no_call(question, registry, db, tmp_path):
    config = Round2Config(cache_enabled=True)
    start_debate_run(db)

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        given_round1(db, question, registry)
        first = ScriptedClient(provider="testprovider")
        run_round2_question(question, registry=registry, client=first, db=db,
                            run_id=RUN, config=config, cache=cache)
        assert first.calls == 5

        # A second run over the same stored Round 1 rebuilds identical messages.
        start_debate_run(db, "second_run")
        given_round1(db, question, registry, run_id="second_run")
        second = ScriptedClient(provider="testprovider")
        run_round2_question(question, registry=registry, client=second, db=db,
                            run_id="second_run", config=config, cache=cache)

    assert second.calls == 0
    assert all(row["cache_hit"] == 1 for row in db.read_responses("second_run", round=2))


def test_a_failed_round2_call_is_not_cached(question, registry, db, tmp_path):
    config = Round2Config(cache_enabled=True)
    start_debate_run(db)

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        given_round1(db, question, registry)
        run_round2_question(
            question, registry=registry,
            client=ScriptedClient(dict(ANSWER_TEXTS, agent_gemma=RATE_LIMITED),
                                  provider="testprovider"),
            db=db, run_id=RUN, config=config, cache=cache,
        )

        start_debate_run(db, "retry_run")
        given_round1(db, question, registry, run_id="retry_run")
        retry = ScriptedClient(provider="testprovider")
        run_round2_question(question, registry=registry, client=retry, db=db,
                            run_id="retry_run", config=config, cache=cache)

    assert retry.calls == 1, "only the uncached failed agent should be called again"
    row = next(r for r in db.read_responses("retry_run", round=2)
               if r["agent_id"] == "agent_gemma")
    assert row["status"] == STATUS_OK


# 7. The fixture client still drives a whole debate offline


def test_the_dry_run_fixture_client_completes_both_rounds(question, registry, db):
    start_debate_run(db)
    run_round1_question(
        question, registry=registry, client=FixtureClient(question), db=db,
        run_id=RUN, config=Round1Config(config_version=DEBATE_CONFIG_VERSION),
    )
    report = run_round2_question(
        question, registry=registry, client=FixtureClient(question), db=db, run_id=RUN
    )

    assert len(report.agents) == 5
    # The fixture's fifth agent refuses in Round 1, so nobody sees four peers.
    assert {agent.peer_count for agent in report.agents} == {3, 4}
