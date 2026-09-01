"""Tests for the OpenRouter client. None of these touch the network.

Real API calls live in scripts/check_models.py, because they cost money.
"""

from __future__ import annotations

import pytest

from mad import api_client
from mad.api_client import (
    DEFAULT_MAX_ATTEMPTS,
    ApiRequestError,
    ModelSpec,
    OpenRouterClient,
)


class FakeResponse:
    """Stands in for a requests Response. Nothing here touches the network."""

    def __init__(self, status_code=200, body=None, text="", bad_json=False):
        # text is the raw body as it arrived; body is what .json() would give.
        self.status_code = status_code
        self._body = body
        self.text = text
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


def _client_with(monkeypatch, responses):
    """A client whose session replays the given responses, one per attempt.

    A response may be an exception, which is raised instead of returned.
    """
    sent: list = []

    def fake_post(url, json, timeout):
        item = responses[len(sent)]
        sent.append(json)
        if isinstance(item, Exception):
            raise item
        return item

    # No waiting between retries - these tests must stay instant.
    monkeypatch.setattr(api_client.time, "sleep", lambda seconds: None)
    client = OpenRouterClient(api_key="test-key-not-real")
    monkeypatch.setattr(client._session, "post", fake_post)
    return client, sent


def _ok_body(content="B", cost=0.0001):
    return {
        "id": "gen-1",
        "model": "vendor/model-1",
        "provider": "SomeProvider",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": cost},
    }


def _spec(**overrides) -> ModelSpec:
    base = dict(
        agent_id="agent_test",
        slug="vendor/model-1",
        display_name="Test Model",
        developer="Vendor",
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        allow_provider_fallbacks=True,
        require_parameters=True,
    )
    base.update(overrides)
    return ModelSpec(**base)


def test_retry_budget_is_one_initial_attempt_plus_one_retry():
    """D012: the proposal promises exactly one retry, so two attempts in total."""
    assert DEFAULT_MAX_ATTEMPTS == 2


def test_request_sends_fixed_generation_settings_and_no_provider_pin(monkeypatch):
    """Temperature 0 and top-p 1 are the experiment. Provider choice is OpenRouter's."""
    captured: dict = {}

    def fake_post(self, path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "id": "gen-1",
            "model": "vendor/model-1",
            "provider": "SomeProvider",
            "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
        }, ()

    monkeypatch.setattr(OpenRouterClient, "_post_with_retries", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    client = OpenRouterClient(api_key="test-key-not-real")
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    provider_block = captured["payload"]["provider"]
    assert "only" not in provider_block, "provider pinning is deferred (D015)"
    assert provider_block["allow_fallbacks"] is True
    assert provider_block["require_parameters"] is True
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["top_p"] == 1.0
    assert captured["payload"]["max_tokens"] == 1024

    # The served provider is recorded for audit, never used to reject a response.
    assert result.provider == "SomeProvider"
    assert result.served_slug == "vendor/model-1"
    assert result.text == "B"


def test_empty_content_is_returned_as_empty_string_not_none(monkeypatch):
    """A null content field must not crash; the parser decides it is a failure."""

    def fake_post(self, path, payload):
        return {
            "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {},
        }, ()

    monkeypatch.setattr(OpenRouterClient, "_post_with_retries", fake_post)
    client = OpenRouterClient(api_key="test-key-not-real")
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.text == ""
    assert result.finish_reason == "length"


def test_malformed_body_raises_rather_than_returning_a_blank_answer(monkeypatch):
    def fake_post(self, path, payload):
        return {"unexpected": True}, ()

    monkeypatch.setattr(OpenRouterClient, "_post_with_retries", fake_post)
    client = OpenRouterClient(api_key="test-key-not-real")

    with pytest.raises(ApiRequestError):
        client.complete(_spec(), [{"role": "user", "content": "hi"}])


def test_missing_api_key_is_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(api_client, "_repository_root", lambda: tmp_path)

    with pytest.raises(api_client.ApiConfigurationError):
        api_client.load_api_key()


# Retry behaviour. These drive the real _post_with_retries through a fake session.


def test_a_temporary_failure_is_retried_and_the_second_attempt_can_succeed(monkeypatch):
    client, sent = _client_with(
        monkeypatch,
        [FakeResponse(429, text="rate limited"), FakeResponse(200, _ok_body())],
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.text == "B"
    assert result.attempts == 2
    assert len(sent) == 2
    assert [record.outcome for record in result.attempt_log] == ["http_error", "ok"]
    assert result.attempt_log[0].status_code == 429


def test_two_temporary_failures_exhaust_the_budget(monkeypatch):
    client, sent = _client_with(
        monkeypatch,
        [FakeResponse(503, text="unavailable"), FakeResponse(503, text="unavailable")],
    )
    with pytest.raises(ApiRequestError) as caught:
        client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert "failed after 2 attempts" in str(caught.value)
    assert len(sent) == DEFAULT_MAX_ATTEMPTS
    # The failed call is still costable: both attempts are on the error.
    assert len(caught.value.attempt_log) == 2


def test_a_permanent_failure_is_not_retried(monkeypatch):
    client, sent = _client_with(monkeypatch, [FakeResponse(401, text="bad key")])
    with pytest.raises(ApiRequestError) as caught:
        client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert len(sent) == 1, "a bad key is not worth a second attempt"
    assert caught.value.attempt_log[0].status_code == 401


def test_a_transport_error_is_retried(monkeypatch):
    client, sent = _client_with(
        monkeypatch,
        [api_client.requests.Timeout("timed out"), FakeResponse(200, _ok_body())],
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.attempts == 2
    assert result.attempt_log[0].outcome == "transport_error"


def test_http_200_carrying_an_error_key_is_not_treated_as_success(monkeypatch):
    error_body = {"error": {"code": 502, "message": "upstream is down"}}
    client, sent = _client_with(
        monkeypatch, [FakeResponse(200, error_body), FakeResponse(200, _ok_body())]
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.text == "B"
    assert result.attempt_log[0].outcome == "upstream_error"
    assert "upstream is down" in result.attempt_log[0].error


def test_a_body_that_is_not_json_is_retried_not_crashed(monkeypatch):
    client, sent = _client_with(
        monkeypatch,
        [FakeResponse(200, bad_json=True, text="<html>gateway</html>"), FakeResponse(200, _ok_body())],
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.text == "B"
    assert result.attempt_log[0].outcome == "bad_json"


def test_a_body_that_is_never_json_ends_as_an_api_error(monkeypatch):
    client, sent = _client_with(
        monkeypatch,
        [FakeResponse(200, bad_json=True), FakeResponse(200, bad_json=True)],
    )
    with pytest.raises(ApiRequestError) as caught:
        client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert "invalid JSON" in str(caught.value)


def test_the_whole_raw_body_is_kept_for_the_cache(monkeypatch):
    """P5: the cache stores the complete API body, not a selection of fields."""
    body = _ok_body()
    client, sent = _client_with(monkeypatch, [FakeResponse(200, body)])
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.raw_response == body


def test_every_attempt_is_costed_not_only_the_successful_one(monkeypatch):
    """P6: failures cost money, so tokens and cost are recorded per attempt."""
    failed = {"error": "upstream", "usage": {"prompt_tokens": 10, "completion_tokens": 0, "cost": 0.00002}}
    client, sent = _client_with(
        monkeypatch, [FakeResponse(200, failed), FakeResponse(200, _ok_body(cost=0.0001))]
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.cost_usd == 0.0001
    assert sum(record.cost_usd for record in result.attempt_log) == pytest.approx(0.00012)


def test_a_failed_attempt_keeps_its_whole_raw_body(monkeypatch):
    """P4 stores a raw response per attempt, so a failure must keep one too."""
    long_body = "gateway timeout: " + "x" * 900
    client, _ = _client_with(
        monkeypatch, [FakeResponse(504, text=long_body), FakeResponse(200, _ok_body())]
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    failed = result.attempt_log[0]
    assert failed.raw_response == long_body, "the raw body must not be truncated"
    assert len(failed.error) <= 400, "the summary stays short; the raw body is the record"


def test_a_successful_attempt_keeps_its_raw_body_and_finish_reason(monkeypatch):
    import json as _json

    body = _ok_body()
    client, _ = _client_with(monkeypatch, [FakeResponse(200, body, text=_json.dumps(body))])
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert _json.loads(result.attempt_log[0].raw_response) == body
    assert result.attempt_log[0].finish_reason == "stop"


def test_a_transport_error_has_no_raw_body_and_none_is_invented(monkeypatch):
    client, _ = _client_with(
        monkeypatch,
        [api_client.requests.Timeout("timed out"), FakeResponse(200, _ok_body())],
    )
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.attempt_log[0].raw_response == "", "no response arrived, so nothing to store"
    assert "timed out" in result.attempt_log[0].error


def test_a_malformed_200_body_keeps_the_attempt_log_with_a_truthful_outcome(monkeypatch):
    """The reply arrived and was paid for; losing its audit trail loses money."""
    malformed = {
        "id": "gen-9",
        "model": "vendor/model-1",
        "usage": {"prompt_tokens": 11, "completion_tokens": 0, "cost": 0.00003},
        # no "choices" at all
    }
    import json as _json

    client, _ = _client_with(monkeypatch, [FakeResponse(200, malformed, text=_json.dumps(malformed))])
    with pytest.raises(ApiRequestError, match="Malformed response") as caught:
        client.complete(_spec(), [{"role": "user", "content": "hi"}])

    log = caught.value.attempt_log
    assert len(log) == 1
    assert log[0].attempt == 1
    assert log[0].outcome == "malformed_body", "it was logged as ok, but it was not ok"
    assert _json.loads(log[0].raw_response) == malformed
    assert log[0].cost_usd == pytest.approx(0.00003)
    assert log[0].latency_seconds >= 0.0
