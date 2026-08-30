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
            "_attempts": 1,
        }

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
            "_attempts": 1,
        }

    monkeypatch.setattr(OpenRouterClient, "_post_with_retries", fake_post)
    client = OpenRouterClient(api_key="test-key-not-real")
    result = client.complete(_spec(), [{"role": "user", "content": "hi"}])

    assert result.text == ""
    assert result.finish_reason == "length"


def test_malformed_body_raises_rather_than_returning_a_blank_answer(monkeypatch):
    def fake_post(self, path, payload):
        return {"unexpected": True}

    monkeypatch.setattr(OpenRouterClient, "_post_with_retries", fake_post)
    client = OpenRouterClient(api_key="test-key-not-real")

    with pytest.raises(ApiRequestError):
        client.complete(_spec(), [{"role": "user", "content": "hi"}])


def test_missing_api_key_is_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(api_client, "_repository_root", lambda: tmp_path)

    with pytest.raises(api_client.ApiConfigurationError):
        api_client.load_api_key()
