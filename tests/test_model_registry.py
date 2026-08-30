"""Tests for the five-agent config in configs/models/agents_v1.yaml.

Changing a value here changes the experiment, so these tests are meant to fail
loudly if someone edits the file without meaning to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mad.api_client import ApiConfigurationError, load_model_registry

CONFIG = Path(__file__).resolve().parents[1] / "configs/models/agents_v1.yaml"

EXPECTED_SLUGS = {
    "agent_llama": "meta-llama/llama-4-maverick",
    "agent_qwen": "qwen/qwen3.8-27b",
    "agent_mistral": "mistralai/mistral-large-2512",
    "agent_deepseek": "deepseek/deepseek-v4-pro-0813",
    "agent_gemma": "google/gemma-4-31b-it",
}


@pytest.fixture(scope="module")
def registry():
    return load_model_registry(CONFIG)


def test_exactly_five_agents(registry):
    assert len(registry) == 5


def test_agent_slugs_are_the_fixed_five(registry):
    assert {a: s.slug for a, s in registry.items()} == EXPECTED_SLUGS


def test_generation_is_deterministic_for_every_agent(registry):
    """D002: temperature 0 so the only variation is model identity and debate."""
    for agent_id, spec in registry.items():
        assert spec.temperature == 0.0, agent_id
        assert spec.top_p == 1.0, agent_id


def test_every_agent_requires_parameter_support(registry):
    """Route only to providers that honour temperature 0, never silently drop it."""
    for agent_id, spec in registry.items():
        assert spec.require_parameters is True, agent_id


def test_completion_budget_is_shared(registry):
    assert {spec.max_tokens for spec in registry.values()} == {1024}


def test_agents_use_five_distinct_model_families(registry):
    families = {spec.slug.split("/")[0] for spec in registry.values()}
    assert len(families) == 5


def test_registry_without_generation_settings_is_rejected(tmp_path):
    """Temperature and top_p define the experiment, so they may never be implicit."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        "agents:\n"
        "  agent_x:\n"
        "    slug: vendor/model\n",
        encoding="utf-8",
    )
    with pytest.raises(ApiConfigurationError):
        load_model_registry(config)


def test_no_api_key_is_present_in_the_registry_file():
    text = CONFIG.read_text(encoding="utf-8")
    assert "sk-or-" not in text
    assert "OPENROUTER_API_KEY" not in text
