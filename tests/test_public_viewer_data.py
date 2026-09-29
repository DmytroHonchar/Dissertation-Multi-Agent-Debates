"""Safety and fidelity tests for the public deployment artifact."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from mad.public_viewer_data import (
    DEFAULT_PUBLIC_BUNDLE,
    OPENROUTER_USER_ID_REDACTION,
    PUBLIC_QUESTION_IDS,
    find_forbidden_public_content,
    load_public_experiment_overview,
    load_public_question_replay,
    sanitize_public_value,
)
from mad.viewer_data import (
    DEFAULT_CACHE_PATH,
    DEFAULT_RESULTS_PATH,
    ViewerDataError,
    load_experiment_overview,
    load_question_replay,
)


PRIVATE_REPLAY_SOURCES_AVAILABLE = (
    Path(DEFAULT_RESULTS_PATH).is_file() and Path(DEFAULT_CACHE_PATH).is_file()
)
requires_private_replay_sources = pytest.mark.skipif(
    not PRIVATE_REPLAY_SOURCES_AVAILABLE,
    reason="private research databases are intentionally absent from a public clone",
)


def test_public_bundle_contains_only_the_approved_run_and_questions():
    payload = json.loads(DEFAULT_PUBLIC_BUNDLE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["provenance"]["run_id"] == "experiment_agents_v7_20260923T114934Z"
    assert set(payload["questions"]) == set(PUBLIC_QUESTION_IDS)
    assert len(payload["questions"]) == 6


def test_public_bundle_has_no_secret_local_or_private_storage_text():
    text = DEFAULT_PUBLIC_BUNDLE.read_text(encoding="utf-8")

    assert find_forbidden_public_content(text) == []
    assert OPENROUTER_USER_ID_REDACTION in text
    assert DEFAULT_PUBLIC_BUNDLE.stat().st_size < 2_000_000


def test_public_overview_is_identical_to_the_verified_local_overview():
    assert load_public_experiment_overview() == load_experiment_overview()


@pytest.mark.parametrize("question_id", PUBLIC_QUESTION_IDS)
@requires_private_replay_sources
def test_public_replay_is_identical_to_the_verified_local_replay(question_id):
    public = asdict(load_public_question_replay(question_id))
    verified_local = sanitize_public_value(asdict(load_question_replay(question_id)))

    assert public == verified_local


def test_public_loader_refuses_an_unapproved_question():
    with pytest.raises(ViewerDataError, match="not approved"):
        load_public_question_replay("mmlu_pro_v1:test:not-public")


def test_public_loader_refuses_private_text(tmp_path: Path):
    payload = json.loads(DEFAULT_PUBLIC_BUNDLE.read_text(encoding="utf-8"))
    payload["questions"][PUBLIC_QUESTION_IDS[0]]["question"] = "sk-or-private"
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ViewerDataError, match="forbidden private text"):
        load_public_question_replay(PUBLIC_QUESTION_IDS[0], unsafe)


def test_public_loader_refuses_an_openrouter_user_identifier(tmp_path: Path):
    payload = json.loads(DEFAULT_PUBLIC_BUNDLE.read_text(encoding="utf-8"))
    payload["questions"][PUBLIC_QUESTION_IDS[0]]["question"] = (
        "user_3HaWA1ZQOnBcZqdUhbE4ArfDVyB"
    )
    unsafe = tmp_path / "unsafe-user-id.json"
    unsafe.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ViewerDataError, match="OpenRouter user identifier"):
        load_public_question_replay(PUBLIC_QUESTION_IDS[0], unsafe)
