"""Load the small, sanitized data bundle used by the public Streamlit app.

The deployed viewer must not depend on the private ``storage/`` directory. Its
only data source is a versioned JSON artifact containing the verified overview
and six curated debates from the accepted experiment. This module performs no
SQLite access and has no write path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mad.viewer_data import (
    AgentAccuracyView,
    AgentResponseView,
    AttemptView,
    ConsensusStateView,
    ExperimentOverview,
    GroupAccuracyView,
    MAIN_RUN_ID,
    MessageView,
    QuestionReplay,
    RoundOutcomeView,
    RoundReplay,
    UsageView,
    ViewerDataError,
)


PUBLIC_BUNDLE_SCHEMA_VERSION = 1
PUBLIC_QUESTION_IDS = (
    "mmlu_pro_v1:test:5503",
    "mmlu_pro_v1:test:11081",
    "mmlu_pro_v1:test:5104",
    "mmlu_pro_v1:test:1203",
    "mmlu_pro_v1:test:2697",
    "mmlu_pro_v1:test:11994",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_BUNDLE = (
    REPOSITORY_ROOT / "data" / "public_viewer" / "accepted_experiment_v1.json"
)

# These strings have no legitimate reason to occur in a deployment artifact.
# Refusing them makes an accidental secret or local-path export fail closed.
FORBIDDEN_PUBLIC_TEXT = (
    "sk-or-",
    "OPENROUTER_API_KEY",
    '"api_key"',
    "/Users/",
    "\\Users\\",
    ".env",
    "storage/results.sqlite",
    "storage/cache.sqlite",
)
OPENROUTER_USER_ID_PATTERN = re.compile(r"\buser_[A-Za-z0-9]{10,}\b")
OPENROUTER_USER_ID_REDACTION = "[redacted-openrouter-user-id]"


def sanitize_public_value(value: Any) -> Any:
    """Remove account identifiers from a JSON-compatible public value.

    OpenRouter can include a ``user_id`` in an upstream error body. The private
    database keeps that body unchanged as audit evidence; the public
    presentation copy does not need the account identifier.
    """
    if isinstance(value, str):
        return OPENROUTER_USER_ID_PATTERN.sub(OPENROUTER_USER_ID_REDACTION, value)
    if isinstance(value, dict):
        return {
            key: (
                OPENROUTER_USER_ID_REDACTION
                if key == "user_id"
                else sanitize_public_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_public_value(item) for item in value)
    return value


def find_forbidden_public_content(text: str) -> list[str]:
    """Name private values that must never enter the tracked public bundle."""
    found = [token for token in FORBIDDEN_PUBLIC_TEXT if token in text]
    if OPENROUTER_USER_ID_PATTERN.search(text):
        found.append("OpenRouter user identifier")
    return found


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ViewerDataError(f"public viewer bundle {label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ViewerDataError(f"public viewer bundle {label} must be a list")
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ViewerDataError(f"public viewer bundle does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    found = find_forbidden_public_content(text)
    if found:
        raise ViewerDataError(
            "public viewer bundle contains forbidden private text: "
            + ", ".join(found)
        )
    try:
        payload = _mapping(json.loads(text), label="root")
    except json.JSONDecodeError as error:
        raise ViewerDataError(f"public viewer bundle is not valid JSON: {path}") from error

    if payload.get("schema_version") != PUBLIC_BUNDLE_SCHEMA_VERSION:
        raise ViewerDataError(
            "unsupported public viewer bundle schema: "
            f"{payload.get('schema_version')!r}"
        )
    provenance = _mapping(payload.get("provenance"), label="provenance")
    if provenance.get("run_id") != MAIN_RUN_ID:
        raise ViewerDataError(
            f"public viewer bundle belongs to {provenance.get('run_id')!r}, "
            f"not {MAIN_RUN_ID!r}"
        )
    questions = _mapping(payload.get("questions"), label="questions")
    if set(questions) != set(PUBLIC_QUESTION_IDS):
        raise ViewerDataError(
            "public viewer bundle must contain exactly the six approved questions"
        )
    return payload


def _overview(payload: dict[str, Any]) -> ExperimentOverview:
    row = _mapping(payload.get("overview"), label="overview")
    try:
        return ExperimentOverview(
            run_id=str(row["run_id"]),
            settings_version=str(row["settings_version"]),
            evaluation_version=str(row["evaluation_version"]),
            question_count=int(row["question_count"]),
            wall_clock_hours=float(row["wall_clock_hours"]),
            group_accuracy=tuple(
                GroupAccuracyView(**_mapping(item, label="group_accuracy item"))
                for item in _sequence(row["group_accuracy"], label="group_accuracy")
            ),
            agent_accuracy=tuple(
                AgentAccuracyView(**_mapping(item, label="agent_accuracy item"))
                for item in _sequence(row["agent_accuracy"], label="agent_accuracy")
            ),
            consensus_states=tuple(
                ConsensusStateView(**_mapping(item, label="consensus_states item"))
                for item in _sequence(row["consensus_states"], label="consensus_states")
            ),
            usage=tuple(
                UsageView(**_mapping(item, label="usage item"))
                for item in _sequence(row["usage"], label="usage")
            ),
            stayed_correct=int(row["stayed_correct"]),
            became_correct=int(row["became_correct"]),
            became_incorrect=int(row["became_incorrect"]),
            stayed_incorrect=int(row["stayed_incorrect"]),
            effect_points=float(row["effect_points"]),
            confidence_low_points=float(row["confidence_low_points"]),
            confidence_high_points=float(row["confidence_high_points"]),
            mcnemar_p_value=float(row["mcnemar_p_value"]),
            corrected_from_undecided=int(row["corrected_from_undecided"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ViewerDataError("public viewer overview is malformed") from error


def _message(row: object) -> MessageView:
    try:
        return MessageView(**_mapping(row, label="message"))
    except TypeError as error:
        raise ViewerDataError("public viewer message is malformed") from error


def _attempt(row: object) -> AttemptView:
    try:
        return AttemptView(**_mapping(row, label="attempt"))
    except TypeError as error:
        raise ViewerDataError("public viewer attempt is malformed") from error


def _response(row: object) -> AgentResponseView:
    data = _mapping(row, label="response").copy()
    try:
        data["peer_response_ids"] = tuple(int(value) for value in data["peer_response_ids"])
        data["messages"] = tuple(_message(item) for item in data["messages"])
        data["attempts"] = tuple(_attempt(item) for item in data["attempts"])
        return AgentResponseView(**data)
    except (KeyError, TypeError, ValueError) as error:
        raise ViewerDataError("public viewer response is malformed") from error


def _round(row: object) -> RoundReplay:
    data = _mapping(row, label="round")
    try:
        return RoundReplay(
            round=int(data["round"]),
            responses=tuple(
                _response(item)
                for item in _sequence(data["responses"], label="round responses")
            ),
            outcome=RoundOutcomeView(
                **_mapping(data["outcome"], label="round outcome")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ViewerDataError("public viewer round is malformed") from error


def _replay(row: object) -> QuestionReplay:
    data = _mapping(row, label="question replay")
    try:
        replay = QuestionReplay(
            run_id=str(data["run_id"]),
            settings_version=str(data["settings_version"]),
            question_id=str(data["question_id"]),
            category=str(data["category"]),
            question=str(data["question"]),
            options=tuple(str(option) for option in data["options"]),
            correct_answer=str(data["correct_answer"]),
            round1=_round(data["round1"]),
            round2=_round(data["round2"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ViewerDataError("public viewer question replay is malformed") from error
    if replay.run_id != MAIN_RUN_ID:
        raise ViewerDataError(
            f"public replay {replay.question_id!r} belongs to the wrong run"
        )
    if replay.round1.round != 1 or replay.round2.round != 2:
        raise ViewerDataError(
            f"public replay {replay.question_id!r} has invalid round numbers"
        )
    if len(replay.round1.responses) != 5 or len(replay.round2.responses) != 5:
        raise ViewerDataError(
            f"public replay {replay.question_id!r} must contain five agents per round"
        )
    return replay


def load_public_experiment_overview(
    bundle_path: str | Path = DEFAULT_PUBLIC_BUNDLE,
) -> ExperimentOverview:
    """Read the aggregate results from the sanitized deployment artifact."""
    payload = _load_payload(Path(bundle_path))
    overview = _overview(payload)
    if overview.run_id != MAIN_RUN_ID:
        raise ViewerDataError("public overview belongs to the wrong run")
    return overview


def load_public_question_replay(
    question_id: str,
    bundle_path: str | Path = DEFAULT_PUBLIC_BUNDLE,
) -> QuestionReplay:
    """Read one of the six approved replays from the public artifact."""
    question_id = question_id.strip()
    if question_id not in PUBLIC_QUESTION_IDS:
        raise ViewerDataError(
            f"question {question_id!r} is not approved for the public viewer"
        )
    payload = _load_payload(Path(bundle_path))
    questions = _mapping(payload["questions"], label="questions")
    replay = _replay(questions[question_id])
    if replay.question_id != question_id:
        raise ViewerDataError(
            f"public replay key {question_id!r} contains {replay.question_id!r}"
        )
    return replay
