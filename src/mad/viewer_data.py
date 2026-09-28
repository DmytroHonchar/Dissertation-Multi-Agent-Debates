"""Read-only data for the Streamlit replay interface.

This module reconstructs what one accepted debate question did without calling
a model or changing experimental evidence. Correctness comes from the verified
evaluation export, never from an answer-key file. Prompts come from the real
prompt builders, and their request keys are checked against the response cache.

The SQLite connections use ``mode=ro`` and ``PRAGMA query_only``. The UI can
therefore inspect runs, responses, attempts and votes, but this layer has no
route for writing any of them.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mad.api_client import ModelSpec, load_model_registry
from mad.cache import cache_key
from mad.debate import peers_for, valid_round1_responses
from mad.prompts_v1 import (
    answer_letters,
    assert_no_answer_key,
    build_round1_messages,
    build_round2_messages,
)


# 1. Fixed accepted artifacts

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MAIN_RUN_ID = "experiment_agents_v7_20260923T114934Z"

DEFAULT_RESULTS_PATH = REPOSITORY_ROOT / "storage" / "results.sqlite"
DEFAULT_CACHE_PATH = REPOSITORY_ROOT / "storage" / "cache.sqlite"
DEFAULT_QUESTIONS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "frozen"
    / "mmlu_pro_v1"
    / "model_inputs"
    / "experimental_questions.jsonl"
)
DEFAULT_EVALUATION_EXPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "main_experiment_20260923"
    / "evaluation_summary.json"
)
DEFAULT_REGISTRY_PATH = REPOSITORY_ROOT / "configs" / "models" / "agents_v7.yaml"


class ViewerDataError(RuntimeError):
    """The stored evidence cannot be replayed safely and consistently."""


# 2. Read-only view models


@dataclass(frozen=True)
class MessageView:
    role: str
    content: str


@dataclass(frozen=True)
class AttemptView:
    attempt_number: int
    outcome: str
    status: str
    http_status: int | None
    error: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float


@dataclass(frozen=True)
class AgentResponseView:
    response_id: int
    round: int
    agent_id: str
    display_name: str
    developer: str
    requested_slug: str
    served_slug: str
    provider: str
    temperature: float
    top_p: float
    max_tokens: int
    pinned_provider: str | None
    reasoning_max_tokens: int | None
    raw_response: str
    status: str
    extracted_letter: str | None
    finish_reason: str
    attempt_count: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    cache_hit: bool
    own_round1_response_id: int | None
    peer_response_ids: tuple[int, ...]
    messages: tuple[MessageView, ...]
    request_cache_key: str
    request_found_in_cache: bool
    attempts: tuple[AttemptView, ...]


@dataclass(frozen=True)
class RoundOutcomeView:
    round: int
    consensus_state: str
    consensus_answer: str | None
    decided: bool
    valid_answer_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    total_latency_seconds: float


@dataclass(frozen=True)
class RoundReplay:
    round: int
    responses: tuple[AgentResponseView, ...]
    outcome: RoundOutcomeView


@dataclass(frozen=True)
class QuestionReplay:
    run_id: str
    settings_version: str
    question_id: str
    category: str
    question: str
    options: tuple[str, ...]
    correct_answer: str
    round1: RoundReplay
    round2: RoundReplay


@dataclass(frozen=True)
class GroupAccuracyView:
    round: int
    correct: int
    questions: int
    accuracy_percent: float
    decided: int
    undecided: int


@dataclass(frozen=True)
class AgentAccuracyView:
    round: int
    agent_id: str
    accuracy_percent: float
    correct: int
    valid_answers: int
    failure_count: int
    failure_rate_percent: float


@dataclass(frozen=True)
class ConsensusStateView:
    round: int
    state: str
    count: int


@dataclass(frozen=True)
class UsageView:
    round: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    api_attempts: int
    retried_responses: int


@dataclass(frozen=True)
class ExperimentOverview:
    run_id: str
    settings_version: str
    evaluation_version: str
    question_count: int
    wall_clock_hours: float
    group_accuracy: tuple[GroupAccuracyView, ...]
    agent_accuracy: tuple[AgentAccuracyView, ...]
    consensus_states: tuple[ConsensusStateView, ...]
    usage: tuple[UsageView, ...]
    stayed_correct: int
    became_correct: int
    became_incorrect: int
    stayed_incorrect: int
    effect_points: float
    confidence_low_points: float
    confidence_high_points: float
    mcnemar_p_value: float
    corrected_from_undecided: int

    @property
    def total_cost_usd(self) -> float:
        return sum(round_usage.cost_usd for round_usage in self.usage)

    @property
    def total_tokens(self) -> int:
        return sum(
            round_usage.prompt_tokens + round_usage.completion_tokens
            for round_usage in self.usage
        )

    @property
    def total_failures(self) -> int:
        return sum(result.failure_count for result in self.agent_accuracy)


# 3. Enforced read-only SQLite access


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ViewerDataError(f"required read-only file does not exist: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


class _ReadOnlyResults:
    """The small ResultsDatabase reading surface needed for one replay."""

    def __init__(self, path: Path) -> None:
        self._connection = _open_read_only(path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> _ReadOnlyResults:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def read_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def read_responses(
        self,
        run_id: str,
        *,
        round: int | None = None,
        question_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM model_responses WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        for column, value in (
            ("round", round),
            ("question_id", question_id),
            ("agent_id", agent_id),
        ):
            if value is not None:
                sql += f" AND {column} = ?"
                parameters.append(value)
        sql += " ORDER BY question_id, round, agent_id"

        rows = [
            dict(row)
            for row in self._connection.execute(sql, tuple(parameters)).fetchall()
        ]
        for row in rows:
            row["peer_response_ids"] = json.loads(row["peer_response_ids"])
        return rows

    def read_outcomes(
        self, run_id: str, *, round: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM question_outcomes WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if round is not None:
            sql += " AND round = ?"
            parameters.append(round)
        return [
            dict(row)
            for row in self._connection.execute(
                sql + " ORDER BY question_id, round", tuple(parameters)
            ).fetchall()
        ]

    def read_attempts(self, response_id: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM response_attempts "
                "WHERE response_id = ? ORDER BY attempt_number",
                (response_id,),
            ).fetchall()
        ]


# 4. Loading safe question and evaluation artifacts


def _load_question(path: Path, question_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise ViewerDataError(f"model-input file does not exist: {path}")

    found: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as error:
                raise ViewerDataError(
                    f"{path} line {line_number} is not valid JSON"
                ) from error
            if candidate.get("stable_id") == question_id:
                if found is not None:
                    raise ViewerDataError(f"{path} contains {question_id!r} twice")
                found = candidate

    if found is None:
        raise ViewerDataError(f"question {question_id!r} is not in {path}")
    assert_no_answer_key(found)
    return found


def _load_exported_question(
    path: Path, *, run_id: str, question_id: str
) -> dict[str, Any]:
    if not path.is_file():
        raise ViewerDataError(f"evaluation export does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        exported_run = payload["provenance"]["run"]["run_id"]
        questions = payload["questions"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ViewerDataError(f"evaluation export is malformed: {path}") from error

    if exported_run != run_id:
        raise ViewerDataError(
            f"evaluation export belongs to {exported_run!r}, not {run_id!r}"
        )
    matches = [row for row in questions if row.get("question_id") == question_id]
    if len(matches) != 1:
        raise ViewerDataError(
            f"evaluation export has {len(matches)} rows for {question_id!r}, expected one"
        )
    answer = matches[0].get("correct_answer")
    if not isinstance(answer, str) or len(answer) != 1 or not answer.isalpha():
        raise ViewerDataError(
            f"evaluation export has no valid correct answer for {question_id!r}"
        )
    return matches[0]


def load_experiment_overview(
    *,
    run_id: str = MAIN_RUN_ID,
    evaluation_export_path: str | Path = DEFAULT_EVALUATION_EXPORT,
) -> ExperimentOverview:
    """Load verified aggregate results for the dashboard; no SQLite is needed."""
    path = Path(evaluation_export_path)
    if not path.is_file():
        raise ViewerDataError(f"evaluation export does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["provenance"]
        run = provenance["run"]
        if run["run_id"] != run_id:
            raise ViewerDataError(
                f"evaluation export belongs to {run['run_id']!r}, not {run_id!r}"
            )

        group_accuracy = tuple(
            GroupAccuracyView(
                round=int(row["round"]),
                correct=int(row["correct"]),
                questions=int(row["questions"]),
                accuracy_percent=float(row["accuracy_percent"]),
                decided=int(row["decided"]),
                undecided=int(row["undecided"]),
            )
            for row in payload["group_accuracy"]
        )
        agent_accuracy = tuple(
            AgentAccuracyView(
                round=int(row["round"]),
                agent_id=str(row["agent_id"]),
                accuracy_percent=float(row["accuracy_percent"]),
                correct=int(row["correct"]),
                valid_answers=int(row["valid_answers"]),
                failure_count=int(row["failure_count"]),
                failure_rate_percent=float(row["failure_rate_percent"]),
            )
            for row in payload["agent_accuracy"]
        )
        consensus_states = tuple(
            ConsensusStateView(
                round=int(row["round"]),
                state=str(row["state"]),
                count=int(row["count"]),
            )
            for row in payload["consensus_states"]
        )
        usage = tuple(
            UsageView(
                round=int(row["round"]),
                prompt_tokens=int(row["prompt_tokens"]),
                completion_tokens=int(row["completion_tokens"]),
                cost_usd=float(row["cost_usd"]),
                latency_seconds=float(row["latency_seconds"]),
                api_attempts=int(row["api_attempts"]),
                retried_responses=int(row["retried_responses"]),
            )
            for row in payload["usage"]
        )
        transitions = payload["group_transitions"]
        debate = payload["debate"]
        bootstrap = debate["bootstrap"]
        mcnemar = debate["mcnemar"]
        questions = payload["questions"]
    except ViewerDataError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ViewerDataError(f"evaluation export is malformed: {path}") from error

    if {row.round for row in group_accuracy} != {1, 2} or len(group_accuracy) != 2:
        raise ViewerDataError("evaluation export must contain one group result per round")
    if len(agent_accuracy) != 10:
        raise ViewerDataError("evaluation export must contain five agent results per round")
    if {row.round for row in usage} != {1, 2} or len(usage) != 2:
        raise ViewerDataError("evaluation export must contain one usage result per round")

    corrected_from_undecided = sum(
        row.get("round1_state") in {"NO_CONSENSUS", "INSUFFICIENT_ANSWERS"}
        and row.get("round1_correct") is False
        and row.get("round2_correct") is True
        for row in questions
    )
    return ExperimentOverview(
        run_id=run_id,
        settings_version=str(run["settings_version"]),
        evaluation_version=str(provenance["evaluation_version"]),
        question_count=int(provenance["expected_questions"]),
        wall_clock_hours=float(provenance["wall_clock_hours"]),
        group_accuracy=group_accuracy,
        agent_accuracy=agent_accuracy,
        consensus_states=consensus_states,
        usage=usage,
        stayed_correct=int(transitions["stayed_correct"]),
        became_correct=int(transitions["became_correct"]),
        became_incorrect=int(transitions["became_incorrect"]),
        stayed_incorrect=int(transitions["stayed_incorrect"]),
        effect_points=float(debate["effect_points"]),
        confidence_low_points=float(bootstrap["low_points"]),
        confidence_high_points=float(bootstrap["high_points"]),
        mcnemar_p_value=float(mcnemar["p_value"]),
        corrected_from_undecided=corrected_from_undecided,
    )


# 5. Reconstructing one question


def load_question_replay(
    question_id: str,
    *,
    run_id: str = MAIN_RUN_ID,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    questions_path: str | Path = DEFAULT_QUESTIONS_PATH,
    evaluation_export_path: str | Path = DEFAULT_EVALUATION_EXPORT,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> QuestionReplay:
    """Load one real two-round debate without writing or contacting a model."""
    question_id = question_id.strip()
    if not question_id:
        raise ViewerDataError("question_id cannot be blank")

    question = _load_question(Path(questions_path), question_id)
    exported = _load_exported_question(
        Path(evaluation_export_path), run_id=run_id, question_id=question_id
    )
    if exported["correct_answer"] not in answer_letters(question):
        raise ViewerDataError(
            f"exported answer {exported['correct_answer']!r} is not one of "
            f"{question_id!r}'s options"
        )
    registry_file = Path(registry_path)
    registry = load_model_registry(registry_file)

    results_file = Path(results_path)
    cache_file = Path(cache_path)
    with _ReadOnlyResults(results_file) as results, closing(
        _open_read_only(cache_file)
    ) as cache_connection:
        run = _validated_run(results, run_id, registry_file)
        responses = results.read_responses(run_id, question_id=question_id)
        outcomes = [
            row
            for row in results.read_outcomes(run_id)
            if row["question_id"] == question_id
        ]
        _validate_question_rows(question_id, registry, responses, outcomes)
        _validate_export_matches_stored(exported, outcomes)

        response_by_round = {
            round_number: {
                row["agent_id"]: row
                for row in responses
                if row["round"] == round_number
            }
            for round_number in (1, 2)
        }
        outcome_by_round = {row["round"]: row for row in outcomes}

        valid_round1 = valid_round1_responses(
            results,
            run_id=run_id,
            question_id=question_id,
            registry=registry,
        )
        round1_messages = build_round1_messages(question)

        round1_views: list[AgentResponseView] = []
        round2_views: list[AgentResponseView] = []
        for agent_id, spec in registry.items():
            first = response_by_round[1][agent_id]
            _validate_response_settings(first, spec)
            round1_views.append(
                _response_view(
                    first,
                    spec=spec,
                    messages=round1_messages,
                    own_round1_response_id=None,
                    results=results,
                    cache_connection=cache_connection,
                )
            )

            own = valid_round1.get(agent_id)
            peers = peers_for(agent_id, valid_round1)
            second = response_by_round[2][agent_id]
            expected_peer_ids = tuple(peer.response_id for peer in peers)
            stored_peer_ids = tuple(int(value) for value in second["peer_response_ids"])
            if stored_peer_ids != expected_peer_ids:
                raise ViewerDataError(
                    f"Round 2 {agent_id!r} stored peers {stored_peer_ids}, but the "
                    f"reconstructed conversation requires {expected_peer_ids}"
                )
            _validate_response_settings(second, spec)
            round2_messages = build_round2_messages(
                question,
                own.text if own is not None else None,
                [peer.text for peer in peers],
            )
            round2_views.append(
                _response_view(
                    second,
                    spec=spec,
                    messages=round2_messages,
                    own_round1_response_id=(own.response_id if own is not None else None),
                    results=results,
                    cache_connection=cache_connection,
                )
            )

    return QuestionReplay(
        run_id=run_id,
        settings_version=str(run["settings_version"]),
        question_id=question_id,
        category=str(question["category"]),
        question=str(question["question"]),
        options=tuple(str(option) for option in question["options"]),
        correct_answer=str(exported["correct_answer"]),
        round1=RoundReplay(
            round=1,
            responses=tuple(round1_views),
            outcome=_outcome_view(outcome_by_round[1]),
        ),
        round2=RoundReplay(
            round=2,
            responses=tuple(round2_views),
            outcome=_outcome_view(outcome_by_round[2]),
        ),
    )


def _validated_run(
    results: _ReadOnlyResults, run_id: str, registry_path: Path
) -> dict[str, Any]:
    run = results.read_run(run_id)
    if run is None:
        raise ViewerDataError(f"run {run_id!r} is not in the results database")
    if run["ended_at"] is None:
        raise ViewerDataError(f"run {run_id!r} is unfinished and cannot be replayed")
    expected_settings = registry_path.stem
    if run["settings_version"] != expected_settings:
        raise ViewerDataError(
            f"run uses {run['settings_version']!r}, but the selected registry is "
            f"{expected_settings!r}"
        )
    return run


def _validate_question_rows(
    question_id: str,
    registry: Mapping[str, ModelSpec],
    responses: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> None:
    expected_agents = set(registry)
    if len(responses) != len(registry) * 2:
        raise ViewerDataError(
            f"{question_id!r} has {len(responses)} responses, expected {len(registry) * 2}"
        )
    for round_number in (1, 2):
        actual_agents = {
            str(row["agent_id"])
            for row in responses
            if row["round"] == round_number
        }
        if actual_agents != expected_agents:
            raise ViewerDataError(
                f"Round {round_number} agents do not match the registry: "
                f"expected {sorted(expected_agents)}, got {sorted(actual_agents)}"
            )
    if {row["round"] for row in outcomes} != {1, 2} or len(outcomes) != 2:
        raise ViewerDataError(
            f"{question_id!r} must have exactly one stored outcome for each round"
        )


def _validate_export_matches_stored(
    exported: Mapping[str, Any], outcomes: Sequence[Mapping[str, Any]]
) -> None:
    by_round = {row["round"]: row for row in outcomes}
    for round_number in (1, 2):
        stored = by_round[round_number]
        for suffix, stored_value in (
            ("state", stored["consensus_state"]),
            ("answer", stored["consensus_answer"]),
        ):
            exported_value = exported.get(f"round{round_number}_{suffix}")
            if exported_value != stored_value:
                raise ViewerDataError(
                    f"evaluation export Round {round_number} {suffix} "
                    f"{exported_value!r} does not match stored value {stored_value!r}"
                )

        expected_correct = stored["consensus_answer"] == exported["correct_answer"]
        if exported.get(f"round{round_number}_correct") is not expected_correct:
            raise ViewerDataError(
                f"evaluation export Round {round_number} correctness does not match "
                "its stored answer and exported key"
            )


def _validate_response_settings(row: Mapping[str, Any], spec: ModelSpec) -> None:
    expected = {
        "requested_slug": spec.slug,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
    }
    mismatches = [
        f"{field}: stored {row[field]!r}, registry {value!r}"
        for field, value in expected.items()
        if row[field] != value
    ]
    if mismatches:
        raise ViewerDataError(
            f"stored settings for {row['agent_id']!r} do not match the registry: "
            + "; ".join(mismatches)
        )


def _response_view(
    row: Mapping[str, Any],
    *,
    spec: ModelSpec,
    messages: list[dict[str, str]],
    own_round1_response_id: int | None,
    results: _ReadOnlyResults,
    cache_connection: sqlite3.Connection,
) -> AgentResponseView:
    key = cache_key(spec, messages)
    cached = cache_connection.execute(
        "SELECT agent_id, slug FROM cached_responses WHERE cache_key = ?", (key,)
    ).fetchone()
    if cached is not None and (
        cached["agent_id"] != spec.agent_id or cached["slug"] != spec.slug
    ):
        raise ViewerDataError(
            f"cache key {key} points to a different agent or model than {spec.agent_id!r}"
        )

    return AgentResponseView(
        response_id=int(row["response_id"]),
        round=int(row["round"]),
        agent_id=spec.agent_id,
        display_name=spec.display_name,
        developer=spec.developer,
        requested_slug=str(row["requested_slug"]),
        served_slug=str(row["served_slug"]),
        provider=str(row["provider"]),
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        max_tokens=int(row["max_tokens"]),
        pinned_provider=spec.pinned_provider,
        reasoning_max_tokens=spec.reasoning_max_tokens,
        raw_response=str(row["raw_response"]),
        status=str(row["status"]),
        extracted_letter=row["extracted_letter"],
        finish_reason=str(row["finish_reason"]),
        attempt_count=int(row["attempt_count"]),
        prompt_tokens=int(row["prompt_tokens"]),
        completion_tokens=int(row["completion_tokens"]),
        cost_usd=float(row["cost_usd"]),
        latency_seconds=float(row["latency_seconds"]),
        cache_hit=bool(row["cache_hit"]),
        own_round1_response_id=own_round1_response_id,
        peer_response_ids=tuple(int(value) for value in row["peer_response_ids"]),
        messages=tuple(MessageView(**message) for message in messages),
        request_cache_key=key,
        request_found_in_cache=cached is not None,
        attempts=tuple(
            AttemptView(
                attempt_number=int(attempt["attempt_number"]),
                outcome=str(attempt["outcome"]),
                status=str(attempt["status"]),
                http_status=attempt["http_status"],
                error=str(attempt["error"]),
                prompt_tokens=int(attempt["prompt_tokens"]),
                completion_tokens=int(attempt["completion_tokens"]),
                cost_usd=float(attempt["cost_usd"]),
                latency_seconds=float(attempt["latency_seconds"]),
            )
            for attempt in results.read_attempts(int(row["response_id"]))
        ),
    )


def _outcome_view(row: Mapping[str, Any]) -> RoundOutcomeView:
    return RoundOutcomeView(
        round=int(row["round"]),
        consensus_state=str(row["consensus_state"]),
        consensus_answer=row["consensus_answer"],
        decided=bool(row["decided"]),
        valid_answer_count=int(row["valid_answer_count"]),
        total_prompt_tokens=int(row["total_prompt_tokens"]),
        total_completion_tokens=int(row["total_completion_tokens"]),
        total_cost_usd=float(row["total_cost_usd"]),
        total_latency_seconds=float(row["total_latency_seconds"]),
    )
