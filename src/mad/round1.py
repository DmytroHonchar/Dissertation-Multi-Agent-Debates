"""Runs one question through Round 1: five calls, parsed, stored, voted.

This is the piece that connects everything already built. It owns no rules of
its own - the prompt, parser, database and voting each enforce theirs.

Money protections, because live calls spend real credit:
  - the default is a dry run on labelled fixture replies, costing nothing
  - live mode needs two explicit flags, not one
  - only questions from the 20-question pilot file are accepted
  - exactly five calls per run, one question at a time
  - there is no cache yet, so repeating a live run pays again
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mad.api_client import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SECONDS,
    ApiRequestError,
    AttemptRecord,
    CompletionResult,
    ModelSpec,
)
from mad.database import (
    OutcomeRecord,
    ResultsDatabase,
    attempt_rows,
    response_from_completion,
    response_from_failed_call,
)
from mad.parser_v1 import PARSER_VERSION, ParsedResponse, api_error, parse_response
from mad.prompts_v1 import PROMPT_VERSION, answer_letters, build_round1_messages
from mad.voting import EXPECTED_AGENT_COUNT, VoteOutcome, tally


# 1. Versions

# P8: everything that shaped a run is named, so its results stay citable.
# v1 was the cacheless Milestone 1 runner; run milestone1_20260901T161852Z is
# recorded under it and that label stays true. v2 added the response cache
# path, so new runs must not claim the old recipe.
CONFIG_VERSION = "round1_config_v2"
QUESTION_SET_VERSION = "mmlu_pro_v1"
SETTINGS_VERSION = "agents_v1"    # the yaml holding temperature 0, top-p 1, 1024 tokens


@dataclass(frozen=True)
class Round1Config:
    """The full recipe of a Round 1 run, as P8 requires."""

    config_version: str = CONFIG_VERSION
    question_set_version: str = QUESTION_SET_VERSION
    prompt_version: str = PROMPT_VERSION
    settings_version: str = SETTINGS_VERSION
    parser_version: str = PARSER_VERSION
    cache_enabled: bool = False       # cache.py does not exist yet (P5)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    parallel_calls: bool = False      # sequential; parallel comes with the pilot


# 2. Paths and errors

FROZEN_ROOT = Path("data") / "frozen" / "mmlu_pro_v1"
PILOT_QUESTIONS_FILE = FROZEN_ROOT / "model_inputs" / "pilot_questions.jsonl"
EXPERIMENTAL_IDS_FILE = FROZEN_ROOT / "metadata" / "experimental_ids.json"

# The real results database. A dry run is refused this path, so fixture rows
# can never sit next to experimental ones.
PRODUCTION_DATABASE = Path("storage") / "results.sqlite"


class RunnerError(RuntimeError):
    """The run was set up wrongly and no call should be made."""


class SpendNotConfirmedError(RunnerError):
    """Live mode was requested without both confirmation flags."""


# 3. Money guards


def require_spend_confirmation(*, live: bool, spend_confirmed: bool) -> None:
    """Live calls need both flags. One is a request; two is a decision."""
    if live and not spend_confirmed:
        raise SpendNotConfirmedError(
            "--live also needs --yes-spend-real-money. "
            "Five calls will charge the OpenRouter account."
        )
    if spend_confirmed and not live:
        raise SpendNotConfirmedError(
            "--yes-spend-real-money without --live makes no sense. "
            "Drop it for a dry run, or pass both for a live one."
        )


def production_database_path() -> Path:
    """The real results database, resolved from the repository root.

    Absolute, so launching the script from another directory cannot quietly
    create a second storage/ somewhere else.
    """
    return _repository_root() / PRODUCTION_DATABASE


def ensure_safe_database_path(path: str | Path, *, live: bool) -> Path:
    """A dry run may never write into the real results database."""
    resolved = Path(path)
    if not live and resolved.resolve() == production_database_path().resolve():
        raise RunnerError(
            f"a dry run may not write to {PRODUCTION_DATABASE}. "
            "Fixture rows must never sit next to real results."
        )
    return resolved


# 4. Loading the one question


def load_pilot_question(stable_id: str) -> dict[str, Any]:
    """One question from the 20-question pilot file, by its stable ID.

    Experimental questions are refused by name: the 300-question set is only
    run once, at the end, on an explicit instruction.
    """
    stable_id = stable_id.strip()
    if not stable_id or "," in stable_id or " " in stable_id:
        raise RunnerError(
            f"one question at a time, got {stable_id!r}. "
            "Milestone 1 is a single pilot question."
        )

    root = _repository_root()
    with (root / PILOT_QUESTIONS_FILE).open(encoding="utf-8") as pilot_file:
        for line in pilot_file:
            question = json.loads(line)
            if question.get("stable_id") == stable_id:
                return question

    experimental_ids = set(json.loads((root / EXPERIMENTAL_IDS_FILE).read_text()))
    if stable_id in experimental_ids:
        raise RunnerError(
            f"{stable_id} is one of the 300 experimental questions. "
            "Those are run once, at the end, and never for testing."
        )
    raise RunnerError(f"{stable_id} is not in the pilot file. Pick one of its 20 IDs.")


# 5. The fixture client

# Labelled test fixture. Deterministic, free, and never mistakable for a model:
# provider and generation IDs say "fixture" on every row it produces.
FIXTURE_PROVIDER = "fixture"

# Deterministic plan: three agents agree on the first option, one picks the
# second, one refuses. Exercises an agreeing vote, a dissent and a failure.
FIXTURE_REFUSAL_TEXT = "I cannot answer this question."


class CompletionClient(Protocol):
    """What the runner needs from a client: complete() and close()."""

    def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> CompletionResult: ...
    def close(self) -> None: ...


class ResponseCacheLike(Protocol):
    """What the runner needs from a cache. Defined here to avoid an import cycle."""

    def lookup(self, spec: ModelSpec, messages: list[dict[str, str]]) -> CompletionResult | None: ...
    def store(self, spec: ModelSpec, messages: list[dict[str, str]], result: CompletionResult) -> None: ...


class FixtureClient:
    """Stands in for OpenRouter during dry runs. Free and deterministic."""

    def __init__(self, question: Mapping[str, Any]) -> None:
        self._letters = answer_letters(question)
        self._calls = 0

    def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> CompletionResult:
        agree, dissent = self._letters[0], self._letters[1]
        plan = [agree, agree, agree, dissent, None]
        letter = plan[self._calls % len(plan)]
        self._calls += 1

        if letter is None:
            text = FIXTURE_REFUSAL_TEXT
        else:
            text = f"REASONING: deterministic fixture reply.\nFINAL ANSWER: {letter}"

        return CompletionResult(
            text=text,
            agent_id=spec.agent_id,
            requested_slug=spec.slug,
            served_slug=spec.slug,
            provider=FIXTURE_PROVIDER,
            generation_id=f"fixture-{spec.agent_id}",
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_seconds=0.0,
            attempts=1,
            raw_response={"fixture": True},
            attempt_log=(
                AttemptRecord(
                    attempt=1,
                    outcome="ok",
                    latency_seconds=0.0,
                    status_code=200,
                    raw_response=text,
                    finish_reason="stop",
                ),
            ),
        )

    def close(self) -> None:
        pass


# 6. Running one question


@dataclass(frozen=True)
class AgentResult:
    """What one agent did with the question, for the report."""

    agent_id: str
    status: str
    letter: str | None
    provider: str
    finish_reason: str
    cost_usd: float
    attempts: int


@dataclass(frozen=True)
class RoundReport:
    """Everything the milestone needs to be inspected by hand."""

    run_id: str
    question_id: str
    agents: tuple[AgentResult, ...]
    outcome: VoteOutcome
    total_cost_usd: float


def run_round1_question(
    question: Mapping[str, Any],
    *,
    registry: Mapping[str, ModelSpec],
    client: CompletionClient,
    db: ResultsDatabase,
    run_id: str,
    config: Round1Config = Round1Config(),
    cache: ResponseCacheLike | None = None,
) -> RoundReport:
    """One question through Round 1: cache lookup, call on miss, parse, store, vote.

    Exactly one call per agent, five in total. One agent failing is stored as
    API_ERROR and the other four continue - a failure must never end the run.

    With a cache, the order per agent is: lookup, API call only on a miss,
    then the genuine reply is cached - refusals and unparseable replies too,
    but never an ApiRequestError, which is temporary and must stay retryable.
    """
    if len(registry) != EXPECTED_AGENT_COUNT:
        raise RunnerError(
            f"the registry must hold {EXPECTED_AGENT_COUNT} agents, got {len(registry)}. "
            "No call is made until that is right."
        )
    # The stored labels must describe the run that actually happened.
    if config.cache_enabled != (cache is not None):
        raise RunnerError(
            f"config says cache_enabled={config.cache_enabled} but a cache "
            f"{'was' if cache is not None else 'was not'} supplied. The labels must tell the truth."
        )
    if config.parallel_calls:
        raise RunnerError("parallel_calls=True, but Round 1 runs sequentially for now")

    question_id = str(question["stable_id"])
    letters = answer_letters(question)
    messages = build_round1_messages(question)

    db.start_run(
        run_id,
        config_name=config.config_version,
        question_set_version=config.question_set_version,
        prompt_version=config.prompt_version,
        settings_version=config.settings_version,
        parser_version=config.parser_version,
    )

    parsed_by_agent: dict[str, ParsedResponse] = {}
    agent_results: list[AgentResult] = []
    totals = {"prompt": 0, "completion": 0, "cost": 0.0, "latency": 0.0}

    for agent_id, spec in registry.items():
        result = cache.lookup(spec, messages) if cache is not None else None
        cache_hit = result is not None
        failure: ApiRequestError | None = None

        if not cache_hit:
            try:
                result = client.complete(spec, messages)
            except ApiRequestError as error:
                # This agent is done, the other four are not.
                failure = error

        if failure is not None:
            parsed = api_error(str(failure))
            record = response_from_failed_call(
                failure.attempt_log,
                run_id=run_id, question_id=question_id, round=1,
                spec=spec, prompt_version=config.prompt_version,
            )
            rows = attempt_rows(failure.attempt_log)
        else:
            parsed = parse_response(result.text, letters, result.finish_reason)
            record = response_from_completion(
                result, parsed,
                run_id=run_id, question_id=question_id, round=1,
                spec=spec, prompt_version=config.prompt_version,
                selected_attempt=max(result.attempts, 1),
                cache_hit=cache_hit,
            )
            # The winning attempt gets its parse; failed earlier ones get none.
            # A cache hit stores no attempt rows at all - this run made no API
            # attempt. Its attempt_count of 1 names the original call that
            # produced the cached body, and cache_hit=1 marks the difference.
            rows = attempt_rows(
                result.attempt_log,
                parsed_by_attempt={max(result.attempts, 1): parsed},
                cache_hit=cache_hit,
            )

        db.record_response(record, rows)
        # Cached only after the paid call is on the audit record. The other way
        # round, a crash between the two would leave a cached reply whose cost
        # never reached the results database.
        if cache is not None and not cache_hit and failure is None:
            # A genuine outcome, even a refusal, is worth keeping. A failure is
            # not cached - it might work next time.
            cache.store(spec, messages, result)
        parsed_by_agent[agent_id] = parsed
        agent_results.append(
            AgentResult(
                agent_id=agent_id,
                status=parsed.status,
                letter=parsed.letter,
                provider=record.provider,
                finish_reason=record.finish_reason,
                cost_usd=record.cost_usd,
                attempts=record.attempt_count,
            )
        )
        totals["prompt"] += record.prompt_tokens
        totals["completion"] += record.completion_tokens
        totals["cost"] += record.cost_usd
        totals["latency"] += record.latency_seconds

    outcome = tally(parsed_by_agent, expected_agents=registry.keys())

    db.record_outcome(
        OutcomeRecord(
            run_id=run_id,
            question_id=question_id,
            round=1,
            consensus_state=outcome.state,
            consensus_answer=outcome.consensus_answer,
            decided=outcome.decided,
            valid_answer_count=outcome.valid_answer_count,
            total_prompt_tokens=totals["prompt"],
            total_completion_tokens=totals["completion"],
            total_cost_usd=totals["cost"],
            total_latency_seconds=totals["latency"],
        )
    )
    db.finish_run(run_id)

    return RoundReport(
        run_id=run_id,
        question_id=question_id,
        agents=tuple(agent_results),
        outcome=outcome,
        total_cost_usd=totals["cost"],
    )


# 7. Helpers


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
