"""The machinery both rounds share: run guards, money guards, fixtures, reports.

Round 1 and Round 2 are two stages of one experiment, and they need the same
things around them - the same refusal to spend money by accident, the same
labelled fixture client, the same shaped report, and the same check that the
database run they are writing into is the run they think it is.

That check exists because the caller, not the round, owns the run lifecycle. A
script opens one run, processes 1, 20 or 300 questions through both rounds, and
finishes it only after everything succeeded. The freedom to label a run one way
and then execute another is the price of that, so it is taken away here.

Nothing in this file knows which round is calling it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mad.api_client import ApiRequestError, AttemptRecord, CompletionResult, ModelSpec
from mad.database import (
    OutcomeRecord,
    ResponseRecord,
    ResultsDatabase,
    attempt_rows,
    response_from_completion,
    response_from_failed_call,
)
from mad.parser_v1 import ParsedResponse, api_error, parse_response
from mad.prompts_v1 import answer_letters
from mad.voting import VoteOutcome


# 1. Versions shared by every round

QUESTION_SET_VERSION = "mmlu_pro_v1"
SETTINGS_VERSION = "agents_v1"    # the yaml holding temperature 0, top-p 1, 1024 tokens


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


def require_spend_confirmation(
    *, live: bool, spend_confirmed: bool, maximum_attempts: int = 10
) -> None:
    """Live calls need both flags and state the worst case that can be charged.

    Attempts, not responses. A temporary failure is retried once (D012) and both
    attempts are billable, so a command that makes five responses can be charged
    for ten. Quoting the smaller number would understate what the flag confirms.
    """
    if live and not spend_confirmed:
        raise SpendNotConfirmedError(
            "--live also needs --yes-spend-real-money. "
            f"Up to {maximum_attempts} paid attempts will charge the OpenRouter account."
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


# 4. The run guard


class RunConfig(Protocol):
    """The five labels a run row records, as any round's config carries them."""

    config_version: str
    question_set_version: str
    prompt_version: str
    settings_version: str
    parser_version: str


def assert_run_is_open(db: ResultsDatabase, run_id: str, config: RunConfig) -> None:
    """The run exists, is unfinished, and is labelled with this exact recipe.

    Called before the first API call of every round, not after. Waiting for the
    foreign key on the first stored response would fail only once that response
    had already been paid for.

    The label check is the other half of moving the lifecycle to the caller. A
    script that opens a run as agents_v1 and then processes questions with an
    agents_v5 configuration would leave a run row describing an experiment that
    never happened, and nothing downstream could tell.
    """
    run = db.read_run(run_id)
    if run is None:
        raise RunnerError(
            f"run {run_id!r} has not been started. The caller must start it "
            "before processing questions."
        )
    if run["ended_at"] is not None:
        raise RunnerError(
            f"run {run_id!r} is already finished. Finished runs cannot accept "
            "more questions."
        )

    expected_labels = {
        "config_name": config.config_version,
        "question_set_version": config.question_set_version,
        "settings_version": config.settings_version,
        "parser_version": config.parser_version,
    }
    mismatches = [
        f"{field}: stored {run[field]!r}, config {expected!r}"
        for field, expected in expected_labels.items()
        if run[field] != expected
    ]

    # A two-round run records both prompts as "round1_v1+round2_v1", so each
    # round asks whether its own prompt is one component of that label.
    if config.prompt_version not in str(run["prompt_version"]).split("+"):
        mismatches.append(
            f"prompt_version: stored {run['prompt_version']!r}, "
            f"this round requires {config.prompt_version!r}"
        )

    if mismatches:
        raise RunnerError(
            f"run {run_id!r} labels do not match the configuration: "
            + "; ".join(mismatches)
        )


# 5. Loading the one question


# The pilot is exactly this many questions. A run that holds fewer is not the
# pilot, and evaluation refuses to score it as one.
PILOT_QUESTION_COUNT = 20


def load_pilot_questions() -> list[dict[str, Any]]:
    """All 20 frozen pilot questions, in file order.

    File order, not a fresh shuffle: the pilot is meant to be repeatable, and a
    reordering would change every Round 2 conversation through the cache keys.
    """
    root = _repository_root()
    with (root / PILOT_QUESTIONS_FILE).open(encoding="utf-8") as pilot_file:
        questions = [json.loads(line) for line in pilot_file if line.strip()]

    if len(questions) != PILOT_QUESTION_COUNT:
        raise RunnerError(
            f"the pilot file holds {len(questions)} questions, not "
            f"{PILOT_QUESTION_COUNT}. The frozen set must not have changed."
        )
    return questions


def load_pilot_question(stable_id: str) -> dict[str, Any]:
    """One question from the 20-question pilot file, by its stable ID.

    Experimental questions are refused by name: the 300-question set is only
    run once, at the end, on an explicit instruction.
    """
    stable_id = stable_id.strip()
    if not stable_id or "," in stable_id or " " in stable_id:
        raise RunnerError(
            f"one question at a time, got {stable_id!r}. "
            "This command runs a single pilot question."
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


# 6. The fixture client

# Labelled test fixture. Deterministic, free, and never mistakable for a model:
# provider and generation IDs say "fixture" on every row it produces.
FIXTURE_PROVIDER = "fixture"

# Deterministic plan: three agents agree on the first option, one picks the
# second, one refuses. Exercises an agreeing vote, a dissent and a failure.
FIXTURE_REFUSAL_TEXT = "I cannot answer this question."


class CompletionClient(Protocol):
    """What a round needs from a client: complete() and close()."""

    def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> CompletionResult: ...
    def close(self) -> None: ...


class ResponseCacheLike(Protocol):
    """What a round needs from a cache. Defined here to avoid an import cycle."""

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


# 7. What a round reports back


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
    # Round 1 always sees none. Round 2 sees four, or fewer when a peer failed.
    peer_count: int = 0


@dataclass(frozen=True)
class RoundReport:
    """Everything one round needs for terminal output or later orchestration."""

    run_id: str
    question_id: str
    round: int
    agents: tuple[AgentResult, ...]
    outcome: VoteOutcome
    total_cost_usd: float


@dataclass
class RoundTotals:
    """What one question cost, summed across its agents as they run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0

    def add(self, record: ResponseRecord) -> None:
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.cost_usd += record.cost_usd
        self.latency_seconds += record.latency_seconds


# 8. One agent, one call


def run_one_agent(
    *,
    spec: ModelSpec,
    messages: list[dict[str, str]],
    letters: Sequence[str],
    client: CompletionClient,
    db: ResultsDatabase,
    cache: ResponseCacheLike | None,
    run_id: str,
    question_id: str,
    round: int,
    prompt_version: str,
    peer_response_ids: Sequence[int] = (),
) -> tuple[ParsedResponse, AgentResult, ResponseRecord]:
    """Look up, call on a miss, parse, store, cache. Identical in both rounds.

    Only the messages differ between Round 1 and Round 2, so only the messages
    are passed in. Writing this twice would let the two rounds drift apart in
    how they retry, what they store or what they cache, and the results would
    then be measuring that difference instead of the debate.

    An ``ApiRequestError`` is caught here and returned as an ``API_ERROR`` row.
    This agent is done; the other four are not.
    """
    result = cache.lookup(spec, messages) if cache is not None else None
    cache_hit = result is not None
    failure: ApiRequestError | None = None

    if not cache_hit:
        try:
            result = client.complete(spec, messages)
        except ApiRequestError as error:
            failure = error

    if failure is not None:
        parsed = api_error(str(failure))
        record = response_from_failed_call(
            failure.attempt_log,
            run_id=run_id, question_id=question_id, round=round,
            spec=spec, prompt_version=prompt_version,
            peer_response_ids=peer_response_ids,
        )
        rows = attempt_rows(failure.attempt_log)
    else:
        parsed = parse_response(result.text, letters, result.finish_reason)
        record = response_from_completion(
            result, parsed,
            run_id=run_id, question_id=question_id, round=round,
            spec=spec, prompt_version=prompt_version,
            selected_attempt=max(result.attempts, 1),
            cache_hit=cache_hit,
            peer_response_ids=peer_response_ids,
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

    agent_result = AgentResult(
        agent_id=spec.agent_id,
        status=parsed.status,
        letter=parsed.letter,
        provider=record.provider,
        finish_reason=record.finish_reason,
        cost_usd=record.cost_usd,
        attempts=record.attempt_count,
        peer_count=len(peer_response_ids),
    )
    return parsed, agent_result, record


def record_round_outcome(
    db: ResultsDatabase,
    *,
    run_id: str,
    question_id: str,
    round: int,
    outcome: VoteOutcome,
    totals: RoundTotals,
) -> None:
    """Store where the group landed after this round. One row per round."""
    db.record_outcome(
        OutcomeRecord(
            run_id=run_id,
            question_id=question_id,
            round=round,
            consensus_state=outcome.state,
            consensus_answer=outcome.consensus_answer,
            decided=outcome.decided,
            valid_answer_count=outcome.valid_answer_count,
            total_prompt_tokens=totals.prompt_tokens,
            total_completion_tokens=totals.completion_tokens,
            total_cost_usd=totals.cost_usd,
            total_latency_seconds=totals.latency_seconds,
        )
    )


# 9. Helpers


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
