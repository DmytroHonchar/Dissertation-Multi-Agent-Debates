"""Turns a finished run into the numbers the dissertation reports.

This is the only module allowed to open an answer key. Nothing here calls a
model, spends money, changes a stored row, or recomputes a vote - the vote was
decided by `voting.py` when the run happened and is read back as it was stored.

The three headline measures (D009), never collapsed into one figure:

  1. per-agent Round 1 accuracy, over that agent's valid answers only
  2. Round 1 group vote accuracy - aggregation, no communication
  3. Round 2 group vote accuracy - after one round of communication

(2) - (1) is the aggregation gain and is reported against both the mean and the
best agent, because those two answer different questions (D021). (3) - (2) is
the debate effect and is the answer to the research question.

Two scoring rules that are easy to get wrong and are fixed by D010:

  - an undecided question counts as incorrect for the group, so both rounds
    share a denominator of every question in the run and stay comparable
  - a failed response is never a wrong answer; agent accuracy is over valid
    answers only, and the failure counts are reported next to it
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, comb
from pathlib import Path
from typing import Any

from mad.database import ResultsDatabase
from mad.parser_v1 import (
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)
from mad.voting import (
    EXPECTED_AGENT_COUNT,
    STATE_CONSENSUS,
    STATE_INSUFFICIENT_ANSWERS,
    STATE_NO_CONSENSUS,
    STATE_UNANIMOUS,
)


# 1. Settings

EVALUATION_VERSION = "evaluation_v1"

# The date D011 was recorded. Fixed before the formal pilot and the main
# experiment, which is what D011's rule protects: a seed that is a date was
# not chosen by looking at a result.
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

# The OpenRouter key limit that stands in for the £15 budget. Stored costs are
# in US dollars, so the budget is tracked in the same currency as the data.
BUDGET_USD = 20.0

ROUNDS = (1, 2)
FAILURE_STATUSES = (STATUS_REFUSAL, STATUS_TRUNCATED, STATUS_PARSE_FAIL, STATUS_API_ERROR)
CONSENSUS_STATES = (
    STATE_UNANIMOUS,
    STATE_CONSENSUS,
    STATE_NO_CONSENSUS,
    STATE_INSUFFICIENT_ANSWERS,
)


class EvaluationError(RuntimeError):
    """The run or the answer key is not fit to be scored."""


# 2. What comes back


@dataclass(frozen=True)
class AgentRoundResult:
    """One agent's accuracy and failures in one round.

    `accuracy` is None when the agent produced no valid answer at all. Zero
    would be a lie: it did not answer wrongly, it did not answer.
    """

    agent_id: str
    round: int
    responses: int
    valid_answers: int
    correct: int
    accuracy: float | None
    failures: Mapping[str, int]

    @property
    def failure_count(self) -> int:
        return sum(self.failures.values())

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.responses if self.responses else 0.0


@dataclass(frozen=True)
class GroupRoundResult:
    """The group's accuracy in one round, over every question in the run."""

    round: int
    questions: int
    correct: int
    decided: int
    accuracy: float
    consensus_states: Mapping[str, int]

    @property
    def undecided(self) -> int:
        return self.questions - self.decided


@dataclass(frozen=True)
class TransitionCounts:
    """The four-way move between rounds. The off-diagonals are the substance."""

    stayed_correct: int = 0
    became_incorrect: int = 0
    became_correct: int = 0
    stayed_incorrect: int = 0

    @property
    def total(self) -> int:
        return (
            self.stayed_correct
            + self.became_incorrect
            + self.became_correct
            + self.stayed_incorrect
        )

    @property
    def discordant(self) -> int:
        """The questions that moved. McNemar looks only at these."""
        return self.became_correct + self.became_incorrect


@dataclass(frozen=True)
class AgentTransitions:
    """One agent's transition table, over questions it answered validly twice.

    Questions where either round failed are excluded and counted, because a
    failure is not a position and cannot have moved.
    """

    agent_id: str
    counts: TransitionCounts
    excluded_questions: int


@dataclass(frozen=True)
class AggregationGain:
    """What voting alone bought, against two different baselines (D021)."""

    group_accuracy: float
    mean_agent_accuracy: float | None
    best_agent_accuracy: float | None
    best_agent_id: str | None
    versus_mean_points: float | None
    versus_best_points: float | None


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval for the debate effect, in percentage points."""

    difference_points: float
    low_points: float
    high_points: float
    resamples: int
    seed: int
    confidence: float

    @property
    def includes_zero(self) -> bool:
        """If zero is inside, the change cannot be claimed as an improvement."""
        return self.low_points <= 0.0 <= self.high_points


@dataclass(frozen=True)
class McNemarResult:
    """The exact test on the questions that moved between rounds."""

    became_correct: int
    became_incorrect: int
    p_value: float

    @property
    def discordant_pairs(self) -> int:
        return self.became_correct + self.became_incorrect


@dataclass(frozen=True)
class UsageTotals:
    """What one round cost in tokens, money, time and calls.

    `api_attempts` counts only attempts this run actually made. A cache hit
    carries `attempt_count = 1` naming the original paid call that produced the
    stored body, so counting it here would report calls that never happened -
    the run that served all five Round 1 responses from cache made zero.
    """

    round: int
    responses: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    cache_hits: int
    api_attempts: int
    retried_responses: int


@dataclass(frozen=True)
class BudgetUse:
    """Spend against the key limit that stands in for the £15 budget.

    Two different numbers, because subtracting one run from the whole budget
    would report a balance the key does not have. `run_usd` is what this run
    cost; `recorded_usd` is every run in this database, which is what has
    actually been spent, and is what `remaining_usd` is measured from.

    It is still a floor, not a bank balance: a run stored in another database,
    or a call whose row never landed, is money spent that no row here knows
    about.
    """

    run_usd: float
    recorded_usd: float
    budget_usd: float

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.recorded_usd

    @property
    def fraction_used(self) -> float:
        return self.recorded_usd / self.budget_usd if self.budget_usd else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    """Every number, computed once, so a printer or a table cannot disagree."""

    run_id: str
    evaluation_version: str
    question_count: int
    agent_ids: tuple[str, ...]

    agent_results: Mapping[int, tuple[AgentRoundResult, ...]]
    group_results: Mapping[int, GroupRoundResult]

    group_transitions: TransitionCounts
    agent_transitions: tuple[AgentTransitions, ...]

    aggregation: AggregationGain
    debate_effect_points: float
    bootstrap: BootstrapInterval
    mcnemar: McNemarResult

    usage: Mapping[int, UsageTotals]
    budget: BudgetUse


# 3. The answer key


def load_answer_key(path: str | Path) -> dict[str, str]:
    """Read the frozen answer key. The only file read here that holds answers."""
    key: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as key_file:
        for line_number, line in enumerate(key_file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                stable_id, answer = record["stable_id"], record["correct_answer"]
            except KeyError as missing:
                raise EvaluationError(
                    f"{path} line {line_number} has no {missing} field"
                ) from missing
            if stable_id in key:
                raise EvaluationError(f"{path} lists {stable_id} twice")
            key[stable_id] = answer
    if not key:
        raise EvaluationError(f"{path} is empty")
    return key


# 4. Scoring one run


def evaluate_run(
    db: ResultsDatabase,
    run_id: str,
    answer_key: Mapping[str, str],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    budget_usd: float = BUDGET_USD,
    expected_questions: int | None = None,
) -> EvaluationReport:
    """Score one finished run against the answer key.

    Refuses before computing anything if the run is unfinished, missing a round,
    scoring different questions in each round, short of a response from any
    agent, or carrying a question the key does not cover. The last check is what
    stops a pilot run being scored against the experimental key: the two frozen
    sets do not overlap, so every ID would be missing.

    Pass `expected_questions` for a formal run - 20 for the pilot, 300 for the
    main experiment - and a run that lost questions is refused rather than
    scored over a smaller denominator, which would quietly change every figure.
    """
    run = db.read_run(run_id)
    if run is None:
        raise EvaluationError(f"run {run_id!r} is not in this database")
    if run["ended_at"] is None:
        raise EvaluationError(
            f"run {run_id!r} never finished. An incomplete run is not scored, "
            "because its missing rows would silently lower every accuracy."
        )

    outcomes = _outcomes_by_round(db, run_id)
    responses = _responses_by_round(db, run_id)
    question_ids = _question_ids(outcomes)

    missing = [qid for qid in question_ids if qid not in answer_key]
    if missing:
        raise EvaluationError(
            f"the answer key does not cover {len(missing)} of the run's "
            f"{len(question_ids)} questions, starting with {missing[0]!r}. "
            "This is what a pilot run scored against the experimental key looks like."
        )

    if expected_questions is not None and len(question_ids) != expected_questions:
        raise EvaluationError(
            f"run {run_id!r} scored {len(question_ids)} questions, not the "
            f"{expected_questions} expected. Scoring a short run would divide by "
            "the wrong denominator and change every number in the report."
        )

    agent_ids = tuple(sorted({row["agent_id"] for row in responses[1]}))
    if not agent_ids:
        raise EvaluationError(f"run {run_id!r} stored no Round 1 responses")
    _assert_every_agent_answered(run_id, question_ids, responses, agent_ids)

    agent_results = {
        round_number: tuple(
            _agent_round_result(agent_id, round_number, responses[round_number], answer_key)
            for agent_id in agent_ids
        )
        for round_number in ROUNDS
    }
    group_results = {
        round_number: _group_round_result(round_number, outcomes[round_number], answer_key)
        for round_number in ROUNDS
    }

    group_correct = {
        round_number: _group_correct_by_question(outcomes[round_number], answer_key)
        for round_number in ROUNDS
    }
    group_transitions = _transitions(
        [(group_correct[1][qid], group_correct[2][qid]) for qid in question_ids]
    )

    debate_effect = (group_results[2].accuracy - group_results[1].accuracy) * 100.0

    return EvaluationReport(
        run_id=run_id,
        evaluation_version=EVALUATION_VERSION,
        question_count=len(question_ids),
        agent_ids=agent_ids,
        agent_results=agent_results,
        group_results=group_results,
        group_transitions=group_transitions,
        agent_transitions=tuple(
            _agent_transitions(agent_id, question_ids, responses, answer_key)
            for agent_id in agent_ids
        ),
        aggregation=_aggregation_gain(group_results[1], agent_results[1]),
        debate_effect_points=debate_effect,
        bootstrap=bootstrap_difference(
            [group_correct[1][qid] for qid in question_ids],
            [group_correct[2][qid] for qid in question_ids],
            seed=seed,
            resamples=resamples,
        ),
        mcnemar=mcnemar_exact(
            became_correct=group_transitions.became_correct,
            became_incorrect=group_transitions.became_incorrect,
        ),
        usage={
            round_number: _usage(round_number, outcomes[round_number], responses[round_number])
            for round_number in ROUNDS
        },
        budget=BudgetUse(
            run_usd=sum(
                float(row["total_cost_usd"])
                for round_number in ROUNDS
                for row in outcomes[round_number]
            ),
            recorded_usd=recorded_spend(db),
            budget_usd=budget_usd,
        ),
    )


# 5. Statistics


def bootstrap_difference(
    round1: Sequence[bool],
    round2: Sequence[bool],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
) -> BootstrapInterval:
    """A paired question-level bootstrap interval for Round 2 minus Round 1.

    Whole questions are resampled, so each question's two outcomes stay together
    (D011). Resampling the rounds independently would break the pairing and give
    a wider, wrong interval.

    Nothing is recalculated in the experiment's favour. The measured difference
    never changes; this only shows how much it would have moved had a different
    set of questions been frozen.
    """
    if len(round1) != len(round2):
        raise EvaluationError(
            f"paired data must be the same length, got {len(round1)} and {len(round2)}"
        )
    questions = len(round1)
    if questions == 0:
        raise EvaluationError("cannot bootstrap an empty run")
    if resamples < 1:
        raise EvaluationError(f"resamples must be at least 1, got {resamples}")

    measured = (sum(round2) - sum(round1)) / questions * 100.0

    rng = random.Random(seed)
    indices = range(questions)
    differences = []
    for _ in range(resamples):
        picked = [rng.randrange(questions) for _ in indices]
        first = sum(round1[i] for i in picked)
        second = sum(round2[i] for i in picked)
        differences.append((second - first) / questions * 100.0)
    differences.sort()

    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        difference_points=measured,
        low_points=_percentile(differences, tail),
        high_points=_percentile(differences, 1.0 - tail),
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def mcnemar_exact(*, became_correct: int, became_incorrect: int) -> McNemarResult:
    """McNemar's exact test: a two-sided binomial test on the moved questions.

    Only the discordant pairs carry information. Under the null the group is as
    likely to have been repaired as broken, so this is Binomial(n, 0.5), which
    is symmetric and exact from `math.comb` - no `scipy` needed (D021).

    With no discordant pairs there is nothing to test and p = 1, reported rather
    than raised, because a run where debate changed no group answer is a real
    and interesting result.
    """
    if became_correct < 0 or became_incorrect < 0:
        raise EvaluationError("transition counts cannot be negative")

    pairs = became_correct + became_incorrect
    if pairs == 0:
        p_value = 1.0
    else:
        smaller = min(became_correct, became_incorrect)
        tail = sum(comb(pairs, i) for i in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2**pairs))

    return McNemarResult(
        became_correct=became_correct,
        became_incorrect=became_incorrect,
        p_value=p_value,
    )


# 6. Helpers


def _outcomes_by_round(db: ResultsDatabase, run_id: str) -> dict[int, list[dict[str, Any]]]:
    """Both rounds' outcome rows, refusing a run that is missing one."""
    outcomes = {r: db.read_outcomes(run_id, round=r) for r in ROUNDS}
    for round_number, rows in outcomes.items():
        if not rows:
            raise EvaluationError(
                f"run {run_id!r} has no Round {round_number} outcomes. "
                "Both rounds are needed; the comparison is the whole point."
            )
    first, second = ({row["question_id"] for row in outcomes[r]} for r in ROUNDS)
    if first != second:
        raise EvaluationError(
            f"run {run_id!r} scored different questions in each round "
            f"({len(first)} and {len(second)}); they would not be comparable."
        )
    return outcomes


def _responses_by_round(db: ResultsDatabase, run_id: str) -> dict[int, list[dict[str, Any]]]:
    return {r: db.read_responses(run_id, round=r) for r in ROUNDS}


def _question_ids(outcomes: Mapping[int, Sequence[Mapping[str, Any]]]) -> tuple[str, ...]:
    """Every question in the run, in a fixed order so the bootstrap repeats."""
    return tuple(sorted({row["question_id"] for row in outcomes[1]}))


def _agent_round_result(
    agent_id: str,
    round_number: int,
    rows: Sequence[Mapping[str, Any]],
    answer_key: Mapping[str, str],
) -> AgentRoundResult:
    mine = [row for row in rows if row["agent_id"] == agent_id]
    valid = [row for row in mine if row["status"] == STATUS_OK]
    correct = sum(1 for row in valid if row["extracted_letter"] == answer_key[row["question_id"]])
    failures = {
        status: sum(1 for row in mine if row["status"] == status)
        for status in FAILURE_STATUSES
    }
    return AgentRoundResult(
        agent_id=agent_id,
        round=round_number,
        responses=len(mine),
        valid_answers=len(valid),
        correct=correct,
        accuracy=correct / len(valid) if valid else None,
        failures=failures,
    )


def _group_correct_by_question(
    rows: Sequence[Mapping[str, Any]], answer_key: Mapping[str, str]
) -> dict[str, bool]:
    """One true/false per question. Undecided is false, never dropped (D010)."""
    return {
        row["question_id"]: bool(row["decided"])
        and row["consensus_answer"] == answer_key[row["question_id"]]
        for row in rows
    }


def _group_round_result(
    round_number: int, rows: Sequence[Mapping[str, Any]], answer_key: Mapping[str, str]
) -> GroupRoundResult:
    correct_by_question = _group_correct_by_question(rows, answer_key)
    states = {
        state: sum(1 for row in rows if row["consensus_state"] == state)
        for state in CONSENSUS_STATES
    }
    return GroupRoundResult(
        round=round_number,
        questions=len(rows),
        correct=sum(correct_by_question.values()),
        decided=sum(1 for row in rows if row["decided"]),
        # Denominator is every question, so the two rounds stay comparable.
        accuracy=sum(correct_by_question.values()) / len(rows) if rows else 0.0,
        consensus_states=states,
    )


def _transitions(pairs: Sequence[tuple[bool, bool]]) -> TransitionCounts:
    counts = {"cc": 0, "ci": 0, "ic": 0, "ii": 0}
    for before, after in pairs:
        counts["cc" if before and after else "ci" if before else "ic" if after else "ii"] += 1
    return TransitionCounts(
        stayed_correct=counts["cc"],
        became_incorrect=counts["ci"],
        became_correct=counts["ic"],
        stayed_incorrect=counts["ii"],
    )


def _agent_transitions(
    agent_id: str,
    question_ids: Sequence[str],
    responses: Mapping[int, Sequence[Mapping[str, Any]]],
    answer_key: Mapping[str, str],
) -> AgentTransitions:
    """One agent's moves, over the questions it answered validly in both rounds."""
    by_round = {
        round_number: {
            row["question_id"]: row
            for row in responses[round_number]
            if row["agent_id"] == agent_id and row["status"] == STATUS_OK
        }
        for round_number in ROUNDS
    }

    pairs = []
    excluded = 0
    for question_id in question_ids:
        first, second = by_round[1].get(question_id), by_round[2].get(question_id)
        if first is None or second is None:
            # A failure in either round is an absence, not a move.
            excluded += 1
            continue
        answer = answer_key[question_id]
        pairs.append((first["extracted_letter"] == answer, second["extracted_letter"] == answer))

    return AgentTransitions(
        agent_id=agent_id,
        counts=_transitions(pairs),
        excluded_questions=excluded,
    )


def _aggregation_gain(
    group: GroupRoundResult, agents: Sequence[AgentRoundResult]
) -> AggregationGain:
    """What the Round 1 vote added over its members, on two baselines (D021)."""
    scored = [agent for agent in agents if agent.accuracy is not None]
    if not scored:
        return AggregationGain(
            group_accuracy=group.accuracy,
            mean_agent_accuracy=None,
            best_agent_accuracy=None,
            best_agent_id=None,
            versus_mean_points=None,
            versus_best_points=None,
        )

    mean_accuracy = sum(agent.accuracy for agent in scored) / len(scored)
    # Ties break on agent_id, so the named best agent is stable across runs.
    best = min(scored, key=lambda agent: (-agent.accuracy, agent.agent_id))
    return AggregationGain(
        group_accuracy=group.accuracy,
        mean_agent_accuracy=mean_accuracy,
        best_agent_accuracy=best.accuracy,
        best_agent_id=best.agent_id,
        versus_mean_points=(group.accuracy - mean_accuracy) * 100.0,
        versus_best_points=(group.accuracy - best.accuracy) * 100.0,
    )


def _usage(
    round_number: int,
    outcomes: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> UsageTotals:
    paid = [row for row in responses if not row["cache_hit"]]
    return UsageTotals(
        round=round_number,
        responses=len(responses),
        prompt_tokens=sum(int(row["total_prompt_tokens"]) for row in outcomes),
        completion_tokens=sum(int(row["total_completion_tokens"]) for row in outcomes),
        cost_usd=sum(float(row["total_cost_usd"]) for row in outcomes),
        latency_seconds=sum(float(row["total_latency_seconds"]) for row in outcomes),
        cache_hits=len(responses) - len(paid),
        # Only calls this run made. A cache hit made none.
        api_attempts=sum(int(row["attempt_count"]) for row in paid),
        retried_responses=sum(1 for row in paid if int(row["attempt_count"]) > 1),
    )


def recorded_spend(db: ResultsDatabase) -> float:
    """Every dollar this database has a row for, across all runs.

    A floor on real spending, not a bank balance: a run in another database or
    a call whose row never landed is money spent that this cannot see.
    """
    return sum(
        float(row["total_cost_usd"])
        for run in db.read_runs()
        for row in db.read_outcomes(run["run_id"])
    )


def _assert_every_agent_answered(
    run_id: str,
    question_ids: Sequence[str],
    responses: Mapping[int, Sequence[Mapping[str, Any]]],
    agent_ids: Sequence[str],
) -> None:
    """Every agent must have a row for every question in both rounds.

    A missing row is not a failure - a failure is stored as `API_ERROR` and
    counted. A missing row is a run that did not finish what it claims to have
    finished, and scoring it would understate the failure rate.
    """
    if len(agent_ids) != EXPECTED_AGENT_COUNT:
        raise EvaluationError(
            f"run {run_id!r} has {len(agent_ids)} agents, not {EXPECTED_AGENT_COUNT}: "
            f"{list(agent_ids)}. The three-of-five threshold assumes five."
        )

    # The same five in both rounds. An agent that only appears in Round 2 never
    # answered alone, so it has no Round 1 accuracy and nothing to reconsider.
    second_round_agents = {row["agent_id"] for row in responses[2]}
    if second_round_agents != set(agent_ids):
        only_first = sorted(set(agent_ids) - second_round_agents)
        only_second = sorted(second_round_agents - set(agent_ids))
        raise EvaluationError(
            f"run {run_id!r} debated a different group than it started with: "
            f"{only_first} answered only Round 1, {only_second} only Round 2. "
            "The two rounds must compare the same five agents."
        )

    expected_rows = len(question_ids) * len(agent_ids)
    for round_number in ROUNDS:
        rows = responses[round_number]
        stored = {(row["question_id"], row["agent_id"]) for row in rows}
        gaps = [
            (question_id, agent_id)
            for question_id in question_ids
            for agent_id in agent_ids
            if (question_id, agent_id) not in stored
        ]
        if gaps:
            question_id, agent_id = gaps[0]
            raise EvaluationError(
                f"run {run_id!r} is missing {len(gaps)} Round {round_number} "
                f"responses, starting with {agent_id} on {question_id}. "
                "A failed call is stored as API_ERROR; a missing row means the "
                "run did not complete."
            )
        # No gaps and the wrong count means rows for something else entirely -
        # an extra agent, or a question with no outcome row.
        if len(rows) != expected_rows:
            raise EvaluationError(
                f"run {run_id!r} stored {len(rows)} Round {round_number} responses "
                f"for {len(question_ids)} questions and {len(agent_ids)} agents, "
                f"where {expected_rows} were expected. Something is stored that "
                "the run does not account for."
            )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile on an already sorted list.

    Rank is ceil(fraction x n), one-based, so the 2.5th percentile of 10,000
    values is the 250th, at index 249. Using int() instead took index 250 and
    was one position out whenever fraction x n landed on a whole number, which
    for 10,000 resamples it always does.
    """
    if not sorted_values:
        raise EvaluationError("cannot take a percentile of nothing")
    index = ceil(fraction * len(sorted_values)) - 1
    return sorted_values[min(max(index, 0), len(sorted_values) - 1)]
