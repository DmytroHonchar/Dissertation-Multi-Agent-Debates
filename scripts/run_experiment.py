"""Run the frozen 300-question main experiment through both debate rounds.

Dry validation (free fixture replies and a temporary database):

    .venv/bin/python scripts/run_experiment.py

Live experiment (3,000 responses; at most 6,000 attempts with retries):

    .venv/bin/python scripts/run_experiment.py \
        --live --yes-spend-real-money

This command is locked to the frozen ``agents_v7`` configuration. It stores
responses but never reads an answer key or prints accuracy. Score the completed
run later, in a separate process, with ``expected_questions=300``.

The runner is sequential, matching the accepted pilot. If the process stops
between complete questions, pass ``--resume RUN_ID`` to continue the same run;
completed questions are preserved and skipped. A stop in the middle of a
question is refused rather than deleting or overwriting its audit rows.
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.api_client import (  # noqa: E402
    ApiConfigurationError,
    DEFAULT_MAX_ATTEMPTS,
    ModelSpec,
    OpenRouterClient,
    load_env_file,
    load_model_registry,
)
from mad.cache import ResponseCache  # noqa: E402
from mad.database import ResultsDatabase  # noqa: E402
from mad.debate import (  # noqa: E402
    DEBATE_CONFIG_VERSION,
    DebateReport,
    Round2Config,
    run_debate_question,
)
from mad.parser_v1 import (  # noqa: E402
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)
from mad.prompts_v1 import DEBATE_PROMPT_VERSION  # noqa: E402
from mad.round1 import Round1Config  # noqa: E402
from mad.runner import (  # noqa: E402
    EXPERIMENTAL_QUESTION_COUNT,
    FixtureClient,
    RunnerError,
    assert_run_is_open,
    ensure_safe_database_path,
    load_experimental_questions,
    production_database_path,
    require_spend_confirmation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SETTINGS_VERSION = "agents_v7"
RUN_PREFIX = f"experiment_{FROZEN_SETTINGS_VERSION}_"

# Five agents, two rounds and 300 questions. Each response can make the initial
# attempt plus one retry under D012, so the spend confirmation names attempts.
MAXIMUM_RESPONSES = EXPERIMENTAL_QUESTION_COUNT * 10
MAXIMUM_ATTEMPTS = MAXIMUM_RESPONSES * DEFAULT_MAX_ATTEMPTS

FAILURE_STATUSES = (
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
    STATUS_PARSE_FAIL,
    STATUS_API_ERROR,
)


# 1. Stored-run safety and resumption


def _main_runs(db: ResultsDatabase) -> list[dict[str, Any]]:
    return [row for row in db.read_runs() if str(row["run_id"]).startswith("experiment_")]


def _question_states(
    db: ResultsDatabase,
    run_id: str,
    *,
    question_ids: Sequence[str],
    agent_ids: Sequence[str],
) -> tuple[set[str], set[str]]:
    """Return complete and empty questions; reject foreign or partial rows."""
    expected_questions = set(question_ids)
    expected_responses = {
        (round_number, agent_id)
        for round_number in (1, 2)
        for agent_id in agent_ids
    }
    responses_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in db.read_responses(run_id):
        responses_by_question[row["question_id"]].append(row)
    for row in db.read_outcomes(run_id):
        outcomes_by_question[row["question_id"]].append(row)

    stored_questions = set(responses_by_question) | set(outcomes_by_question)
    unexpected = sorted(stored_questions - expected_questions)
    if unexpected:
        raise RunnerError(
            f"run {run_id!r} contains non-experimental question rows: {unexpected[:5]}"
        )

    complete: set[str] = set()
    empty: set[str] = set()
    partial: list[str] = []
    for question_id in question_ids:
        responses = responses_by_question[question_id]
        outcomes = outcomes_by_question[question_id]
        if not responses and not outcomes:
            empty.add(question_id)
            continue
        response_keys = {
            (int(row["round"]), str(row["agent_id"])) for row in responses
        }
        outcome_rounds = {int(row["round"]) for row in outcomes}
        if (
            len(responses) == len(expected_responses)
            and response_keys == expected_responses
            and len(outcomes) == 2
            and outcome_rounds == {1, 2}
        ):
            complete.add(question_id)
        else:
            partial.append(question_id)

    if partial:
        raise RunnerError(
            "cannot safely resume: these questions have partial audit rows: "
            + ", ".join(partial[:10])
            + ". No rows were deleted or overwritten."
        )
    return complete, empty


def _inspect_existing_run(
    db_path: Path,
    *,
    resume_run_id: str | None,
    questions: Sequence[Mapping[str, Any]],
    registry: Mapping[str, ModelSpec],
    round1_config: Round1Config,
    round2_config: Round2Config,
) -> set[str]:
    """Validate a requested resume or refuse an accidental second main run."""
    if not db_path.exists():
        if resume_run_id:
            raise RunnerError(f"cannot resume {resume_run_id!r}: database does not exist")
        return set()

    question_ids = [str(question["stable_id"]) for question in questions]
    with ResultsDatabase(db_path) as db:
        if not resume_run_id:
            previous = _main_runs(db)
            if previous:
                run_ids = ", ".join(str(row["run_id"]) for row in previous[:3])
                raise RunnerError(
                    "a main-experiment run already exists in this database: "
                    f"{run_ids}. Use --resume for an unfinished run; never create "
                    "a second formal result silently."
                )
            return set()

        if not resume_run_id.startswith(RUN_PREFIX):
            raise RunnerError(
                f"--resume must name an {FROZEN_SETTINGS_VERSION} main run beginning "
                f"with {RUN_PREFIX!r}"
            )
        run = db.read_run(resume_run_id)
        if run is None:
            raise RunnerError(f"cannot resume unknown run {resume_run_id!r}")
        if run["ended_at"] is not None:
            raise RunnerError(f"run {resume_run_id!r} is already complete")
        assert_run_is_open(db, resume_run_id, round1_config)
        assert_run_is_open(db, resume_run_id, round2_config)
        complete, _ = _question_states(
            db,
            resume_run_id,
            question_ids=question_ids,
            agent_ids=list(registry),
        )
        return complete


# 2. Progress and final stored-data summary


def _vote_text(round_report) -> str:
    outcome = round_report.outcome
    arrow = f"->{outcome.consensus_answer}" if outcome.decided else ""
    return f"{outcome.state}{arrow}"


def _question_line(
    index: int, total: int, report: DebateReport, seconds: float
) -> str:
    return (
        f"{index:>3}/{total}  {report.round1.question_id:<24} "
        f"R1 {_vote_text(report.round1):<22} "
        f"R2 {_vote_text(report.round2):<22} "
        f"${report.total_cost_usd:.6f}  {seconds:5.1f}s"
    )


def _print_stored_summary(
    db: ResultsDatabase,
    run_id: str,
    *,
    registry: Mapping[str, ModelSpec],
    question_count: int,
    session_seconds: float,
    live: bool,
) -> None:
    responses = db.read_responses(run_id)
    outcomes = db.read_outcomes(run_id)

    print("\nfailures by agent and round (a failure is never a wrong answer)")
    header = (
        "  agent            round  "
        + "  ".join(f"{status:<10}" for status in FAILURE_STATUSES)
        + "  ok"
    )
    print(header)
    for agent_id in registry:
        for round_number in (1, 2):
            counts = Counter(
                row["status"]
                for row in responses
                if row["agent_id"] == agent_id and row["round"] == round_number
            )
            cells = "  ".join(
                f"{counts.get(status, 0):<10}" for status in FAILURE_STATUSES
            )
            print(
                f"  {agent_id:<16} {round_number}      {cells}  "
                f"{counts.get(STATUS_OK, 0)}"
            )

    truncated = Counter(
        row["agent_id"] for row in responses if row["status"] == STATUS_TRUNCATED
    )
    if truncated:
        print("\ntruncation observed; these rows remain failures under frozen settings")
        for agent_id, count in truncated.most_common():
            print(
                f"  {agent_id:<16} {count:>3} of {question_count * 2} responses "
                f"(limit {registry[agent_id].max_tokens})"
            )
    else:
        print("\ntruncation: none")

    print("\nconsensus states")
    by_round: dict[int, list[dict[str, Any]]] = {
        round_number: [row for row in outcomes if row["round"] == round_number]
        for round_number in (1, 2)
    }
    for round_number in (1, 2):
        counts = Counter(row["consensus_state"] for row in by_round[round_number])
        cells = "  ".join(f"{state}={counts[state]}" for state in sorted(counts))
        print(f"  round {round_number}: {cells}")
    round1 = {row["question_id"]: row for row in by_round[1]}
    round2 = {row["question_id"]: row for row in by_round[2]}
    split = sum(not bool(row["decided"]) for row in round1.values())
    moved = sum(
        round1[question_id]["consensus_answer"]
        != round2[question_id]["consensus_answer"]
        for question_id in round1
    )
    print(f"  {split} of {question_count} questions had no Round 1 majority")
    print(f"  {moved} of {question_count} group answers changed between rounds")

    print("\ntokens and cost")
    for round_number in (1, 2):
        rows = by_round[round_number]
        prompt = sum(int(row["total_prompt_tokens"]) for row in rows)
        completion = sum(int(row["total_completion_tokens"]) for row in rows)
        cost = sum(float(row["total_cost_usd"]) for row in rows)
        print(
            f"  round {round_number}: {prompt:>8,} prompt  "
            f"{completion:>7,} completion  ${cost:.6f}"
        )
    total_prompt = sum(int(row["total_prompt_tokens"]) for row in outcomes)
    total_completion = sum(int(row["total_completion_tokens"]) for row in outcomes)
    total_cost = sum(float(row["total_cost_usd"]) for row in outcomes)
    paid = [row for row in responses if not row["cache_hit"]]
    attempts = sum(int(row["attempt_count"]) for row in paid)
    print(
        f"  total:   {total_prompt:>8,} prompt  {total_completion:>7,} completion  "
        f"${total_cost:.6f}"
    )
    if live:
        print(
            f"  {len(responses) - len(paid)} of {len(responses)} responses came from "
            f"cache; {attempts} paid attempts for {len(paid)} paid responses"
        )
    else:
        print(f"  {len(responses)} fixture responses; 0 API calls and $0 real spend")
        print("  (fixture token/cost numbers are not experimental measurements)")
    print(f"this command session: {session_seconds / 3600:.2f} hours")


# 3. Command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run all 300 frozen experimental questions under frozen agents_v7 settings."
        )
    )
    parser.add_argument("--live", action="store_true", help="make real, paid API calls")
    parser.add_argument(
        "--yes-spend-real-money",
        action="store_true",
        help=f"required with --live; confirms up to {MAXIMUM_ATTEMPTS} paid attempts",
    )
    parser.add_argument(
        "--db",
        help=(
            "database path (default: temp file for dry runs, "
            "storage/results.sqlite for live)"
        ),
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue one unfinished main run after complete question boundaries",
    )
    args = parser.parse_args(argv)

    # The main experiment is frozen. There is deliberately no --agents or
    # --no-cache switch here: either would change the accepted recipe.
    use_cache = args.live
    round1_config = Round1Config(
        config_version=DEBATE_CONFIG_VERSION,
        settings_version=FROZEN_SETTINGS_VERSION,
        cache_enabled=use_cache,
    )
    round2_config = Round2Config(
        settings_version=FROZEN_SETTINGS_VERSION,
        cache_enabled=use_cache,
    )

    try:
        require_spend_confirmation(
            live=args.live,
            spend_confirmed=args.yes_spend_real_money,
            maximum_attempts=MAXIMUM_ATTEMPTS,
        )
        questions = load_experimental_questions()
        if args.db:
            db_path = ensure_safe_database_path(args.db, live=args.live)
        elif args.live:
            db_path = production_database_path()
        else:
            if args.resume:
                raise RunnerError("a dry --resume requires the original --db path")
            db_path = Path(tempfile.mkdtemp()) / "dryrun.sqlite"
    except RunnerError as error:
        print(f"refused: {error}")
        return 1

    registry = load_model_registry(
        REPO_ROOT / "configs" / "models" / f"{FROZEN_SETTINGS_VERSION}.yaml"
    )
    try:
        completed = _inspect_existing_run(
            db_path,
            resume_run_id=args.resume,
            questions=questions,
            registry=registry,
            round1_config=round1_config,
            round2_config=round2_config,
        )
    except RunnerError as error:
        print(f"refused: {error}")
        return 1

    mode = "LIVE - this spends real money" if args.live else "dry run - fixture replies, free"
    print(f"mode: {mode}")
    print(f"settings: {FROZEN_SETTINGS_VERSION} (frozen; no settings override)")
    print(
        "max_tokens: "
        + ", ".join(f"{agent}={spec.max_tokens}" for agent, spec in registry.items())
    )
    print(f"cache: {'on' if use_cache else 'off'}")
    print("execution: sequential")
    print(
        f"questions: {len(questions)} experimental questions, both rounds. "
        f"{MAXIMUM_RESPONSES} responses, up to {MAXIMUM_ATTEMPTS} paid attempts "
        "if every response is retried."
    )

    shared_client = None
    if args.live:
        load_env_file()
        shared_client = OpenRouterClient(
            timeout_seconds=round1_config.timeout_seconds,
            max_attempts=round1_config.max_attempts,
        )
        try:
            shared_client.assert_registry_routes_available(registry)
        except ApiConfigurationError as error:
            shared_client.close()
            print(f"refused: {error}")
            return 1

    run_id = args.resume or (
        RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    cache = ResponseCache() if use_cache else None
    started = monotonic()

    print(f"\nrun: {run_id}\ndatabase: {db_path}")
    if completed:
        print(f"resume: preserving and skipping {len(completed)} complete questions")
    print()

    try:
        with ResultsDatabase(db_path) as db:
            if args.resume:
                # Recheck after the preflight, in case another process changed
                # the database while the free endpoint checks were running.
                completed, _ = _question_states(
                    db,
                    run_id,
                    question_ids=[str(question["stable_id"]) for question in questions],
                    agent_ids=list(registry),
                )
            else:
                if _main_runs(db):
                    raise RunnerError("a main run appeared while preflight was running")
                db.start_run(
                    run_id,
                    config_name=DEBATE_CONFIG_VERSION,
                    question_set_version=round1_config.question_set_version,
                    prompt_version=DEBATE_PROMPT_VERSION,
                    settings_version=FROZEN_SETTINGS_VERSION,
                    parser_version=round1_config.parser_version,
                )

            for index, question in enumerate(questions, start=1):
                question_id = str(question["stable_id"])
                if question_id in completed:
                    print(
                        f"{index:>3}/{len(questions)}  {question_id:<24} "
                        "already complete — skipped",
                        flush=True,
                    )
                    continue
                client = shared_client or FixtureClient(question)
                question_started = monotonic()
                report = run_debate_question(
                    question,
                    registry=registry,
                    client=client,
                    db=db,
                    run_id=run_id,
                    round1_config=round1_config,
                    round2_config=round2_config,
                    cache=cache,
                )
                print(
                    _question_line(
                        index, len(questions), report, monotonic() - question_started
                    ),
                    flush=True,
                )

            complete, empty = _question_states(
                db,
                run_id,
                question_ids=[str(question["stable_id"]) for question in questions],
                agent_ids=list(registry),
            )
            if empty or len(complete) != len(questions):
                raise RunnerError(
                    f"run has {len(complete)} complete and {len(empty)} empty questions; "
                    "it will remain unfinished"
                )
            db.finish_run(run_id)
            _print_stored_summary(
                db,
                run_id,
                registry=registry,
                question_count=len(questions),
                session_seconds=monotonic() - started,
                live=args.live,
            )
    except RunnerError as error:
        print(f"refused: {error}")
        return 1
    finally:
        if shared_client is not None:
            shared_client.close()
        if cache is not None:
            cache.close()

    print(
        f"\nstored {len(questions)} questions under completed run {run_id}."
        "\nNo accuracy was computed: the answer key never entered this process."
        "\nEvaluate this run separately with expected_questions=300 against "
        "experimental_answers.jsonl."
    )
    print("Back up the results database and cache before evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
