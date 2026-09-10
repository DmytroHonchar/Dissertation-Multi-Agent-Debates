"""Run all 20 frozen pilot questions through both rounds, in one database run.

Dry run (free, fixture replies, throwaway database):

    .venv/bin/python scripts/run_pilot.py

Live run (200 responses, up to 400 paid attempts with retries):

    .venv/bin/python scripts/run_pilot.py --live --yes-spend-real-money

This script stores results. It does not score them: the answer key never enters
a process that calls models. Score the finished run separately against
`data/frozen/mmlu_pro_v1/answer_keys/pilot_answers.jsonl`, with
`evaluate_run(..., expected_questions=20)`.

What it prints is what P11 asks you to check by hand and needs no answer key:
truncation by agent, failures by status, consensus states per round, real token
usage read back from the stored rows, and a cost projection taken per paid
response so cached questions cannot make the 300-question estimate read low.

A crash part-way leaves `ended_at` NULL, so the run is honestly marked
incomplete and evaluation will refuse it. Pilot results are development data and
never enter the final results.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.api_client import (
    DEFAULT_MAX_ATTEMPTS,
    ModelSpec,
    OpenRouterClient,
    load_env_file,
    load_model_registry,
)
from mad.cache import ResponseCache
from mad.database import ResultsDatabase
from mad.debate import (
    DEBATE_CONFIG_VERSION,
    DebateReport,
    Round2Config,
    run_debate_question,
)
from mad.parser_v1 import (
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)
from mad.prompts_v1 import DEBATE_PROMPT_VERSION
from mad.round1 import Round1Config
from mad.runner import (
    PILOT_QUESTION_COUNT,
    FixtureClient,
    RunnerError,
    ensure_safe_database_path,
    load_pilot_questions,
    production_database_path,
    require_spend_confirmation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_REGISTRIES = ("agents_v1", "agents_v2", "agents_v3", "agents_v4", "agents_v5")

# Five agents, two rounds, twenty questions: 200 responses. Each may be retried
# once (D012), so the ceiling on billable attempts is twice that. The money
# guard quotes the attempt figure, because that is what can be charged.
MAXIMUM_RESPONSES = PILOT_QUESTION_COUNT * 10
MAXIMUM_ATTEMPTS = MAXIMUM_RESPONSES * DEFAULT_MAX_ATTEMPTS

# The main experiment, for the cost projection printed at the end.
EXPERIMENTAL_QUESTION_COUNT = 300

FAILURE_STATUSES = (STATUS_REFUSAL, STATUS_TRUNCATED, STATUS_PARSE_FAIL, STATUS_API_ERROR)


# 1. Printing


def _question_line(index: int, report: DebateReport, seconds: float) -> str:
    """One line per question, so a long live run shows progress as it goes."""
    def vote(round_report):
        outcome = round_report.outcome
        arrow = f"->{outcome.consensus_answer}" if outcome.decided else ""
        return f"{outcome.state}{arrow}"

    return (
        f"{index:>3}/{PILOT_QUESTION_COUNT}  {report.round1.question_id:<24} "
        f"R1 {vote(report.round1):<22} R2 {vote(report.round2):<22} "
        f"${report.total_cost_usd:.6f}  {seconds:5.1f}s"
    )


def _print_failures(reports: Sequence[DebateReport], registry: Mapping[str, ModelSpec]) -> None:
    """Per-agent failure counts by status and round. P11's first check."""
    print("\nfailures by agent and round (a failure is never a wrong answer)")
    header = "  agent            round  " + "  ".join(f"{s:<10}" for s in FAILURE_STATUSES) + "  ok"
    print(header)
    for agent_id in registry:
        for round_number in (1, 2):
            rounds = [r.round1 if round_number == 1 else r.round2 for r in reports]
            agents = [a for r in rounds for a in r.agents if a.agent_id == agent_id]
            counts = Counter(a.status for a in agents)
            cells = "  ".join(f"{counts.get(s, 0):<10}" for s in FAILURE_STATUSES)
            print(f"  {agent_id:<16} {round_number}      {cells}  {counts.get(STATUS_OK, 0)}")


def _print_truncation(reports: Sequence[DebateReport], registry: Mapping[str, ModelSpec]) -> None:
    """Truncation is the one failure that means a setting is wrong, not a model."""
    truncated = Counter(
        agent.agent_id
        for report in reports
        for round_report in (report.round1, report.round2)
        for agent in round_report.agents
        if agent.status == STATUS_TRUNCATED
    )
    if not truncated:
        print("\ntruncation: none. The agents_v5 ceilings held for all 20 questions.")
        return

    print("\nWARNING truncation - read these responses before changing a ceiling:")
    for agent_id, count in truncated.most_common():
        limit = registry[agent_id].max_tokens
        print(f"  {agent_id:<16} {count:>3} of 40 responses  (limit {limit})")
    print("  Raising a ceiling means a new settings version and a new pilot (D018).")


def _print_consensus(reports: Sequence[DebateReport]) -> None:
    """How often the group agreed, before and after talking."""
    print("\nconsensus states")
    for round_number in (1, 2):
        rounds = [r.round1 if round_number == 1 else r.round2 for r in reports]
        counts = Counter(r.outcome.state for r in rounds)
        cells = "  ".join(f"{state}={counts[state]}" for state in sorted(counts))
        print(f"  round {round_number}: {cells}")

    split = sum(1 for r in reports if not r.round1.outcome.decided)
    moved = sum(
        1
        for r in reports
        if r.round1.outcome.consensus_answer != r.round2.outcome.consensus_answer
    )
    print(f"  {split} of {len(reports)} questions had no Round 1 majority")
    print(f"  {moved} of {len(reports)} group answers changed between rounds")


def _read_usage(db: ResultsDatabase, run_id: str) -> dict[str, Any]:
    """Token, cache and attempt totals, read back from the rows just written.

    Taken from the database rather than the in-memory reports because that is
    the data evaluation will see, and because the reports carry cost but not
    tokens or cache hits.
    """
    outcomes = db.read_outcomes(run_id)
    responses = db.read_responses(run_id)
    paid = [row for row in responses if not row["cache_hit"]]
    return {
        "per_round": {
            round_number: {
                "prompt_tokens": sum(
                    int(row["total_prompt_tokens"])
                    for row in outcomes if row["round"] == round_number
                ),
                "completion_tokens": sum(
                    int(row["total_completion_tokens"])
                    for row in outcomes if row["round"] == round_number
                ),
                "cost_usd": sum(
                    float(row["total_cost_usd"])
                    for row in outcomes if row["round"] == round_number
                ),
            }
            for round_number in (1, 2)
        },
        "responses": len(responses),
        "cache_hits": len(responses) - len(paid),
        "paid_responses": len(paid),
        "api_attempts": sum(int(row["attempt_count"]) for row in paid),
        "cost_usd": sum(float(row["total_cost_usd"]) for row in outcomes),
    }


def _print_usage(usage: Mapping[str, Any], questions: int, wall_clock: float, live: bool) -> None:
    print("\ntokens and cost")
    for round_number in (1, 2):
        row = usage["per_round"][round_number]
        print(
            f"  round {round_number}: {row['prompt_tokens']:>8,} prompt  "
            f"{row['completion_tokens']:>7,} completion  ${row['cost_usd']:.6f}"
        )
    total_prompt = sum(usage["per_round"][r]["prompt_tokens"] for r in (1, 2))
    total_completion = sum(usage["per_round"][r]["completion_tokens"] for r in (1, 2))
    print(
        f"  total:   {total_prompt:>8,} prompt  {total_completion:>7,} completion  "
        f"${usage['cost_usd']:.6f}"
    )
    print(
        f"  {usage['cache_hits']} of {usage['responses']} responses came from cache; "
        f"{usage['api_attempts']} paid attempts for {usage['paid_responses']} paid responses"
    )

    # Per paid response, not per question: a cached question cost nothing and
    # would drag the projection below what the main run will actually cost.
    if usage["paid_responses"]:
        per_paid = usage["cost_usd"] / usage["paid_responses"]
        projected = per_paid * 10 * EXPERIMENTAL_QUESTION_COUNT
        print(
            f"  ${per_paid:.6f} per paid response -> ${projected:.2f} projected for "
            f"{EXPERIMENTAL_QUESTION_COUNT} uncached questions"
        )
    else:
        print("  every response came from cache, so this run cannot project a cost")
    if not live:
        print("  (fixture replies are free; these numbers mean nothing in a dry run)")

    print(f"time: {wall_clock / 60:.1f} minutes wall clock over {questions} questions")


# 2. The command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all 20 pilot questions through both debate rounds."
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
        "--agents",
        choices=KNOWN_REGISTRIES,
        default="agents_v5",
        help="which versioned model settings to run (default: agents_v5)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="skip the response cache; repeating the same live calls then costs money",
    )
    args = parser.parse_args(argv)

    use_cache = args.live and not args.no_cache
    round1_config = Round1Config(
        config_version=DEBATE_CONFIG_VERSION,
        settings_version=args.agents,
        cache_enabled=use_cache,
    )
    round2_config = Round2Config(settings_version=args.agents, cache_enabled=use_cache)

    # Refuse unsafe input before an API client or production database is opened.
    try:
        require_spend_confirmation(
            live=args.live,
            spend_confirmed=args.yes_spend_real_money,
            maximum_attempts=MAXIMUM_ATTEMPTS,
        )
        questions = load_pilot_questions()
        if args.db:
            db_path = ensure_safe_database_path(args.db, live=args.live)
        elif args.live:
            db_path = production_database_path()
        else:
            db_path = Path(tempfile.mkdtemp()) / "dryrun.sqlite"
    except RunnerError as error:
        print(f"refused: {error}")
        return 1

    registry = load_model_registry(REPO_ROOT / "configs" / "models" / f"{args.agents}.yaml")
    mode = "LIVE - this spends real money" if args.live else "dry run - fixture replies, free"
    print(f"mode: {mode}")
    print(
        f"settings: {args.agents} | max_tokens: "
        + ", ".join(f"{agent}={spec.max_tokens}" for agent, spec in registry.items())
    )
    print(f"cache: {'on' if use_cache else 'off'}")
    print(
        f"questions: {len(questions)} pilot questions, both rounds. "
        f"{MAXIMUM_RESPONSES} responses, up to {MAXIMUM_ATTEMPTS} paid attempts "
        f"if every one is retried."
    )

    shared_client = None
    if args.live:
        load_env_file()
        shared_client = OpenRouterClient(
            timeout_seconds=round1_config.timeout_seconds,
            max_attempts=round1_config.max_attempts,
        )

    run_id = f"pilot_{args.agents}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cache = ResponseCache() if use_cache else None
    reports: list[DebateReport] = []
    started = monotonic()

    print(f"\nrun: {run_id}\ndatabase: {db_path}\n")
    try:
        with ResultsDatabase(db_path) as db:
            db.start_run(
                run_id,
                config_name=DEBATE_CONFIG_VERSION,
                question_set_version=round1_config.question_set_version,
                prompt_version=DEBATE_PROMPT_VERSION,
                settings_version=args.agents,
                parser_version=round1_config.parser_version,
            )
            for index, question in enumerate(questions, start=1):
                # A dry run needs one fixture per question; a live run shares
                # the one HTTP client.
                client = shared_client or FixtureClient(question)
                question_started = monotonic()
                # Not caught: a raised error must leave ended_at NULL, so a
                # half-finished pilot is refused later rather than scored.
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
                reports.append(report)
                print(_question_line(index, report, monotonic() - question_started), flush=True)

            db.finish_run(run_id)
            usage = _read_usage(db, run_id)
    finally:
        if shared_client is not None:
            shared_client.close()
        if cache is not None:
            cache.close()

    _print_failures(reports, registry)
    _print_truncation(reports, registry)
    _print_consensus(reports)
    _print_usage(usage, len(reports), monotonic() - started, args.live)

    print(
        f"\nstored {len(reports)} questions under run {run_id}."
        "\nNo accuracy is printed here: the answer key never enters a process "
        "that calls models."
        "\nScore it with evaluate_run(db, run_id, key, expected_questions="
        f"{PILOT_QUESTION_COUNT}) against pilot_answers.jsonl."
    )
    if args.live:
        print("\nPilot results are development data. They never enter the final results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
