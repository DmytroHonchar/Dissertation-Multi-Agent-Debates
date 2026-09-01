"""Milestone 1: one pilot question through Round 1.

Dry run (free, fixture replies, throwaway database):

    .venv/bin/python scripts/run_milestone1.py --question mmlu_pro_v1:test:7296

Live run (five real calls, charges the OpenRouter account):

    .venv/bin/python scripts/run_milestone1.py --question mmlu_pro_v1:test:7296 \
        --live --yes-spend-real-money

There is no cache yet: repeating a live run pays for the same answers again.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.api_client import OpenRouterClient, load_env_file, load_model_registry
from mad.database import ResultsDatabase
from mad.parser_v1 import STATUS_TRUNCATED
from mad.round1 import (
    FixtureClient,
    Round1Config,
    RunnerError,
    ensure_safe_database_path,
    load_pilot_question,
    production_database_path,
    require_spend_confirmation,
    run_round1_question,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "models" / "agents_v1.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One pilot question through Round 1.")
    parser.add_argument("--question", required=True, help="one stable ID from the pilot file")
    parser.add_argument("--live", action="store_true", help="make five real, paid API calls")
    parser.add_argument("--yes-spend-real-money", action="store_true",
                        help="required with --live; confirms the spend")
    parser.add_argument("--db", help="database path (default: temp file for dry runs, "
                        "storage/results.sqlite for live)")
    args = parser.parse_args(argv)

    # One config drives everything: the client below is built from it, and the
    # same version labels end up stored on the run.
    config = Round1Config()

    # Every check that can refuse the command runs before a client exists.
    try:
        require_spend_confirmation(live=args.live, spend_confirmed=args.yes_spend_real_money)
        question = load_pilot_question(args.question)
        if args.db:
            db_path = ensure_safe_database_path(args.db, live=args.live)
        elif args.live:
            db_path = production_database_path()
        else:
            db_path = Path(tempfile.mkdtemp()) / "dryrun.sqlite"
    except RunnerError as error:
        print(f"refused: {error}")
        return 1

    registry = load_model_registry(REGISTRY_PATH)
    mode = "LIVE - this spends real money" if args.live else "dry run - fixture replies, free"
    print(f"mode: {mode}")
    print(f"question: {question['stable_id']} ({len(question['options'])} options)")

    if args.live:
        load_env_file()
        client = OpenRouterClient(
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
        )
    else:
        client = FixtureClient(question)

    run_id = f"milestone1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    try:
        with ResultsDatabase(db_path) as db:
            report = run_round1_question(
                question, registry=registry, client=client, db=db,
                run_id=run_id, config=config,
            )
    finally:
        client.close()

    print(f"\nrun: {report.run_id}")
    print(f"database: {db_path}")
    for agent in report.agents:
        letter = agent.letter or "-"
        print(f"  {agent.agent_id:<16} {agent.status:<10} {letter:<3} "
              f"provider={agent.provider} finish={agent.finish_reason or '-'} "
              f"cost=${agent.cost_usd:.6f} attempts={agent.attempts}")

    print(f"\nvote: {report.outcome.state}"
          + (f" -> {report.outcome.consensus_answer}" if report.outcome.decided else "")
          + f" ({report.outcome.valid_answer_count} valid answers)")
    print(f"total cost: ${report.total_cost_usd:.6f}")

    truncated = [a.agent_id for a in report.agents if a.status == STATUS_TRUNCATED]
    if truncated:
        print(f"\nWARNING: truncated: {', '.join(truncated)}. "
              "max_tokens=1024 is provisional - this is the signal it is too low.")
    if args.live:
        print("\nnote: there is no cache yet. Repeating this run pays for the "
              "same five answers again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
