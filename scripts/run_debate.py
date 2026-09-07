"""Run one pilot question through Round 1 and Round 2.

Dry run (free, fixture replies, throwaway database):

    .venv/bin/python scripts/run_debate.py --question mmlu_pro_v1:test:7296

Live run (ten real calls at most, charges the OpenRouter account):

    .venv/bin/python scripts/run_debate.py --question mmlu_pro_v1:test:7296 \
        --live --yes-spend-real-money

Live mode uses the response cache unless ``--no-cache`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.api_client import ModelSpec, OpenRouterClient, load_env_file, load_model_registry
from mad.cache import ResponseCache
from mad.database import ResultsDatabase
from mad.debate import (
    DEBATE_CONFIG_VERSION,
    DebateReport,
    Round2Config,
    run_debate_question,
)
from mad.parser_v1 import STATUS_TRUNCATED
from mad.prompts_v1 import DEBATE_PROMPT_VERSION
from mad.round1 import Round1Config
from mad.runner import (
    FixtureClient,
    RoundReport,
    RunnerError,
    ensure_safe_database_path,
    load_pilot_question,
    production_database_path,
    require_spend_confirmation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_REGISTRIES = ("agents_v1", "agents_v2", "agents_v3", "agents_v4", "agents_v5")


def _print_round(report: RoundReport) -> None:
    """Print one round in the same shape as the stored report."""
    print(f"\nRound {report.round}")
    for agent in report.agents:
        letter = agent.letter or "-"
        peers = f" peers={agent.peer_count}" if report.round == 2 else ""
        print(
            f"  {agent.agent_id:<16} {agent.status:<10} {letter:<3} "
            f"provider={agent.provider} finish={agent.finish_reason or '-'} "
            f"cost=${agent.cost_usd:.6f} attempts={agent.attempts}{peers}"
        )

    print(
        f"vote: {report.outcome.state}"
        + (f" -> {report.outcome.consensus_answer}" if report.outcome.decided else "")
        + f" ({report.outcome.valid_answer_count} valid answers)"
    )
    print(f"round cost: ${report.total_cost_usd:.6f}")


def _print_truncation_warning(
    report: RoundReport, registry: Mapping[str, ModelSpec]
) -> None:
    """Name every truncated agent and the ceiling that stopped it."""
    truncated = [agent for agent in report.agents if agent.status == STATUS_TRUNCATED]
    if not truncated:
        return
    details = ", ".join(
        f"{agent.agent_id} (limit {registry[agent.agent_id].max_tokens})"
        for agent in truncated
    )
    print(
        f"WARNING Round {report.round}: truncated: {details}. "
        "Inspect this response before changing a frozen setting."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one pilot question through a complete two-round debate."
    )
    parser.add_argument("--question", required=True, help="one stable ID from the pilot file")
    parser.add_argument("--live", action="store_true", help="make real, paid API calls")
    parser.add_argument(
        "--yes-spend-real-money",
        action="store_true",
        help="required with --live; confirms up to ten paid calls",
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
    round2_config = Round2Config(
        settings_version=args.agents,
        cache_enabled=use_cache,
    )

    # Refuse unsafe input before an API client or production database is opened.
    try:
        require_spend_confirmation(
            live=args.live,
            spend_confirmed=args.yes_spend_real_money,
            maximum_calls=10,
        )
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

    registry = load_model_registry(
        REPO_ROOT / "configs" / "models" / f"{args.agents}.yaml"
    )
    mode = "LIVE - this spends real money" if args.live else "dry run - fixture replies, free"
    print(f"mode: {mode}")
    print(
        f"settings: {args.agents} | max_tokens: "
        + ", ".join(f"{agent}={spec.max_tokens}" for agent, spec in registry.items())
    )
    print(f"cache: {'on' if use_cache else 'off'}")
    print(f"question: {question['stable_id']} ({len(question['options'])} options)")

    if args.live:
        load_env_file()
        client = OpenRouterClient(
            timeout_seconds=round1_config.timeout_seconds,
            max_attempts=round1_config.max_attempts,
        )
    else:
        client = FixtureClient(question)

    run_id = f"debate_{args.agents}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cache = ResponseCache() if use_cache else None
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
            report: DebateReport = run_debate_question(
                question,
                registry=registry,
                client=client,
                db=db,
                run_id=run_id,
                round1_config=round1_config,
                round2_config=round2_config,
                cache=cache,
            )
            # A crash leaves ended_at NULL, accurately marking an incomplete run.
            db.finish_run(run_id)
    finally:
        client.close()
        if cache is not None:
            cache.close()

    print(f"\nrun: {run_id}")
    print(f"database: {db_path}")
    _print_round(report.round1)
    _print_round(report.round2)
    print(f"\ntotal debate cost: ${report.total_cost_usd:.6f}")
    _print_truncation_warning(report.round1, registry)
    _print_truncation_warning(report.round2, registry)

    if args.live and use_cache:
        print(
            "\nnote: an exact rerun is served from cache. Changed prompts, settings "
            "or peer responses create new paid requests."
        )
    elif args.live:
        print("\nnote: --no-cache means an exact rerun pays for the calls again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
