"""Round 1: five agents answer one question alone, then the group votes.

No agent sees another's answer here, and none is told a second round follows.
That is the whole point of the round - it produces the per-agent accuracy and
the aggregation-only vote, which are two of the three headline numbers.

The caller owns the run lifecycle. It starts one database run before processing
one or many questions and finishes it only after every required stage succeeds.
Everything shared with Round 2 - the run guard, the per-agent call, the fixture
client, the report - lives in `runner.py`, so the two rounds cannot drift apart.

Money protections, because live calls spend real credit:
  - the default is a dry run on labelled fixture replies, costing nothing
  - live mode needs two explicit flags, not one
  - only questions from the 20-question pilot file are accepted
  - at most five API calls per question
  - an optional response cache prevents paying twice for an identical request
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mad.api_client import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT_SECONDS, ModelSpec
from mad.database import ResultsDatabase
from mad.parser_v1 import PARSER_VERSION, ParsedResponse
from mad.prompts_v1 import PROMPT_VERSION, answer_letters, build_round1_messages
from mad.runner import (
    QUESTION_SET_VERSION,
    SETTINGS_VERSION,
    AgentResult,
    CompletionClient,
    ResponseCacheLike,
    RoundReport,
    RoundTotals,
    RunnerError,
    assert_run_is_open,
    record_round_outcome,
    run_one_agent,
)
from mad.voting import EXPECTED_AGENT_COUNT, tally


# 1. Versions

# P8: everything that shaped a run is named, so its results stay citable.
# v1 was the cacheless Milestone 1 runner; run milestone1_20260901T161852Z is
# recorded under it and that label stays true. v2 added the response cache
# path, so new runs must not claim the old recipe.
CONFIG_VERSION = "round1_config_v2"


@dataclass(frozen=True)
class Round1Config:
    """The full recipe of a Round 1 run, as P8 requires."""

    config_version: str = CONFIG_VERSION
    question_set_version: str = QUESTION_SET_VERSION
    prompt_version: str = PROMPT_VERSION
    settings_version: str = SETTINGS_VERSION
    parser_version: str = PARSER_VERSION
    cache_enabled: bool = False       # callers explicitly opt into the cache
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    parallel_calls: bool = False      # sequential; parallel comes with the pilot


# 2. Running one question


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

    The caller must start ``run_id`` first and finish it after all questions and
    rounds succeed. There is at most one API call per agent here; a cache hit
    makes none. One agent failing is stored as API_ERROR and the other four
    continue - a failure must never end the run.

    All five agents receive exactly the same two messages, so nothing here
    depends on the order the registry happens to be in.
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

    assert_run_is_open(db, run_id, config)

    question_id = str(question["stable_id"])
    letters = answer_letters(question)
    messages = build_round1_messages(question)

    parsed_by_agent: dict[str, ParsedResponse] = {}
    agent_results: list[AgentResult] = []
    totals = RoundTotals()

    for agent_id, spec in registry.items():
        parsed, agent_result, record = run_one_agent(
            spec=spec,
            messages=messages,
            letters=letters,
            client=client,
            db=db,
            cache=cache,
            run_id=run_id,
            question_id=question_id,
            round=1,
            prompt_version=config.prompt_version,
        )
        parsed_by_agent[agent_id] = parsed
        agent_results.append(agent_result)
        totals.add(record)

    outcome = tally(parsed_by_agent, expected_agents=registry.keys())
    record_round_outcome(
        db,
        run_id=run_id,
        question_id=question_id,
        round=1,
        outcome=outcome,
        totals=totals,
    )

    return RoundReport(
        run_id=run_id,
        question_id=question_id,
        round=1,
        agents=tuple(agent_results),
        outcome=outcome,
        total_cost_usd=totals.cost_usd,
    )
