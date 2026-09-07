"""Round 2: each agent reconsiders its own answer alongside the other four.

This is the only stage where the agents communicate, so the difference between
the Round 1 vote and the Round 2 vote is the answer to the research question.
Everything here exists to make that difference mean what it claims to mean:

  - an agent is shown its own Round 1 response as its own earlier turn, never
    as an anonymous peer, so Round 2 continues an opinion instead of forming a
    fresh one
  - the other four arrive without identity labels and in a deterministic order,
    so a stored conversation can be reconstructed exactly; fixed order may still
    create a positional effect and is reported as a design limitation
  - only a genuinely valid Round 1 response is shown at all (D019), so a
    refusal or a truncated reply cannot be argued with as though it were an
    answer
  - an agent whose own Round 1 failed still answers, and so does an agent with
    no surviving peers - dropping either would quietly change which questions
    the two rounds are compared on

How many peers each agent actually saw is stored per response, because it is a
confound the results chapter has to report.

The caller owns the run lifecycle, exactly as in Round 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mad.api_client import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT_SECONDS, ModelSpec
from mad.database import ResultsDatabase
from mad.parser_v1 import PARSER_VERSION, STATUS_OK, ParsedResponse
from mad.prompts_v1 import (
    PROMPT_VERSION,
    ROUND2_PROMPT_VERSION,
    answer_letters,
    build_round2_messages,
)
from mad.round1 import Round1Config, run_round1_question
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

# A run that contains both rounds is not a Round 1 run, so it is not labelled
# like one. Its run row records this name and the combined prompt version
# DEBATE_PROMPT_VERSION; each response row still records only its own round's
# prompt (D020).
DEBATE_CONFIG_VERSION = "debate_config_v1"


@dataclass(frozen=True)
class Round2Config:
    """The full recipe of the Round 2 stage of a debate run."""

    config_version: str = DEBATE_CONFIG_VERSION
    question_set_version: str = QUESTION_SET_VERSION
    prompt_version: str = ROUND2_PROMPT_VERSION
    settings_version: str = SETTINGS_VERSION
    parser_version: str = PARSER_VERSION
    cache_enabled: bool = False       # callers explicitly opt into the cache
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    parallel_calls: bool = False      # sequential; parallel comes with the pilot


@dataclass(frozen=True)
class DebateReport:
    """The Round 1 and Round 2 reports for one complete debate question."""

    round1: RoundReport
    round2: RoundReport

    @property
    def total_cost_usd(self) -> float:
        """The total cost of both rounds."""
        return self.round1.total_cost_usd + self.round2.total_cost_usd


# 2. What each agent is shown


@dataclass(frozen=True)
class PeerResponse:
    """One valid Round 1 response, ready to be shown anonymously.

    The agent_id is kept on this side of the boundary only so the runner can
    exclude an agent from its own peer set and store which rows it saw. It is
    never put into a message.
    """

    response_id: int
    agent_id: str
    text: str


def valid_round1_responses(
    db: ResultsDatabase, *, run_id: str, question_id: str, registry: Mapping[str, ModelSpec]
) -> dict[str, PeerResponse]:
    """Every agent's Round 1 response that is usable as debate material.

    Valid means ``status == "OK"`` and nothing else (D019). A REFUSAL, a
    TRUNCATED reply, a PARSE_FAIL or an API_ERROR is an absence, not a position,
    and showing one as though it were a peer's argument would put text in front
    of four models that the parser has already judged unusable.

    Keyed by agent_id, in registry order, so the order every caller sees is the
    configured order rather than whatever the database returned.
    """
    stored = {
        row["agent_id"]: row
        for row in db.read_responses(run_id, round=1, question_id=question_id)
    }
    missing = [agent_id for agent_id in registry if agent_id not in stored]
    if missing:
        raise RunnerError(
            f"Round 1 of {question_id} is incomplete: no stored response for {missing}. "
            "Round 2 continues a debate; it cannot start one."
        )

    return {
        agent_id: PeerResponse(
            response_id=int(stored[agent_id]["response_id"]),
            agent_id=agent_id,
            text=stored[agent_id]["raw_response"],
        )
        for agent_id in registry
        if stored[agent_id]["status"] == STATUS_OK
    }


def peers_for(
    agent_id: str, valid_responses: Mapping[str, PeerResponse]
) -> tuple[PeerResponse, ...]:
    """The other agents' valid responses, in registry order, never this one.

    Registry order makes the sequence deterministic, which is what lets a stored
    run be reconstructed exactly. A model sees only "PEER RESPONSE 1" to
    "PEER RESPONSE 4" and is told the order carries no meaning, so no identity
    is disclosed. A peer's numerical position can shift by one depending on
    which answering agent was removed. A fixed order can therefore still carry
    a small positional effect. It is a known, recorded property of the design,
    not a claim that ordering cannot matter.
    """
    return tuple(
        peer for other_id, peer in valid_responses.items() if other_id != agent_id
    )


# 3. Running one question


def run_round2_question(
    question: Mapping[str, Any],
    *,
    registry: Mapping[str, ModelSpec],
    client: CompletionClient,
    db: ResultsDatabase,
    run_id: str,
    config: Round2Config = Round2Config(),
    cache: ResponseCacheLike | None = None,
) -> RoundReport:
    """One question through Round 2: read Round 1, debate, parse, store, vote.

    Round 1 for this question and run must already be stored. Each agent gets
    its own conversation, because each is shown a different four responses, so
    unlike Round 1 the messages are built inside the loop.

    One agent failing is stored as API_ERROR and the other four continue. The
    three-of-five threshold is unchanged: it is never lowered to a majority of
    whatever came back.
    """
    if len(registry) != EXPECTED_AGENT_COUNT:
        raise RunnerError(
            f"the registry must hold {EXPECTED_AGENT_COUNT} agents, got {len(registry)}. "
            "No call is made until that is right."
        )
    if config.cache_enabled != (cache is not None):
        raise RunnerError(
            f"config says cache_enabled={config.cache_enabled} but a cache "
            f"{'was' if cache is not None else 'was not'} supplied. The labels must tell the truth."
        )
    if config.parallel_calls:
        raise RunnerError("parallel_calls=True, but Round 2 runs sequentially for now")

    assert_run_is_open(db, run_id, config)

    question_id = str(question["stable_id"])
    letters = answer_letters(question)
    valid_responses = valid_round1_responses(
        db, run_id=run_id, question_id=question_id, registry=registry
    )

    parsed_by_agent: dict[str, ParsedResponse] = {}
    agent_results: list[AgentResult] = []
    totals = RoundTotals()

    for agent_id, spec in registry.items():
        own = valid_responses.get(agent_id)
        peers = peers_for(agent_id, valid_responses)
        messages = build_round2_messages(
            question,
            own.text if own is not None else None,
            [peer.text for peer in peers],
        )

        parsed, agent_result, record = run_one_agent(
            spec=spec,
            messages=messages,
            letters=letters,
            client=client,
            db=db,
            cache=cache,
            run_id=run_id,
            question_id=question_id,
            round=2,
            prompt_version=config.prompt_version,
            # Exactly the rows shown, in the order shown, so the conversation
            # can be rebuilt from the database alone.
            peer_response_ids=[peer.response_id for peer in peers],
        )
        parsed_by_agent[agent_id] = parsed
        agent_results.append(agent_result)
        totals.add(record)

    outcome = tally(parsed_by_agent, expected_agents=registry.keys())
    record_round_outcome(
        db,
        run_id=run_id,
        question_id=question_id,
        round=2,
        outcome=outcome,
        totals=totals,
    )

    return RoundReport(
        run_id=run_id,
        question_id=question_id,
        round=2,
        agents=tuple(agent_results),
        outcome=outcome,
        total_cost_usd=totals.cost_usd,
    )


# 4. Running both rounds


def run_debate_question(
    question: Mapping[str, Any],
    *,
    registry: Mapping[str, ModelSpec],
    client: CompletionClient,
    db: ResultsDatabase,
    run_id: str,
    round1_config: Round1Config,
    round2_config: Round2Config,
    cache: ResponseCacheLike | None = None,
) -> DebateReport:
    """Run one question through Round 1 and then Round 2.

    The caller starts and finishes the database run. Both stage configurations
    are checked before Round 1, so a bad Round 2 setup cannot waste five paid
    Round 1 calls before being discovered.
    """
    mismatches: list[str] = []
    for field in (
        "config_version",
        "question_set_version",
        "settings_version",
        "parser_version",
        "timeout_seconds",
        "max_attempts",
    ):
        round1_value = getattr(round1_config, field)
        round2_value = getattr(round2_config, field)
        if round1_value != round2_value:
            mismatches.append(
                f"{field}: Round 1 {round1_value!r}, Round 2 {round2_value!r}"
            )

    if round1_config.prompt_version != PROMPT_VERSION:
        mismatches.append(
            f"Round 1 prompt must be {PROMPT_VERSION!r}, "
            f"got {round1_config.prompt_version!r}"
        )
    if round2_config.prompt_version != ROUND2_PROMPT_VERSION:
        mismatches.append(
            f"Round 2 prompt must be {ROUND2_PROMPT_VERSION!r}, "
            f"got {round2_config.prompt_version!r}"
        )

    cache_enabled = cache is not None
    if round1_config.cache_enabled != cache_enabled:
        mismatches.append(
            f"Round 1 cache_enabled={round1_config.cache_enabled}, "
            f"cache supplied={cache_enabled}"
        )
    if round2_config.cache_enabled != cache_enabled:
        mismatches.append(
            f"Round 2 cache_enabled={round2_config.cache_enabled}, "
            f"cache supplied={cache_enabled}"
        )
    if round1_config.parallel_calls or round2_config.parallel_calls:
        mismatches.append(
            "both rounds must remain sequential until the pilot adds parallel calls"
        )

    if mismatches:
        raise RunnerError(
            "the two-round configuration is inconsistent: " + "; ".join(mismatches)
        )

    # Validate both stages before the first potentially paid call.
    assert_run_is_open(db, run_id, round1_config)
    assert_run_is_open(db, run_id, round2_config)

    round1_report = run_round1_question(
        question,
        registry=registry,
        client=client,
        db=db,
        run_id=run_id,
        config=round1_config,
        cache=cache,
    )
    round2_report = run_round2_question(
        question,
        registry=registry,
        client=client,
        db=db,
        run_id=run_id,
        config=round2_config,
        cache=cache,
    )
    return DebateReport(round1=round1_report, round2=round2_report)
