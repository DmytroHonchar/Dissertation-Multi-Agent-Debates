"""Three-of-five voting. Turns five agents' answers into one group answer.

The whole rule is: an answer needs three of the five configured agents to give
it. Not three of the ones that worked - three of five. If two agents fail, the
remaining three must all agree.

That distinction is the point of the file. Lowering the bar to a majority of the
survivors would make a round with more failures look more decisive, which is
exactly the flaw this project is measuring. D004 fixes the threshold at three.

Nothing here knows the correct answer. Whether a group answer is right is
evaluation.py's job, and it is the only module allowed to read the key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mad.parser_v1 import ParsedResponse


# 1. Settings

# Five agents, three votes. Fixed by D004, never counted from what came back.
EXPECTED_AGENT_COUNT = 5
CONSENSUS_THRESHOLD = 3

STATE_UNANIMOUS = "UNANIMOUS"                        # all five agree
STATE_CONSENSUS = "CONSENSUS"                        # three or four agree
STATE_NO_CONSENSUS = "NO_CONSENSUS"                  # enough answers, no majority
STATE_INSUFFICIENT_ANSWERS = "INSUFFICIENT_ANSWERS"  # too few answers to reach three

# The last two both mean undecided, but they are different findings.
CONSENSUS_STATES = (
    STATE_UNANIMOUS,
    STATE_CONSENSUS,
    STATE_NO_CONSENSUS,
    STATE_INSUFFICIENT_ANSWERS,
)


# 2. Errors


class VotingError(ValueError):
    """The group being voted on is not the five configured agents."""


# 3. What comes back


@dataclass(frozen=True)
class VoteOutcome:
    """Where the group stood on one question after one round.

    Round 1 and Round 2 each produce their own outcome. Comparing the two is the
    research question, so they are never combined.
    """

    state: str
    consensus_answer: str | None
    decided: bool
    valid_answer_count: int
    # Votes per letter, e.g. {"A": 3, "B": 1}. Read-only once the vote is done.
    vote_counts: Mapping[str, int]
    # Who failed, so failures can be traced without re-reading every response.
    failed_agents: tuple[str, ...] = ()

    @property
    def is_undecided(self) -> bool:
        """True for both undecided states. D010 scores these as incorrect."""
        return not self.decided


# 4. Voting


def tally(
    responses: Mapping[str, ParsedResponse],
    *,
    expected_agents: Collection[str] | None = None,
) -> VoteOutcome:
    """Count the votes of exactly five agents and return the group's position.

    Only a response the parser marked OK carries a vote. REFUSAL, TRUNCATED,
    PARSE_FAIL and API_ERROR contribute nothing - they are not wrong answers,
    they are absences.

    Pass `expected_agents` to check that the five are the configured five, not
    just any five. The runner should pass `load_model_registry()`'s keys. This
    module does not read the registry itself, because voting has no business
    knowing which models the agents are.

    The responses are read, never modified.
    """
    _check_group(responses, expected_agents)

    votes = Counter(
        response.letter
        for response in responses.values()
        if response.is_vote and response.letter is not None
    )
    valid_answer_count = sum(votes.values())
    failed_agents = tuple(
        sorted(agent_id for agent_id, response in responses.items() if not response.is_vote)
    )

    # Three out of five, asked first, so a failure can never shrink the target.
    leader, leader_votes = _leader(votes)

    if leader_votes == EXPECTED_AGENT_COUNT:
        state, answer = STATE_UNANIMOUS, leader
    elif leader_votes >= CONSENSUS_THRESHOLD:
        state, answer = STATE_CONSENSUS, leader
    elif valid_answer_count < CONSENSUS_THRESHOLD:
        # Three votes was already unreachable, whatever the agents said.
        state, answer = STATE_INSUFFICIENT_ANSWERS, None
    else:
        state, answer = STATE_NO_CONSENSUS, None

    return VoteOutcome(
        state=state,
        consensus_answer=answer,
        decided=answer is not None,
        valid_answer_count=valid_answer_count,
        # A read-only view, so the tally cannot be edited after the vote.
        vote_counts=MappingProxyType(dict(votes)),
        failed_agents=failed_agents,
    )


def _leader(votes: Counter[str]) -> tuple[str | None, int]:
    """The most-voted letter and its count, or (None, 0) if nobody answered.

    A tie needs no resolving: two letters cannot both reach three out of five,
    so a tied leader is below the threshold either way. There is no tie-break
    and no judge model (D004).
    """
    if not votes:
        return None, 0
    letter, count = votes.most_common(1)[0]
    return letter, count


def _check_group(
    responses: Mapping[str, ParsedResponse], expected_agents: Collection[str] | None
) -> None:
    """The group must be five identified agents. Not four, not six.

    The threshold is three out of five. Voting on a different number of agents
    would silently change what a "majority" means, so it is refused rather than
    accommodated.

    Which five they are is only checked when the caller says what to expect.
    Without that, this guarantees the count and the identifiers, not the names.
    """
    if len(responses) != EXPECTED_AGENT_COUNT:
        raise VotingError(
            f"voting needs exactly {EXPECTED_AGENT_COUNT} agents, got {len(responses)}: "
            f"{sorted(responses)}. The threshold is never adjusted to fit."
        )
    if any(not agent_id for agent_id in responses):
        raise VotingError("every agent must have an identifier")

    if expected_agents is not None:
        expected = set(expected_agents)
        if len(expected) != EXPECTED_AGENT_COUNT:
            raise VotingError(
                f"expected_agents must name {EXPECTED_AGENT_COUNT} agents, got {sorted(expected)}"
            )
        if set(responses) != expected:
            missing = sorted(expected - set(responses))
            unexpected = sorted(set(responses) - expected)
            raise VotingError(
                f"these are not the configured agents; missing {missing}, unexpected {unexpected}"
            )
