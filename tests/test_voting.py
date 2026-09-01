"""Tests for three-of-five voting. Pure logic - no API, no database, no key."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mad import voting
from mad.parser_v1 import (
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
    ParsedResponse,
)
from mad.voting import (
    CONSENSUS_THRESHOLD,
    EXPECTED_AGENT_COUNT,
    STATE_CONSENSUS,
    STATE_INSUFFICIENT_ANSWERS,
    STATE_NO_CONSENSUS,
    STATE_UNANIMOUS,
    VoteOutcome,
    VotingError,
    tally,
)

AGENTS = ("agent_llama", "agent_qwen", "agent_mistral", "agent_deepseek", "agent_gemma")

FAILURE_STATUSES = (STATUS_REFUSAL, STATUS_TRUNCATED, STATUS_PARSE_FAIL, STATUS_API_ERROR)


def answered(letter: str) -> ParsedResponse:
    return ParsedResponse(STATUS_OK, letter=letter, extraction_method="LAST_FINAL_ANSWER_MATCH")


def failed(status: str = STATUS_PARSE_FAIL) -> ParsedResponse:
    return ParsedResponse(status, note="no usable answer")


def group(*responses: ParsedResponse) -> dict[str, ParsedResponse]:
    """Five responses, one per configured agent, in a fixed order."""
    return dict(zip(AGENTS, responses))


# 1. The four states


def test_five_identical_answers_are_unanimous():
    outcome = tally(group(*[answered("A")] * 5))
    assert outcome.state == STATE_UNANIMOUS
    assert outcome.consensus_answer == "A"
    assert outcome.decided
    assert outcome.valid_answer_count == 5


def test_four_agreeing_answers_are_consensus_not_unanimous():
    outcome = tally(group(answered("A"), answered("A"), answered("A"), answered("A"), answered("B")))
    assert outcome.state == STATE_CONSENSUS
    assert outcome.consensus_answer == "A"


def test_three_a_and_two_b_is_consensus_for_a():
    outcome = tally(group(answered("A"), answered("A"), answered("A"), answered("B"), answered("B")))
    assert outcome.state == STATE_CONSENSUS
    assert outcome.consensus_answer == "A"
    assert outcome.vote_counts == {"A": 3, "B": 2}


def test_three_votes_plus_two_failures_is_still_consensus():
    outcome = tally(group(answered("A"), answered("A"), answered("A"), failed(), failed()))
    assert outcome.state == STATE_CONSENSUS
    assert outcome.consensus_answer == "A"
    assert outcome.valid_answer_count == 3


def test_five_different_answers_is_no_consensus():
    outcome = tally(group(*[answered(letter) for letter in "ABCDE"]))
    assert outcome.state == STATE_NO_CONSENSUS
    assert outcome.consensus_answer is None
    assert not outcome.decided
    assert outcome.valid_answer_count == 5


def test_two_and_two_and_one_is_no_consensus():
    outcome = tally(group(answered("A"), answered("A"), answered("B"), answered("B"), answered("C")))
    assert outcome.state == STATE_NO_CONSENSUS
    assert outcome.consensus_answer is None


def test_two_a_one_b_and_two_failures_is_no_consensus():
    outcome = tally(group(answered("A"), answered("A"), answered("B"), failed(), failed()))
    assert outcome.state == STATE_NO_CONSENSUS
    assert outcome.valid_answer_count == 3


def test_two_a_two_b_and_one_failure_is_no_consensus():
    outcome = tally(group(answered("A"), answered("A"), answered("B"), answered("B"), failed()))
    assert outcome.state == STATE_NO_CONSENSUS
    assert outcome.valid_answer_count == 4


def test_two_answers_and_three_failures_is_insufficient_answers():
    outcome = tally(group(answered("A"), answered("A"), failed(), failed(), failed()))
    assert outcome.state == STATE_INSUFFICIENT_ANSWERS
    assert outcome.consensus_answer is None
    assert outcome.valid_answer_count == 2


def test_five_failures_is_insufficient_answers_with_no_votes():
    outcome = tally(group(*[failed()] * 5))
    assert outcome.state == STATE_INSUFFICIENT_ANSWERS
    assert outcome.valid_answer_count == 0
    assert outcome.vote_counts == {}
    assert len(outcome.failed_agents) == 5


# 2. The threshold is three of five, never three of the survivors


def test_two_agreeing_survivors_do_not_make_a_group_answer():
    """D004: both survivors agree, but two is not three. Never adjusted."""
    outcome = tally(group(answered("A"), answered("A"), failed(), failed(), failed()))
    assert outcome.state == STATE_INSUFFICIENT_ANSWERS
    assert outcome.consensus_answer is None, "100% of survivors is still only two votes"


def test_one_lone_survivor_does_not_make_a_group_answer():
    outcome = tally(group(answered("A"), failed(), failed(), failed(), failed()))
    assert outcome.state == STATE_INSUFFICIENT_ANSWERS
    assert outcome.consensus_answer is None


def test_exactly_three_is_the_boundary():
    """Two fails, three passes. This is the line the whole rule turns on."""
    two = tally(group(answered("A"), answered("A"), answered("B"), answered("C"), failed()))
    three = tally(group(answered("A"), answered("A"), answered("A"), answered("C"), failed()))
    assert two.consensus_answer is None
    assert three.consensus_answer == "A"


def test_the_threshold_and_group_size_are_fixed_constants():
    assert EXPECTED_AGENT_COUNT == 5
    assert CONSENSUS_THRESHOLD == 3


# 3. Failures never vote


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_no_failure_status_contributes_a_vote(status):
    outcome = tally(group(answered("A"), answered("A"), failed(status), failed(status), failed(status)))
    assert outcome.valid_answer_count == 2
    assert outcome.vote_counts == {"A": 2}
    assert outcome.state == STATE_INSUFFICIENT_ANSWERS


@pytest.mark.parametrize("status", FAILURE_STATUSES)
def test_a_failure_cannot_tip_a_question_into_consensus(status):
    """A failure is an absence, not a wrong answer, and not a vote either."""
    outcome = tally(group(answered("A"), answered("A"), answered("B"), failed(status), failed(status)))
    assert outcome.state == STATE_NO_CONSENSUS


def test_failed_agents_are_named_so_failures_can_be_attributed():
    outcome = tally(
        dict(
            zip(
                AGENTS,
                [answered("A"), failed(STATUS_REFUSAL), answered("A"), answered("A"), failed()],
            )
        )
    )
    assert outcome.failed_agents == ("agent_gemma", "agent_qwen")
    assert outcome.state == STATE_CONSENSUS


# 4. Every option letter is votable


@pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
def test_any_option_letter_can_win_including_the_tenth(letter):
    """MMLU-Pro questions run up to J. Nothing here assumes four options."""
    outcome = tally(group(*[answered(letter)] * 5))
    assert outcome.state == STATE_UNANIMOUS
    assert outcome.consensus_answer == letter


def test_a_late_letter_beats_an_early_one_on_votes_not_on_order():
    outcome = tally(group(answered("J"), answered("J"), answered("J"), answered("A"), answered("A")))
    assert outcome.consensus_answer == "J"


# 5. The group must be the five configured agents


@pytest.mark.parametrize("size", [0, 1, 3, 4, 6, 7])
def test_voting_on_anything_other_than_five_agents_is_refused(size):
    responses = {f"agent_{index}": answered("A") for index in range(size)}
    with pytest.raises(VotingError, match="exactly 5 agents"):
        tally(responses)


def test_a_duplicated_agent_leaves_four_entries_and_is_refused():
    """A mapping collapses a repeated agent_id, which must not pass as five."""
    responses = {
        "agent_llama": answered("A"),
        "agent_qwen": answered("A"),
        "agent_mistral": answered("A"),
        "agent_gemma": answered("B"),
    }
    with pytest.raises(VotingError):
        tally(responses)


def test_an_agent_without_an_identifier_is_refused():
    responses = {agent: answered("A") for agent in AGENTS[:4]}
    responses[""] = answered("A")
    with pytest.raises(VotingError, match="identifier"):
        tally(responses)


# 6. What this module must not do


def test_voting_does_not_modify_the_responses_it_is_given():
    responses = group(answered("A"), answered("A"), answered("A"), failed(), answered("B"))
    before = {agent: (r.status, r.letter) for agent, r in responses.items()}
    tally(responses)
    assert {agent: (r.status, r.letter) for agent, r in responses.items()} == before


def test_the_outcome_says_nothing_about_correctness():
    """Whether the group is right is evaluation.py's job, and it needs the key."""
    fields = set(VoteOutcome.__dataclass_fields__)
    assert not fields & {"correct", "is_correct", "answer_key", "gold", "score", "target"}


def test_the_module_imports_nothing_that_could_store_or_score():
    """Voting stays independent: no database, no evaluation, no answer key."""
    tree = ast.parse(Path(voting.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not {name for name in imported if "database" in name or "evaluation" in name}
    assert "mad.parser_v1" in imported, "it needs the parser's statuses and nothing more"


# 7. It feeds the outcomes table without depending on it


def test_the_outcome_carries_exactly_what_question_outcomes_stores():
    from mad.database import CONSENSUS_STATES as DATABASE_STATES, OutcomeRecord

    outcome = tally(group(answered("A"), answered("A"), answered("A"), failed(), answered("B")))
    assert outcome.state in DATABASE_STATES

    record = OutcomeRecord(
        run_id="run_test",
        question_id="q0001",
        round=1,
        consensus_state=outcome.state,
        consensus_answer=outcome.consensus_answer,
        decided=outcome.decided,
        valid_answer_count=outcome.valid_answer_count,
    )
    assert record.consensus_state == STATE_CONSENSUS
    assert record.consensus_answer == "A"


def test_both_undecided_states_report_as_undecided():
    """D010 scores both as incorrect, but they are reported as separate categories."""
    no_consensus = tally(group(*[answered(letter) for letter in "ABCDE"]))
    insufficient = tally(group(answered("A"), failed(), failed(), failed(), failed()))

    assert no_consensus.is_undecided and insufficient.is_undecided
    assert no_consensus.state != insufficient.state


def test_the_four_state_names_match_the_database_constraint():
    from mad.database import CONSENSUS_STATES as DATABASE_STATES

    assert set(voting.CONSENSUS_STATES) == set(DATABASE_STATES)


# 8. The outcome really is frozen, and the group can be checked by name


def test_the_vote_counts_cannot_be_edited_after_the_vote():
    """frozen=True protects the fields; a plain dict inside would not."""
    outcome = tally(group(answered("A"), answered("A"), answered("A"), answered("B"), failed()))

    with pytest.raises(TypeError):
        outcome.vote_counts["A"] = 0
    with pytest.raises(TypeError):
        del outcome.vote_counts["A"]
    assert outcome.vote_counts == {"A": 3, "B": 1}


def test_the_outcome_fields_cannot_be_reassigned():
    outcome = tally(group(*[answered("A")] * 5))
    with pytest.raises(Exception):
        outcome.consensus_answer = "B"


def test_the_configured_agents_are_checked_when_the_caller_names_them():
    """Without expected_agents, tally guarantees five identifiers, not which five."""
    strangers = {f"someone_{index}": answered("A") for index in range(5)}

    assert tally(strangers).state == STATE_UNANIMOUS, "five unnamed agents still vote"

    with pytest.raises(VotingError, match="not the configured agents"):
        tally(strangers, expected_agents=AGENTS)


def test_a_swapped_agent_is_caught_when_the_expected_set_is_given():
    responses = dict(zip(AGENTS, [answered("A")] * 5))
    responses["agent_impostor"] = responses.pop("agent_gemma")

    with pytest.raises(VotingError, match="agent_gemma"):
        tally(responses, expected_agents=AGENTS)


def test_the_configured_five_pass_the_check():
    outcome = tally(dict(zip(AGENTS, [answered("A")] * 5)), expected_agents=AGENTS)
    assert outcome.state == STATE_UNANIMOUS


def test_an_expected_set_that_is_not_five_agents_is_refused():
    responses = dict(zip(AGENTS, [answered("A")] * 5))
    with pytest.raises(VotingError, match="expected_agents"):
        tally(responses, expected_agents=AGENTS[:4])
