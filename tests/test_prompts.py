"""Tests for the Round 1 prompt.

The prompt gets frozen after the pilot, so these tests pin down the things that
must stay true: no answer leaks in, no hint that a debate follows, and the
option count always comes from the data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mad.prompts_v1 import (
    DEBATE_PROMPT_VERSION,
    FINAL_ANSWER_MARKER,
    PROMPT_VERSION,
    ROUND1_PROMPT_VERSION,
    ROUND2_PROMPT_VERSION,
    ROUND1_SYSTEM_PROMPT,
    ROUND2_SYSTEM_PROMPT,
    AnswerLeakageError,
    PromptConstructionError,
    answer_letters,
    build_round1_messages,
    build_round2_messages,
    format_options,
    format_question,
)

FROZEN_INPUTS = (
    Path(__file__).resolve().parents[1] / "data/frozen/mmlu_pro_v1/model_inputs"
)


def question(**overrides):
    base = {
        "stable_id": "mmlu_pro_v1:test:1",
        "benchmark": "MMLU-Pro",
        "category": "biology",
        "question": "What is the primary function of the mitochondria?",
        "options": ["Protein synthesis", "ATP production", "Lipid storage"],
    }
    base.update(overrides)
    return base


def frozen_questions(filename: str):
    path = FROZEN_INPUTS / filename
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- option formatting -------------------------------------------------------


def test_options_are_lettered_in_stored_order():
    rendered = format_options(["first", "second", "third"])
    assert rendered == "A. first\nB. second\nC. third"


@pytest.mark.parametrize("count", [3, 4, 5, 6, 7, 8, 9, 10])
def test_option_count_is_never_hardcoded(count):
    """MMLU-Pro questions carry 3 to 10 options; the count comes from the data."""
    q = question(options=[f"option {i}" for i in range(count)])
    lines = format_options(q["options"]).splitlines()
    assert len(lines) == count
    assert lines[-1].startswith("ABCDEFGHIJ"[count - 1] + ".")
    assert answer_letters(q) == list("ABCDEFGHIJ"[:count])


def test_every_frozen_question_renders():
    """No question in either frozen set may fail to build a prompt."""
    for filename in ("experimental_questions.jsonl", "pilot_questions.jsonl"):
        for q in frozen_questions(filename):
            rendered = format_question(q)
            assert q["question"].strip() in rendered
            assert len(answer_letters(q)) == len(q["options"])


def test_frozen_questions_span_several_option_counts():
    """Guards against a regression that assumes ten options everywhere."""
    counts = {len(q["options"]) for q in frozen_questions("experimental_questions.jsonl")}
    assert len(counts) > 1
    assert min(counts) < 10


# --- message construction ----------------------------------------------------


def test_round1_messages_are_system_then_user():
    messages = build_round1_messages(question())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == ROUND1_SYSTEM_PROMPT


def test_prompt_demands_the_marker_the_parser_looks_for():
    """Prompt and parser must agree on the anchor, or nothing parses."""
    assert FINAL_ANSWER_MARKER in ROUND1_SYSTEM_PROMPT


def test_prompt_forces_a_choice_when_uncertain():
    """A hedge produces no vote, and the three-of-five threshold never drops."""
    assert "even if you are not certain" in ROUND1_SYSTEM_PROMPT


def test_round1_never_reveals_that_a_second_round_follows():
    """D003: Round 1 answers must be independent and uninformed."""
    lowered = ROUND1_SYSTEM_PROMPT.lower()
    for leak in (
        "debate",
        "other agent",
        "other model",
        "peer",
        "round 2",
        "second round",
        "later round",
        "revise",
        "reconsider",
        "will be reviewed",
    ):
        assert leak not in lowered, f"Round 1 prompt mentions {leak!r}"


def test_prompt_version_is_recorded():
    assert PROMPT_VERSION == ROUND1_PROMPT_VERSION == "round1_v1"


def test_two_round_run_records_both_prompt_versions():
    assert DEBATE_PROMPT_VERSION == "round1_v1+round2_v1"


# --- Round 2 ---------------------------------------------------------------


def test_round2_restores_the_agents_complete_previous_response():
    own = "REASONING: My original argument.\nFINAL ANSWER: B"
    messages = build_round2_messages(
        question(), own, ["REASONING: Peer argument.\nFINAL ANSWER: A"]
    )

    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "user"
    ]
    assert messages[0]["content"] == ROUND2_SYSTEM_PROMPT
    assert messages[2]["content"] == own
    assert ROUND2_PROMPT_VERSION == "round2_v1"


def test_round2_labels_peers_anonymously_and_preserves_their_order():
    first = "REASONING: first peer\nFINAL ANSWER: A"
    second = "REASONING: second peer\nFINAL ANSWER: C"
    messages = build_round2_messages(question(), "my reply", [first, second])
    follow_up = messages[-1]["content"]

    assert "PEER RESPONSE 1" in follow_up
    assert "PEER RESPONSE 2" in follow_up
    assert follow_up.index(first) < follow_up.index(second)
    assert "agent_" not in follow_up
    assert "model" not in follow_up.lower()


def test_round2_includes_the_original_question_and_options():
    messages = build_round2_messages(question(), "my reply", [])
    assert messages[1]["content"] == format_question(question())


def test_round2_without_a_valid_previous_response_does_not_invent_one():
    messages = build_round2_messages(question(), None, ["one valid peer response"])
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "did not produce a usable response" in messages[-1]["content"]


def test_round2_runs_with_no_valid_peers():
    messages = build_round2_messages(question(), "my reply", [])
    assert "No valid peer responses are available" in messages[-1]["content"]
    assert "answer the original question again" in messages[-1]["content"]


def test_round2_uses_the_same_final_answer_contract_as_round1():
    assert FINAL_ANSWER_MARKER in ROUND2_SYSTEM_PROMPT
    assert "at most 200 words" in ROUND2_SYSTEM_PROMPT
    assert "Keeping your previous answer" in ROUND2_SYSTEM_PROMPT


def test_round2_refuses_more_than_four_peers():
    with pytest.raises(PromptConstructionError, match="at most 4"):
        build_round2_messages(question(), "my reply", ["reply"] * 5)


@pytest.mark.parametrize("peers", ["one string", [""], ["ok", 42]])
def test_round2_refuses_malformed_peer_responses(peers):
    with pytest.raises(PromptConstructionError):
        build_round2_messages(question(), "my reply", peers)


@pytest.mark.parametrize("own", ["", "   ", 42])
def test_round2_refuses_a_malformed_own_response(own):
    with pytest.raises(PromptConstructionError):
        build_round2_messages(question(), own, [])


def test_round2_refuses_a_question_carrying_an_answer():
    with pytest.raises(AnswerLeakageError):
        build_round2_messages(
            question(correct_answer="B"), "my reply", ["one valid peer response"]
        )


# --- answer leakage ----------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["answer", "answer_index", "correct_answer", "cot_content", "label"]
)
def test_a_question_carrying_an_answer_is_refused(field):
    with pytest.raises(AnswerLeakageError):
        build_round1_messages(question(**{field: "B"}))


def test_no_frozen_question_text_contains_the_answer_marker():
    """A question containing the marker would let the parser read the wrong line."""
    for filename in ("experimental_questions.jsonl", "pilot_questions.jsonl"):
        for q in frozen_questions(filename):
            assert FINAL_ANSWER_MARKER not in format_question(q).upper()


# --- malformed input ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"question": ""},
        {"question": "   "},
        {"options": []},
        {"options": ["only one"]},
        {"options": ["fine", ""]},
        {"options": ["fine", "   "]},
        {"options": "not a list"},
        {"options": ["fine", 42]},
    ],
)
def test_malformed_questions_are_refused(bad):
    with pytest.raises(PromptConstructionError):
        build_round1_messages(question(**bad))


def test_more_options_than_letters_is_refused():
    with pytest.raises(PromptConstructionError):
        build_round1_messages(question(options=[f"o{i}" for i in range(27)]))


@pytest.mark.parametrize("text", [None, 123, 12.5, ["a"], ""])
def test_a_question_whose_text_is_not_a_string_is_rejected(text):
    """Coercing this would send the model a prompt saying "None" or "123"."""
    with pytest.raises(PromptConstructionError):
        format_question({"stable_id": "q1", "question": text, "options": ["a", "b"]})
