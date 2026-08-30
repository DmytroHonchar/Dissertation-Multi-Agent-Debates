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
    FINAL_ANSWER_MARKER,
    PROMPT_VERSION,
    ROUND1_SYSTEM_PROMPT,
    AnswerLeakageError,
    PromptConstructionError,
    answer_letters,
    build_round1_messages,
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
    assert PROMPT_VERSION == "round1_v1"


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
