"""Proves the frozen question files carry no answers.

This is the most important test in the project. If it ever fails, every
result produced afterwards is worthless.
"""

import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_INPUT_DIRECTORY = (
    REPOSITORY_ROOT / "data/frozen/mmlu_pro_v1/model_inputs"
)
ALLOWED_MODEL_INPUT_FIELDS = {
    "stable_id",
    "benchmark",
    "category",
    "question",
    "options",
}


@pytest.mark.parametrize(
    "filename",
    ["experimental_questions.jsonl", "pilot_questions.jsonl"],
)
def test_frozen_model_inputs_contain_only_safe_fields(filename: str) -> None:
    path = MODEL_INPUT_DIRECTORY / filename
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert records
    for record in records:
        assert set(record) == ALLOWED_MODEL_INPUT_FIELDS


# Round 2 sends far more text than Round 1 - the question, the agent's own
# reply and four peer replies - so the same guarantee is asserted again there.


def _pilot_question():
    from mad.runner import load_pilot_question

    return load_pilot_question("mmlu_pro_v1:test:7296")


@pytest.mark.parametrize("field", sorted({"answer", "answer_index", "cot_content", "correct_answer"}))
def test_a_poisoned_question_cannot_build_a_round2_prompt(field: str) -> None:
    from mad.prompts_v1 import AnswerLeakageError, build_round2_messages

    poisoned = dict(_pilot_question())
    poisoned[field] = "A"

    with pytest.raises(AnswerLeakageError):
        build_round2_messages(poisoned, "FINAL ANSWER: A", ["FINAL ANSWER: B"])


def test_no_forbidden_field_name_reaches_a_round2_prompt() -> None:
    """The whole conversation, not just the question, is checked."""
    from mad.prompts_v1 import FORBIDDEN_FIELDS, build_round2_messages

    messages = build_round2_messages(
        _pilot_question(),
        "REASONING: mine.\nFINAL ANSWER: A",
        [f"REASONING: peer {index}.\nFINAL ANSWER: B" for index in range(4)],
    )
    whole = "\n".join(message["content"] for message in messages)

    assert set(_pilot_question()).isdisjoint(FORBIDDEN_FIELDS)
    for forbidden in {"answer_index", "cot_content", "correct_answer", "answer_key"}:
        assert forbidden not in whole
