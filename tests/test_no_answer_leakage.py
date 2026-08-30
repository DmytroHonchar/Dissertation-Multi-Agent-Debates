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
