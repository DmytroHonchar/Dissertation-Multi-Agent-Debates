"""Tests that the 300 and the 20 are picked the same way every time.

Seed 42, no overlap between the two sets, every subject represented.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from mad.benchmark import (
    FrozenArtifactExistsError,
    freeze_mmlu_pro_question_sets,
    load_benchmark_questions,
    select_stratified_question_sets,
    separate_valid_questions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark/mmlu_pro_v1.yaml"
FROZEN_ROOT = REPOSITORY_ROOT / "data/frozen/mmlu_pro_v1"


def _load_valid_test_pool() -> tuple[list[dict], list[dict]]:
    loaded = load_benchmark_questions(CONFIG_PATH)
    return separate_valid_questions(loaded["test"])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_seeded_stratified_selection_is_valid_and_deterministic() -> None:
    valid_questions, broken_questions = _load_valid_test_pool()
    first_experimental, first_pilot = select_stratified_question_sets(
        valid_questions, seed=42
    )
    second_experimental, second_pilot = select_stratified_question_sets(
        list(reversed(valid_questions)), seed=42
    )

    valid_ids = {question["stable_id"] for question in valid_questions}
    broken_ids = {question["stable_id"] for question in broken_questions}
    experimental_ids = [
        question["stable_id"] for question in first_experimental
    ]
    pilot_ids = [question["stable_id"] for question in first_pilot]

    assert len(experimental_ids) == 300
    assert len(pilot_ids) == 20
    assert len(experimental_ids) == len(set(experimental_ids))
    assert len(pilot_ids) == len(set(pilot_ids))
    assert not set(experimental_ids) & set(pilot_ids)
    assert set(experimental_ids) | set(pilot_ids) <= valid_ids
    assert not broken_ids & (set(experimental_ids) | set(pilot_ids))
    assert experimental_ids == [
        question["stable_id"] for question in second_experimental
    ]
    assert pilot_ids == [
        question["stable_id"] for question in second_pilot
    ]

    for selected_questions in (first_experimental, first_pilot):
        category_counts = Counter(
            question["category"] for question in selected_questions
        )
        assert len(category_counts) == 14
        assert max(category_counts.values()) - min(category_counts.values()) <= 1


def test_frozen_ids_and_answer_keys_have_exact_coverage() -> None:
    experimental_ids = json.loads(
        (FROZEN_ROOT / "metadata/experimental_ids.json").read_text()
    )
    pilot_ids = json.loads((FROZEN_ROOT / "metadata/pilot_ids.json").read_text())
    experimental_inputs = _read_jsonl(
        FROZEN_ROOT / "model_inputs/experimental_questions.jsonl"
    )
    pilot_inputs = _read_jsonl(
        FROZEN_ROOT / "model_inputs/pilot_questions.jsonl"
    )
    experimental_answers = _read_jsonl(
        FROZEN_ROOT / "answer_keys/experimental_answers.jsonl"
    )
    pilot_answers = _read_jsonl(
        FROZEN_ROOT / "answer_keys/pilot_answers.jsonl"
    )

    assert experimental_ids == [
        question["stable_id"] for question in experimental_inputs
    ]
    assert pilot_ids == [question["stable_id"] for question in pilot_inputs]
    assert {answer["stable_id"] for answer in experimental_answers} == set(
        experimental_ids
    )
    assert {answer["stable_id"] for answer in pilot_answers} == set(pilot_ids)
    assert all(
        set(answer) == {"stable_id", "correct_answer"}
        for answer in experimental_answers + pilot_answers
    )
    assert len(experimental_answers) == len(experimental_ids) == 300
    assert len(pilot_answers) == len(pilot_ids) == 20


def test_frozen_checksums_and_raw_checksums_can_be_recalculated() -> None:
    checksums = json.loads(
        (FROZEN_ROOT / "metadata/checksums.json").read_text()
    )
    for relative_path, expected in checksums["files"].items():
        content = (REPOSITORY_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
        assert len(content) == expected["bytes"]

    config = yaml.safe_load(CONFIG_PATH.read_text())
    for file_config in config["files"].values():
        content = (REPOSITORY_ROOT / file_config["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == file_config["sha256"]


def test_existing_frozen_files_cannot_be_overwritten(tmp_path: Path) -> None:
    frozen_root = tmp_path / "mmlu_pro_v1"
    existing_path = (
        frozen_root / "model_inputs/experimental_questions.jsonl"
    )
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("do not replace\n")

    with pytest.raises(FrozenArtifactExistsError, match="Refusing to replace"):
        freeze_mmlu_pro_question_sets(CONFIG_PATH, frozen_root=frozen_root)

    assert existing_path.read_text() == "do not replace\n"
    assert [path for path in frozen_root.rglob("*") if path.is_file()] == [
        existing_path
    ]
