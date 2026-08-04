from pathlib import Path

import pytest

from mad.benchmark import BenchmarkDataError, assign_stable_ids, load_benchmark_questions


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/benchmark/mmlu_pro_v1.yaml"


def test_local_questions_have_unique_deterministic_stable_ids() -> None:
    first_load = load_benchmark_questions(CONFIG_PATH)
    second_load = load_benchmark_questions(CONFIG_PATH)

    assert {split: len(rows) for split, rows in first_load.items()} == {
        "test": 12_032,
        "validation": 70,
    }

    for split, questions in first_load.items():
        stable_ids = [question["stable_id"] for question in questions]
        assert all(stable_ids)
        assert len(stable_ids) == len(set(stable_ids))
        assert stable_ids == [
            f"mmlu_pro_v1:{split}:{question['question_id']}"
            for question in questions
        ]
        assert stable_ids == [
            question["stable_id"] for question in second_load[split]
        ]


def test_content_hash_fallback_uses_only_question_and_ordered_options() -> None:
    first_question = {
        "question": "Which option is correct?",
        "options": ["first", "second"],
        "answer": "first",
        "answer_index": 0,
    }
    changed_answer = {
        **first_question,
        "answer": "second",
        "answer_index": 1,
    }
    reordered_options = {
        **first_question,
        "options": ["second", "first"],
    }

    first_id = assign_stable_ids(
        [first_question],
        dataset_version="mmlu_pro_v1",
        split="test",
        original_id_field=None,
    )[0]["stable_id"]
    changed_answer_id = assign_stable_ids(
        [changed_answer],
        dataset_version="mmlu_pro_v1",
        split="test",
        original_id_field=None,
    )[0]["stable_id"]
    reordered_options_id = assign_stable_ids(
        [reordered_options],
        dataset_version="mmlu_pro_v1",
        split="test",
        original_id_field=None,
    )[0]["stable_id"]

    assert first_id == changed_answer_id
    assert first_id != reordered_options_id


def test_duplicate_original_ids_are_rejected() -> None:
    duplicate_questions = [
        {"question_id": 7, "question": "First", "options": ["A", "B"]},
        {"question_id": 7, "question": "Second", "options": ["A", "B"]},
    ]

    with pytest.raises(BenchmarkDataError, match="Duplicate stable IDs"):
        assign_stable_ids(
            duplicate_questions,
            dataset_version="mmlu_pro_v1",
            split="test",
            original_id_field="question_id",
        )
