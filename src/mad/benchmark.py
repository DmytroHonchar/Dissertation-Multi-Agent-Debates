from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import yaml


class BenchmarkDataError(ValueError):
    """Raised when benchmark questions cannot receive valid stable IDs."""


def load_benchmark_questions(
    config_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load every locally configured split and immediately assign stable IDs."""
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        dataset_version = config["dataset"]["local_version"]
        configured_files = config["files"]
    except (KeyError, TypeError) as error:
        raise BenchmarkDataError("Invalid benchmark configuration") from error

    repository_root = resolved_config_path.parents[2]
    loaded_splits: dict[str, list[dict[str, Any]]] = {}
    for split, file_config in configured_files.items():
        try:
            configured_path = Path(file_config["path"])
        except (KeyError, TypeError) as error:
            raise BenchmarkDataError(f"Missing local path for split {split!r}") from error

        parquet_path = (
            configured_path
            if configured_path.is_absolute()
            else repository_root / configured_path
        )
        loaded_splits[split] = load_questions(
            parquet_path,
            dataset_version=dataset_version,
            split=split,
        )

    return loaded_splits


def load_questions(
    parquet_path: str | Path,
    *,
    dataset_version: str,
    split: str,
) -> list[dict[str, Any]]:
    """Load one local Parquet split and assign IDs before returning any rows."""
    table = parquet.read_table(Path(parquet_path))
    original_id_field = "question_id" if "question_id" in table.column_names else None
    return assign_stable_ids(
        table.to_pylist(),
        dataset_version=dataset_version,
        split=split,
        original_id_field=original_id_field,
    )


def assign_stable_ids(
    questions: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str,
    split: str,
    original_id_field: str | None,
) -> list[dict[str, Any]]:
    """Return question copies with deterministic, validated stable IDs."""
    identified_questions: list[dict[str, Any]] = []

    for question in questions:
        identified_question = dict(question)
        if original_id_field is not None:
            original_id = _format_original_id(
                identified_question.get(original_id_field), original_id_field
            )
            stable_suffix = original_id
        else:
            stable_suffix = _question_content_hash(identified_question)

        identified_question["stable_id"] = (
            f"{dataset_version}:{split}:{stable_suffix}"
        )
        identified_questions.append(identified_question)

    validate_stable_ids(identified_questions)
    return identified_questions


def validate_stable_ids(questions: Sequence[Mapping[str, Any]]) -> None:
    """Require a non-empty collection of present and unique stable IDs."""
    if not questions:
        raise BenchmarkDataError("The benchmark split contains no questions")

    stable_ids = [question.get("stable_id") for question in questions]
    missing_positions = [
        position
        for position, stable_id in enumerate(stable_ids)
        if not isinstance(stable_id, str) or not stable_id
    ]
    if missing_positions:
        raise BenchmarkDataError(
            f"Missing stable IDs at positions: {missing_positions[:10]}"
        )

    duplicate_ids = [
        stable_id
        for stable_id, count in Counter(stable_ids).items()
        if count > 1
    ]
    if duplicate_ids:
        raise BenchmarkDataError(
            f"Duplicate stable IDs found: {duplicate_ids[:10]}"
        )


def _format_original_id(value: Any, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise BenchmarkDataError(f"Invalid value in original ID field {field_name!r}")
    formatted_value = str(value)
    if not formatted_value.strip():
        raise BenchmarkDataError(f"Empty value in original ID field {field_name!r}")
    return formatted_value


def _question_content_hash(question: Mapping[str, Any]) -> str:
    question_text = question.get("question")
    options = question.get("options")
    if not isinstance(question_text, str):
        raise BenchmarkDataError("Question text is required for content-based IDs")
    if (
        not isinstance(options, Sequence)
        or isinstance(options, (str, bytes))
        or not all(isinstance(option, str) for option in options)
    ):
        raise BenchmarkDataError("Ordered string options are required for content-based IDs")

    content = json.dumps(
        [question_text, list(options)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()
