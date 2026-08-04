from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet
import yaml


class BenchmarkDataError(ValueError):
    """Raised when benchmark data or its stable IDs are invalid."""


VALIDATION_REASONS = (
    "empty_question_text",
    "invalid_options",
    "empty_option",
    "duplicate_options",
    "invalid_correct_answer",
)

EXPERIMENTAL_QUESTION_COUNT = 300
PILOT_QUESTION_COUNT = 20
SAMPLING_SEED = 42


class FrozenArtifactExistsError(FileExistsError):
    """Raised when freezing would replace an existing permanent artifact."""


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


def validate_question(question: Mapping[str, Any]) -> list[str]:
    """Return every validation failure found in one identified question."""
    failures: list[str] = []

    question_text = question.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        failures.append("empty_question_text")

    options = question.get("options")
    valid_options = (
        isinstance(options, list)
        and bool(options)
        and all(isinstance(option, str) for option in options)
    )
    if not valid_options:
        failures.append("invalid_options")
        option_count = 0
    else:
        stripped_options = [option.strip() for option in options]
        option_count = len(options)
        if any(not option for option in stripped_options):
            failures.append("empty_option")
        if len(stripped_options) != len(set(stripped_options)):
            failures.append("duplicate_options")

    if not _correct_answer_is_valid(question, option_count):
        failures.append("invalid_correct_answer")

    return failures


def separate_valid_questions(
    questions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate valid and broken question copies without changing the source rows."""
    validate_stable_ids(questions)
    valid_questions: list[dict[str, Any]] = []
    broken_questions: list[dict[str, Any]] = []

    for question in questions:
        failures = validate_question(question)
        question_copy = dict(question)
        if failures:
            question_copy["validation_failures"] = failures
            broken_questions.append(question_copy)
        else:
            valid_questions.append(question_copy)

    return valid_questions, broken_questions


def validate_benchmark_and_save_report(
    config_path: str | Path,
    report_path: str | Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Validate all local splits and save preparation metadata as JSON."""
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    loaded_splits = load_benchmark_questions(resolved_config_path)
    valid_splits: dict[str, list[dict[str, Any]]] = {}
    broken_splits: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "benchmark": config["benchmark"]["name"],
        "dataset_version": config["dataset"]["local_version"],
        "splits": {},
    }

    for split, loaded_questions in loaded_splits.items():
        valid_questions, broken_questions = separate_valid_questions(
            loaded_questions
        )
        valid_splits[split] = valid_questions
        broken_splits[split] = broken_questions

        reason_counts = Counter(
            reason
            for question in broken_questions
            for reason in question["validation_failures"]
        )
        report["splits"][split] = {
            "questions_loaded": len(loaded_questions),
            "questions_valid": len(valid_questions),
            "unique_questions_removed": len(broken_questions),
            "removed_by_reason": {
                reason: reason_counts[reason] for reason in VALIDATION_REASONS
            },
            "removed_questions": [
                {
                    "stable_id": question["stable_id"],
                    "failure_reasons": question["validation_failures"],
                }
                for question in broken_questions
            ],
        }

    resolved_report_path = Path(report_path)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")

    return valid_splits, broken_splits, report


def select_stratified_question_sets(
    valid_questions: Sequence[Mapping[str, Any]],
    *,
    experimental_count: int = EXPERIMENTAL_QUESTION_COUNT,
    pilot_count: int = PILOT_QUESTION_COUNT,
    seed: int = SAMPLING_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select balanced, disjoint question sets in one deterministic operation."""
    validate_stable_ids(valid_questions)
    categories = sorted(
        {
            question["category"]
            for question in valid_questions
            if isinstance(question.get("category"), str)
            and question["category"].strip()
        }
    )
    if len(categories) != 14:
        raise BenchmarkDataError(
            f"Expected 14 non-empty MMLU-Pro categories, found {len(categories)}"
        )

    experimental_quotas = _balanced_category_quotas(
        categories, experimental_count
    )
    pilot_quotas = _balanced_category_quotas(categories, pilot_count)
    random_generator = random.Random(seed)
    experimental_questions: list[dict[str, Any]] = []
    pilot_questions: list[dict[str, Any]] = []

    for category in categories:
        category_questions = sorted(
            (
                dict(question)
                for question in valid_questions
                if question.get("category") == category
            ),
            key=lambda question: question["stable_id"],
        )
        random_generator.shuffle(category_questions)

        experimental_quota = experimental_quotas[category]
        pilot_quota = pilot_quotas[category]
        required_count = experimental_quota + pilot_quota
        if len(category_questions) < required_count:
            raise BenchmarkDataError(
                f"Category {category!r} has {len(category_questions)} valid "
                f"questions but {required_count} are required"
            )

        experimental_questions.extend(category_questions[:experimental_quota])
        pilot_questions.extend(
            category_questions[
                experimental_quota : experimental_quota + pilot_quota
            ]
        )

    random_generator.shuffle(experimental_questions)
    random_generator.shuffle(pilot_questions)
    _validate_selected_sets(
        experimental_questions,
        pilot_questions,
        experimental_count=experimental_count,
        pilot_count=pilot_count,
    )
    return experimental_questions, pilot_questions


def freeze_mmlu_pro_question_sets(
    config_path: str | Path,
    *,
    frozen_root: str | Path | None = None,
    seed: int = SAMPLING_SEED,
    creation_time: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create protected model inputs, answer keys, IDs, manifest and checksums."""
    resolved_config_path = Path(config_path).resolve()
    repository_root = resolved_config_path.parents[2]
    with resolved_config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_version = config["dataset"]["local_version"]
    resolved_frozen_root = (
        Path(frozen_root)
        if frozen_root is not None
        else repository_root / "data/frozen" / dataset_version
    )
    target_paths = _frozen_target_paths(resolved_frozen_root)
    _require_absent_frozen_targets(target_paths)

    _verify_raw_file_checksums(config, repository_root)
    loaded_splits = load_benchmark_questions(resolved_config_path)
    if config["splits"]["primary"] != "test":
        raise BenchmarkDataError("The configured primary split must be 'test'")
    valid_test_questions, broken_test_questions = separate_valid_questions(
        loaded_splits["test"]
    )
    validation_report_path = (
        repository_root
        / "data/frozen"
        / dataset_version
        / "metadata/validation_report.json"
    )
    _verify_validation_report(
        validation_report_path,
        loaded_splits["test"],
        valid_test_questions,
        broken_test_questions,
    )

    experimental_questions, pilot_questions = select_stratified_question_sets(
        valid_test_questions,
        seed=seed,
    )
    benchmark_name = config["benchmark"]["name"]
    experimental_model_inputs = [
        _model_input_record(question, benchmark_name)
        for question in experimental_questions
    ]
    pilot_model_inputs = [
        _model_input_record(question, benchmark_name)
        for question in pilot_questions
    ]
    experimental_answers = [
        _answer_key_record(question) for question in experimental_questions
    ]
    pilot_answers = [_answer_key_record(question) for question in pilot_questions]
    experimental_ids = [
        question["stable_id"] for question in experimental_questions
    ]
    pilot_ids = [question["stable_id"] for question in pilot_questions]

    artifact_paths = {
        name: _display_path(path, repository_root)
        for name, path in target_paths.items()
    }
    raw_checksums = {
        file_config["path"]: file_config["sha256"]
        for _, file_config in sorted(config["files"].items())
    }
    manifest = {
        "benchmark": benchmark_name,
        "dataset_source": config["dataset"]["url"],
        "dataset_revision": config["dataset"]["revision"],
        "dataset_split": "test",
        "raw_file_checksums": raw_checksums,
        "random_seed": seed,
        "sampling_method": {
            "name": "deterministic_stratified_without_replacement",
            "category_order": "lexicographic",
            "category_allocation": (
                "floor(set_size/category_count), with remainder assigned "
                "to the earliest sorted categories"
            ),
            "question_order_before_shuffle": "stable_id lexicographic",
            "random_generator": "Python random.Random(seed), MT19937",
            "selection_order": (
                "single shuffle per category; experimental slice followed by "
                "non-overlapping pilot slice; each completed set then shuffled"
            ),
        },
        "selection": {
            "experimental": {
                "count": len(experimental_questions),
                "per_category": _category_counts(experimental_questions),
            },
            "pilot": {
                "count": len(pilot_questions),
                "per_category": _category_counts(pilot_questions),
            },
        },
        "creation_time_utc": creation_time
        or datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "stable_id_method": {
            "source_field": "question_id",
            "format": "mmlu_pro_v1:<split>:<question_id>",
            "assigned_before_validation": True,
        },
        "artifacts": artifact_paths,
    }

    artifact_contents: dict[Path, tuple[bytes, str, int | None]] = {
        target_paths["experimental_model_inputs"]: (
            _canonical_jsonl_bytes(experimental_model_inputs),
            "canonical_jsonl",
            len(experimental_model_inputs),
        ),
        target_paths["pilot_model_inputs"]: (
            _canonical_jsonl_bytes(pilot_model_inputs),
            "canonical_jsonl",
            len(pilot_model_inputs),
        ),
        target_paths["experimental_answer_key"]: (
            _canonical_jsonl_bytes(experimental_answers),
            "canonical_jsonl",
            len(experimental_answers),
        ),
        target_paths["pilot_answer_key"]: (
            _canonical_jsonl_bytes(pilot_answers),
            "canonical_jsonl",
            len(pilot_answers),
        ),
        target_paths["experimental_ids"]: (
            _canonical_json_bytes(experimental_ids),
            "canonical_json",
            len(experimental_ids),
        ),
        target_paths["pilot_ids"]: (
            _canonical_json_bytes(pilot_ids),
            "canonical_json",
            len(pilot_ids),
        ),
        target_paths["manifest"]: (
            _canonical_json_bytes(manifest),
            "canonical_json",
            None,
        ),
    }
    for path in sorted(artifact_contents, key=str):
        content, _, _ = artifact_contents[path]
        _write_new_file(path, content)

    checksums = {
        "algorithm": "sha256",
        "canonicalization": {
            "encoding": "UTF-8",
            "json": (
                "sorted object keys, compact separators, ensure_ascii=false, "
                "one trailing LF"
            ),
            "jsonl": "one canonical JSON object per line, LF line endings",
            "scope": "exact file bytes",
        },
        "files": {
            _display_path(path, repository_root): {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "format": file_format,
                **({"records": record_count} if record_count is not None else {}),
            }
            for path, (content, file_format, record_count) in sorted(
                artifact_contents.items(), key=lambda item: str(item[0])
            )
        },
        "self_checksum": "excluded because a file cannot contain its own hash",
    }
    _write_new_file(
        target_paths["checksums"], _canonical_json_bytes(checksums)
    )
    return manifest, checksums


def _format_original_id(value: Any, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise BenchmarkDataError(f"Invalid value in original ID field {field_name!r}")
    formatted_value = str(value)
    if not formatted_value.strip():
        raise BenchmarkDataError(f"Empty value in original ID field {field_name!r}")
    return formatted_value


def _balanced_category_quotas(
    categories: Sequence[str], total_count: int
) -> dict[str, int]:
    base_count, remainder = divmod(total_count, len(categories))
    return {
        category: base_count + (position < remainder)
        for position, category in enumerate(categories)
    }


def _validate_selected_sets(
    experimental_questions: Sequence[Mapping[str, Any]],
    pilot_questions: Sequence[Mapping[str, Any]],
    *,
    experimental_count: int,
    pilot_count: int,
) -> None:
    if len(experimental_questions) != experimental_count:
        raise BenchmarkDataError("Incorrect experimental question count")
    if len(pilot_questions) != pilot_count:
        raise BenchmarkDataError("Incorrect pilot question count")

    experimental_ids = [
        question["stable_id"] for question in experimental_questions
    ]
    pilot_ids = [question["stable_id"] for question in pilot_questions]
    if len(experimental_ids) != len(set(experimental_ids)):
        raise BenchmarkDataError("Duplicate experimental stable IDs")
    if len(pilot_ids) != len(set(pilot_ids)):
        raise BenchmarkDataError("Duplicate pilot stable IDs")
    if set(experimental_ids) & set(pilot_ids):
        raise BenchmarkDataError("Experimental and pilot sets overlap")


def _category_counts(
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(question["category"] for question in questions)
    return {category: counts[category] for category in sorted(counts)}


def _model_input_record(
    question: Mapping[str, Any], benchmark_name: str
) -> dict[str, Any]:
    return {
        "stable_id": question["stable_id"],
        "benchmark": benchmark_name,
        "category": question["category"],
        "question": question["question"],
        "options": list(question["options"]),
    }


def _answer_key_record(question: Mapping[str, Any]) -> dict[str, str]:
    return {
        "stable_id": question["stable_id"],
        "correct_answer": question["answer"],
    }


def _frozen_target_paths(frozen_root: Path) -> dict[str, Path]:
    return {
        "experimental_model_inputs": (
            frozen_root / "model_inputs/experimental_questions.jsonl"
        ),
        "pilot_model_inputs": frozen_root / "model_inputs/pilot_questions.jsonl",
        "experimental_answer_key": (
            frozen_root / "answer_keys/experimental_answers.jsonl"
        ),
        "pilot_answer_key": frozen_root / "answer_keys/pilot_answers.jsonl",
        "experimental_ids": frozen_root / "metadata/experimental_ids.json",
        "pilot_ids": frozen_root / "metadata/pilot_ids.json",
        "manifest": frozen_root / "metadata/manifest.json",
        "checksums": frozen_root / "metadata/checksums.json",
    }


def _require_absent_frozen_targets(target_paths: Mapping[str, Path]) -> None:
    existing_paths = sorted(
        path for path in target_paths.values() if path.exists()
    )
    if existing_paths:
        formatted_paths = ", ".join(str(path) for path in existing_paths)
        raise FrozenArtifactExistsError(
            f"Refusing to replace existing frozen artifacts: {formatted_paths}"
        )


def _verify_raw_file_checksums(
    config: Mapping[str, Any], repository_root: Path
) -> None:
    mismatches: list[str] = []
    for split, file_config in sorted(config["files"].items()):
        raw_path = repository_root / file_config["path"]
        actual_checksum = _sha256_file(raw_path)
        if actual_checksum != file_config["sha256"]:
            mismatches.append(split)
    if mismatches:
        raise BenchmarkDataError(
            f"Raw Parquet checksum mismatch for splits: {mismatches}"
        )


def _verify_validation_report(
    report_path: Path,
    loaded_questions: Sequence[Mapping[str, Any]],
    valid_questions: Sequence[Mapping[str, Any]],
    broken_questions: Sequence[Mapping[str, Any]],
) -> None:
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    test_report = report["splits"]["test"]
    broken_ids = {question["stable_id"] for question in broken_questions}
    reported_broken_ids = {
        question["stable_id"] for question in test_report["removed_questions"]
    }
    if (
        test_report["questions_loaded"] != len(loaded_questions)
        or test_report["questions_valid"] != len(valid_questions)
        or test_report["unique_questions_removed"] != len(broken_questions)
        or reported_broken_ids != broken_ids
    ):
        raise BenchmarkDataError(
            "Validation report does not match the current valid test pool"
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _write_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output_file:
            output_file.write(content)
    except FileExistsError as error:
        raise FrozenArtifactExistsError(
            f"Refusing to replace existing frozen artifact: {path}"
        ) from error


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correct_answer_is_valid(question: Mapping[str, Any], option_count: int) -> bool:
    answer_indices: list[int] = []

    if "answer_index" in question:
        answer_index = question.get("answer_index")
        if isinstance(answer_index, bool) or not isinstance(answer_index, int):
            return False
        answer_indices.append(answer_index)

    if "answer" in question:
        answer = question.get("answer")
        if not isinstance(answer, str):
            return False
        answer_letter = answer.strip().upper()
        if len(answer_letter) != 1 or not "A" <= answer_letter <= "Z":
            return False
        answer_indices.append(ord(answer_letter) - ord("A"))

    return (
        bool(answer_indices)
        and len(set(answer_indices)) == 1
        and 0 <= answer_indices[0] < option_count
    )


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
