"""Evaluate the one accepted 300-question experiment and export result tables.

This command is offline and read-only with respect to experimental data. It
opens the completed results database, joins the separate frozen answer key,
requires exactly 300 questions, and writes Markdown, JSON and CSV reports. It
never imports the API client, calls a model, changes a response or recomputes a
stored vote.

Default command:

    .venv/bin/python scripts/evaluate_experiment.py

The accepted run is deliberately fixed below. Use ``--overwrite`` only to
regenerate the same deterministic report files after reviewing code changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.database import ResultsDatabase  # noqa: E402
from mad.evaluation import (  # noqa: E402
    EvaluationError,
    EvaluationReport,
    evaluate_run,
    load_answer_key,
)
from mad.parser_v1 import (  # noqa: E402
    STATUS_API_ERROR,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN_ID = "experiment_agents_v7_20260923T114934Z"
FROZEN_SETTINGS_VERSION = "agents_v7"
EXPECTED_QUESTIONS = 300
DEFAULT_DATABASE = REPO_ROOT / "storage" / "results.sqlite"
DEFAULT_ANSWER_KEY = (
    REPO_ROOT
    / "data"
    / "frozen"
    / "mmlu_pro_v1"
    / "answer_keys"
    / "experimental_answers.jsonl"
)
DEFAULT_OUTPUT_DIRECTORY = REPO_ROOT / "reports" / "main_experiment_20260923"

FAILURE_STATUSES = (
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
    STATUS_PARSE_FAIL,
    STATUS_API_ERROR,
)

OUTPUT_FILES = (
    "evaluation_report.md",
    "evaluation_summary.json",
    "group_accuracy.csv",
    "agent_accuracy.csv",
    "group_transitions.csv",
    "agent_transitions.csv",
    "consensus_states.csv",
    "usage.csv",
    "question_comparisons.csv",
)


class ReportingError(RuntimeError):
    """The accepted experiment cannot be exported safely."""


# 1. Read-only evaluation


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wall_clock_hours(run: Mapping[str, Any]) -> float | None:
    started = run.get("started_at")
    ended = run.get("ended_at")
    if not started or not ended:
        return None
    return (
        datetime.fromisoformat(str(ended)) - datetime.fromisoformat(str(started))
    ).total_seconds() / 3600.0


def evaluate_database(
    database_path: Path,
    answer_key_path: Path,
) -> tuple[EvaluationReport, dict[str, Any], str]:
    """Return the verified report, run metadata and unchanged database hash."""
    if not database_path.is_file():
        raise ReportingError(f"results database does not exist: {database_path}")
    if not answer_key_path.is_file():
        raise ReportingError(f"experimental answer key does not exist: {answer_key_path}")

    before = _sha256(database_path)
    answer_key = load_answer_key(answer_key_path)
    with ResultsDatabase(database_path) as database:
        run = database.read_run(MAIN_RUN_ID)
        if run is None:
            raise ReportingError(
                f"accepted main run {MAIN_RUN_ID!r} is not in {database_path}"
            )
        if run["settings_version"] != FROZEN_SETTINGS_VERSION:
            raise ReportingError(
                f"accepted run claims settings {run['settings_version']!r}, not "
                f"{FROZEN_SETTINGS_VERSION!r}"
            )
        report = evaluate_run(
            database,
            MAIN_RUN_ID,
            answer_key,
            expected_questions=EXPECTED_QUESTIONS,
        )
    after = _sha256(database_path)
    if after != before:
        raise ReportingError(
            "opening the results database for evaluation changed its bytes; "
            "no report was written"
        )
    return report, run, before


# 2. Structured rows


def _accuracy_percent(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _transition_dict(counts) -> dict[str, int]:
    return {
        "stayed_correct": counts.stayed_correct,
        "became_incorrect": counts.became_incorrect,
        "became_correct": counts.became_correct,
        "stayed_incorrect": counts.stayed_incorrect,
    }


def _agent_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_number in (1, 2):
        for result in report.agent_results[round_number]:
            row = {
                "round": round_number,
                "agent_id": result.agent_id,
                "responses": result.responses,
                "valid_answers": result.valid_answers,
                "correct": result.correct,
                "accuracy_percent": _accuracy_percent(result.accuracy),
                "failure_count": result.failure_count,
                "failure_rate_percent": result.failure_rate * 100.0,
            }
            row.update(
                {status.lower(): int(result.failures.get(status, 0)) for status in FAILURE_STATUSES}
            )
            rows.append(row)
    return rows


def _group_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    return [
        {
            "round": round_number,
            "questions": result.questions,
            "correct": result.correct,
            "accuracy_percent": result.accuracy * 100.0,
            "decided": result.decided,
            "undecided": result.undecided,
        }
        for round_number, result in sorted(report.group_results.items())
    ]


def _group_transition_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    return [
        {"transition": name, "count": value}
        for name, value in _transition_dict(report.group_transitions).items()
    ]


def _agent_transition_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    rows = []
    for result in report.agent_transitions:
        row = {"agent_id": result.agent_id, **_transition_dict(result.counts)}
        row["excluded_questions"] = result.excluded_questions
        rows.append(row)
    return rows


def _consensus_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    rows = []
    for round_number, result in sorted(report.group_results.items()):
        for state, count in result.consensus_states.items():
            rows.append({"round": round_number, "state": state, "count": count})
    return rows


def _usage_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    return [asdict(report.usage[round_number]) for round_number in (1, 2)]


def _question_rows(report: EvaluationReport) -> list[dict[str, Any]]:
    rows = []
    for comparison in report.question_comparisons:
        rows.append(
            {
                "question_id": comparison.question_id,
                "complete_case": comparison.complete_case,
                "round1_state": comparison.round1.consensus_state,
                "round1_answer": comparison.round1.consensus_answer,
                "round1_correct": comparison.round1.correct,
                "round1_failures": "; ".join(
                    f"{agent}:{status}" for agent, status in comparison.round1.failures
                ),
                "round2_state": comparison.round2.consensus_state,
                "round2_answer": comparison.round2.consensus_answer,
                "round2_correct": comparison.round2.correct,
                "round2_failures": "; ".join(
                    f"{agent}:{status}" for agent, status in comparison.round2.failures
                ),
            }
        )
    return rows


def _summary_payload(
    report: EvaluationReport,
    run: Mapping[str, Any],
    database_sha256: str,
) -> dict[str, Any]:
    complete = report.complete_cases
    return {
        "provenance": {
            "run": dict(run),
            "database_sha256_at_evaluation": database_sha256,
            "expected_questions": EXPECTED_QUESTIONS,
            "evaluation_version": report.evaluation_version,
            "wall_clock_hours": _wall_clock_hours(run),
        },
        "group_accuracy": _group_rows(report),
        "agent_accuracy": _agent_rows(report),
        "group_transitions": _transition_dict(report.group_transitions),
        "agent_transitions": _agent_transition_rows(report),
        "aggregation": {
            **asdict(report.aggregation),
            "group_accuracy_percent": report.aggregation.group_accuracy * 100.0,
            "mean_agent_accuracy_percent": _accuracy_percent(
                report.aggregation.mean_agent_accuracy
            ),
            "best_agent_accuracy_percent": _accuracy_percent(
                report.aggregation.best_agent_accuracy
            ),
        },
        "debate": {
            "effect_points": report.debate_effect_points,
            "bootstrap": asdict(report.bootstrap),
            "mcnemar": asdict(report.mcnemar),
        },
        "complete_cases": {
            "question_count": complete.question_count,
            "round1_accuracy_percent": _accuracy_percent(complete.round1_accuracy),
            "round2_accuracy_percent": _accuracy_percent(complete.round2_accuracy),
            "difference_points": complete.difference_points,
            "transitions": _transition_dict(complete.transitions),
            "included_question_ids": list(complete.included_question_ids),
            "excluded_question_ids": list(complete.excluded_question_ids),
        },
        "consensus_states": _consensus_rows(report),
        "usage": _usage_rows(report),
        "budget": asdict(report.budget),
        "questions": _question_rows(report),
    }


# 3. Human-readable report


def _display_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100.0:.1f}%"


def _display_points(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _markdown_report(
    report: EvaluationReport,
    run: Mapping[str, Any],
    database_sha256: str,
) -> str:
    first, second = report.group_results[1], report.group_results[2]
    complete = report.complete_cases
    lines = [
        "# Main experiment evaluation",
        "",
        f"- Run: `{report.run_id}`",
        f"- Settings: `{run['settings_version']}`",
        f"- Evaluation: `{report.evaluation_version}`",
        f"- Questions: {report.question_count}",
        f"- Started: `{run['started_at']}`",
        f"- Finished: `{run['ended_at']}`",
        f"- Wall clock: {_wall_clock_hours(run):.2f} hours",
        f"- Database SHA-256 at evaluation: `{database_sha256}`",
        "",
        "## Headline results",
        "",
        "| Measure | Correct | Accuracy |",
        "|---|---:|---:|",
        f"| Round 1 group vote | {first.correct}/{first.questions} | {first.accuracy * 100:.1f}% |",
        f"| Round 2 group vote | {second.correct}/{second.questions} | {second.accuracy * 100:.1f}% |",
        f"| Change after debate | {second.correct - first.correct:+d} net | {report.debate_effect_points:+.2f} points |",
        "",
        "## Paired statistical result",
        "",
        f"- 95% paired-bootstrap interval: `[{report.bootstrap.low_points:+.2f}, {report.bootstrap.high_points:+.2f}]` points",
        f"- Bootstrap resamples: {report.bootstrap.resamples:,}; seed: `{report.bootstrap.seed}`",
        f"- McNemar exact p-value: `{report.mcnemar.p_value:.6f}`",
        f"- Became correct: {report.mcnemar.became_correct}; became incorrect: {report.mcnemar.became_incorrect}",
        "",
        "## Group transitions",
        "",
        "| Transition | Questions |",
        "|---|---:|",
    ]
    for name, value in _transition_dict(report.group_transitions).items():
        lines.append(f"| {name.replace('_', ' ').title()} | {value} |")

    lines.extend(
        [
            "",
            "## Per-agent accuracy and failures",
            "",
            "Accuracy is calculated over each agent's valid answers only. It does not share the group's fixed denominator.",
            "",
            "| Round | Agent | Correct / valid | Accuracy | Refusal | Truncated | Parse fail | API error |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for round_number in (1, 2):
        for result in report.agent_results[round_number]:
            lines.append(
                f"| {round_number} | {result.agent_id} | {result.correct}/{result.valid_answers} "
                f"| {_display_percent(result.accuracy)} | "
                f"{result.failures.get(STATUS_REFUSAL, 0)} | "
                f"{result.failures.get(STATUS_TRUNCATED, 0)} | "
                f"{result.failures.get(STATUS_PARSE_FAIL, 0)} | "
                f"{result.failures.get(STATUS_API_ERROR, 0)} |"
            )

    lines.extend(
        [
            "",
            "## Aggregation comparison",
            "",
            f"- Round 1 group accuracy: {report.aggregation.group_accuracy * 100:.1f}%",
            f"- Mean agent valid-answer accuracy: {_display_percent(report.aggregation.mean_agent_accuracy)}",
            f"- Group versus mean: {_display_points(report.aggregation.versus_mean_points)} points",
            f"- Best valid-answer accuracy: {_display_percent(report.aggregation.best_agent_accuracy)} ({report.aggregation.best_agent_id})",
            f"- Group versus best valid-answer accuracy: {_display_points(report.aggregation.versus_best_points)} points",
            "",
            "The group and individual figures use different denominators; this comparison does not mean that the named best agent solved more of the complete 300-question set.",
            "",
            "## Consensus states",
            "",
            "| State | Round 1 | Round 2 |",
            "|---|---:|---:|",
        ]
    )
    states = list(first.consensus_states)
    for state in states:
        lines.append(
            f"| {state} | {first.consensus_states[state]} | {second.consensus_states[state]} |"
        )

    lines.extend(
        [
            "",
            "## Complete-case supplement",
            "",
            f"- Questions with all five agents valid in both rounds: {complete.question_count}",
            f"- Round 1: {_display_percent(complete.round1_accuracy)}",
            f"- Round 2: {_display_percent(complete.round2_accuracy)}",
            f"- Change: {_display_points(complete.difference_points)} points",
            f"- Transitions: {complete.transitions.stayed_correct} stayed correct, "
            f"{complete.transitions.became_correct} became correct, "
            f"{complete.transitions.became_incorrect} became incorrect, "
            f"{complete.transitions.stayed_incorrect} stayed incorrect.",
            "",
            "This selected subset supplements rather than replaces the primary 300-question result.",
            "",
            "## Usage",
            "",
            "| Round | Prompt tokens | Completion tokens | Cost | API attempts | Retried responses |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for round_number in (1, 2):
        usage = report.usage[round_number]
        lines.append(
            f"| {round_number} | {usage.prompt_tokens:,} | {usage.completion_tokens:,} "
            f"| ${usage.cost_usd:.6f} | {usage.api_attempts} | {usage.retried_responses} |"
        )
    lines.extend(
        [
            "",
            f"Total run cost: **${report.budget.run_usd:.6f}**.",
            "",
            "## Interpretation boundary",
            "",
            "The measured Round 2 minus Round 1 change is the observed effect of the full implemented system. It is not automatically a pure causal effect of communication: Round 2 adds another inference pass and failures differ between rounds. Provider failures remain reported evidence and are not repaired by rerunning the experiment.",
            "",
        ]
    )
    return "\n".join(lines)


# 4. Writing exports


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ReportingError(f"refusing to write empty table {path.name}")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def export_report(
    report: EvaluationReport,
    run: Mapping[str, Any],
    database_sha256: str,
    output_directory: Path,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    existing = [output_directory / name for name in OUTPUT_FILES if (output_directory / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise ReportingError(
            f"report files already exist in {output_directory}: {names}. "
            "Pass --overwrite to regenerate them."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    payload = _summary_payload(report, run, database_sha256)
    (output_directory / "evaluation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "evaluation_report.md").write_text(
        _markdown_report(report, run, database_sha256), encoding="utf-8"
    )
    _write_csv(output_directory / "group_accuracy.csv", _group_rows(report))
    _write_csv(output_directory / "agent_accuracy.csv", _agent_rows(report))
    _write_csv(
        output_directory / "group_transitions.csv", _group_transition_rows(report)
    )
    _write_csv(
        output_directory / "agent_transitions.csv", _agent_transition_rows(report)
    )
    _write_csv(output_directory / "consensus_states.csv", _consensus_rows(report))
    _write_csv(output_directory / "usage.csv", _usage_rows(report))
    _write_csv(
        output_directory / "question_comparisons.csv", _question_rows(report)
    )
    return tuple(output_directory / name for name in OUTPUT_FILES)


# 5. Command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and export the accepted 300-question experiment offline."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--answer-key", type=Path, default=DEFAULT_ANSWER_KEY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the known generated report files",
    )
    args = parser.parse_args(argv)

    try:
        report, run, database_sha256 = evaluate_database(args.db, args.answer_key)
        written = export_report(
            report,
            run,
            database_sha256,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (EvaluationError, ReportingError, OSError, json.JSONDecodeError) as error:
        print(f"refused: {error}")
        return 1

    first, second = report.group_results[1], report.group_results[2]
    print("offline evaluation complete — no API calls and no database changes")
    print(f"run: {report.run_id}")
    print(
        f"group accuracy: R1 {first.correct}/{first.questions} "
        f"({first.accuracy * 100:.1f}%) -> R2 {second.correct}/{second.questions} "
        f"({second.accuracy * 100:.1f}%)"
    )
    print(
        f"debate effect: {report.debate_effect_points:+.2f} points; "
        f"95% CI [{report.bootstrap.low_points:+.2f}, "
        f"{report.bootstrap.high_points:+.2f}]; "
        f"McNemar p={report.mcnemar.p_value:.6f}"
    )
    print(f"reports: {args.output_dir}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
