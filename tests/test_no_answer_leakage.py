"""Proves the frozen question files carry no answers.

These tests check the frozen question files and every file that could reach the
answer key. Run them alone with:

    .venv/bin/python -m pytest tests/test_no_answer_leakage.py

This is the most important test in the project. If it ever fails, every
result produced afterwards is worthless.
"""

import ast
import json
import re
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


# Round 2 sends much more text than Round 1, so the same guarantee is asserted again there.


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


# Only one module reads the answer key. No other module on the run path may know where it lives.


def test_only_evaluation_and_the_freezing_code_touch_the_answer_key_directory() -> None:
    """benchmark.py wrote the keys once; evaluation.py is the only reader.

    If a new module ever names that directory, this fails and someone has to
    justify it. The rule matters more than the convenience.
    """
    package = REPOSITORY_ROOT / "src" / "mad"
    allowed = {"benchmark.py", "evaluation.py"}

    offenders = sorted(
        module.name
        for module in package.glob("*.py")
        if module.name not in allowed and "answer_keys" in module.read_text()
    )

    assert offenders == [], (
        f"{offenders} reference the answer-key directory. Only evaluation.py may "
        "read it, and only benchmark.py may write it."
    )


def test_the_run_path_never_imports_the_evaluation_module() -> None:
    """Prompt-building and model-calling code must not reach the one reader."""
    package = REPOSITORY_ROOT / "src" / "mad"
    run_path = ("prompts_v1.py", "parser_v1.py", "api_client.py", "cache.py",
                "round1.py", "debate.py", "runner.py", "voting.py", "database.py")

    # An import, not a mention: voting.py names the module in a comment about not scoring.
    imports = re.compile(r"^\s*(?:from\s+mad\.evaluation\b|import\s+mad\.evaluation\b)", re.M)
    offenders = sorted(
        name for name in run_path if imports.search((package / name).read_text())
    )

    assert offenders == [], f"{offenders} import evaluation, which reads the answer key"


def test_no_script_that_calls_models_reads_the_answer_key() -> None:
    """Scripts are the other place the key could sneak into a calling process.

    A run script stores results; scoring happens afterwards in a separate
    process. Keeping the key out of anything that can reach OpenRouter is
    cheaper than proving, later, that it never left.
    """
    scripts = REPOSITORY_ROOT / "scripts"

    def code_without_the_docstring(source: str) -> str:
        """A docstring may name the key file; code may not touch it."""
        tree = ast.parse(source)
        if ast.get_docstring(tree) is not None:
            tree.body = tree.body[1:]
        return ast.unparse(tree)

    offenders = []
    for script in sorted(scripts.glob("*.py")):
        source = script.read_text()
        if "OpenRouterClient" not in source:
            continue
        code = code_without_the_docstring(source)
        if "answer_keys" in code or "load_answer_key" in code:
            offenders.append(script.name)

    assert offenders == [], (
        f"{offenders} both call models and read the answer key. Score a stored "
        "run in a separate process instead."
    )
