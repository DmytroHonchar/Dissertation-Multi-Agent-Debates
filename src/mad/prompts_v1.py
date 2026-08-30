"""Builds the Round 1 prompt sent to all five models.

Two rules this file exists to protect:
  - Round 1 models must not know a debate follows.
  - The correct answer must never get in here.

Change the prompt text and you change the experiment - bump PROMPT_VERSION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# 1. Settings

PROMPT_VERSION = "round1_v1"

# Questions have 3 to 10 options - never assume 10.
ANSWER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The parser looks for this exact text. Both files import it so they can't drift.
FINAL_ANSWER_MARKER = "FINAL ANSWER:"

# If a question carries any of these, an answer key has leaked in.
FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "answer_index",
        "answer_key",
        "correct",
        "correct_answer",
        "correct_index",
        "correct_letter",
        "cot_content",
        "gold",
        "label",
        "solution",
        "target",
    }
)


# 2. The prompt

ROUND1_SYSTEM_PROMPT = """\
You are answering a multiple-choice question from a graduate-level academic \
exam. Questions come from mathematics, physics, chemistry, biology, health, \
economics, business, law, psychology, philosophy, history and engineering.

Work through the question in this order:

1. Identify what is being asked, and which field and concept it belongs to.
2. State the specific principle, rule, formula or definition that applies.
3. Apply it and derive the answer before looking at the options.
4. Compare each option against what you derived. Eliminate those that \
contradict it. Several options will be close to correct but wrong in one \
detail - find that detail.
5. Select the single best remaining option.

Then respond in exactly this format:

REASONING: <your reasoning, following the steps above, at most 200 words>
FINAL ANSWER: <a single letter>

The letter must be one of the options offered. Give exactly one letter, with no \
brackets or punctuation. You must choose an option even if you are not certain. \
Write nothing after the answer line."""


# 3. Errors


class PromptConstructionError(ValueError):
    """The question is broken and no prompt can be built from it."""


class AnswerLeakageError(RuntimeError):
    """The correct answer reached this file. Stop the run."""


# 4. Safety


def assert_no_answer_key(question: Mapping[str, Any]) -> None:
    """Refuse to build a prompt from anything carrying the answer.

    Questions and answers live in separate files, so this should never fire.
    It catches the case where something merges them together in memory before
    the prompt is built. That bug would leave no trace in the results, so this
    crashes rather than warns.
    """
    leaked = sorted(FORBIDDEN_FIELDS.intersection(question))
    if leaked:
        raise AnswerLeakageError(
            f"Answer key reached prompt construction via {leaked}. "
            "Model inputs must never carry an answer."
        )


# 5. Building the prompt


def answer_letters(question: Mapping[str, Any]) -> list[str]:
    """Which letters this question offers, e.g. A to G."""
    return [ANSWER_LETTERS[index] for index in range(len(_options(question)))]


def format_options(options: Sequence[str]) -> str:
    """Turn the option list into lettered lines: 'A. text'."""
    return "\n".join(
        f"{ANSWER_LETTERS[index]}. {text.strip()}" for index, text in enumerate(options)
    )


def format_question(question: Mapping[str, Any]) -> str:
    """The part the model sees: question text, blank line, then the options."""
    assert_no_answer_key(question)

    text = str(question.get("question", "")).strip()
    if not text:
        raise PromptConstructionError(
            f"Question {question.get('stable_id', '<unknown>')!r} has no text."
        )

    return f"{text}\n\n{format_options(_options(question))}"


def build_round1_messages(question: Mapping[str, Any]) -> list[dict[str, str]]:
    """The two messages sent to a model. Identical for all five agents."""
    return [
        {"role": "system", "content": ROUND1_SYSTEM_PROMPT},
        {"role": "user", "content": format_question(question)},
    ]


def _options(question: Mapping[str, Any]) -> list[str]:
    """Pull out the option list, refusing anything malformed."""
    stable_id = question.get("stable_id", "<unknown>")
    options = question.get("options")

    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise PromptConstructionError(f"Question {stable_id!r} has no option list.")

    options = list(options)
    if len(options) < 2:
        raise PromptConstructionError(
            f"Question {stable_id!r} has {len(options)} option(s); at least 2 are needed."
        )
    if len(options) > len(ANSWER_LETTERS):
        raise PromptConstructionError(
            f"Question {stable_id!r} has {len(options)} options; "
            f"only {len(ANSWER_LETTERS)} letters are available."
        )

    for index, text in enumerate(options):
        if not isinstance(text, str) or not text.strip():
            raise PromptConstructionError(
                f"Question {stable_id!r} option {ANSWER_LETTERS[index]} is empty."
            )

    return options
