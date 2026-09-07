"""Builds the versioned prompts sent to all five models.

Three rules this file exists to protect:
  - Round 1 models must not know a debate follows.
  - Round 2 peers must stay anonymous and separate from the agent's own reply.
  - The correct answer must never get in here.

Change a prompt and you change the experiment - bump that round's version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# 1. Settings

ROUND1_PROMPT_VERSION = "round1_v1"

# Kept as the Round 1 alias so existing stored runs and imports remain truthful.
PROMPT_VERSION = ROUND1_PROMPT_VERSION

ROUND2_PROMPT_VERSION = "round2_v1"
ROUND2_MAX_PEERS = 4

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


ROUND2_SYSTEM_PROMPT = """\
You are revisiting a multiple-choice question from a graduate-level academic \
exam.

You previously attempted this question independently. When your previous \
response was valid, it appears earlier in this conversation. You are now being \
shown anonymous responses produced independently by other solvers.

Reconsider the question using all the available reasoning:

1. Re-examine the reasoning in your previous response, if one is available.
2. Evaluate each peer response by checking its principles, calculations, \
evidence and logical steps against the original question.
3. Compare their reasoning with your own. Do not assume an answer is correct \
because it is yours or because several responses agree. Do not change your \
answer only because another response disagrees.
4. Decide which option is best supported. Keeping your previous answer and \
changing it are equally acceptable.

Then respond in exactly this format:

REASONING: <briefly explain your conclusion and whether you kept or changed \
your previous answer, if one was available, at most 200 words>
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

    # Must already be text. Converting a number or None into a string here
    # would build a prompt out of "123" or "None" and never say anything.
    text = question.get("question")
    if not isinstance(text, str) or not text.strip():
        raise PromptConstructionError(
            f"Question {question.get('stable_id', '<unknown>')!r} has no usable text "
            f"(got {type(text).__name__})."
        )

    return f"{text.strip()}\n\n{format_options(_options(question))}"


def build_round1_messages(question: Mapping[str, Any]) -> list[dict[str, str]]:
    """The two messages sent to a model. Identical for all five agents."""
    return [
        {"role": "system", "content": ROUND1_SYSTEM_PROMPT},
        {"role": "user", "content": format_question(question)},
    ]


def build_round2_messages(
    question: Mapping[str, Any],
    own_round1_response: str | None,
    peer_responses: Sequence[str],
) -> list[dict[str, str]]:
    """Continue one agent's conversation with anonymous Round 1 peer replies.

    A valid own response is restored as an ``assistant`` turn, so Round 2 is a
    reconsideration of that agent's earlier position rather than a fresh answer.
    ``None`` means the agent had no usable Round 1 response. Peer identities and
    ordering are owned by the caller; this function preserves the supplied order
    and adds anonymous labels only.
    """
    formatted_question = format_question(question)
    peers = _validate_peer_responses(peer_responses)

    messages = [
        {"role": "system", "content": ROUND2_SYSTEM_PROMPT},
        {"role": "user", "content": formatted_question},
    ]

    if own_round1_response is not None:
        if not isinstance(own_round1_response, str) or not own_round1_response.strip():
            raise PromptConstructionError(
                "own_round1_response must be non-empty text or None when unavailable"
            )
        # Preserve the model's response exactly. It is conversation history, not
        # an anonymous peer response and not text reconstructed from its letter.
        messages.append({"role": "assistant", "content": own_round1_response})
        previous_note = "Your complete previous response appears above."
        final_instruction = (
            "Reconsider your previous response and answer the original question again."
        )
    else:
        previous_note = "Your previous attempt did not produce a usable response."
        final_instruction = "Answer the original question using the available reasoning."

    if peers:
        peer_blocks = "\n\n".join(
            f"--- PEER RESPONSE {index} ---\n{text}"
            for index, text in enumerate(peers, start=1)
        )
        peer_section = (
            "Here are anonymous responses from other independent solvers. "
            "Their order carries no meaning.\n\n"
            f"{peer_blocks}"
        )
    else:
        peer_section = "No valid peer responses are available."

    messages.append(
        {
            "role": "user",
            "content": f"{previous_note}\n\n{peer_section}\n\n{final_instruction}",
        }
    )
    return messages


def _validate_peer_responses(peer_responses: Sequence[str]) -> list[str]:
    """Return up to four complete peer replies, rejecting malformed input."""
    if isinstance(peer_responses, (str, bytes)) or not isinstance(peer_responses, Sequence):
        raise PromptConstructionError("peer_responses must be a sequence of response texts")

    peers = list(peer_responses)
    if len(peers) > ROUND2_MAX_PEERS:
        raise PromptConstructionError(
            f"Round 2 accepts at most {ROUND2_MAX_PEERS} peer responses, got {len(peers)}"
        )
    for index, text in enumerate(peers, start=1):
        if not isinstance(text, str) or not text.strip():
            raise PromptConstructionError(
                f"peer response {index} must be non-empty text"
            )
    return peers


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
