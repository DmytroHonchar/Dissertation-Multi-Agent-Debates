"""Reads the answer letter out of a model reply and labels what went wrong.

One parser for all five agents. If each model got its own reading rules, the
results would measure how forgiving the parser was, not how good the models are.

Two things it never does:
  - guess an answer from the reasoning
  - treat a failure as a wrong answer

The raw text is stored elsewhere, whole. This file only reads it.

Change any rule here and you change the results - bump PARSER_VERSION.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from mad.prompts_v1 import FINAL_ANSWER_MARKER


# 1. Settings

PARSER_VERSION = "parser_v1"

# The five outcomes a response can have. Stored as-is in the results database.
STATUS_OK = "OK"
STATUS_REFUSAL = "REFUSAL"
STATUS_TRUNCATED = "TRUNCATED"
STATUS_PARSE_FAIL = "PARSE_FAIL"
STATUS_API_ERROR = "API_ERROR"

ALL_STATUSES = frozenset(
    {STATUS_OK, STATUS_REFUSAL, STATUS_TRUNCATED, STATUS_PARSE_FAIL, STATUS_API_ERROR}
)

# How the letter was found. Only one method exists, but it is recorded so a
# second one can never be added silently.
EXTRACTION_LAST_MATCH = "LAST_FINAL_ANSWER_MATCH"

# Built from the marker in prompts_v1, so the prompt and the parser can't drift
# apart. Comes out as: FINAL\s+ANSWER\s*:\s*([A-Za-z])(?![A-Za-z])
#   \s+ and \s*  - tolerate odd spacing
#   ([A-Za-z]) - one bare letter, so "FINAL ANSWER: [B]" does not match
#   (?!\w)     - nothing word-like after it, so "Berlin", "BC" and "B2" do not
ANSWER_LINE_PATTERN = re.compile(
    r"\s+".join(FINAL_ANSWER_MARKER.rstrip(":").split())
    + r"\s*:\s*([A-Za-z])(?!\w)",
    re.IGNORECASE,
)

# Providers use these finish reasons. Anything else is treated as a normal stop.
FINISH_REASON_TRUNCATED = frozenset({"length", "max_tokens"})
FINISH_REASON_REFUSAL = frozenset({"content_filter"})

# A reply is only called a refusal on one of these. The list is deliberately
# short - a phrase that could show up in real reasoning would throw away valid
# answers, which is far worse than missing a refusal.
REFUSAL_PHRASES = (
    "i cannot answer",
    "i can't answer",
    "i cannot assist",
    "i can't assist",
    "i cannot provide an answer",
    "i can't provide an answer",
    "i will not answer",
    "i won't answer",
    "i am unable to answer",
    "i'm unable to answer",
    "unable to provide an answer",
)

# When no finish reason comes back, truncation has to be guessed. A truncated
# reply ran into the 1024-token limit, so it is long and stops mid-sentence.
# Both conditions are required: "Yes" is short junk, not a cut-off answer.
# This is weak evidence and must be checked against the pilot transcripts.
MIN_TRUNCATION_LENGTH = 200
SENTENCE_ENDINGS = tuple(".!?\"')]}")


# 2. What comes back


@dataclass(frozen=True)
class ParsedResponse:
    """The outcome of reading one reply."""

    status: str
    letter: str | None = None       # uppercase, and a real option for this question
    extraction_method: str | None = None
    note: str = ""                  # short reason, for debugging and the failure tables

    @property
    def is_vote(self) -> bool:
        """True only if this response contributes a vote. Failures never do."""
        return self.status == STATUS_OK and self.letter is not None


# 3. Reading a reply


def parse_response(
    text: str | None,
    valid_letters: Sequence[str],
    finish_reason: str = "",
    refusal_signal: bool = False,
) -> ParsedResponse:
    """Read one reply. Returns a letter and a status, never an exception.

    valid_letters are the options this question actually offers, e.g. A to G.
    A letter outside that set is not an answer.

    Checks run in a fixed order, because a reply can look like more than one
    kind of failure at once:
      1. refusal    - the model declined
      2. truncated  - it ran out of tokens mid-sentence
      3. empty      - nothing came back
      4. the letter - found, or not found

    Refusal and truncation win even when a letter is present. A cut-off answer
    is not a real answer, and scoring it would inflate the results.
    """
    body = (text or "").strip()
    reason = (finish_reason or "").strip().lower()
    allowed = {letter.upper() for letter in valid_letters}

    # Provider signals come first. They are hard evidence and win over a letter.
    if refusal_signal or reason in FINISH_REASON_REFUSAL:
        return ParsedResponse(
            STATUS_REFUSAL, note=f"provider signalled refusal ({reason or 'flagged'})"
        )
    if reason in FINISH_REASON_TRUNCATED:
        return ParsedResponse(STATUS_TRUNCATED, note=f"finish reason {reason!r}")

    if not body:
        return ParsedResponse(STATUS_PARSE_FAIL, note="empty response")

    # Models bold their answer line constantly ("**FINAL ANSWER:** B"). The
    # asterisks are formatting, not an answer, so drop them before matching.
    # Several answer lines can appear if the model restates itself; the last one
    # is its final word.
    matches = ANSWER_LINE_PATTERN.findall(body.replace("*", ""))
    letter = matches[-1].upper() if matches else None

    # A refusal phrase in the text is much weaker evidence than a provider flag,
    # so it only counts when nothing was answered. "I cannot answer A, so
    # FINAL ANSWER: B" is an answer, not a refusal.
    if letter is None and _is_refusal_text(body):
        return ParsedResponse(STATUS_REFUSAL, note="reply text is a refusal")

    # Guess truncation from the text only when the provider told us nothing.
    if letter is None and not reason and _looks_truncated(body):
        return ParsedResponse(
            STATUS_TRUNCATED, note="no finish reason and reply is long and unfinished"
        )

    if letter is None:
        return ParsedResponse(STATUS_PARSE_FAIL, note=f"no '{FINAL_ANSWER_MARKER} X' line")
    if letter not in allowed:
        return ParsedResponse(
            STATUS_PARSE_FAIL,
            note=f"answered {letter}, but this question offers {_letter_range(allowed)}",
        )

    return ParsedResponse(STATUS_OK, letter=letter, extraction_method=EXTRACTION_LAST_MATCH)


def api_error(note: str) -> ParsedResponse:
    """For a call that failed both attempts. No reply text exists to read.

    Kept here so all five statuses are produced in one file.
    """
    return ParsedResponse(STATUS_API_ERROR, note=note)


# 4. Helpers


def _is_refusal_text(body: str) -> bool:
    """True if the reply says outright that it won't answer."""
    lowered = body.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def _looks_truncated(body: str) -> bool:
    """Guess whether a reply was cut off. Only used when no finish reason came back.

    Long and stopping mid-sentence. A short reply is a parse failure instead -
    claiming truncation without evidence would move it into the wrong column of
    the failure table.
    """
    return len(body) >= MIN_TRUNCATION_LENGTH and not body.endswith(SENTENCE_ENDINGS)


def _letter_range(allowed: set[str]) -> str:
    """Describe the valid options for an error message, e.g. 'A-G'."""
    letters = sorted(allowed)
    if not letters:
        return "no options"
    if len(letters) == 1:
        return letters[0]
    return f"{letters[0]}-{letters[-1]}"
