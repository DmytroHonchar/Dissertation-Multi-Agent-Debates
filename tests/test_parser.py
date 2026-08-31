"""Guards the parser rules. Offline - no model is ever called."""

from __future__ import annotations

import pytest

from mad.parser_v1 import (
    EXTRACTION_LAST_MATCH,
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
    api_error,
    parse_response,
)

TEN = list("ABCDEFGHIJ")
FOUR = list("ABCD")


def reply(answer: str, reasoning: str = "The compound is an alkene.") -> str:
    return f"REASONING: {reasoning}\nFINAL ANSWER: {answer}"


# 1. Answers that must be accepted


@pytest.mark.parametrize("answer", TEN)
def test_every_letter_of_a_ten_option_question_parses(answer):
    result = parse_response(reply(answer), TEN, finish_reason="stop")
    assert result.status == STATUS_OK
    assert result.letter == answer
    assert result.extraction_method == EXTRACTION_LAST_MATCH
    assert result.is_vote


@pytest.mark.parametrize(
    "text",
    [
        "FINAL ANSWER: c",
        "final answer: C",
        "Final Answer: C",
        "FINAL ANSWER:C",
        "FINAL ANSWER:    C",
        "FINAL   ANSWER : C",
        "REASONING: ...\n\nFINAL ANSWER: C\n",
    ],
)
def test_case_and_spacing_are_tolerated(text):
    assert parse_response(text, FOUR, finish_reason="stop").letter == "C"


@pytest.mark.parametrize(
    "text",
    [
        "**FINAL ANSWER:** B",
        "**FINAL ANSWER: B**",
        "FINAL ANSWER: **B**",
        "FINAL ANSWER: B.",
        "FINAL ANSWER: B, because it is the only stable option.",
    ],
)
def test_markdown_and_trailing_punctuation_do_not_lose_an_answer(text):
    assert parse_response(text, FOUR, finish_reason="stop").letter == "B"


def test_text_after_the_answer_line_is_allowed():
    text = "FINAL ANSWER: B\nI hope this helps!"
    assert parse_response(text, FOUR, finish_reason="stop").letter == "B"


def test_the_last_answer_line_wins():
    text = "FINAL ANSWER: A\nOn reflection that is wrong.\nFINAL ANSWER: D"
    assert parse_response(text, FOUR, finish_reason="stop").letter == "D"


def test_the_reasoning_is_never_used_to_infer_an_answer():
    # Reasons toward C, answers A. The answer is A.
    text = "REASONING: Option C is clearly correct.\nFINAL ANSWER: A"
    assert parse_response(text, FOUR, finish_reason="stop").letter == "A"


# 2. Answers that must be rejected


@pytest.mark.parametrize(
    "text",
    [
        "FINAL ANSWER: [B]",
        "FINAL ANSWER: (B)",
        "FINAL ANSWER: Berlin",
        "FINAL ANSWER: BC",
        "FINAL ANSWER: B2",
        "FINAL ANSWER: B_x",
        "The answer is B.",
        "REASONING: it is clearly B.",
        "Yes.",                      # Mistral did exactly this in the smoke test
        "   ",
        "",
    ],
)
def test_unusable_replies_are_parse_fail(text):
    result = parse_response(text, FOUR, finish_reason="stop")
    assert result.status == STATUS_PARSE_FAIL
    assert result.letter is None
    assert not result.is_vote


def test_a_letter_outside_the_option_set_is_rejected():
    # Valid syntax, but this question only offers A to D.
    result = parse_response(reply("H"), FOUR, finish_reason="stop")
    assert result.status == STATUS_PARSE_FAIL
    assert "A-D" in result.note


def test_the_same_letter_is_fine_on_a_question_that_offers_it():
    assert parse_response(reply("H"), TEN, finish_reason="stop").letter == "H"


# 3. Failure statuses


@pytest.mark.parametrize("reason", ["length", "max_tokens"])
def test_finish_reason_marks_truncation(reason):
    result = parse_response("REASONING: the compound is an", TEN, finish_reason=reason)
    assert result.status == STATUS_TRUNCATED


def test_truncation_beats_a_letter_that_is_present():
    result = parse_response(reply("B"), FOUR, finish_reason="length")
    assert result.status == STATUS_TRUNCATED
    assert result.letter is None


def test_truncation_is_guessed_only_when_no_finish_reason_came_back():
    cut_off = "REASONING: " + "the compound is an alkene and therefore " * 8
    assert parse_response(cut_off, FOUR, finish_reason="").status == STATUS_TRUNCATED
    assert parse_response(cut_off, FOUR, finish_reason="stop").status == STATUS_PARSE_FAIL


@pytest.mark.parametrize("text", ["Yes", "Yes.", "B", "I think so"])
def test_a_short_reply_is_never_guessed_to_be_truncated(text):
    # Truncation means the 1024-token limit was hit, so the reply is long.
    # A short reply is a parse failure - a different column in the failure table.
    assert parse_response(text, FOUR, finish_reason="").status == STATUS_PARSE_FAIL


def test_the_note_records_the_finish_reason_that_was_actually_returned():
    assert "max_tokens" in parse_response("x", FOUR, finish_reason="max_tokens").note
    assert "length" in parse_response("x", FOUR, finish_reason="length").note


def test_a_complete_answer_with_no_finish_reason_is_not_called_truncated():
    assert parse_response(reply("B"), FOUR, finish_reason="").status == STATUS_OK


@pytest.mark.parametrize(
    "text",
    [
        "I cannot answer this question.",
        "I'm unable to answer that.",
        "I won't answer this.",
    ],
)
def test_refusal_text_is_detected(text):
    assert parse_response(text, FOUR, finish_reason="stop").status == STATUS_REFUSAL


def test_a_refusal_phrase_does_not_override_an_answer_that_was_given():
    # The phrase search is not sentence-aware, so it can appear inside a reply
    # that does answer. A provider flag is trusted over a letter; text is not.
    text = "I cannot answer A, so FINAL ANSWER: B"
    assert parse_response(text, FOUR, finish_reason="stop").letter == "B"


def test_a_refusal_with_no_answer_is_still_a_refusal():
    text = "I cannot answer this question. It is outside what I will discuss."
    result = parse_response(text, FOUR, finish_reason="stop")
    assert result.status == STATUS_REFUSAL
    assert result.letter is None


def test_provider_refusal_signals_are_honoured():
    assert parse_response(reply("B"), FOUR, finish_reason="content_filter").status == STATUS_REFUSAL
    assert parse_response(reply("B"), FOUR, refusal_signal=True).status == STATUS_REFUSAL


def test_ordinary_reasoning_is_not_mistaken_for_a_refusal():
    # "cannot" appears constantly in real reasoning. It must not cost an answer.
    text = "REASONING: The value cannot be negative, so option B is out.\nFINAL ANSWER: C"
    assert parse_response(text, FOUR, finish_reason="stop").status == STATUS_OK


def test_api_error_carries_its_reason():
    result = api_error("mistral failed after 2 attempts: HTTP 429")
    assert result.status == STATUS_API_ERROR
    assert result.letter is None
    assert not result.is_vote
    assert "429" in result.note


# 4. Nothing here may ever count as a wrong answer


@pytest.mark.parametrize(
    "result",
    [
        parse_response("", FOUR, finish_reason="stop"),
        parse_response("Yes.", FOUR, finish_reason="stop"),
        parse_response(reply("B"), FOUR, finish_reason="length"),
        parse_response("I cannot answer this one.", FOUR, finish_reason="stop"),
        api_error("timeout"),
    ],
)
def test_no_failure_ever_contributes_a_vote(result):
    assert result.status != STATUS_OK
    assert result.letter is None
    assert not result.is_vote
