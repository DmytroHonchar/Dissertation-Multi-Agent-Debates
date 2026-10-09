"""The animated stage may only replay stored evidence, never invent it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mad.debate_stage import (
    LOGO_FILES,
    STAGE_DATA_PLACEHOLDER,
    build_stage_payload,
    load_logos,
    plain_text,
    reasoning_excerpt,
    stage_document,
)
from mad.public_viewer_data import PUBLIC_QUESTION_IDS, load_public_question_replay


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (REPOSITORY_ROOT / "app" / "debate_stage.html").read_text(encoding="utf-8")
SHORT_NAMES = {
    "agent_llama": "Llama",
    "agent_qwen": "Qwen",
    "agent_mistral": "Mistral",
    "agent_deepseek": "DeepSeek",
    "agent_gemma": "Gemma",
}


@pytest.fixture(scope="module", params=PUBLIC_QUESTION_IDS)
def case(request):
    replay = load_public_question_replay(request.param)
    return replay, build_stage_payload(replay, SHORT_NAMES)


def test_every_bubble_quotes_the_stored_response_verbatim(case):
    replay, payload = case
    stored = {r.response_id: r for r in (*replay.round1.responses, *replay.round2.responses)}
    for agent in payload["agents"]:
        for round_key in ("round1", "round2"):
            speech = agent[round_key]
            if speech["excerpt"]:
                assert speech["excerpt"] in plain_text(stored[speech["response_id"]].raw_response)


def test_letters_and_statuses_are_the_stored_ones(case):
    replay, payload = case
    stored = {r.response_id: r for r in (*replay.round1.responses, *replay.round2.responses)}
    for agent in payload["agents"]:
        for round_key in ("round1", "round2"):
            speech = agent[round_key]
            response = stored[speech["response_id"]]
            assert speech["letter"] == response.extracted_letter
            assert speech["status"] == response.status
            if speech["letter"] is None:
                assert speech["no_vote_note"]


def test_a_provider_error_never_appears_as_model_speech():
    replay = load_public_question_replay("mmlu_pro_v1:test:11994")
    payload = build_stage_payload(replay, SHORT_NAMES)
    mistral = next(agent for agent in payload["agents"] if agent["id"] == "agent_mistral")

    assert mistral["round1"]["status"] == "API_ERROR"
    assert mistral["round1"]["excerpt"] == ""
    assert mistral["round1"]["full_text"] is None


def test_vote_board_matches_the_stored_outcomes(case):
    replay, payload = case
    for number, round_replay in (("1", replay.round1), ("2", replay.round2)):
        tally = payload["rounds"][number]
        assert sum(tally["counts"].values()) == round_replay.outcome.valid_answer_count
        assert tally["state"] == round_replay.outcome.consensus_state
        assert tally["answer"] == round_replay.outcome.consensus_answer


def test_peer_arrows_follow_the_stored_round2_rows(case):
    replay, payload = case
    valid_senders = {r.agent_id for r in replay.round1.responses if r.extracted_letter is not None}
    for agent, stored in zip(payload["agents"], replay.round2.responses, strict=True):
        assert len(agent["round2"]["peers_from"]) == len(stored.peer_response_ids)
        assert agent["id"] not in agent["round2"]["peers_from"]
        assert set(agent["round2"]["peers_from"]) <= valid_senders


def test_failed_round1_agents_send_nothing():
    replay = load_public_question_replay("mmlu_pro_v1:test:11994")
    payload = build_stage_payload(replay, SHORT_NAMES)
    senders = {sender for agent in payload["agents"] for sender in agent["round2"]["peers_from"]}

    assert senders == {"agent_llama", "agent_deepseek"}


def test_excerpt_drops_the_answer_line_and_cuts_at_a_word():
    raw = "REASONING: **First** step. " + "word " * 80 + "\nFINAL ANSWER: B"
    excerpt, cut = reasoning_excerpt(raw, limit=60)

    assert cut
    assert excerpt.startswith("First step.")
    assert "FINAL ANSWER" not in excerpt
    assert not excerpt.endswith("wor")


def test_stored_text_cannot_break_out_of_the_data_block():
    hostile = {"question": "</script><script>alert(1)</script>"}
    document = stage_document(hostile, TEMPLATE)

    assert document.count("</script>") == TEMPLATE.count("</script>")
    block = re.search(r'<script id="stage-data" type="application/json">(.*?)</script>', document, re.S)
    assert json.loads(block.group(1)) == hostile


def test_template_has_exactly_one_data_slot():
    assert TEMPLATE.count(STAGE_DATA_PLACEHOLDER) == 1
    with pytest.raises(ValueError):
        stage_document({}, TEMPLATE + STAGE_DATA_PLACEHOLDER)


def test_the_stage_makes_no_network_calls_for_data():
    """Fonts are the only external request; the debate itself is inline."""
    assert "fetch(" not in TEMPLATE
    assert "XMLHttpRequest" not in TEMPLATE
    assert "innerHTML = speech" not in TEMPLATE


LOGO_DIRECTORY = REPOSITORY_ROOT / "app" / "logos"


def test_every_agent_has_an_inline_logo():
    logos = load_logos(LOGO_DIRECTORY)
    replay = load_public_question_replay(PUBLIC_QUESTION_IDS[0])
    payload = build_stage_payload(replay, SHORT_NAMES, logos)

    assert set(logos) == set(LOGO_FILES)
    for agent in payload["agents"]:
        assert agent["logo"].startswith("data:image/svg+xml;base64,")


@pytest.mark.parametrize("filename", sorted(LOGO_FILES.values()))
def test_logo_files_are_plain_shapes(filename):
    """A logo is shown as an image, but it must stay inert and self-contained."""
    text = (LOGO_DIRECTORY / filename).read_text(encoding="utf-8").lower()

    for forbidden in ("<script", "href=", "url(http", "onload", "foreignobject"):
        assert forbidden not in text


def test_a_missing_logo_leaves_the_letter(tmp_path):
    assert load_logos(tmp_path) == {}
