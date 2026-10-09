"""Turn one stored debate into the data the animated debate stage plays back.

The stage is presentation only: it makes no model calls and invents nothing.
Every speech bubble is a verbatim slice of a stored response, never a summary,
so the animation cannot put words in a model's mouth. Who received which peer
response comes from the stored Round 2 rows, not from the protocol on paper,
so a failed agent visibly sends nothing.
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mad.viewer_data import AgentResponseView, QuestionReplay, RoundReplay


# Long enough to show how an agent starts its reasoning, short enough that
# five bubbles fit side by side.
EXCERPT_CHARACTERS = 140

STAGE_DATA_PLACEHOLDER = "__STAGE_DATA__"

# Why a response cast no vote, in the words the report uses.
NO_VOTE_NOTES = {
    "TRUNCATED": "Cut off by the token limit before a final answer, so no vote.",
    "API_ERROR": "The provider returned an error instead of a reply, so no vote.",
    "PARSE_FAILURE": "No valid final answer line, so no vote.",
    "REFUSAL": "Declined to answer, so no vote.",
}


# One logo per agent, identifying the model; see app/logos/README.md.
LOGO_FILES = {
    "agent_llama": "llama.svg",
    "agent_qwen": "qwen.svg",
    "agent_mistral": "mistral.svg",
    "agent_deepseek": "deepseek.svg",
    "agent_gemma": "gemma.svg",
}


def load_logos(directory: Path) -> dict[str, str]:
    """Read each agent's logo as an inline image.

    Inlining keeps the stage self-contained: the public app never fetches an
    image from another site. A missing file simply leaves the agent's letter.
    """
    logos = {}
    for agent_id, filename in LOGO_FILES.items():
        path = directory / filename
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            logos[agent_id] = f"data:image/svg+xml;base64,{encoded}"
    return logos


def plain_text(text: str) -> str:
    """Collapse whitespace and drop Markdown bold markers, nothing else."""
    return " ".join(text.replace("**", "").split())


def reasoning_excerpt(raw_response: str, limit: int = EXCERPT_CHARACTERS) -> tuple[str, bool]:
    """Return the opening of a response's reasoning and whether it was cut.

    The final answer line is dropped because the stage shows the letter on
    its own. Whitespace is collapsed and Markdown bold markers are removed so
    a bubble reads as text, but no word is changed, added or reordered.
    """
    text = raw_response
    final_line = text.upper().rfind("FINAL ANSWER")
    if final_line != -1:
        text = text[:final_line]
    text = plain_text(text)
    if text.upper().startswith("REASONING:"):
        text = text[len("REASONING:"):].lstrip()
    if len(text) <= limit:
        return text, False
    cut = text.rfind(" ", 0, limit)
    if cut < limit * 0.6:
        cut = limit
    return text[:cut].rstrip(" ,;:"), True


def _speech(response: AgentResponseView) -> dict[str, Any]:
    """What one agent says in one round, as the stage shows it."""
    # An API error body is provider text, not model text, so it never
    # appears in a speech bubble.
    has_model_text = response.status != "API_ERROR" and bool(response.raw_response.strip())
    excerpt, cut = reasoning_excerpt(response.raw_response) if has_model_text else ("", False)
    return {
        "response_id": response.response_id,
        "status": response.status,
        "letter": response.extracted_letter,
        "excerpt": excerpt,
        "excerpt_cut": cut,
        "full_text": response.raw_response if has_model_text else None,
        "no_vote_note": NO_VOTE_NOTES.get(response.status) if response.extracted_letter is None else None,
        "latency_seconds": round(response.latency_seconds, 1),
    }


def _round_tally(round_replay: RoundReplay) -> dict[str, Any]:
    counts = Counter(
        response.extracted_letter
        for response in round_replay.responses
        if response.extracted_letter is not None
    )
    outcome = round_replay.outcome
    return {
        "state": outcome.consensus_state,
        "answer": outcome.consensus_answer,
        "valid_votes": outcome.valid_answer_count,
        "counts": dict(sorted(counts.items())),
    }


def build_stage_payload(
    replay: QuestionReplay,
    short_names: Mapping[str, str],
    logos: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect everything the stage animates for one stored question."""
    logos = logos or {}
    first = {response.agent_id: response for response in replay.round1.responses}
    second = {response.agent_id: response for response in replay.round2.responses}
    sender_of = {response.response_id: response.agent_id for response in replay.round1.responses}

    agents = []
    for agent_id, initial in first.items():
        revised = second[agent_id]
        round2 = _speech(revised)
        # PEER RESPONSE 1, 2, ... in the order the agent actually read them.
        round2["peers_from"] = [sender_of[peer_id] for peer_id in revised.peer_response_ids]
        round2["own_previous_included"] = revised.own_round1_response_id is not None
        agents.append(
            {
                "id": agent_id,
                "name": short_names.get(agent_id, initial.display_name),
                "model": initial.display_name,
                "developer": initial.developer,
                "logo": logos.get(agent_id),
                "round1": _speech(initial),
                "round2": round2,
            }
        )

    return {
        "question_id": replay.question_id,
        "category": replay.category,
        "question": replay.question,
        "options": list(replay.options),
        "correct_answer": replay.correct_answer,
        "agents": agents,
        "rounds": {"1": _round_tally(replay.round1), "2": _round_tally(replay.round2)},
    }


def stage_document(payload: Mapping[str, Any], template: str) -> str:
    """Place the payload into the stage page as inert JSON.

    Escaping every ``<`` stops stored text such as ``</script>`` from ending
    the data block early and running as page code.
    """
    if template.count(STAGE_DATA_PLACEHOLDER) != 1:
        raise ValueError("the stage template must contain the data placeholder exactly once")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace(STAGE_DATA_PLACEHOLDER, data)
