"""Offline tests for the read-only Streamlit data layer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mad.api_client import load_model_registry
from mad.cache import cache_key
from mad.database import OutcomeRecord, ResponseRecord, ResultsDatabase
from mad.prompts_v1 import build_round1_messages, build_round2_messages
from mad.viewer_data import (
    ViewerDataError,
    load_experiment_overview,
    load_question_replay,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "configs" / "models" / "agents_v7.yaml"
RUN_ID = "viewer_test_run"
QUESTION_ID = "mmlu_pro_v1:test:viewer"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _response(
    *,
    round_number: int,
    agent_id: str,
    spec,
    text: str,
    letter: str,
    peer_ids: tuple[int, ...] = (),
) -> ResponseRecord:
    return ResponseRecord(
        run_id=RUN_ID,
        question_id=QUESTION_ID,
        round=round_number,
        agent_id=agent_id,
        requested_slug=spec.slug,
        served_slug=spec.slug,
        provider="fixture-provider",
        generation_id=f"generation-{round_number}-{agent_id}",
        raw_response=text,
        status="OK",
        extracted_letter=letter,
        extraction_method="LAST_FINAL_ANSWER_MATCH",
        finish_reason="stop",
        attempt_count=1,
        selected_attempt=1,
        prompt_version="round1_v1" if round_number == 1 else "round2_v1",
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.001,
        latency_seconds=2.0,
        peer_response_ids=peer_ids,
    )


@pytest.fixture
def replay_files(tmp_path):
    registry = load_model_registry(REGISTRY_PATH)
    question = {
        "benchmark": "MMLU-Pro",
        "category": "test subject",
        "question": "Which option is correct?",
        "options": ["first", "second", "third"],
        "stable_id": QUESTION_ID,
    }
    questions_path = tmp_path / "experimental_questions.jsonl"
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")

    results_path = tmp_path / "results.sqlite"
    round1_texts: dict[str, str] = {}
    round1_ids: dict[str, int] = {}
    round1_letters = dict(
        zip(registry, ("C", "A", "C", "C", "A"), strict=True)
    )
    with ResultsDatabase(results_path) as database:
        database.start_run(
            RUN_ID,
            config_name="debate_config_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version="round1_v1+round2_v1",
            settings_version="agents_v7",
            parser_version="parser_v1",
        )
        for agent_id, spec in registry.items():
            letter = round1_letters[agent_id]
            text = f"REASONING: first response from {agent_id}.\nFINAL ANSWER: {letter}"
            round1_texts[agent_id] = text
            round1_ids[agent_id] = database.record_response(
                _response(
                    round_number=1,
                    agent_id=agent_id,
                    spec=spec,
                    text=text,
                    letter=letter,
                )
            )
        database.record_outcome(
            OutcomeRecord(
                run_id=RUN_ID,
                question_id=QUESTION_ID,
                round=1,
                consensus_state="CONSENSUS",
                consensus_answer="C",
                decided=True,
                valid_answer_count=5,
                total_prompt_tokens=500,
                total_completion_tokens=250,
                total_cost_usd=0.005,
                total_latency_seconds=10.0,
            )
        )

        for agent_id, spec in registry.items():
            peer_ids = tuple(
                round1_ids[other_id] for other_id in registry if other_id != agent_id
            )
            text = f"REASONING: revised response from {agent_id}.\nFINAL ANSWER: A"
            database.record_response(
                _response(
                    round_number=2,
                    agent_id=agent_id,
                    spec=spec,
                    text=text,
                    letter="A",
                    peer_ids=peer_ids,
                )
            )
        database.record_outcome(
            OutcomeRecord(
                run_id=RUN_ID,
                question_id=QUESTION_ID,
                round=2,
                consensus_state="UNANIMOUS",
                consensus_answer="A",
                decided=True,
                valid_answer_count=5,
                total_prompt_tokens=1000,
                total_completion_tokens=200,
                total_cost_usd=0.007,
                total_latency_seconds=12.0,
            )
        )
        database.finish_run(RUN_ID)

    evaluation_path = tmp_path / "evaluation_summary.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "provenance": {"run": {"run_id": RUN_ID}},
                "questions": [
                    {
                        "question_id": QUESTION_ID,
                        "correct_answer": "A",
                        "round1_state": "CONSENSUS",
                        "round1_answer": "C",
                        "round1_correct": False,
                        "round2_state": "UNANIMOUS",
                        "round2_answer": "A",
                        "round2_correct": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cache_path = tmp_path / "cache.sqlite"
    connection = sqlite3.connect(cache_path)
    connection.execute(
        """
        CREATE TABLE cached_responses (
            cache_key TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            latency_seconds REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    round1_messages = build_round1_messages(question)
    for agent_id, spec in registry.items():
        peer_texts = [
            round1_texts[other_id] for other_id in registry if other_id != agent_id
        ]
        round2_messages = build_round2_messages(
            question, round1_texts[agent_id], peer_texts
        )
        for messages in (round1_messages, round2_messages):
            connection.execute(
                "INSERT INTO cached_responses VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cache_key(spec, messages),
                    agent_id,
                    spec.slug,
                    "{}",
                    2.0,
                    "2026-09-23T00:00:00+00:00",
                ),
            )
    connection.commit()
    connection.close()

    return {
        "results": results_path,
        "cache": cache_path,
        "questions": questions_path,
        "evaluation": evaluation_path,
        "registry": REGISTRY_PATH,
    }


def _load(files):
    return load_question_replay(
        QUESTION_ID,
        run_id=RUN_ID,
        results_path=files["results"],
        cache_path=files["cache"],
        questions_path=files["questions"],
        evaluation_export_path=files["evaluation"],
        registry_path=files["registry"],
    )


def test_one_debate_is_reconstructed_and_every_request_matches_cache(replay_files):
    before = {name: _sha256(path) for name, path in replay_files.items()}

    replay = _load(replay_files)

    assert replay.question_id == QUESTION_ID
    assert replay.correct_answer == "A"
    assert replay.round1.outcome.consensus_answer == "C"
    assert replay.round2.outcome.consensus_answer == "A"
    assert [response.agent_id for response in replay.round1.responses] == [
        "agent_llama",
        "agent_qwen",
        "agent_mistral",
        "agent_deepseek",
        "agent_gemma",
    ]
    assert all(
        response.request_found_in_cache
        for response in (*replay.round1.responses, *replay.round2.responses)
    )
    assert all(len(response.peer_response_ids) == 4 for response in replay.round2.responses)
    assert all(
        response.own_round1_response_id is not None
        for response in replay.round2.responses
    )
    assert all(
        [message.role for message in response.messages]
        == ["system", "user", "assistant", "user"]
        for response in replay.round2.responses
    )
    assert all(
        "Round 1 group vote" not in "\n".join(
            message.content for message in response.messages
        )
        for response in replay.round2.responses
    )

    after = {name: _sha256(path) for name, path in replay_files.items()}
    assert after == before, "loading a replay must not change any source artifact"


def test_a_stale_evaluation_export_is_refused(replay_files):
    payload = json.loads(replay_files["evaluation"].read_text())
    payload["questions"][0]["round2_answer"] = "B"
    replay_files["evaluation"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ViewerDataError, match="does not match stored value"):
        _load(replay_files)


def test_the_viewer_layer_never_names_or_loads_an_answer_key():
    source = (REPOSITORY_ROOT / "src" / "mad" / "viewer_data.py").read_text()
    assert "answer_keys" not in source
    assert "load_answer_key" not in source
    assert "OpenRouterClient" not in source
    assert "requests." not in source


def test_main_experiment_overview_comes_from_the_verified_export():
    overview = load_experiment_overview()

    assert overview.question_count == 300
    assert [row.accuracy_percent for row in overview.group_accuracy] == pytest.approx(
        [81.6666666667, 85.0]
    )
    assert overview.effect_points == pytest.approx(3.3333333333)
    assert overview.became_correct == 12
    assert overview.became_incorrect == 2
    assert overview.corrected_from_undecided == 10
    assert overview.total_cost_usd == pytest.approx(2.9206494725)
    assert overview.total_failures == 117
