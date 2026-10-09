"""Static and pure-logic tests for the Streamlit replay page."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = REPOSITORY_ROOT / "app" / "viewer.py"


def test_viewer_has_the_nine_planned_steps():
    source = VIEWER_PATH.read_text()
    expected = (
        "The question",
        "Round 1 request",
        "Round 1 responses",
        "Parsing the answers",
        "First group vote",
        "What each agent receives in Round 2",
        "Round 2 responses",
        "Final group vote",
        "Question summary",
    )
    assert all(title in source for title in expected)
    assert 'DEFAULT_QUESTION_ID = "mmlu_pro_v1:test:5503"' in source


def test_viewer_has_an_overview_and_curated_real_cases():
    source = VIEWER_PATH.read_text()
    assert 'page = st.radio("Workspace", ("Overview", "Debate replay")' in source
    assert "load_public_experiment_overview" in source
    assert '"mmlu_pro_v1:test:5503"' in source
    assert '"mmlu_pro_v1:test:11994"' in source
    assert "Group accuracy by round" in source
    assert "What changed between rounds?" in source
    assert "Individual accuracy" in source
    assert "Consensus changed" in source


def test_viewer_keeps_the_fixed_three_of_five_rule_visible():
    source = VIEWER_PATH.read_text()
    assert "fixed majority threshold: 3 of the 5 configured agents" in source
    assert "the Round 1 group vote is not included" in source


def test_viewer_depends_on_the_read_only_layer_not_experimental_writers():
    tree = ast.parse(VIEWER_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert "mad.public_viewer_data" in imported
    assert not imported.intersection(
        {
            "mad.api_client",
            "mad.cache",
            "mad.database",
            "mad.evaluation",
            "mad.runner",
            "mad.viewer_data",
        }
    )


def test_viewer_uses_only_the_sanitized_public_bundle():
    source = VIEWER_PATH.read_text()
    assert "load_public_question_replay" in source
    assert "load_question_replay(" not in source
    assert "results.sqlite" not in source
    assert "cache.sqlite" not in source


def test_viewer_never_names_an_answer_key_or_model_client():
    source = VIEWER_PATH.read_text()
    assert "answer_keys" not in source
    assert "load_answer_key" not in source
    assert "OpenRouterClient" not in source
    assert ".complete(" not in source
    assert "INSERT " not in source
    assert "UPDATE " not in source
    assert "DELETE " not in source


def test_viewer_file_is_valid_python():
    spec = importlib.util.spec_from_file_location("viewer", VIEWER_PATH)
    assert spec is not None and spec.loader is not None


def test_viewer_offers_the_animated_stage_beside_the_evidence_steps():
    source = VIEWER_PATH.read_text()
    assert 'REPLAY_VIEWS = ("Animated debate", "Step-by-step evidence")' in source
    assert "build_stage_payload" in source
    assert (REPOSITORY_ROOT / "app" / "debate_stage.html").is_file()
