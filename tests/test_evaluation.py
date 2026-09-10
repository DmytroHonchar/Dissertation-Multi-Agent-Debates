"""Tests for the evaluation calculator. Offline: it never calls a model.

Every fixture run here is built by hand so the right answer is known in advance
and the arithmetic can be checked against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mad.database import OutcomeRecord, ResponseRecord, ResultsDatabase
from mad.evaluation import (
    BOOTSTRAP_SEED,
    BUDGET_USD,
    EvaluationError,
    bootstrap_difference,
    evaluate_run,
    load_answer_key,
    mcnemar_exact,
)
from mad.parser_v1 import (
    STATUS_API_ERROR,
    STATUS_OK,
    STATUS_PARSE_FAIL,
    STATUS_REFUSAL,
    STATUS_TRUNCATED,
)

RUN = "eval_test_run"
AGENTS = ("agent_a", "agent_b", "agent_c", "agent_d", "agent_e")


# --- Building a run by hand ------------------------------------------------


@pytest.fixture
def db(tmp_path):
    with ResultsDatabase(tmp_path / "eval.sqlite") as database:
        database.start_run(
            RUN,
            config_name="debate_config_v1",
            question_set_version="mmlu_pro_v1",
            prompt_version="round1_v1+round2_v1",
            settings_version="agents_v5",
            parser_version="parser_v1",
        )
        yield database


def store_response(db, question_id, round_number, agent_id, *, status, letter=None,
                   prompt_tokens=10, completion_tokens=20, cost=0.001, cache_hit=False,
                   attempts=1):
    db.record_response(
        ResponseRecord(
            run_id=RUN, question_id=question_id, round=round_number, agent_id=agent_id,
            requested_slug=f"vendor/{agent_id}", served_slug=f"vendor/{agent_id}",
            provider="testprovider", generation_id=f"gen-{question_id}-{round_number}-{agent_id}",
            raw_response=f"FINAL ANSWER: {letter}" if letter else "no answer",
            status=status,
            extracted_letter=letter if status == STATUS_OK else None,
            extraction_method="LAST_FINAL_ANSWER_MATCH" if status == STATUS_OK else None,
            finish_reason="stop", attempt_count=attempts, selected_attempt=attempts,
            prompt_version="round1_v1" if round_number == 1 else "round2_v1",
            temperature=0.0, top_p=1.0, max_tokens=1024,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost, latency_seconds=1.5, cache_hit=cache_hit,
        )
    )


def store_outcome(db, question_id, round_number, *, state, answer, cost=0.005,
                  prompt_tokens=50, completion_tokens=100, latency=7.5):
    db.record_outcome(
        OutcomeRecord(
            run_id=RUN, question_id=question_id, round=round_number,
            consensus_state=state, consensus_answer=answer, decided=answer is not None,
            valid_answer_count=5, total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens, total_cost_usd=cost,
            total_latency_seconds=latency,
        )
    )


def simple_run(db, *, votes, agent_letters=None, statuses=None, finish=True):
    """Build a run from {question_id: (round1_vote, round2_vote)}.

    A vote of None means undecided. Agent rows default to every agent giving the
    group's letter, so agent numbers can be reasoned about from the votes alone.
    """
    for question_id, (first, second) in votes.items():
        for round_number, vote in ((1, first), (2, second)):
            store_outcome(
                db, question_id, round_number,
                state="UNANIMOUS" if vote else "NO_CONSENSUS", answer=vote,
            )
            for agent_id in AGENTS:
                status = (statuses or {}).get((question_id, round_number, agent_id), STATUS_OK)
                letter = (agent_letters or {}).get(
                    (question_id, round_number, agent_id), vote or "Z"
                )
                store_response(
                    db, question_id, round_number, agent_id,
                    status=status, letter=letter if status == STATUS_OK else None,
                )
    if finish:
        db.finish_run(RUN)


def key_file(tmp_path, mapping, name="key.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(
            json.dumps({"stable_id": qid, "correct_answer": ans})
            for qid, ans in mapping.items()
        )
    )
    return path


# 1. McNemar's exact test


@pytest.mark.parametrize(
    ("became_incorrect", "became_correct", "expected"),
    [
        (0, 0, 1.0),               # nothing moved
        (0, 10, 0.001953125),      # every move was a repair
        (1, 9, 0.021484375),
        (2, 8, 0.109375),
        (3, 7, 0.34375),
        (5, 5, 1.0),               # perfectly balanced
    ],
)
def test_mcnemar_matches_the_exact_binomial_values(became_incorrect, became_correct, expected):
    """These are scipy.stats.binomtest(b, b+c, 0.5) to the last digit."""
    result = mcnemar_exact(became_correct=became_correct, became_incorrect=became_incorrect)
    assert result.p_value == pytest.approx(expected, abs=1e-12)


def test_no_discordant_pairs_reports_p_of_one_rather_than_failing():
    """A run where debate moved no group answer is a real result, not an error."""
    result = mcnemar_exact(became_correct=0, became_incorrect=0)
    assert result.p_value == 1.0
    assert result.discordant_pairs == 0


def test_mcnemar_is_symmetric():
    forwards = mcnemar_exact(became_correct=8, became_incorrect=2)
    backwards = mcnemar_exact(became_correct=2, became_incorrect=8)
    assert forwards.p_value == backwards.p_value


# 2. The bootstrap


def test_the_measured_difference_is_never_changed_by_resampling():
    """The bootstrap reports uncertainty. It must not move the result itself."""
    round1 = [True] * 60 + [False] * 40
    round2 = [True] * 62 + [False] * 38
    interval = bootstrap_difference(round1, round2, resamples=500)
    assert interval.difference_points == pytest.approx(2.0)


def test_the_same_seed_gives_the_same_interval():
    round1 = [True, False] * 50
    round2 = [True, True, False, False] * 25
    first = bootstrap_difference(round1, round2, seed=BOOTSTRAP_SEED, resamples=300)
    second = bootstrap_difference(round1, round2, seed=BOOTSTRAP_SEED, resamples=300)
    assert (first.low_points, first.high_points) == (second.low_points, second.high_points)


def test_a_different_seed_gives_a_different_interval():
    round1 = [True, False] * 50
    round2 = [True, True, False, False] * 25
    first = bootstrap_difference(round1, round2, seed=1, resamples=300)
    second = bootstrap_difference(round1, round2, seed=2, resamples=300)
    assert (first.low_points, first.high_points) != (second.low_points, second.high_points)


def test_an_unchanged_run_has_a_zero_wide_interval():
    """If no question moved, every resample gives zero. No spread to report."""
    outcomes = [True, False, True, True, False]
    interval = bootstrap_difference(outcomes, list(outcomes), resamples=200)
    assert interval.difference_points == 0.0
    assert (interval.low_points, interval.high_points) == (0.0, 0.0)
    assert interval.includes_zero


def test_a_large_real_effect_gives_an_interval_clear_of_zero():
    round1 = [False] * 100
    round2 = [True] * 100
    interval = bootstrap_difference(round1, round2, resamples=1000)
    assert interval.difference_points == pytest.approx(100.0)
    assert interval.low_points > 0
    assert not interval.includes_zero


def test_the_interval_uses_true_nearest_rank_positions():
    """The 2.5th percentile of 10,000 values is the 250th, at index 249.

    Built so every resample gives a distinct known value: 100 questions where
    Round 1 is all wrong and Round 2 all right makes every difference +100, so
    instead check the boundary directly on a controlled list.
    """
    from mad.evaluation import _percentile

    values = list(range(1000))                 # already sorted, value == index
    assert _percentile(values, 0.025) == 24    # 25th value, index 24
    assert _percentile(values, 0.975) == 974   # 975th value, index 974
    assert _percentile(values, 0.0) == 0
    assert _percentile(values, 1.0) == 999


def test_mismatched_pairs_are_refused():
    with pytest.raises(EvaluationError, match="same length"):
        bootstrap_difference([True, False], [True], resamples=10)


def test_an_empty_run_cannot_be_bootstrapped():
    with pytest.raises(EvaluationError, match="empty"):
        bootstrap_difference([], [], resamples=10)


# 3. Reading the answer key


def test_the_answer_key_is_read_into_a_mapping(tmp_path):
    path = key_file(tmp_path, {"q1": "A", "q2": "B"})
    assert load_answer_key(path) == {"q1": "A", "q2": "B"}


def test_a_duplicated_question_in_the_key_is_refused(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"stable_id": "q1", "correct_answer": "A"}\n'
        '{"stable_id": "q1", "correct_answer": "B"}\n'
    )
    with pytest.raises(EvaluationError, match="twice"):
        load_answer_key(path)


def test_a_key_line_without_an_answer_is_refused(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"stable_id": "q1"}\n')
    with pytest.raises(EvaluationError, match="correct_answer"):
        load_answer_key(path)


def test_an_empty_key_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(EvaluationError, match="empty"):
        load_answer_key(path)


# 4. Group accuracy and the fixed denominator (D010)


def test_an_undecided_question_counts_as_incorrect_and_stays_in_the_denominator(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("B", None), "q3": (None, "C")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "B", "q3": "C"}))

    report = evaluate_run(db, RUN, key, resamples=100)

    assert report.question_count == 3
    assert report.group_results[1].questions == 3
    assert report.group_results[2].questions == 3
    # Round 1: q1 and q2 right, q3 undecided. Round 2: q1 and q3 right.
    assert report.group_results[1].correct == 2
    assert report.group_results[2].correct == 2
    assert report.group_results[1].accuracy == pytest.approx(2 / 3)
    assert report.group_results[1].undecided == 1
    assert report.group_results[2].undecided == 1


def test_a_decided_but_wrong_vote_is_incorrect(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "B")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert report.group_results[1].correct == 1
    assert report.group_results[2].correct == 0
    assert report.debate_effect_points == pytest.approx(-100.0)


def test_consensus_states_are_counted_separately(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A"), "q2": (None, "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert report.group_results[1].consensus_states["UNANIMOUS"] == 1
    assert report.group_results[1].consensus_states["NO_CONSENSUS"] == 1
    assert report.group_results[2].consensus_states["UNANIMOUS"] == 2


# 5. Per-agent accuracy over valid answers only (D010)


def test_agent_accuracy_ignores_failures_and_never_scores_them_wrong(db, tmp_path):
    statuses = {
        ("q2", 1, "agent_a"): STATUS_REFUSAL,
        ("q3", 1, "agent_a"): STATUS_API_ERROR,
    }
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("A", "A"), "q3": ("A", "A")},
               statuses=statuses)
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A", "q3": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)
    agent_a = next(a for a in report.agent_results[1] if a.agent_id == "agent_a")

    assert agent_a.responses == 3
    assert agent_a.valid_answers == 1
    assert agent_a.correct == 1
    assert agent_a.accuracy == 1.0, "one right out of one valid, not one out of three"
    assert agent_a.failures[STATUS_REFUSAL] == 1
    assert agent_a.failures[STATUS_API_ERROR] == 1
    assert agent_a.failure_count == 2


@pytest.mark.parametrize(
    "status", [STATUS_REFUSAL, STATUS_TRUNCATED, STATUS_PARSE_FAIL, STATUS_API_ERROR]
)
def test_every_failure_status_is_reported_by_name(db, tmp_path, status):
    simple_run(db, votes={"q1": ("A", "A")},
               statuses={("q1", 1, "agent_a"): status})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)
    agent_a = next(a for a in report.agent_results[1] if a.agent_id == "agent_a")

    assert agent_a.failures[status] == 1
    assert agent_a.accuracy is None, "no valid answer means no accuracy, not zero"


def test_both_rounds_are_scored_for_every_agent(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert set(report.agent_results) == {1, 2}
    for round_number in (1, 2):
        assert len(report.agent_results[round_number]) == 5
        assert {a.agent_id for a in report.agent_results[round_number]} == set(AGENTS)


# 6. Transitions


def test_the_group_transition_table_counts_all_four_moves(db, tmp_path):
    simple_run(db, votes={
        "q1": ("A", "A"),      # stayed correct
        "q2": ("A", "B"),      # became incorrect
        "q3": ("B", "A"),      # became correct
        "q4": ("B", "B"),      # stayed incorrect
    })
    key = load_answer_key(key_file(tmp_path, {q: "A" for q in ("q1", "q2", "q3", "q4")}))

    counts = evaluate_run(db, RUN, key, resamples=50).group_transitions

    assert (counts.stayed_correct, counts.became_incorrect) == (1, 1)
    assert (counts.became_correct, counts.stayed_incorrect) == (1, 1)
    assert counts.total == 4
    assert counts.discordant == 2


def test_agent_transitions_exclude_questions_that_failed_in_either_round(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("A", "A")},
               statuses={("q2", 2, "agent_a"): STATUS_TRUNCATED})
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)
    agent_a = next(t for t in report.agent_transitions if t.agent_id == "agent_a")

    assert agent_a.counts.total == 1, "only q1 was answered validly twice"
    assert agent_a.excluded_questions == 1
    assert agent_a.counts.stayed_correct == 1


def test_every_agent_gets_its_own_transition_table(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert {t.agent_id for t in report.agent_transitions} == set(AGENTS)


# 7. The aggregation gain, on both baselines (D021)


def test_aggregation_is_reported_against_the_mean_and_the_best_agent(db, tmp_path):
    # agent_a alone is perfect; the other four are wrong on q2. The group vote
    # follows the majority, so the group gets q2 wrong and loses to agent_a.
    letters = {("q2", 1, "agent_a"): "A"}
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("B", "B")}, agent_letters=letters)
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    gain = evaluate_run(db, RUN, key, resamples=50).aggregation

    assert gain.group_accuracy == pytest.approx(0.5)
    assert gain.best_agent_id == "agent_a"
    assert gain.best_agent_accuracy == pytest.approx(1.0)
    assert gain.mean_agent_accuracy == pytest.approx((1.0 + 0.5 * 4) / 5)
    assert gain.versus_mean_points == pytest.approx((0.5 - 0.6) * 100)
    assert gain.versus_best_points == pytest.approx(-50.0), "the group lost to its best member"


def test_the_best_agent_is_stable_when_agents_tie(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    gain = evaluate_run(db, RUN, key, resamples=50).aggregation

    assert gain.best_agent_id == "agent_a", "ties break on agent_id, not row order"
    assert gain.versus_best_points == pytest.approx(0.0)


# 8. Usage, cost and the budget


def test_usage_and_budget_are_totalled_per_round(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    for round_number in (1, 2):
        usage = report.usage[round_number]
        assert usage.prompt_tokens == 100          # 2 questions x 50
        assert usage.completion_tokens == 200
        assert usage.cost_usd == pytest.approx(0.01)
        assert usage.api_attempts == 10            # 2 questions x 5 agents

    assert report.budget.run_usd == pytest.approx(0.02), "both rounds together"
    assert report.budget.budget_usd == BUDGET_USD
    # Only one run in this database, so the two costs agree here.
    assert report.budget.recorded_usd == pytest.approx(0.02)
    assert report.budget.remaining_usd == pytest.approx(BUDGET_USD - 0.02)


def test_remaining_budget_counts_every_run_in_the_database(db, tmp_path):
    """Subtracting one run from the whole budget would report a balance the key
    does not have. Earlier runs already spent money."""
    db.start_run(
        "earlier_run", config_name="round1_config_v2", question_set_version="mmlu_pro_v1",
        prompt_version="round1_v1", settings_version="agents_v1", parser_version="parser_v1",
    )
    db.record_outcome(
        OutcomeRecord(
            run_id="earlier_run", question_id="old_q", round=1, consensus_state="UNANIMOUS",
            consensus_answer="A", decided=True, valid_answer_count=5, total_cost_usd=0.5,
        )
    )
    db.finish_run("earlier_run")

    simple_run(db, votes={"q1": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))
    report = evaluate_run(db, RUN, key, resamples=50)

    assert report.budget.run_usd == pytest.approx(0.01), "this run only"
    assert report.budget.recorded_usd == pytest.approx(0.51), "every run in the file"
    assert report.budget.remaining_usd == pytest.approx(BUDGET_USD - 0.51)


def test_a_cached_response_is_not_counted_as_an_api_call(db, tmp_path):
    """A cache hit keeps attempt_count=1 from the original paid call. Counting
    it would report calls this run never made."""
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q1", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A", cache_hit=True)
        store_response(db, "q1", 2, agent_id, status=STATUS_OK, letter="A", cache_hit=False)
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert report.usage[1].cache_hits == 5
    assert report.usage[1].api_attempts == 0, "served from cache, nothing was called"
    assert report.usage[1].responses == 5
    assert report.usage[2].cache_hits == 0
    assert report.usage[2].api_attempts == 5


def test_a_retried_response_counts_both_attempts(db, tmp_path):
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q1", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        attempts = 2 if agent_id == "agent_a" else 1
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A", attempts=attempts)
        store_response(db, "q1", 2, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    report = evaluate_run(db, RUN, key, resamples=50)

    assert report.usage[1].api_attempts == 6, "four agents once, one agent twice"
    assert report.usage[1].retried_responses == 1


# 9. Refusals


def test_an_unknown_run_is_refused(db, tmp_path):
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))
    with pytest.raises(EvaluationError, match="not in this database"):
        evaluate_run(db, "no_such_run", key)


def test_an_unfinished_run_is_refused(db, tmp_path):
    simple_run(db, votes={"q1": ("A", "A")}, finish=False)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    with pytest.raises(EvaluationError, match="never finished"):
        evaluate_run(db, RUN, key)


def test_a_run_missing_round_two_is_refused(db, tmp_path):
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    with pytest.raises(EvaluationError, match="no Round 2 outcomes"):
        evaluate_run(db, RUN, key)


def test_rounds_scoring_different_questions_are_refused(db, tmp_path):
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q2", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A")
        store_response(db, "q2", 2, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    with pytest.raises(EvaluationError, match="different questions"):
        evaluate_run(db, RUN, key)


def test_a_question_missing_an_agent_response_is_refused(db, tmp_path):
    """A failure is stored as API_ERROR and counted. A missing row is a run
    that did not finish what it claims to have finished."""
    # agent_e is present in Round 2, just not for q2. The group is intact, so
    # this must be caught by the per-question check rather than the agent check.
    for question_id in ("q1", "q2"):
        store_outcome(db, question_id, 1, state="UNANIMOUS", answer="A")
        store_outcome(db, question_id, 2, state="UNANIMOUS", answer="A")
        for agent_id in AGENTS:
            store_response(db, question_id, 1, agent_id, status=STATUS_OK, letter="A")
            if not (question_id == "q2" and agent_id == "agent_e"):
                store_response(db, question_id, 2, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    with pytest.raises(EvaluationError, match="missing 1 Round 2"):
        evaluate_run(db, RUN, key)


def test_a_run_with_the_wrong_number_of_agents_is_refused(db, tmp_path):
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q1", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS[:4]:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A")
        store_response(db, "q1", 2, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    with pytest.raises(EvaluationError, match="not 5"):
        evaluate_run(db, RUN, key)


def test_an_extra_agent_appearing_only_in_round_two_is_refused(db, tmp_path):
    """The Round 1 agent list alone would not notice a sixth agent later."""
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q1", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A")
        store_response(db, "q1", 2, agent_id, status=STATUS_OK, letter="A")
    store_response(db, "q1", 2, "agent_f", status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    with pytest.raises(EvaluationError, match="different group"):
        evaluate_run(db, RUN, key)


def test_an_agent_that_answered_only_round_one_is_refused(db, tmp_path):
    store_outcome(db, "q1", 1, state="UNANIMOUS", answer="A")
    store_outcome(db, "q1", 2, state="UNANIMOUS", answer="A")
    for agent_id in AGENTS:
        store_response(db, "q1", 1, agent_id, status=STATUS_OK, letter="A")
    for agent_id in (*AGENTS[:4], "agent_f"):
        store_response(db, "q1", 2, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A"}))

    with pytest.raises(EvaluationError, match="different group"):
        evaluate_run(db, RUN, key)


def test_a_response_for_a_question_with_no_outcome_row_is_refused(db, tmp_path):
    """No gap, but more rows than the run accounts for."""
    simple_run(db, votes={"q1": ("A", "A")}, finish=False)
    for agent_id in AGENTS:
        store_response(db, "ghost_q", 1, agent_id, status=STATUS_OK, letter="A")
    db.finish_run(RUN)
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "ghost_q": "A"}))

    with pytest.raises(EvaluationError, match="does not account for"):
        evaluate_run(db, RUN, key)


def test_a_short_formal_run_is_refused_when_the_question_count_is_declared(db, tmp_path):
    """The pilot is 20 questions and the main run is 300. A run that lost some
    must not be scored over a smaller denominator."""
    simple_run(db, votes={"q1": ("A", "A"), "q2": ("A", "A")})
    key = load_answer_key(key_file(tmp_path, {"q1": "A", "q2": "A"}))

    with pytest.raises(EvaluationError, match="scored 2 questions, not the 20"):
        evaluate_run(db, RUN, key, expected_questions=20)

    # Without the declaration the same run scores normally.
    assert evaluate_run(db, RUN, key, resamples=50).question_count == 2


def test_a_pilot_run_scored_against_the_experimental_key_is_refused(db, tmp_path):
    """The two frozen sets do not overlap, so every ID would be missing."""
    simple_run(db, votes={"pilot_q1": ("A", "A")})
    experimental_key = load_answer_key(
        key_file(tmp_path, {"experimental_q1": "A", "experimental_q2": "B"})
    )

    with pytest.raises(EvaluationError, match="does not cover"):
        evaluate_run(db, RUN, experimental_key)


def test_the_real_frozen_keys_do_not_overlap():
    """The guard above only works because the two sets are disjoint."""
    root = Path(__file__).resolve().parents[1] / "data/frozen/mmlu_pro_v1/answer_keys"
    pilot = set(load_answer_key(root / "pilot_answers.jsonl"))
    experimental = set(load_answer_key(root / "experimental_answers.jsonl"))

    assert len(pilot) == 20
    assert len(experimental) == 300
    assert not pilot & experimental


# 10. End to end on a run whose answer is known in advance


def test_a_worked_run_produces_every_headline_number(db, tmp_path):
    """Six questions, key A. Round 1 gets 3, Round 2 gets 4: debate +16.7 points."""
    simple_run(db, votes={
        "q1": ("A", "A"),      # stayed correct
        "q2": ("A", "A"),      # stayed correct
        "q3": ("A", "B"),      # became incorrect
        "q4": ("B", "A"),      # became correct
        "q5": ("B", "A"),      # became correct
        "q6": ("B", "B"),      # stayed incorrect
    })
    key = load_answer_key(key_file(tmp_path, {f"q{i}": "A" for i in range(1, 7)}))

    report = evaluate_run(db, RUN, key, resamples=2000)

    assert report.question_count == 6
    assert report.group_results[1].correct == 3
    assert report.group_results[2].correct == 4
    assert report.debate_effect_points == pytest.approx(100 / 6)

    counts = report.group_transitions
    assert (counts.stayed_correct, counts.became_incorrect) == (2, 1)
    assert (counts.became_correct, counts.stayed_incorrect) == (2, 1)

    # Three moved, two of them repairs: nowhere near significant on six questions.
    assert report.mcnemar.discordant_pairs == 3
    assert report.mcnemar.p_value == pytest.approx(1.0)
    assert report.bootstrap.includes_zero, "six questions cannot settle anything"
    assert report.bootstrap.seed == BOOTSTRAP_SEED
    assert report.evaluation_version == "evaluation_v1"
