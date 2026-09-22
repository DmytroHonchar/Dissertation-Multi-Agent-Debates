"""Compare Mistral Small 3.2's two alternative hosts on the failed pilot prompts.

This is a bounded diagnostic, not a repaired pilot and not an experiment run.
It reads the four stored Mistral API failures from the agents_v6 pilot,
reconstructs their exact messages, and sends each message once to DeepInfra and
once to Venice. It never reads an answer key, never uses the response cache and
never modifies the results database.

Dry validation (free):

    .venv/bin/python scripts/probe_mistral_hosts.py

Eight single-attempt live calls (hard cap $0.02):

    .venv/bin/python scripts/probe_mistral_hosts.py \
        --live --yes-spend-real-money
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.api_client import ApiRequestError, OpenRouterClient, load_model_registry  # noqa: E402
from mad.diagnostics import diagnose_attempt  # noqa: E402
from mad.parser_v1 import parse_response  # noqa: E402
from mad.prompts_v1 import answer_letters, build_round1_messages, build_round2_messages  # noqa: E402
from mad.runner import load_pilot_questions, require_spend_confirmation  # noqa: E402


SOURCE_RUN = "pilot_agents_v6_20260922T171757Z"
SOURCE_SETTINGS = "agents_v6"
TARGET_AGENT = "agent_mistral"
EXPECTED_FAILURES = 4
EXPECTED_TARGETS = (
    ("mmlu_pro_v1:test:10925", 1),
    ("mmlu_pro_v1:test:9622", 1),
    ("mmlu_pro_v1:test:9622", 2),
    ("mmlu_pro_v1:test:6939", 1),
)
CANDIDATE_PINS = ("deepinfra/fp8", "venice/fp8")
MAXIMUM_CALLS = EXPECTED_FAILURES * len(CANDIDATE_PINS)
MAXIMUM_BUDGET_USD = 0.02
REQUIRED_PARAMETERS = frozenset({"temperature", "top_p", "max_tokens"})


def _reconstruct_messages(question, source, registry, round1_rows):
    if source["round"] == 1:
        if json.loads(source["peer_response_ids"]) != []:
            raise ValueError("A stored Round 1 source unexpectedly names peers")
        return build_round1_messages(question)

    stored = {row["agent_id"]: row for row in round1_rows}
    if set(stored) != set(registry):
        raise ValueError(f"Round 1 of {source['question_id']} is incomplete")
    valid = [
        stored[agent_id]
        for agent_id in registry
        if stored[agent_id]["status"] == "OK"
    ]
    own = stored[TARGET_AGENT]
    peers = [row for row in valid if row["agent_id"] != TARGET_AGENT]
    stored_peer_ids = json.loads(source["peer_response_ids"])
    reconstructed_peer_ids = [row["response_id"] for row in peers]
    if stored_peer_ids != reconstructed_peer_ids:
        raise ValueError(
            f"Stored peer order {stored_peer_ids} does not match "
            f"reconstruction {reconstructed_peer_ids}"
        )
    return build_round2_messages(
        question,
        own["raw_response"] if own["status"] == "OK" else None,
        [row["raw_response"] for row in peers],
    )


def prepare():
    """Validate the source and reconstruct the four original paid requests."""
    registry = load_model_registry(ROOT / "configs/models/agents_v6.yaml")
    spec = registry[TARGET_AGENT]
    questions = {question["stable_id"]: question for question in load_pilot_questions()}
    database = sqlite3.connect(f"file:{ROOT}/storage/results.sqlite?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        run = database.execute(
            "SELECT * FROM runs WHERE run_id = ?", (SOURCE_RUN,)
        ).fetchone()
        if not run or not run["ended_at"] or run["settings_version"] != SOURCE_SETTINGS:
            raise ValueError(f"Expected one finished {SOURCE_SETTINGS} pilot: {SOURCE_RUN}")
        response_count = database.execute(
            "SELECT COUNT(*) FROM model_responses WHERE run_id = ?", (SOURCE_RUN,)
        ).fetchone()[0]
        outcome_count = database.execute(
            "SELECT COUNT(*) FROM question_outcomes WHERE run_id = ?", (SOURCE_RUN,)
        ).fetchone()[0]
        if (response_count, outcome_count) != (200, 40):
            raise ValueError("Source is not the complete 20-question, two-round pilot")

        targets = database.execute(
            """
            SELECT * FROM model_responses
            WHERE run_id = ? AND agent_id = ? AND status = 'API_ERROR'
            ORDER BY response_id
            """,
            (SOURCE_RUN, TARGET_AGENT),
        ).fetchall()
        if len(targets) != EXPECTED_FAILURES:
            raise ValueError(
                f"Expected {EXPECTED_FAILURES} stored Mistral API failures, got {len(targets)}"
            )
        actual_targets = tuple((row["question_id"], row["round"]) for row in targets)
        if actual_targets != EXPECTED_TARGETS:
            raise ValueError(f"Unexpected Mistral failure set: {actual_targets}")

        plan = []
        for source in targets:
            if source["requested_slug"] != spec.slug or source["max_tokens"] != spec.max_tokens:
                raise ValueError("Stored Mistral request does not match agents_v6")
            if (
                source["temperature"] != spec.temperature
                or source["top_p"] != spec.top_p
                or source["attempt_count"] != 2
            ):
                raise ValueError("Stored Mistral generation settings or attempts changed")
            attempts = database.execute(
                """
                SELECT * FROM response_attempts
                WHERE response_id = ? ORDER BY attempt_number
                """,
                (source["response_id"],),
            ).fetchall()
            if len(attempts) != 2 or any(
                attempt["http_status"] != 429
                or '"provider_name":"Parasail"' not in attempt["raw_response"]
                or '"limit_source":"upstream_provider_shared_pool"'
                not in attempt["raw_response"]
                for attempt in attempts
            ):
                raise ValueError("Source failure is not the documented Parasail 429 pattern")
            question = questions[source["question_id"]]
            round1_rows = database.execute(
                """
                SELECT * FROM model_responses
                WHERE run_id = ? AND question_id = ? AND round = 1
                ORDER BY response_id
                """,
                (SOURCE_RUN, source["question_id"]),
            ).fetchall()
            messages = _reconstruct_messages(question, source, registry, round1_rows)
            plan.append((question, dict(source), messages))
        return spec, plan
    finally:
        database.close()


def _endpoint_snapshot(client, slug: str):
    response = client._session.get(
        f"https://openrouter.ai/api/v1/models/{slug}/endpoints", timeout=20
    )
    response.raise_for_status()
    endpoints = response.json()["data"]["endpoints"]
    healthy = [
        endpoint
        for endpoint in endpoints
        if endpoint.get("status") == 0
        and REQUIRED_PARAMETERS.issubset(
            set(endpoint.get("supported_parameters") or [])
        )
    ]
    providers = {
        endpoint.get("provider_name") for endpoint in healthy if endpoint.get("provider_name")
    }
    if len(providers) < 3:
        raise ValueError(f"Model has only {len(providers)} independent healthy providers")
    return healthy


def _endpoint_for(endpoints, pin: str):
    endpoint = next((item for item in endpoints if item.get("tag") == pin), None)
    if endpoint is None:
        raise ValueError(f"The healthy compatible endpoint {pin!r} disappeared")
    return endpoint


def _reserved_cost(messages, max_tokens, pricing) -> float:
    """Conservative byte-based bound; intentionally higher than token cost."""
    prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024
    allowed = {"prompt", "completion", "input_cache_read", "input_cache_write", "discount"}
    if any(key not in allowed and float(value) != 0 for key, value in pricing.items()):
        raise ValueError("Endpoint has an unbudgeted non-token charge")
    prompt_rate = float(pricing["prompt"])
    completion_rate = float(pricing["completion"])
    if not all(
        math.isfinite(value) and value >= 0
        for value in (prompt_rate, completion_rate)
    ):
        raise ValueError("Invalid endpoint pricing")
    return 1.2 * (prompt_bound * prompt_rate + max_tokens * completion_rate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--yes-spend-real-money", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=MAXIMUM_BUDGET_USD)
    args = parser.parse_args(argv)

    require_spend_confirmation(
        live=args.live,
        spend_confirmed=args.yes_spend_real_money,
        maximum_attempts=MAXIMUM_CALLS,
    )
    if (
        not math.isfinite(args.budget_usd)
        or args.budget_usd <= 0
        or args.budget_usd > MAXIMUM_BUDGET_USD
    ):
        raise ValueError(f"Diagnostic budget must be between $0 and ${MAXIMUM_BUDGET_USD:.2f}")

    source_spec, plan = prepare()
    print(f"source: {SOURCE_RUN}")
    print(f"model: {source_spec.slug}")
    print(f"calls: {len(plan)} prompts x {len(CANDIDATE_PINS)} hosts = {MAXIMUM_CALLS}")
    for question, source, _ in plan:
        print(f"  {question['stable_id']} round {source['round']}")
    if not args.live:
        print("DRY: exact messages reconstructed; no API calls or writes")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_path = ROOT / "storage" / f"mistral_host_probe_{stamp}.sqlite"
    with audit_path.open("xb"):
        pass
    audit = sqlite3.connect(audit_path)
    audit.execute(
        "CREATE TABLE events(id INTEGER PRIMARY KEY, timestamp TEXT, kind TEXT, body TEXT)"
    )

    def record(kind, body):
        audit.execute(
            "INSERT INTO events(timestamp, kind, body) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), kind, json.dumps(body)),
        )
        audit.commit()

    record(
        "protocol",
        {
            "source_run": SOURCE_RUN,
            "model": source_spec.slug,
            "candidate_pins": CANDIDATE_PINS,
            "calls": MAXIMUM_CALLS,
            "attempts_per_call": 1,
            "cache_bypassed": True,
            "answer_key_loaded": False,
            "budget_usd": args.budget_usd,
            "selection_basis": "completion and required format only, never correctness",
        },
    )
    print(f"audit: {audit_path}")

    results = {pin: Counter() for pin in CANDIDATE_PINS}
    reported_cost = 0.0
    try:
        with OpenRouterClient(max_attempts=1) as client:
            key_response = client._session.get("https://openrouter.ai/api/v1/key", timeout=20)
            key_response.raise_for_status()
            key = key_response.json()["data"]
            remaining = key.get("limit_remaining")
            if key.get("limit") is None or remaining is None or float(remaining) < args.budget_usd:
                raise ValueError("A finite key limit with enough remaining allowance is required")
            record(
                "key_guard",
                {name: key.get(name) for name in ("limit", "limit_remaining", "usage")},
            )

            endpoint_list = _endpoint_snapshot(client, source_spec.slug)
            endpoints = {pin: _endpoint_for(endpoint_list, pin) for pin in CANDIDATE_PINS}
            reservations = {
                (source["response_id"], pin): _reserved_cost(
                    messages, source_spec.max_tokens, endpoints[pin]["pricing"]
                )
                for _, source, messages in plan
                for pin in CANDIDATE_PINS
            }
            total_reservation = sum(reservations.values())
            if total_reservation > args.budget_usd:
                raise ValueError(
                    f"Conservative reservation ${total_reservation:.6f} exceeds "
                    f"the ${args.budget_usd:.2f} diagnostic cap"
                )
            record("endpoint_metadata", endpoints)
            record("reservation", {"usd": total_reservation})

            # Interleave hosts for each prompt so time-of-day affects them as
            # evenly as a short diagnostic can manage.
            for index, (question, source, messages) in enumerate(plan):
                pins = CANDIDATE_PINS if index % 2 == 0 else tuple(reversed(CANDIDATE_PINS))
                for pin in pins:
                    spec = replace(source_spec, pinned_provider=pin)
                    label = f"{question['stable_id']} R{source['round']} {pin}"
                    print(f"CALL {label}", flush=True)
                    record(
                        "request",
                        {
                            "label": label,
                            "source_response_id": source["response_id"],
                            "spec": asdict(spec),
                            "messages": messages,
                        },
                    )
                    try:
                        completion = client.complete(spec, messages)
                    except ApiRequestError as error:
                        attempts = [asdict(attempt) for attempt in error.attempt_log]
                        diagnoses = [
                            asdict(diagnose_attempt(attempt)) for attempt in error.attempt_log
                        ]
                        record(
                            "api_error",
                            {"label": label, "attempts": attempts, "diagnoses": diagnoses},
                        )
                        results[pin]["API_ERROR"] += 1
                        print(f"  API_ERROR {diagnoses}", flush=True)
                        continue

                    record(
                        "raw_completion",
                        {"label": label, "result": asdict(completion)},
                    )
                    expected_provider = endpoints[pin]["provider_name"]
                    if completion.provider != expected_provider:
                        raise ValueError(
                            f"Requested {pin!r} but response names {completion.provider!r}"
                        )
                    reservation = reservations[(source["response_id"], pin)]
                    if completion.cost_usd > reservation:
                        raise ValueError(
                            f"Charge ${completion.cost_usd:.6f} exceeded reserved "
                            f"${reservation:.6f} for {label}"
                        )
                    parsed = parse_response(
                        completion.text,
                        answer_letters(question),
                        finish_reason=completion.finish_reason,
                    )
                    record(
                        "parsed",
                        {
                            "label": label,
                            "parsed": asdict(parsed),
                        },
                    )
                    results[pin][parsed.status] += 1
                    reported_cost += completion.cost_usd
                    print(
                        f"  {parsed.status} provider={completion.provider} "
                        f"finish={completion.finish_reason} "
                        f"tokens={completion.completion_tokens} "
                        f"cost=${completion.cost_usd:.6f}",
                        flush=True,
                    )

            record(
                "summary",
                {
                    "results": {pin: dict(counts) for pin, counts in results.items()},
                    "reported_cost_usd": reported_cost,
                },
            )
    except Exception as error:
        record(
            "local_stop",
            {"type": type(error).__name__, "message": str(error)[:500]},
        )
        raise
    finally:
        audit.close()

    print("\nsummary")
    for pin, counts in results.items():
        print(f"  {pin}: {dict(counts)}")
    print(f"reported cost: ${reported_cost:.6f}")
    winners = [pin for pin, counts in results.items() if counts == Counter({"OK": len(plan)})]
    if not winners:
        print("decision: neither host passed 4/4; do not create agents_v7")
        return 1
    print("eligible hosts: " + ", ".join(winners))
    print("No answer correctness was used to select a provider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
