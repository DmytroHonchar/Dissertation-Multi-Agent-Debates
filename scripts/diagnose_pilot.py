"""Replay failed pilot requests under their original settings, with a separate audit.

Dry by default. This is not a repaired pilot: original rows and cache are read
only. A fresh request bypasses cache intentionally; no answer key is loaded.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mad.api_client import OpenRouterClient, ApiRequestError, load_model_registry
from mad.cache import cache_key
from mad.diagnostics import diagnose_attempt, diagnose_output
from mad.parser_v1 import parse_response
from mad.prompts_v1 import build_round1_messages, build_round2_messages, answer_letters
from mad.runner import require_spend_confirmation, load_pilot_questions

SOURCE_RUN = 'pilot_agents_v5_20260910T161210Z'


def reconstruct_messages(question, round_number, agent_id, registry, round1_rows):
    if round_number == 1:
        return build_round1_messages(question)
    stored = {row['agent_id']: row for row in round1_rows}
    if set(stored) != set(registry):
        raise ValueError('Original Round 1 agent set is incomplete')
    valid = {aid: stored[aid]['raw_response'] for aid in registry if stored[aid]['status'] == 'OK'}
    return build_round2_messages(question, valid.get(agent_id),
                                 [text for aid, text in valid.items() if aid != agent_id])


def prepare():
    registry = load_model_registry(ROOT / 'configs/models/agents_v5.yaml')
    questions = {q['stable_id']: q for q in load_pilot_questions()}
    db = sqlite3.connect(f'file:{ROOT}/storage/results.sqlite?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    cache = sqlite3.connect(f'file:{ROOT}/storage/cache.sqlite?mode=ro', uri=True)
    try:
        run = db.execute('SELECT * FROM runs WHERE run_id=?', (SOURCE_RUN,)).fetchone()
        if not run or not run['ended_at'] or run['settings_version'] != 'agents_v5':
            raise ValueError('Expected finished agents_v5 pilot')
        targets = db.execute("SELECT * FROM model_responses WHERE run_id=? AND status!='OK' ORDER BY response_id", (SOURCE_RUN,)).fetchall()
        if len(targets) != 15:
            raise ValueError('Expected the 15 documented pilot failures; inspect changed source first')
        plan = []
        for source in targets:
            spec = registry[source['agent_id']]
            if source['max_tokens'] != spec.max_tokens or source['requested_slug'] != spec.slug:
                raise ValueError('Stored request does not match the original model config')
            question = questions[source['question_id']]
            rows = db.execute('SELECT * FROM model_responses WHERE run_id=? AND question_id=? AND round=1',
                              (SOURCE_RUN, source['question_id'])).fetchall()
            messages = reconstruct_messages(question, source['round'], spec.agent_id, registry, rows)
            original_key = cache_key(spec, messages)
            if source['status'] != 'API_ERROR' and not cache.execute(
                'SELECT 1 FROM cached_responses WHERE cache_key=?', (original_key,)).fetchone():
                raise ValueError('Reconstructed original request does not match cache')
            original_attempts = [dict(a) for a in db.execute(
                'SELECT * FROM response_attempts WHERE response_id=? ORDER BY attempt_number', (source['response_id'],))]
            plan.append((spec, messages, question, dict(source), original_attempts))
        return plan
    finally:
        db.close()
        cache.close()


def reserve_cost(messages, max_tokens, pricing):
    """Conservative text-token estimate, not a billing guarantee."""
    input_bound = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
    rates = [float(pricing[k]) for k in ('prompt', 'completion')]
    if not all(math.isfinite(rate) and rate >= 0 for rate in rates):
        raise ValueError('Invalid pricing; refuse to spend')
    return 1.2 * (input_bound * rates[0] + max_tokens * rates[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--yes-spend-real-money', action='store_true')
    parser.add_argument('--budget-usd', type=float, default=0.30)
    args = parser.parse_args()
    require_spend_confirmation(live=args.live, spend_confirmed=args.yes_spend_real_money, maximum_attempts=15)
    if not math.isfinite(args.budget_usd) or not 0 < args.budget_usd <= 0.30:
        raise ValueError('This diagnostic is limited to $0.30')
    plan = prepare()  # All input/source validation happens before client creation.
    for i, (spec, _, q, source, _) in enumerate(plan, 1):
        print(i, spec.agent_id, q['stable_id'], 'R'+str(source['round']), source['status'], 'limit', spec.max_tokens, flush=True)
    if not args.live:
        print('DRY: original messages reconstructed. Zero calls or writes.')
        return

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = ROOT / 'storage' / f'failure_replay_{stamp}.sqlite'
    # Reserve a new audit path exclusively; never reuse a prior diagnostic.
    with path.open('xb'):
        pass
    audit = sqlite3.connect(path)
    audit.execute('CREATE TABLE events(id INTEGER PRIMARY KEY, timestamp TEXT, kind TEXT, body TEXT)')
    def record(kind, body):
        audit.execute('INSERT INTO events(timestamp,kind,body) VALUES (?,?,?)',
                      (datetime.now(timezone.utc).isoformat(), kind, json.dumps(body)))
        audit.commit()
    record('protocol', {'source_run': SOURCE_RUN, 'request_settings': 'agents_v5 unchanged',
                        'max_attempts_per_request': 1, 'cache_bypassed': True,
                        'budget_usd': args.budget_usd, 'purpose': 'diagnostic, not formal results'})
    print('AUDIT', path, flush=True)
    spent_or_reserved = 0.0
    reported = 0.0
    try:
        with OpenRouterClient(max_attempts=1) as client:
            key_response = client._session.get('https://openrouter.ai/api/v1/key', timeout=20)
            key_response.raise_for_status()
            key = key_response.json()['data']
            if key.get('limit') is None or (key.get('limit_remaining') or 0) < args.budget_usd:
                raise ValueError('Finite key limit and sufficient allowance required')
            record('key_guard', {k: key.get(k) for k in ('limit', 'limit_remaining', 'usage')})
            for i, (spec, messages, question, source, old_attempts) in enumerate(plan, 1):
                response = client._session.get(f'https://openrouter.ai/api/v1/models/{spec.slug}/endpoints', timeout=20)
                response.raise_for_status()
                metadata = response.json()
                endpoints = metadata['data']['endpoints']
                endpoint = next((e for e in endpoints if e.get('tag') == spec.pinned_provider), None)
                # We intentionally test the pinned route even if metadata says
                # unavailable: its actual HTTP reply is the evidence requested.
                pricing = endpoint['pricing'] if endpoint else {'prompt': '0.00002', 'completion': '0.00002'}
                price_source = 'endpoint' if endpoint else 'conservative allowance ($20/million); no listed price'
                allowed = {'prompt', 'completion', 'input_cache_read', 'input_cache_write', 'discount'}
                if any(k not in allowed and float(v) != 0 for k, v in pricing.items()):
                    raise ValueError('Unbudgeted non-token charge')
                reserve = reserve_cost(messages, spec.max_tokens, pricing)
                if spent_or_reserved + reserve > args.budget_usd:
                    record('stop', {'reason': 'budget', 'index': i, 'reserved': spent_or_reserved, 'next_reserve': reserve})
                    print('STOP: spending allowance would be exceeded', flush=True)
                    break
                spent_or_reserved += reserve
                record('request', {'index': i, 'spec': asdict(spec), 'messages': messages,
                                   'original': source, 'original_attempts': old_attempts,
                                   'endpoint_metadata': metadata, 'price_source': price_source, 'reserve_usd': reserve})
                print('CALL', i, spec.agent_id, question['stable_id'], 'R'+str(source['round']), flush=True)
                try:
                    result = client.complete(spec, messages)
                except ApiRequestError as error:
                    attempts = [asdict(a) for a in error.attempt_log]
                    diagnoses = [asdict(diagnose_attempt(a)) for a in error.attempt_log]
                    record('api_error', {'index': i, 'attempts': attempts, 'diagnoses': diagnoses})
                    print('RESULT', i, 'API_ERROR', [d['category'] for d in diagnoses], str(error)[:220], flush=True)
                    continue  # Keep the reservation: missing cost is not proven free.
                record('raw_completion', {'index': i, 'result': asdict(result)})
                parsed = parse_response(result.text, answer_letters(question), finish_reason=result.finish_reason)
                record('parsed', {'index': i, 'parsed': asdict(parsed), 'diagnosis': asdict(diagnose_output(parsed.status))})
                reported += result.cost_usd
                spent_or_reserved += result.cost_usd - reserve
                print('RESULT', i, parsed.status, result.finish_reason, 'tokens', result.completion_tokens,
                      'cost', result.cost_usd, 'provider', result.provider, flush=True)
                if result.cost_usd > reserve:
                    record('stop', {'reason': 'charge exceeded reservation', 'index': i})
                    break
            record('finished', {'reported_completion_cost': reported, 'spent_or_reserved': spent_or_reserved})
            print('Reported completion cost', reported, 'including unknown-cost reservations', spent_or_reserved, flush=True)
    except Exception as error:
        record('local_stop', {'exception_type': type(error).__name__, 'message': str(error)[:500]})
        raise
    finally:
        audit.close()


if __name__ == '__main__':
    main()
