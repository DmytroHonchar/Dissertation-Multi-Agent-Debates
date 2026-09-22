# Paid diagnostic results — 22 September 2026

Protocol D023. Six fresh single-attempt requests, no retries, about three
minutes in one session. Reported charge **$0.029602848**, against conservative
reservations of $0.0771048 within the $0.10 test ceiling. The account key had a
separate $1 limit with $0.71531925 remaining before the tests. These are not
formal pilot or main-experiment results.

## DeepSeek availability

Three calls to the existing DigitalOcean endpoint, distributed through the
session, all completed and parsed OK. They used the original Round 1 prompt
for pilot question 3932 and unchanged 2048-token settings. Output tokens were
215, 228 and 234; charges were $0.001322640, $0.001047464 and $0.001071224.

The endpoint works in this session. This does not establish all-day reliability
or disprove the earlier outage. No provider or production retry change made.

## Qwen token comparison

Only the total ceiling changed: 3072 to 4096. Same Parasail/fp8 pin, prompts,
temperature, top-p and best-effort reasoning request (2048). Round 2 used the
original pilot's Round 1 responses, not today's new ones, to hold its input
fixed. Original request identities were verified against stored cache keys;
the cache was deliberately bypassed for these fresh diagnostic requests.

| Question | Round | Original status | New status | Output tokens | Reported reasoning tokens | Cost USD |
|---|---:|---|---|---:|---:|---:|
| 9622 | 1 | TRUNCATED | OK | 3348 | 3073 | 0.007476000 |
| 11875 | 1 | TRUNCATED | TRUNCATED | 4096 | 3967 | 0.009130000 |
| 11875 | 2 | TRUNCATED | TRUNCATED | 4096 | 3962 | 0.009555520 |

The extra allowance enabled one completion but did not reliably fix truncation.
Do not keep escalating. Reported reasoning again exceeded the requested 2048,
so that request is not an enforced partition. A single fresh response is not
a deterministic replay; the comparison does not prove the ceiling alone caused
every difference. No answer key was read or accuracy used to select settings.

## Deferred tests: endpoint availability

Free endpoint metadata returned an empty endpoint list for the configured
Mistral model. Gemma's pinned `deepinfra/fp8` appeared with status -2 rather
than active status 0. Those token tests were skipped, not counted as failures
or silently sent to different providers. This is a current availability
warning, not proof of permanent withdrawal.

Before another pilot, resolve these endpoints. A changed provider requires a
new configuration and a separate comparison; otherwise provider and token
changes would be mixed together. No replacement was selected here.

## Audit and preserved data

Exact requests, settings, price snapshots, raw replies, attempt logs and parsed
statuses are in `storage/diagnostic_20260922T150205759313Z.sqlite`.
The one-off script is `storage/probe_20260922.py` (dry preparation by default).
Both are local ignored diagnostic artifacts; back them up with experimental
storage, not as fabricated rows in the main results database.

Production results, cache and agents_v5 hashes were unchanged after testing:

- results.sqlite: `276ca22a891af536239dab1adab9d9b6426ee897f1c6787a3fa3ef027f428f4f`
- cache.sqlite: `f71a04ca81593c3894d3bac0a8108a1d6ab8c75225a0918ad2f1fa626cb76cea`
- agents_v5.yaml: `6d4ebcdd005ced5eb0ebd266ddac9ba770723f20561797d233c82c4d7eca0185`

This was superseded later on 22 September by D024: `agents_v6` now exists as an
unvalidated replacement candidate. No paid call or formal pilot has used it
yet. The main experiment remains uncleared.
