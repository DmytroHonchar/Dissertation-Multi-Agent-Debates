"""Verify every configured agent slug exists on OpenRouter and answers a prompt.

A model passes the smoke test when the request succeeds, the response body is
well formed, and the visible content is not empty. Empty content is a FAILURE,
not a pass: some reasoning models spend their whole completion budget before
emitting anything visible, which is exactly the condition the pilot must catch.
See docs/decisions.md D002.

OpenRouter chooses the upstream provider. The served model and provider are
logged on every call, but a response is never rejected for coming from a
different provider than last time — pinning is deferred until before the pilot
(D015).

Usage:
    python scripts/check_models.py                 # registry check only, no cost
    python scripts/check_models.py --smoke-test    # also send one real prompt each
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mad.api_client import (  # noqa: E402
    ApiRequestError,
    OpenRouterClient,
    load_model_registry,
)

# 1. Settings

MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/models/agents_v1.yaml"
SMOKE_TEST_MESSAGES = [
    {"role": "user", "content": "Reply with exactly one word: ready"},
]
# Big enough that a reasoning model can think and still say something.
# At 16 tokens, Qwen and DeepSeek returned nothing at all.
SMOKE_TEST_MAX_TOKENS = 256


# 2. The check


def main() -> int:
    """Look up all five models, and optionally send each one a real prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG_PATH)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-tokens", type=int, default=SMOKE_TEST_MAX_TOKENS)
    args = parser.parse_args()

    registry = load_model_registry(args.config)
    failures = 0

    with OpenRouterClient() as client:
        available = client.list_available_models()

        for agent_id, spec in registry.items():
            model = available.get(spec.slug)
            if model is None:
                failures += 1
                print(f"[MISSING] {agent_id:<16} {spec.slug} is not served by OpenRouter")
                for candidate in _suggest(spec.slug, available):
                    print(f"           did you mean: {candidate}")
                continue

            pricing = model.get("pricing", {})
            print(
                f"[OK]      {agent_id:<16} {spec.slug}\n"
                f"           context={model.get('context_length')} "
                f"prompt=${pricing.get('prompt')}/tok "
                f"completion=${pricing.get('completion')}/tok"
            )

            if args.smoke_test:
                try:
                    result = client.complete(
                        spec, SMOKE_TEST_MESSAGES, max_tokens=args.smoke_max_tokens
                    )
                except ApiRequestError as error:
                    failures += 1
                    print(f"           smoke test FAILED: {error}")
                    continue

                print(
                    f"           requested={result.requested_slug} "
                    f"served={result.served_slug} provider={result.provider!r}"
                )
                print(
                    f"           finish={result.finish_reason!r} "
                    f"tokens={result.prompt_tokens}+{result.completion_tokens} "
                    f"{result.latency_seconds:.1f}s ${result.cost_usd:.6f} "
                    f"attempts={result.attempts}"
                )

                if not result.text.strip():
                    failures += 1
                    print(
                        "           smoke test FAILED: empty visible content "
                        f"(finish_reason={result.finish_reason!r}, "
                        f"{result.completion_tokens} completion tokens). "
                        "Raise --smoke-max-tokens or treat this model as unusable."
                    )
                else:
                    print(f"           smoke test ok: {result.text.strip()!r}")

    print(f"\n{len(registry) - failures}/{len(registry)} agents usable.")
    return 1 if failures else 0


# 3. Helpers


def _suggest(slug: str, available: dict[str, dict]) -> list[str]:
    """If a model ID is wrong, list others from the same company as hints."""
    family = slug.split("/")[0]
    return sorted(other for other in available if other.startswith(f"{family}/"))[:8]


if __name__ == "__main__":
    raise SystemExit(main())
