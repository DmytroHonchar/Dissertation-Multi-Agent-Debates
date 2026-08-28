"""Verify every configured agent slug exists on OpenRouter and answers a prompt.

Usage:
    python scripts/check_models.py                 # registry check only
    python scripts/check_models.py --smoke-test    # also send one cheap prompt each
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

MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/models/agents_v1.yaml"
SMOKE_TEST_MESSAGES = [
    {"role": "user", "content": "Reply with exactly one word: ready"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG_PATH)
    parser.add_argument("--smoke-test", action="store_true")
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
                    result = client.complete(spec, SMOKE_TEST_MESSAGES, max_tokens=16)
                except ApiRequestError as error:
                    failures += 1
                    print(f"           smoke test FAILED: {error}")
                else:
                    print(
                        f"           smoke test ok: {result.text.strip()!r} "
                        f"via {result.provider} "
                        f"({result.latency_seconds:.1f}s, ${result.cost_usd:.6f})"
                    )

    print(f"\n{len(registry) - failures}/{len(registry)} agents usable.")
    return 1 if failures else 0


def _suggest(slug: str, available: dict[str, dict]) -> list[str]:
    """Offer same-family slugs so a renamed or hallucinated model is easy to fix."""
    family = slug.split("/")[0]
    return sorted(other for other in available if other.startswith(f"{family}/"))[:8]


if __name__ == "__main__":
    raise SystemExit(main())
