"""The response cache. Exists so the same call is never paid for twice.

Separate file from results.sqlite - this is a wallet protection, not part of
the experimental record. Deleting it loses money, not results.

What gets cached: genuine model outcomes, refusals and unparseable replies
included - a refusal is what the model really did, and asking again costs money
to hear it again. What never gets cached: transport errors, rate limits and
provider failures - caching those would make a temporary problem permanent.

A hit returns the stored reply with its original tokens and latency, costs
nothing, and makes no API call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from mad.api_client import CompletionResult, ModelSpec, _parse_completion


# 1. Settings

CACHE_SCHEMA_VERSION = 1

DEFAULT_CACHE_PATH = Path("storage") / "cache.sqlite"


# 2. Errors


class CacheError(RuntimeError):
    """Something was handed to the cache that must not go in it."""


# 3. The key


def cache_key(spec: ModelSpec, messages: list[dict[str, str]]) -> str:
    """One key per distinct request.

    Hashes everything that changes what a call would return: the agent, the
    model, the full messages, and the sampling settings. agent_id is included
    even though the API never sees it, so two agents making an identical
    request cannot share one reply. The run ID is left out on purpose - a
    reply is reusable across runs, that is the point of the cache.
    """
    payload = {
        "agent_id": spec.agent_id,
        "slug": spec.slug,
        "messages": messages,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
    }
    # Both of these change what a call would return, so they are part of the
    # request's identity. Added only when set, so every reply cached before
    # they existed keeps its key.
    if spec.pinned_provider:
        payload["pinned_provider"] = spec.pinned_provider
    if spec.reasoning_max_tokens is not None:
        payload["reasoning_max_tokens"] = spec.reasoning_max_tokens
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# 4. The cache


class ResponseCache:
    """One SQLite file mapping request keys to complete raw API replies."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _repository_root() / DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cached_responses (
                cache_key       TEXT PRIMARY KEY,
                agent_id        TEXT NOT NULL,
                slug            TEXT NOT NULL,
                raw_response    TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                created_at      TEXT NOT NULL
            )
            """
        )
        self._connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
        self._connection.commit()

    def lookup(self, spec: ModelSpec, messages: list[dict[str, str]]) -> CompletionResult | None:
        """The stored reply for this exact request, or None. Never calls the API.

        A hit keeps the original tokens and latency but costs nothing, so run
        totals never count the same spend twice.
        """
        row = self._connection.execute(
            "SELECT raw_response, latency_seconds FROM cached_responses WHERE cache_key = ?",
            (cache_key(spec, messages),),
        ).fetchone()
        if row is None:
            return None

        result = _parse_completion(
            json.loads(row["raw_response"]),
            spec=spec,
            latency_seconds=row["latency_seconds"],
            attempt_log=(),   # a hit made no API attempt
        )
        return replace(result, cost_usd=0.0)

    def store(self, spec: ModelSpec, messages: list[dict[str, str]], result: CompletionResult) -> None:
        """Keep one genuine reply. First write wins; nothing is ever overwritten."""
        # "fixture" marks dry-run stand-ins. One in the cache could later be
        # served as if a model had said it, which is fabrication.
        if result.provider == "fixture":
            raise CacheError("fixture replies never enter the cache")
        if not result.raw_response:
            raise CacheError("a reply without its raw body cannot be cached")

        self._connection.execute(
            # OR IGNORE: if the key exists, the original stays. A cache that
            # rewrote its entries would let a rerun quietly replay different
            # replies than the run before it.
            "INSERT OR IGNORE INTO cached_responses "
            "(cache_key, agent_id, slug, raw_response, latency_seconds, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                cache_key(spec, messages),
                spec.agent_id,
                spec.slug,
                json.dumps(result.raw_response),
                result.latency_seconds,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# 5. Helpers


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
