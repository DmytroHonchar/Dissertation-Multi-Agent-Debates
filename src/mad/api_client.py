"""Talks to OpenRouter. Every paid API call in the project goes through here.

One key, five models. Nothing else in the project may contact a provider
directly - if it did, the run would stop being auditable.

Every reply comes back with its cost, timing and which provider actually served
it, so the results can be checked later.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml


# 1. Settings

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0

# One try, then one retry. That is the whole retry budget.
DEFAULT_MAX_ATTEMPTS = 2

# Temporary problems worth retrying: timeouts, rate limits, provider outages.
# Anything else (bad key, unknown model) fails immediately - retrying won't help.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


# 2. Errors

class ApiConfigurationError(RuntimeError):
    """The key or the model config is missing or broken."""


class ApiRequestError(RuntimeError):
    """The call failed and the retry didn't save it.

    Carries the record of every attempt made, so a call that never succeeded can
    still be stored and costed instead of vanishing.
    """

    def __init__(self, message: str, attempt_log: tuple[AttemptRecord, ...] = ()) -> None:
        super().__init__(message)
        self.attempt_log = attempt_log


# 3. What goes out, what comes back


@dataclass(frozen=True)
class ModelSpec:
    """One agent: which model it is and how to sample it.

    Built from configs/models/agents_v1.yaml. Frozen means it can't be edited
    after creation, so nothing can quietly change a model's settings mid-run.
    """

    agent_id: str
    slug: str
    display_name: str
    developer: str
    temperature: float
    top_p: float
    max_tokens: int
    allow_provider_fallbacks: bool
    require_parameters: bool


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt at one call, successful or not.

    Failures cost money and time, so every attempt is recorded, not just the one
    that worked. Token counts stay 0 unless the reply carried a usage block.
    """

    attempt: int
    outcome: str                 # ok, http_error, upstream_error, transport_error, bad_json
    latency_seconds: float
    status_code: int | None = None
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CompletionResult:
    """One reply from a model, plus everything needed to audit it later."""

    text: str
    agent_id: str
    requested_slug: str      # the model we asked for
    served_slug: str         # the model OpenRouter actually used
    provider: str            # who served it, e.g. DeepInfra
    generation_id: str
    finish_reason: str       # "stop" = finished, "length" = ran out of tokens
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    attempts: int

    # OpenRouter's reply exactly as it arrived. The cache stores this whole body,
    # so nothing is lost by the fields above being a selection.
    raw_response: dict[str, Any] = field(default_factory=dict)
    # Every attempt, including ones that failed before this one succeeded.
    attempt_log: tuple[AttemptRecord, ...] = ()


# 4. Loading the key and the model config


def load_env_file(env_path: str | Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Skips blanks and # comments. Never overwrites a variable that is already
    set, so a real environment variable always wins over the file.
    """
    path = Path(env_path) if env_path else _repository_root() / ".env"
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_api_key() -> str:
    """Get the OpenRouter key. Crashes if it's missing."""
    load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ApiConfigurationError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def load_model_registry(config_path: str | Path) -> dict[str, ModelSpec]:
    """Turn the agents YAML into one ModelSpec per agent."""
    resolved_path = Path(config_path).resolve()
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        defaults = dict(config.get("defaults") or {})
        agents = config["agents"]
    except (KeyError, TypeError) as error:
        raise ApiConfigurationError(f"Invalid model registry: {resolved_path}") from error

    registry: dict[str, ModelSpec] = {}
    for agent_id, agent_config in agents.items():
        # The agent's own settings win over the shared defaults.
        settings = {**defaults, **dict(agent_config)}
        try:
            registry[agent_id] = ModelSpec(
                agent_id=agent_id,
                slug=settings["slug"],
                display_name=settings.get("display_name", settings["slug"]),
                developer=settings.get("developer", "unknown"),
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                max_tokens=int(settings["max_tokens"]),
                allow_provider_fallbacks=bool(settings.get("allow_provider_fallbacks", False)),
                require_parameters=bool(settings.get("require_parameters", True)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApiConfigurationError(f"Invalid agent config: {agent_id!r}") from error

    return registry


# 5. The client


class OpenRouterClient:
    """Sends requests to OpenRouter and retries once on a temporary failure."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        # One Session, so connections can be reused between calls instead of a new
        # handshake each time. It reopens one when it has to.
        self._session = requests.Session()
        self._session.headers.update(_build_headers(api_key or load_api_key()))

    def complete(
        self,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        *,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Send one question to one model. Returns the reply, its cost and timing.

        `messages` is what prompts_v1.build_round1_messages produced.
        """
        payload: dict[str, Any] = {
            "model": spec.slug,
            "messages": messages,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": max_tokens if max_tokens is not None else spec.max_tokens,
            "usage": {"include": True},   # ask for token counts and cost back
            "provider": {
                # OpenRouter picks the provider for now. Pinning comes before the pilot.
                "allow_fallbacks": spec.allow_provider_fallbacks,
                # Only use providers that honour temperature and top_p, so
                # temperature 0 can't be silently ignored.
                "require_parameters": spec.require_parameters,
            },
        }
        if seed is not None:
            payload["seed"] = seed

        started_at = time.monotonic()
        body, attempt_log = self._post_with_retries("/chat/completions", payload)
        latency_seconds = time.monotonic() - started_at

        return _parse_completion(
            body,
            spec=spec,
            latency_seconds=latency_seconds,
            attempt_log=attempt_log,
        )

    def list_available_models(self) -> dict[str, dict[str, Any]]:
        """Every model OpenRouter currently serves. Free - uses no tokens."""
        response = self._session.get(f"{self._base_url}/models", timeout=self._timeout_seconds)
        response.raise_for_status()
        return {model["id"]: model for model in response.json()["data"]}

    def close(self) -> None:
        self._session.close()

    # These two let you write:  with OpenRouterClient() as client:
    # so the connection always closes, even if something crashes.
    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post_with_retries(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], tuple[AttemptRecord, ...]]:
        """Send the request. Try again once if the failure looks temporary.

        Returns the reply body and the record of every attempt made. On total
        failure the same record is attached to the ApiRequestError, so a dead
        call can still be costed.
        """
        url = f"{self._base_url}{path}"
        log: list[AttemptRecord] = []
        last_error = "unknown error"

        for attempt in range(1, self._max_attempts + 1):
            started_at = time.monotonic()
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout_seconds)
            except requests.RequestException as error:
                # Network died, DNS failed, timed out. Worth another go.
                last_error = f"transport error: {error}"
                log.append(
                    AttemptRecord(
                        attempt=attempt,
                        outcome="transport_error",
                        latency_seconds=time.monotonic() - started_at,
                        error=last_error[:400],
                    )
                )
            else:
                elapsed = time.monotonic() - started_at
                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError as error:
                        # HTTP 200 with a body that isn't JSON. Usually a proxy
                        # or gateway page, so it is worth one more go.
                        last_error = f"invalid JSON: {error}"
                        log.append(
                            AttemptRecord(
                                attempt=attempt,
                                outcome="bad_json",
                                latency_seconds=elapsed,
                                status_code=200,
                                error=last_error[:400],
                            )
                        )
                    else:
                        # OpenRouter sometimes reports upstream failures as HTTP 200
                        # with an "error" key inside, so 200 alone isn't success.
                        usage = body.get("usage") or {}
                        record = AttemptRecord(
                            attempt=attempt,
                            outcome="ok" if "error" not in body else "upstream_error",
                            latency_seconds=elapsed,
                            status_code=200,
                            error="" if "error" not in body else str(body["error"])[:400],
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                            cost_usd=float(usage.get("cost", 0.0)),
                        )
                        log.append(record)
                        if record.outcome == "ok":
                            return body, tuple(log)
                        last_error = f"upstream error: {body['error']}"
                elif response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}: {response.text[:400]}"
                    log.append(
                        AttemptRecord(
                            attempt=attempt,
                            outcome="http_error",
                            latency_seconds=elapsed,
                            status_code=response.status_code,
                            error=last_error[:400],
                        )
                    )
                else:
                    # Permanent problem, e.g. bad key or unknown model. Give up now.
                    log.append(
                        AttemptRecord(
                            attempt=attempt,
                            outcome="http_error",
                            latency_seconds=elapsed,
                            status_code=response.status_code,
                            error=response.text[:400],
                        )
                    )
                    raise ApiRequestError(
                        f"{payload['model']} failed with HTTP {response.status_code}: "
                        f"{response.text[:400]}",
                        tuple(log),
                    )

            if attempt < self._max_attempts:
                # Wait before retrying, plus a random fraction of a second so
                # five agents retrying at once don't all hit the API together.
                time.sleep(min(2.0**attempt, 30.0) + random.uniform(0.0, 1.0))

        raise ApiRequestError(
            f"{payload['model']} failed after {self._max_attempts} attempts: {last_error}",
            tuple(log),
        )


# 6. Helpers


def _build_headers(api_key: str) -> dict[str, str]:
    """The HTTP headers, including the key. The two app headers are optional."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    app_url = os.environ.get("OPENROUTER_APP_URL", "").strip()
    app_title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title
    return headers


def _parse_completion(
    body: dict[str, Any],
    *,
    spec: ModelSpec,
    latency_seconds: float,
    attempt_log: tuple[AttemptRecord, ...] = (),
) -> CompletionResult:
    """Pull the reply text and the usage numbers out of OpenRouter's JSON."""
    try:
        choice = body["choices"][0]
        # content can be null. Keep it as "" and let the parser decide it failed.
        text = choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise ApiRequestError(f"Malformed response for {spec.slug}: {body}") from error

    usage = body.get("usage") or {}
    return CompletionResult(
        text=text,
        agent_id=spec.agent_id,
        requested_slug=spec.slug,
        served_slug=body.get("model", spec.slug),
        provider=body.get("provider", "unknown"),
        generation_id=body.get("id", ""),
        finish_reason=choice.get("finish_reason") or "",
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        cost_usd=float(usage.get("cost", 0.0)),
        latency_seconds=latency_seconds,
        attempts=len(attempt_log) or 1,
        raw_response=body,
        attempt_log=attempt_log,
    )


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
