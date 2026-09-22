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
from dataclasses import dataclass, field, replace
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

# A model is accepted for a live experiment only while at least three separate
# companies can serve it. The run stays pinned to one exact provider; this rule
# ensures that losing one host does not force another model replacement.
MINIMUM_HEALTHY_PROVIDERS = 3


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

    # Set to a provider name to route every call for this agent there and
    # nowhere else. None keeps automatic routing (D015).
    pinned_provider: str | None = None
    # Ceiling on hidden reasoning tokens, spent inside max_tokens. Guards the
    # space for the visible answer; None keeps the model's default behaviour.
    reasoning_max_tokens: int | None = None


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt at one call, successful or not.

    Failures cost money and time, so every attempt is recorded, not just the one
    that worked. Token counts stay 0 unless the reply carried a usage block.
    """

    attempt: int
    outcome: str                 # ok, http_error, upstream_error, transport_error,
                                 # bad_json, malformed_body
    latency_seconds: float
    status_code: int | None = None
    # The response body exactly as it arrived, untruncated. P4 stores this per
    # attempt, so a failed attempt is as auditable as a successful one. Empty
    # only when the request never got a response at all.
    raw_response: str = ""
    finish_reason: str = ""
    error: str = ""              # short summary for logs; raw_response is the record
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # Safe response-side diagnostics; never store request/authentication headers.
    response_headers: dict[str, str] = field(default_factory=dict)
    transport_exception: str = ""


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
                pinned_provider=settings.get("pinned_provider"),
                reasoning_max_tokens=(
                    int(settings["reasoning_max_tokens"])
                    if settings.get("reasoning_max_tokens") is not None else None
                ),
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

        `messages` is what one of the versioned prompt builders produced.
        """
        payload: dict[str, Any] = {
            "model": spec.slug,
            "messages": messages,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": max_tokens if max_tokens is not None else spec.max_tokens,
            "usage": {"include": True},   # ask for token counts and cost back
            "provider": {
                "allow_fallbacks": spec.allow_provider_fallbacks,
                # Only use providers that honour temperature and top_p, so
                # temperature 0 can't be silently ignored.
                "require_parameters": spec.require_parameters,
            },
        }
        if spec.pinned_provider:
            # This agent goes to one named provider and nowhere else.
            payload["provider"]["only"] = [spec.pinned_provider]
        if spec.reasoning_max_tokens is not None:
            # Cap the hidden thinking so the visible answer keeps its space.
            payload["reasoning"] = {"max_tokens": spec.reasoning_max_tokens}
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

    def assert_registry_routes_available(self, registry: dict[str, ModelSpec]) -> None:
        """Refuse a live run if its pin or multi-host safety margin disappeared.

        OpenRouter may keep a model page after its serving endpoints are gone.
        Checking the broad model catalogue is therefore insufficient. A usable
        endpoint must be marked healthy and support the fixed generation
        parameters. Provider names are deduplicated, so three endpoint variants
        from one company still count as one independent host.

        The run remains pinned and never silently fails over mid-experiment.
        Alternative hosts are an escape route for a later, newly versioned
        configuration if the pin is withdrawn.
        """
        missing: list[str] = []
        for agent_id, spec in registry.items():
            url = f"{self._base_url}/models/{spec.slug}/endpoints"
            try:
                response = self._session.get(url, timeout=self._timeout_seconds)
                response.raise_for_status()
                body = response.json()
                endpoints = body["data"]["endpoints"]
                if not isinstance(endpoints, list):
                    raise TypeError("endpoints is not a list")
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                raise ApiConfigurationError(
                    f"could not verify the free endpoint list for {agent_id} "
                    f"({spec.slug}): {error}"
                ) from error
            required_parameters = (
                {"temperature", "top_p", "max_tokens"}
                if spec.require_parameters
                else set()
            )
            healthy_endpoints = [
                endpoint
                for endpoint in endpoints
                if isinstance(endpoint, dict)
                and endpoint.get("status") == 0
                and required_parameters.issubset(
                    set(endpoint.get("supported_parameters") or [])
                )
            ]
            healthy_providers = {
                endpoint["provider_name"].strip()
                for endpoint in healthy_endpoints
                if isinstance(endpoint.get("provider_name"), str)
                and endpoint["provider_name"].strip()
            }
            if spec.pinned_provider:
                pinned_endpoint = next(
                    (
                        endpoint
                        for endpoint in healthy_endpoints
                        if endpoint.get("tag") == spec.pinned_provider
                    ),
                    None,
                )
                if pinned_endpoint is None:
                    missing.append(
                        f"{agent_id}: {spec.slug} has no healthy, compatible "
                        f"{spec.pinned_provider!r} endpoint"
                    )
            if len(healthy_providers) < MINIMUM_HEALTHY_PROVIDERS:
                names = ", ".join(sorted(healthy_providers)) or "none"
                missing.append(
                    f"{agent_id}: {spec.slug} has {len(healthy_providers)} independent "
                    f"healthy compatible provider(s), minimum is "
                    f"{MINIMUM_HEALTHY_PROVIDERS} ({names})"
                )
        if missing:
            raise ApiConfigurationError(
                "configured OpenRouter route unavailable; no paid calls made: "
                + "; ".join(missing)
            )

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
                        transport_exception=type(error).__name__,
                        latency_seconds=time.monotonic() - started_at,
                        error=last_error[:400],  # no response arrived, so nothing raw to keep
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
                                raw_response=response.text,
                                response_headers=_diagnostic_headers(response),
                                error=last_error[:400],
                            )
                        )
                    else:
                        # OpenRouter sometimes reports upstream failures as HTTP 200
                        # with an "error" key inside, so 200 alone isn't success.
                        # A JSON array/null or malformed usage used to crash here
                        # before the paid response could be attached to an error.
                        try:
                            if not isinstance(body, dict):
                                raise TypeError("response must be a JSON object")
                            usage = body.get("usage") or {}
                            if not isinstance(usage, dict):
                                raise TypeError("usage must be a JSON object")
                            int(usage.get("prompt_tokens", 0))
                            int(usage.get("completion_tokens", 0))
                            float(usage.get("cost", 0.0))
                        except (TypeError, ValueError, OverflowError) as error:
                            log.append(AttemptRecord(
                                attempt=attempt, outcome="malformed_body",
                                latency_seconds=elapsed, status_code=200,
                                raw_response=response.text, error=str(error),
                                response_headers=_diagnostic_headers(response),
                            ))
                            raise ApiRequestError(
                                f"Malformed response for {payload['model']}: {error}",
                                tuple(log),
                            ) from error
                        record = AttemptRecord(
                            attempt=attempt,
                            outcome="ok" if "error" not in body else "upstream_error",
                            latency_seconds=elapsed,
                            status_code=200,
                            raw_response=response.text,
                            response_headers=_diagnostic_headers(response),
                            finish_reason=_finish_reason_of(body),
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
                            raw_response=response.text,
                            response_headers=_diagnostic_headers(response),
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
                            raw_response=response.text,
                            response_headers=_diagnostic_headers(response),
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


def _diagnostic_headers(response: requests.Response) -> dict[str, str]:
    """Keep correlation/retry hints, not cookies or arbitrary server headers."""
    allowed = {"retry-after", "x-request-id", "request-id", "cf-ray"}
    return {key.lower(): str(value) for key, value in
            getattr(response, "headers", {}).items() if key.lower() in allowed}


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
        if not isinstance(text, str):
            raise TypeError("message content must be text or null")
    except (KeyError, IndexError, TypeError) as error:
        # The reply arrived and was paid for, but has no usable choices/message
        # structure. Keep every attempt on the error - including this one, with
        # a truthful outcome instead of the "ok" it was logged as - so the run
        # can still store and cost the failure.
        if attempt_log:
            last = replace(
                attempt_log[-1],
                outcome="malformed_body",
                error=f"missing choices/message structure: {str(body)[:400]}",
            )
            attempt_log = attempt_log[:-1] + (last,)
        raise ApiRequestError(
            f"Malformed response for {spec.slug}: {str(body)[:400]}", attempt_log
        ) from error

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


def _finish_reason_of(body: dict[str, Any]) -> str:
    """Read the finish reason out of a reply body, or "" if it has none.

    Metadata, not parsing - the answer letter is the parser's job, not this file's.
    """
    try:
        return body["choices"][0].get("finish_reason") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
