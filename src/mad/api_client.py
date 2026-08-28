from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class ApiConfigurationError(RuntimeError):
    """Raised when credentials or the model registry are missing or malformed."""


class ApiRequestError(RuntimeError):
    """Raised when a completion could not be obtained after every retry."""


@dataclass(frozen=True)
class ModelSpec:
    """One debate agent: which OpenRouter model it is and how it is sampled."""

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
class CompletionResult:
    """A single assistant turn plus the provenance needed to audit the run."""

    text: str
    agent_id: str
    requested_slug: str
    served_slug: str
    provider: str
    generation_id: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_seconds: float
    attempts: int


def load_env_file(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env file without overriding real env vars."""
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
    """Return the OpenRouter key, reading .env first if the var is unset."""
    load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ApiConfigurationError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def load_model_registry(config_path: str | Path) -> dict[str, ModelSpec]:
    """Load configs/models/*.yaml into one ModelSpec per debate agent."""
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


class OpenRouterClient:
    """Thin, retrying wrapper over the OpenRouter chat-completions endpoint."""

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
        """Send one chat completion and return the reply with its usage metadata."""
        payload: dict[str, Any] = {
            "model": spec.slug,
            "messages": messages,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": max_tokens if max_tokens is not None else spec.max_tokens,
            "usage": {"include": True},
            "provider": {
                "allow_fallbacks": spec.allow_provider_fallbacks,
                "require_parameters": spec.require_parameters,
            },
        }
        if seed is not None:
            payload["seed"] = seed

        started_at = time.monotonic()
        body = self._post_with_retries("/chat/completions", payload)
        latency_seconds = time.monotonic() - started_at
        return _parse_completion(
            body,
            spec=spec,
            latency_seconds=latency_seconds,
            attempts=body.pop("_attempts", 1),
        )

    def list_available_models(self) -> dict[str, dict[str, Any]]:
        """Return every model OpenRouter currently serves, keyed by slug."""
        response = self._session.get(f"{self._base_url}/models", timeout=self._timeout_seconds)
        response.raise_for_status()
        return {model["id"]: model for model in response.json()["data"]}

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post_with_retries(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_error = "unknown error"

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout_seconds)
            except requests.RequestException as error:
                last_error = f"transport error: {error}"
            else:
                if response.status_code == 200:
                    body = response.json()
                    # OpenRouter reports upstream failures with HTTP 200 too.
                    if "error" not in body:
                        body["_attempts"] = attempt
                        return body
                    last_error = f"upstream error: {body['error']}"
                elif response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}: {response.text[:400]}"
                else:
                    raise ApiRequestError(
                        f"{payload['model']} failed with HTTP {response.status_code}: "
                        f"{response.text[:400]}"
                    )

            if attempt < self._max_attempts:
                time.sleep(min(2.0**attempt, 30.0) + random.uniform(0.0, 1.0))

        raise ApiRequestError(
            f"{payload['model']} failed after {self._max_attempts} attempts: {last_error}"
        )


def _build_headers(api_key: str) -> dict[str, str]:
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
    attempts: int,
) -> CompletionResult:
    try:
        choice = body["choices"][0]
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
        attempts=attempts,
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
