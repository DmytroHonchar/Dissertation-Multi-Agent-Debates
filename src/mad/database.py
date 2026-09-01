"""Where every result is stored.

SQLite, so it is one file and there is no server to run. The file is
storage/results.sqlite, which git ignores. Results stay off git; the dataset
does not.

Four rules this file has to hold:

    store the whole reply, not just the letter
    keep every attempt, including the ones that failed and cost money
    never overwrite a row - a repeat means a new run ID
    no column can hold a correct answer

Round 1 and Round 2 get separate rows. Comparing them is the research question,
so nothing here is allowed to merge them.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mad.api_client import AttemptRecord, CompletionResult, ModelSpec
from mad.parser_v1 import ParsedResponse, STATUS_API_ERROR, STATUS_OK


# 1. Settings

SCHEMA_VERSION = 1

DEFAULT_DATABASE_PATH = Path("storage") / "results.sqlite"

ROUNDS = (1, 2)

# The four ways a vote can end. All four come from the one rule: an answer needs
# three of the five agents.
CONSENSUS_STATES = (
    "UNANIMOUS",              # everyone said the same thing
    "CONSENSUS",              # three or four said the same thing
    "NO_CONSENSUS",           # enough agents answered, but nobody got three
    "INSUFFICIENT_ANSWERS",   # too many agents failed, so three was impossible
)

# In Round 2 an agent sees the other four, never itself.
MAX_PEER_RESPONSES = 4


# 2. Errors


class DatabaseError(RuntimeError):
    """Something about the row was wrong."""


class DuplicateRecordError(DatabaseError):
    """This row is already there. Nothing gets replaced - use a new run ID."""


# 3. Schema

# The CHECK and UNIQUE lines below are not decoration. They put the project's
# rules into the file itself, so breaking one is an error instead of a quiet
# wrong number months later. Each is explained where it sits.

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id               TEXT PRIMARY KEY,
        config_name          TEXT NOT NULL,
        question_set_version TEXT NOT NULL,
        prompt_version       TEXT NOT NULL,
        settings_version     TEXT NOT NULL,
        parser_version       TEXT NOT NULL,
        started_at           TEXT NOT NULL,
        ended_at             TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_responses (
        response_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id            TEXT NOT NULL REFERENCES runs(run_id),
        question_id       TEXT NOT NULL,
        round             INTEGER NOT NULL CHECK (round IN (1, 2)),
        agent_id          TEXT NOT NULL,

        -- A slug is the model's ID on OpenRouter, like "qwen/qwen3.8-27b".
        -- We ask for one; OpenRouter can serve a different build of it. Both
        -- are kept, and a response is never rejected for the difference.
        requested_slug    TEXT NOT NULL,
        served_slug       TEXT NOT NULL,
        provider          TEXT NOT NULL,
        generation_id     TEXT NOT NULL,

        raw_response      TEXT NOT NULL,
        extracted_letter  TEXT,
        extraction_method TEXT,
        -- A typo in calling code becomes an error here, not a new category.
        status            TEXT NOT NULL CHECK (
                              status IN ('OK', 'REFUSAL', 'TRUNCATED',
                                         'PARSE_FAIL', 'API_ERROR')),
        finish_reason     TEXT NOT NULL DEFAULT '',

        attempt_count     INTEGER NOT NULL CHECK (attempt_count >= 1),
        selected_attempt  INTEGER NOT NULL CHECK (selected_attempt >= 1),

        prompt_version    TEXT NOT NULL,
        temperature       REAL NOT NULL,
        top_p             REAL NOT NULL,
        max_tokens        INTEGER NOT NULL,

        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd          REAL NOT NULL DEFAULT 0.0,
        latency_seconds   REAL NOT NULL DEFAULT 0.0,
        cache_hit         INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),

        -- Which peer answers this agent was shown, by response_id. Round 1
        -- sees none. Round 2 sees four, or fewer when a peer failed - the real
        -- number matters, so it is stored rather than assumed.
        peer_response_ids TEXT NOT NULL DEFAULT '[]',
        peer_count        INTEGER NOT NULL DEFAULT 0
                              CHECK (peer_count BETWEEN 0 AND 4),

        created_at        TEXT NOT NULL,

        -- One answer per agent per question per round. A second insert fails
        -- instead of overwriting the first.
        UNIQUE (run_id, question_id, round, agent_id),
        CHECK (selected_attempt <= attempt_count),
        -- Round 1 agents work alone.
        CHECK (round = 2 OR peer_count = 0),
        -- Only an OK row may carry a letter, so a failure cannot become a vote.
        CHECK ((status = 'OK' AND extracted_letter IS NOT NULL)
            OR (status <> 'OK' AND extracted_letter IS NULL))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS response_attempts (
        attempt_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        response_id       INTEGER NOT NULL REFERENCES model_responses(response_id),
        attempt_number    INTEGER NOT NULL CHECK (attempt_number >= 1),

        raw_response      TEXT NOT NULL,
        extracted_letter  TEXT,
        status            TEXT NOT NULL,
        finish_reason     TEXT NOT NULL DEFAULT '',

        outcome           TEXT NOT NULL,
        http_status       INTEGER,
        error             TEXT NOT NULL DEFAULT '',

        prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd          REAL NOT NULL DEFAULT 0.0,
        latency_seconds   REAL NOT NULL DEFAULT 0.0,
        cache_hit         INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),

        -- The same attempt cannot be logged twice.
        UNIQUE (response_id, attempt_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS question_outcomes (
        run_id                  TEXT NOT NULL REFERENCES runs(run_id),
        question_id             TEXT NOT NULL,
        round                   INTEGER NOT NULL CHECK (round IN (1, 2)),

        consensus_state         TEXT NOT NULL CHECK (
                                    consensus_state IN ('UNANIMOUS', 'CONSENSUS',
                                                        'NO_CONSENSUS',
                                                        'INSUFFICIENT_ANSWERS')),
        consensus_answer        TEXT,
        decided                 INTEGER NOT NULL CHECK (decided IN (0, 1)),
        valid_answer_count      INTEGER NOT NULL CHECK (valid_answer_count >= 0),

        total_prompt_tokens     INTEGER NOT NULL DEFAULT 0,
        total_completion_tokens INTEGER NOT NULL DEFAULT 0,
        total_cost_usd          REAL NOT NULL DEFAULT 0.0,
        total_latency_seconds   REAL NOT NULL DEFAULT 0.0,

        created_at              TEXT NOT NULL,

        -- Round is part of the key, so Round 1 and Round 2 cannot collide.
        PRIMARY KEY (run_id, question_id, round),
        -- Decided means there is an answer. Undecided must not pretend to have one.
        CHECK ((decided = 1 AND consensus_answer IS NOT NULL)
            OR (decided = 0 AND consensus_answer IS NULL))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_responses_run_round ON model_responses(run_id, round)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_run_round ON question_outcomes(run_id, round)",
)


# 4. What gets written


@dataclass(frozen=True)
class ResponseRecord:
    """One agent's answer to one question in one round."""

    run_id: str
    question_id: str
    round: int
    agent_id: str

    requested_slug: str      # the model ID we asked OpenRouter for
    served_slug: str         # the model ID it actually used
    provider: str            # the company that ran it, e.g. DeepInfra
    generation_id: str       # OpenRouter's ID for this reply

    raw_response: str
    status: str
    extracted_letter: str | None
    extraction_method: str | None
    finish_reason: str

    attempt_count: int
    selected_attempt: int

    prompt_version: str
    temperature: float
    top_p: float
    max_tokens: int

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    cache_hit: bool = False

    peer_response_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AttemptRow:
    """One API attempt belonging to a response."""

    attempt_number: int
    raw_response: str
    status: str
    outcome: str
    extracted_letter: str | None = None
    finish_reason: str = ""
    http_status: int | None = None
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    cache_hit: bool = False


@dataclass(frozen=True)
class OutcomeRecord:
    """The group's position on one question after one round."""

    run_id: str
    question_id: str
    round: int
    consensus_state: str
    consensus_answer: str | None
    decided: bool
    valid_answer_count: int
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0


# 5. Opening and closing


class ResultsDatabase:
    """The results file. Open it, write to it, read it back.

    Best used with `with`, so it closes itself:

        with ResultsDatabase(path) as db:
            db.start_run(...)
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _repository_root() / DEFAULT_DATABASE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        # Read rows by column name instead of position.
        self._connection.row_factory = sqlite3.Row
        # SQLite ignores foreign keys unless you ask for them.
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        """Make the tables if the file is new, and write the version onto it."""
        existing = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if existing and existing != SCHEMA_VERSION:
            raise DatabaseError(
                f"{self.path} was written by schema version {existing}, "
                f"but this code is version {SCHEMA_VERSION}. Use a new file."
            )
        with self._connection:
            for statement in SCHEMA_STATEMENTS:
                self._connection.execute(statement)
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ResultsDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # 6. Writing

    def start_run(
        self,
        run_id: str,
        *,
        config_name: str,
        question_set_version: str,
        prompt_version: str,
        settings_version: str,
        parser_version: str,
        started_at: str | None = None,
    ) -> None:
        """Open a run. These version numbers are what make its results citable later."""
        self._insert(
            "runs",
            {
                "run_id": run_id,
                "config_name": config_name,
                "question_set_version": question_set_version,
                "prompt_version": prompt_version,
                "settings_version": settings_version,
                "parser_version": parser_version,
                "started_at": started_at or _now(),
                "ended_at": None,
            },
            duplicate_message=f"run {run_id!r} already exists; use a new run ID",
        )

    def finish_run(self, run_id: str, ended_at: str | None = None) -> None:
        """Write the end time. The only update in this file, and it works once.

        This is not overwriting a result. A run has a start and an end, and the
        end is not known when it starts.
        """
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE runs SET ended_at = ? WHERE run_id = ? AND ended_at IS NULL",
                (ended_at or _now(), run_id),
            )
        if cursor.rowcount == 0:
            raise DatabaseError(f"run {run_id!r} is unknown or already finished")

    def record_response(
        self, response: ResponseRecord, attempts: Sequence[AttemptRow] = ()
    ) -> int:
        """Store one response with all its attempts, or store nothing.

        One transaction for both. Half a record would undercount what the run
        cost, which is worse than an error you can see.
        """
        _validate_response(response, attempts)
        row = _response_to_row(response)

        try:
            with self._connection:
                cursor = self._connection.execute(*_insert_sql("model_responses", row))
                response_id = int(cursor.lastrowid)
                for attempt in attempts:
                    self._connection.execute(
                        *_insert_sql(
                            "response_attempts",
                            _attempt_to_row(attempt, response_id),
                        )
                    )
        except sqlite3.IntegrityError as error:
            raise _integrity_error(
                error,
                f"response for run {response.run_id!r} question {response.question_id!r} "
                f"round {response.round} agent {response.agent_id!r}",
            ) from error
        return response_id

    def record_outcome(self, outcome: OutcomeRecord) -> None:
        """Store where the group landed on one question after one round."""
        if outcome.round not in ROUNDS:
            raise DatabaseError(f"round must be 1 or 2, got {outcome.round}")
        if outcome.consensus_state not in CONSENSUS_STATES:
            raise DatabaseError(f"unknown consensus state {outcome.consensus_state!r}")

        self._insert(
            "question_outcomes",
            {
                "run_id": outcome.run_id,
                "question_id": outcome.question_id,
                "round": outcome.round,
                "consensus_state": outcome.consensus_state,
                "consensus_answer": outcome.consensus_answer,
                "decided": int(outcome.decided),
                "valid_answer_count": outcome.valid_answer_count,
                "total_prompt_tokens": outcome.total_prompt_tokens,
                "total_completion_tokens": outcome.total_completion_tokens,
                "total_cost_usd": outcome.total_cost_usd,
                "total_latency_seconds": outcome.total_latency_seconds,
                "created_at": _now(),
            },
            duplicate_message=(
                f"outcome for run {outcome.run_id!r} question {outcome.question_id!r} "
                f"round {outcome.round} already exists"
            ),
        )

    def _insert(self, table: str, row: dict[str, Any], *, duplicate_message: str) -> None:
        """A plain INSERT. Never INSERT OR REPLACE - a duplicate has to fail."""
        try:
            with self._connection:
                self._connection.execute(*_insert_sql(table, row))
        except sqlite3.IntegrityError as error:
            raise _integrity_error(error, duplicate_message) from error

    # 7. Reading

    def read_runs(self) -> list[dict[str, Any]]:
        """Every run, newest first."""
        return self._query("SELECT * FROM runs ORDER BY started_at DESC")

    def read_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return rows[0] if rows else None

    def read_responses(
        self,
        run_id: str,
        *,
        round: int | None = None,
        question_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Responses for a run, filtered by round, question or agent.

        peer_response_ids comes back as a list, not the JSON text it is stored as.
        """
        sql = "SELECT * FROM model_responses WHERE run_id = ?"
        params: list[Any] = [run_id]
        for column, value in (("round", round), ("question_id", question_id), ("agent_id", agent_id)):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY question_id, round, agent_id"

        rows = self._query(sql, tuple(params))
        for row in rows:
            row["peer_response_ids"] = json.loads(row["peer_response_ids"])
        return rows

    def read_attempts(self, response_id: int) -> list[dict[str, Any]]:
        """Every attempt behind one response, in the order they happened."""
        return self._query(
            "SELECT * FROM response_attempts WHERE response_id = ? ORDER BY attempt_number",
            (response_id,),
        )

    def read_outcomes(self, run_id: str, *, round: int | None = None) -> list[dict[str, Any]]:
        """Group outcomes for a run. Each round is its own row."""
        sql = "SELECT * FROM question_outcomes WHERE run_id = ?"
        params: list[Any] = [run_id]
        if round is not None:
            sql += " AND round = ?"
            params.append(round)
        return self._query(sql + " ORDER BY question_id, round", tuple(params))

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(sql, params).fetchall()]


# 8. Turning API replies into rows


def response_from_completion(
    result: CompletionResult,
    parsed: ParsedResponse,
    *,
    run_id: str,
    question_id: str,
    round: int,
    spec: ModelSpec,
    prompt_version: str,
    selected_attempt: int | None = None,
    cache_hit: bool = False,
    peer_response_ids: Sequence[int] = (),
) -> ResponseRecord:
    """Build a row from a successful call and its parse.

    The API client knows nothing about this database, and this database does no
    parsing. They meet here and nowhere else.

    Tokens and cost cover every attempt, not just the one that succeeded - a
    paid failure before the retry is part of what this answer cost (P6). The
    per-attempt split stays visible in response_attempts.
    """
    return ResponseRecord(
        run_id=run_id,
        question_id=question_id,
        round=round,
        agent_id=result.agent_id,
        requested_slug=result.requested_slug,
        served_slug=result.served_slug,
        provider=result.provider,
        generation_id=result.generation_id,
        raw_response=result.text,
        status=parsed.status,
        extracted_letter=parsed.letter,
        extraction_method=parsed.extraction_method,
        finish_reason=result.finish_reason,
        attempt_count=max(result.attempts, 1),
        selected_attempt=selected_attempt or max(result.attempts, 1),
        prompt_version=prompt_version,
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        prompt_tokens=sum(r.prompt_tokens for r in result.attempt_log) or result.prompt_tokens,
        completion_tokens=sum(r.completion_tokens for r in result.attempt_log) or result.completion_tokens,
        cost_usd=sum(r.cost_usd for r in result.attempt_log) or result.cost_usd,
        latency_seconds=result.latency_seconds,
        cache_hit=cache_hit,
        peer_response_ids=tuple(peer_response_ids),
    )


def response_from_failed_call(
    attempt_log: Sequence[AttemptRecord],
    *,
    run_id: str,
    question_id: str,
    round: int,
    spec: ModelSpec,
    prompt_version: str,
    peer_response_ids: Sequence[int] = (),
) -> ResponseRecord:
    """Build a row for a call that failed every attempt.

    Comes from ApiRequestError.attempt_log. Stored as API_ERROR, never as an
    answer. Nobody served it, so the model and provider are stored as "unknown"
    rather than guessed.
    """
    last = attempt_log[-1] if attempt_log else None
    return ResponseRecord(
        run_id=run_id,
        question_id=question_id,
        round=round,
        agent_id=spec.agent_id,
        requested_slug=spec.slug,
        served_slug="unknown",
        provider="unknown",
        generation_id="",
        raw_response=last.raw_response if last else "",
        status=STATUS_API_ERROR,
        extracted_letter=None,
        extraction_method=None,
        finish_reason=last.finish_reason if last else "",
        attempt_count=max(len(attempt_log), 1),
        selected_attempt=max(len(attempt_log), 1),
        prompt_version=prompt_version,
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        prompt_tokens=sum(record.prompt_tokens for record in attempt_log),
        completion_tokens=sum(record.completion_tokens for record in attempt_log),
        cost_usd=sum(record.cost_usd for record in attempt_log),
        latency_seconds=sum(record.latency_seconds for record in attempt_log),
        peer_response_ids=tuple(peer_response_ids),
    )


def attempt_rows(
    attempt_log: Iterable[AttemptRecord],
    *,
    parsed_by_attempt: dict[int, ParsedResponse] | None = None,
    cache_hit: bool = False,
) -> list[AttemptRow]:
    """Turn the API client's attempt log into rows.

    An attempt that returned nothing has nothing to parse, so it is stored as
    the API_ERROR it was. Nothing is filled in.
    """
    parsed_by_attempt = parsed_by_attempt or {}
    rows = []
    for record in attempt_log:
        parsed = parsed_by_attempt.get(record.attempt)
        rows.append(
            AttemptRow(
                attempt_number=record.attempt,
                raw_response=record.raw_response,
                status=parsed.status if parsed else STATUS_API_ERROR,
                outcome=record.outcome,
                extracted_letter=parsed.letter if parsed else None,
                finish_reason=record.finish_reason,
                http_status=record.status_code,
                error=record.error,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost_usd=record.cost_usd,
                latency_seconds=record.latency_seconds,
                cache_hit=cache_hit,
            )
        )
    return rows


# 9. Helpers


def _validate_response(response: ResponseRecord, attempts: Sequence[AttemptRow]) -> None:
    """Checks SQL cannot do, with an error that says what is actually wrong."""
    if response.round not in ROUNDS:
        raise DatabaseError(f"round must be 1 or 2, got {response.round}")
    if response.status != STATUS_OK and response.extracted_letter is not None:
        raise DatabaseError(
            f"status {response.status} carries letter {response.extracted_letter!r}; "
            "a failure is never an answer"
        )
    if response.status == STATUS_OK and not response.extracted_letter:
        raise DatabaseError("an OK response must carry the letter that was extracted")
    if response.round == 1 and response.peer_response_ids:
        raise DatabaseError("Round 1 agents see no peer responses")
    if len(response.peer_response_ids) > MAX_PEER_RESPONSES:
        raise DatabaseError(
            f"an agent sees at most {MAX_PEER_RESPONSES} peers, "
            f"got {len(response.peer_response_ids)}"
        )
    if len(set(response.peer_response_ids)) != len(response.peer_response_ids):
        raise DatabaseError("the same peer response was supplied twice")
    if attempts and response.selected_attempt not in {row.attempt_number for row in attempts}:
        raise DatabaseError(
            f"selected attempt {response.selected_attempt} is not among the attempts stored"
        )


def _response_to_row(response: ResponseRecord) -> dict[str, Any]:
    return {
        "run_id": response.run_id,
        "question_id": response.question_id,
        "round": response.round,
        "agent_id": response.agent_id,
        "requested_slug": response.requested_slug,
        "served_slug": response.served_slug,
        "provider": response.provider,
        "generation_id": response.generation_id,
        "raw_response": response.raw_response,
        "extracted_letter": response.extracted_letter,
        "extraction_method": response.extraction_method,
        "status": response.status,
        "finish_reason": response.finish_reason,
        "attempt_count": response.attempt_count,
        "selected_attempt": response.selected_attempt,
        "prompt_version": response.prompt_version,
        "temperature": response.temperature,
        "top_p": response.top_p,
        "max_tokens": response.max_tokens,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "cost_usd": response.cost_usd,
        "latency_seconds": response.latency_seconds,
        "cache_hit": int(response.cache_hit),
        "peer_response_ids": json.dumps(list(response.peer_response_ids)),
        "peer_count": len(response.peer_response_ids),
        "created_at": _now(),
    }


def _attempt_to_row(attempt: AttemptRow, response_id: int) -> dict[str, Any]:
    return {
        "response_id": response_id,
        "attempt_number": attempt.attempt_number,
        "raw_response": attempt.raw_response,
        "extracted_letter": attempt.extracted_letter,
        "status": attempt.status,
        "finish_reason": attempt.finish_reason,
        "outcome": attempt.outcome,
        "http_status": attempt.http_status,
        "error": attempt.error,
        "prompt_tokens": attempt.prompt_tokens,
        "completion_tokens": attempt.completion_tokens,
        "cost_usd": attempt.cost_usd,
        "latency_seconds": attempt.latency_seconds,
        "cache_hit": int(attempt.cache_hit),
    }


def _insert_sql(table: str, row: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    """Build an INSERT from a column-to-value mapping."""
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    return f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(row.values())


def _integrity_error(error: sqlite3.IntegrityError, subject: str) -> DatabaseError:
    """Name the rule that was broken. SQLite's own message is too short to help."""
    if "UNIQUE" in str(error):
        return DuplicateRecordError(f"{subject} already exists; rows are never replaced")
    return DatabaseError(f"{subject} was rejected: {error}")


def _now() -> str:
    """The time now, in UTC. One timezone everywhere, so runs compare cleanly."""
    return datetime.now(timezone.utc).isoformat()


def _repository_root() -> Path:
    """The project folder, two levels up from src/mad/."""
    return Path(__file__).resolve().parents[2]
