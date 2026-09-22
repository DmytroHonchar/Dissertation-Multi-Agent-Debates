"""Explain observed API/output failures without claiming an unseen root cause.

Pure logic: no calls, storage or answer keys. Used by the diagnostic runner.
Categories describe evidence, not whether a model answer is correct.
"""
import json
from dataclasses import dataclass

from mad.api_client import AttemptRecord


@dataclass(frozen=True)
class Diagnosis:
    category: str
    explanation: str


def diagnose_attempt(attempt: AttemptRecord) -> Diagnosis:
    if attempt.outcome == "transport_error":
        return Diagnosis("TRANSPORT", "No HTTP response: network, timeout or TLS; attribution to our machine or provider is unresolved.")
    if attempt.outcome in {"bad_json", "malformed_body"}:
        return Diagnosis("RESPONSE_FORMAT", "A response arrived but its JSON/schema was unusable; inspect the raw body.")
    code = attempt.status_code
    if attempt.outcome == "upstream_error":
        try:
            embedded = json.loads(attempt.raw_response)["error"].get("code")
            code = int(embedded) if embedded is not None else None
        except (TypeError, ValueError, KeyError, AttributeError):
            code = None
    categories = {
        400: ("REQUEST_REJECTED", "Server rejected request parameters; inspect its message before blaming client or provider."),
        401: ("AUTHENTICATION", "API authentication rejected; check the key."),
        402: ("CREDIT_OR_LIMIT", "Billing/credit limit rejected the request; check account and key allowances."),
        403: ("ACCESS_OR_POLICY", "Access or policy restriction; not evidence the provider is down."),
        404: ("ROUTING_OR_MODEL", "Requested model/provider route not found; inspect the exact ID, pin and server message."),
        408: ("SERVER_TIMEOUT", "Server reported a timeout; the internal cause is unknown."),
        413: ("REQUEST_SIZE", "Server rejected request size; inspect context/input limits."),
        429: ("RATE_LIMIT", "Throttling reported; raw metadata may distinguish account-level from upstream provider limits."),
    }
    if code in categories:
        return Diagnosis(*categories[code])
    if code is not None and 500 <= code < 600:
        return Diagnosis("REMOTE_SERVICE", "OpenRouter/provider/gateway returned a server error; this does not identify its internal cause.")
    if attempt.outcome != "ok":
        return Diagnosis("REMOTE_ERROR_UNCLASSIFIED", "An error was returned; retain the raw evidence instead of guessing its cause.")
    return Diagnosis("HTTP_SUCCESS", "Request succeeded at API level; check the model output separately.")


def diagnose_output(status: str) -> Diagnosis:
    return Diagnosis(*{
        "OK": ("COMPLETED", "Response finished in the required answer format; correctness is not scored here."),
        "TRUNCATED": ("OUTPUT_LIMIT_OR_CUTOFF", "Parser detected a cut-off response; finish_reason=length confirms a reported output limit."),
        "REFUSAL": ("REFUSAL", "A refusal was returned; the call itself need not have failed."),
        "PARSE_FAIL": ("ANSWER_FORMAT", "A response arrived but the answer parser could not accept its format."),
        "API_ERROR": ("API_FAILURE", "No usable completion; inspect individual attempt diagnoses."),
    }[status])
