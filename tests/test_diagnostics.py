import pytest
from mad.api_client import AttemptRecord
from mad.diagnostics import diagnose_attempt, diagnose_output


@pytest.mark.parametrize('code,category', [
    (400, 'REQUEST_REJECTED'), (401, 'AUTHENTICATION'), (402, 'CREDIT_OR_LIMIT'),
    (403, 'ACCESS_OR_POLICY'), (404, 'ROUTING_OR_MODEL'), (408, 'SERVER_TIMEOUT'),
    (413, 'REQUEST_SIZE'), (429, 'RATE_LIMIT'), (500, 'REMOTE_SERVICE'),
    (502, 'REMOTE_SERVICE'), (503, 'REMOTE_SERVICE'), (504, 'REMOTE_SERVICE'),
])
def test_http_error_categories(code, category):
    assert diagnose_attempt(AttemptRecord(1, 'http_error', 1, status_code=code)).category == category


@pytest.mark.parametrize('outcome,category', [
    ('transport_error', 'TRANSPORT'), ('bad_json', 'RESPONSE_FORMAT'),
    ('malformed_body', 'RESPONSE_FORMAT'), ('ok', 'HTTP_SUCCESS'),
    ('upstream_error', 'REMOTE_ERROR_UNCLASSIFIED'),
])
def test_failure_layers(outcome, category):
    assert diagnose_attempt(AttemptRecord(1, outcome, 1, status_code=200)).category == category


def test_error_inside_http_200_is_not_success():
    attempt = AttemptRecord(1, 'upstream_error', 1, status_code=200,
                            raw_response='{"error":{"code":429}}')
    assert diagnose_attempt(attempt).category == 'RATE_LIMIT'


@pytest.mark.parametrize('status,category', [
    ('OK', 'COMPLETED'), ('TRUNCATED', 'OUTPUT_LIMIT_OR_CUTOFF'),
    ('PARSE_FAIL', 'ANSWER_FORMAT'), ('REFUSAL', 'REFUSAL'), ('API_ERROR', 'API_FAILURE'),
])
def test_output_is_separate_from_transport(status, category):
    assert diagnose_output(status).category == category
