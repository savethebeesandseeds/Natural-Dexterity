"""Private provider diagnostics are tested against bounded synthetic errors only."""
from datetime import datetime, timezone
import json
from urllib.parse import quote

import pytest

from web_radio import provider_diagnostics as diagnostics

NOW = datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc)
KEY = "fixture-private-key?redact=this&also=that"


def error(*details, message="Quota rejected for this scoped request."):
    return json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                "message": message, "details": list(details)}}).encode()


def test_quota_limits_are_scoped_observations_with_daily_precedence():
    body = error(
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "generativelanguage.googleapis.com/generate_requests",
             "quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaValue": "3",
             "quotaDimensions": {"model": "gemini-3.8-flash-tts", "location": "global"}},
            {"quotaMetric": "generativelanguage.googleapis.com/generate_requests",
             "quotaId": "GenerateRequestsPerDayPerProjectPerModel", "quotaValue": "100",
             "quotaDimensions": {"model": "gemini-3.8-flash-lite-tts", "location": "eu"}},
        ]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42.25s"},
    )
    parsed = diagnostics.parse_error(body, 429, operation="/v1beta/voices", now=NOW)
    assert parsed["classification"] == diagnostics.QUOTA_REJECTED
    assert diagnostics.quota_kind(parsed) == "daily"
    assert diagnostics.retry_after_seconds(parsed) == 42.25
    observations = parsed["quota_observations"]
    assert [item["limit"] for item in observations] == [3, 100]
    assert [item["kind"] for item in observations] == ["minute", "daily"]
    assert all(item["unit"] == "requests" for item in observations)
    assert observations[1]["dimensions"] == {"model": "gemini-3.8-flash-lite-tts", "location": "eu"}
    assert "not a universal" in parsed["scope_notice"]


def test_error_info_supports_observed_quota_metadata_and_drops_headers():
    parsed = diagnostics.parse_error(error({
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "RATE_LIMIT_EXCEEDED", "domain": "generativelanguage.googleapis.com",
        "metadata": {"quota_metric": "input_tokens", "quota_limit": "TokensPerMinutePerModel",
                     "quota_limit_value": "50", "model": "fixture-model", "consumer": "projects/fixture",
                     "headers": KEY, "api_key": KEY},
    }), 429, key=KEY, now=NOW)
    item = parsed["quota_observations"][0]
    assert item["limit"] == 50 and item["unit"] == "tokens" and item["kind"] == "minute"
    assert item["dimensions"]["consumer"] == "projects/fixture"
    assert KEY not in json.dumps(parsed)
    assert "headers" not in parsed["error_info"][0]["metadata"]


def test_field_violations_preserve_actual_provider_reason_without_inventing_rules():
    parsed = diagnostics.parse_error(error({
        "@type": "type.googleapis.com/google.rpc.BadRequest",
        "fieldViolations": [{"field": "voice.prompted.input",
                             "description": "Provider rejected this original creature description."}],
    }, message="Provider says invalid description."), 400, now=NOW)
    assert parsed["classification"] == diagnostics.PROVIDER_REJECTED
    assert parsed["field_violations"][0]["field"] == "voice.prompted.input"
    assert parsed["message"] == "Provider says invalid description."
    assert diagnostics.quota_kind(parsed) == "unknown"


def test_known_key_url_encoding_control_characters_and_credential_labels_are_redacted():
    body = error(message=f"{KEY} {quote(KEY, safe='')} Authorization: Bearer another-secret\n\x1b[31m")
    parsed = diagnostics.parse_error(body, 400, key=KEY, now=NOW)
    serialized = json.dumps(parsed)
    assert KEY not in serialized and quote(KEY, safe="") not in serialized
    assert "another-secret" not in serialized and "\\u001b" not in serialized
    assert "[REDACTED]" in parsed["message"]


def test_redaction_precedes_length_truncation():
    secret = "private-prefix-" + "a" * 200
    clean = diagnostics.sanitize("x" * 20 + secret, secrets=(secret,), limit=40)
    assert "private-prefix" not in clean and "[REDACTED]" in clean


@pytest.mark.parametrize("raw", [b"", b"not-json", b"[]", b'{"error":[]}', b'{"error":{},"error":{}}',
                                b'{"error":{"code":NaN}}', b"\xff"])
def test_malformed_error_body_keeps_only_safe_http_classification(raw):
    parsed = diagnostics.parse_error(raw, 503, key=KEY, now=NOW)
    assert parsed["classification"] == diagnostics.DISPATCHED_UNCERTAIN
    assert parsed["details_available"] is False
    assert parsed["message"] == ""


def test_oversized_error_body_is_not_partially_trusted():
    parsed = diagnostics.parse_error(b"x" * (diagnostics.MAX_ERROR_BYTES + 1), 429, now=NOW)
    assert parsed["details_available"] is False and parsed["details_truncated"] is True
    assert parsed["classification"] == diagnostics.QUOTA_REJECTED


def test_large_structured_diagnostics_stay_bounded_and_redacted():
    details = [{"@type": "type.googleapis.com/google.rpc.BadRequest",
                "fieldViolations": [{"field": KEY, "description": "x" * 5000}] * 100}] * 10
    parsed = diagnostics.parse_error(error(*details), 400, key=KEY, now=NOW)
    assert len(json.dumps(parsed).encode()) <= diagnostics.MAX_DIAGNOSTIC_BYTES
    assert KEY not in json.dumps(parsed)


@pytest.mark.parametrize("value", ["20", "Thu, 22 Oct 2026 20:00:30 GMT"])
def test_retry_after_header_is_parsed_without_retaining_header(value):
    parsed = diagnostics.parse_error(b"{}", 429, retry_after=value, now=NOW)
    assert diagnostics.retry_after_seconds(parsed) in (20, 30)
    assert "retry-after" not in parsed


def test_longest_valid_retry_hint_is_retained():
    parsed = diagnostics.parse_error(error(
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "10s"},
    ), 429, retry_after="20", now=NOW)
    assert diagnostics.retry_after_seconds(parsed) == 20


@pytest.mark.parametrize("value", ["-1s", "NaNs", "999999999s", "provider said maybe", True])
def test_invalid_retry_hints_do_not_create_retry_policy(value):
    parsed = diagnostics.parse_error(error({
        "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": value,
    }), 429, now=NOW)
    assert diagnostics.retry_after_seconds(parsed) is None


def test_quota_window_is_not_guessed_from_human_message_or_http_status():
    parsed = diagnostics.parse_error(error(message="Maybe requests per minute are exhausted"), 429, now=NOW)
    assert diagnostics.quota_kind(parsed) == "unknown"
    assert parsed["quota_observations"] == []


@pytest.mark.parametrize("message,kind,limit,delay", [
    ("Rate limit exceeded for model gemini-3.8-flash-tts (limit: 3 requests per minute on Free Tier). Please retry in 33s...",
     "minute", 3, 33),
    ("Rate limit exceeded for model gemini-3.8-flash-tts (limit: 10 requests per day on Free Tier). Please retry in 10h53m25s...",
     "daily", 10, 39205),
])
def test_observed_interactions_plain_message_form_is_scoped_not_universal(message, kind, limit, delay):
    body = json.dumps({"error": {"code": "too_many_requests", "message": message}}).encode()
    parsed = diagnostics.parse_error(body, 429, now=NOW)
    assert parsed["code"] == "too_many_requests"
    assert diagnostics.quota_kind(parsed) == kind
    assert diagnostics.retry_after_seconds(parsed) == delay
    assert parsed["quota_observations"][0]["basis"] == "provider_message"
    assert parsed["quota_observations"][0]["limit"] == limit
    assert parsed["quota_observations"][0]["dimensions"] == {
        "model": "gemini-3.8-flash-tts", "tier": "Free Tier"}


def test_structured_quota_and_retry_details_take_precedence_over_plain_message():
    body = error(
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaMetric": "requests", "quotaId": "RequestsPerDay", "quotaValue": "5"}]},
        message="Rate limit exceeded for model fixture-model (limit: 3 requests per minute on Free Tier). Please retry in 33s...")
    parsed = diagnostics.parse_error(body, 429, now=NOW)
    assert parsed["quota_kind"] == "daily" and parsed["retry_after_seconds"] == 12
    assert [item["limit"] for item in parsed["quota_observations"]] == [5]


@pytest.mark.parametrize("observed,expected", [
    ("2026-01-15T20:00:00+00:00", "2026-01-16T08:00:00+00:00"),
    ("2026-07-15T20:00:00+00:00", "2026-07-16T07:00:00+00:00"),
    ("2026-03-08T08:00:00+00:00", "2026-03-09T07:00:00+00:00"),
    ("2026-11-01T07:00:00+00:00", "2026-11-02T08:00:00+00:00"),
    ("2026-10-23T06:59:59+00:00", "2026-10-23T07:00:00+00:00"),
    ("2026-10-23T07:00:00+00:00", "2026-10-24T07:00:00+00:00"),
])
def test_next_daily_reset_uses_pacific_calendar_midnight_with_dst(observed, expected):
    stamp = datetime.fromisoformat(observed).timestamp()
    assert diagnostics.next_daily_reset(stamp) == datetime.fromisoformat(expected).timestamp()
    assert diagnostics.next_daily_reset(stamp) > stamp


@pytest.mark.parametrize("stamp", [True, None, "123", float("nan"), float("inf"), 1e100, 10**1000])
def test_next_daily_reset_returns_none_for_invalid_or_unrepresentable_timestamps(stamp):
    assert diagnostics.next_daily_reset(stamp) is None


def test_next_daily_reset_does_not_invent_boundary_when_timezone_data_unavailable(monkeypatch):
    def unavailable(_name):
        raise diagnostics.ZoneInfoNotFoundError("Fixture timezone data missing")
    monkeypatch.setattr(diagnostics, "ZoneInfo", unavailable)
    assert diagnostics.next_daily_reset(NOW.timestamp()) is None
