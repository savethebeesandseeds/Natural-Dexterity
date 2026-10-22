"""Private, bounded Google RPC error observations; no credentials or raw headers.

Google's structured details are observations about the rejected request's quota
scope. They are not universal project RPM/RPD settings and never authorize retry.
https://github.com/googleapis/googleapis/blob/master/google/rpc/error_details.proto
https://ai.google.dev/gemini-api/docs/rate-limits
"""
from __future__ import annotations

from datetime import datetime, time as datetime_time, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import math
import re
import unicodedata
from urllib.parse import quote, quote_plus
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MAX_ERROR_BYTES = 64 * 1024
MAX_DIAGNOSTIC_BYTES = 32 * 1024
MAX_RETRY_SECONDS = 7 * 24 * 60 * 60
QUOTA_REJECTED = "quota-rejected"
PROVIDER_REJECTED = "provider-rejected"
DISPATCHED_UNCERTAIN = "dispatched_uncertain"


def sanitize(value, *, secrets=(), limit=1024) -> str:
    """Redact before truncating, so a length boundary cannot expose a key prefix."""
    if not isinstance(value, str):
        return ""
    text = value
    for secret in secrets:
        if isinstance(secret, str) and secret:
            for variant in {secret, quote(secret, safe=""), quote_plus(secret, safe="")}:
                text = text.replace(variant, "[REDACTED]")
    text = re.sub(r"AIza[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"AQ\.[A-Za-z0-9_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"(?i)\b(?:x-goog-api-key|api[_-]?key|authorization|access[_-]?token)"
                  r"\s*[:=]\s*(?:bearer\s+)?[\"']?[^\"'\s,;}]+",
                  "[REDACTED CREDENTIAL]", text)
    text = "".join(" " if unicodedata.category(char) in {"Cc", "Cf", "Cs"} else char for char in text)
    return " ".join(text.split())[:limit]


def _number(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
        value = int(value)
    return value if type(value) is int and value >= 0 else None


def _kind(value) -> str:
    normalized = re.sub(r"[^a-z]", "", value.lower()) if isinstance(value, str) else ""
    if "perday" in normalized or "daily" in normalized:
        return "daily"
    if "perminute" in normalized or "permin" in normalized:
        return "minute"
    return "unknown"


def _delay(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,9})?s", value):
        number = float(value[:-1])
    elif type(value) is dict:
        seconds = _number(value.get("seconds", 0))
        nanos = _number(value.get("nanos", 0))
        if seconds is None or nanos is None or nanos >= 1_000_000_000:
            return None
        number = seconds + nanos / 1_000_000_000
    elif type(value) in (int, float):
        number = float(value)
    else:
        return None
    return number if math.isfinite(number) and 0 <= number <= MAX_RETRY_SECONDS else None


def _retry_header(value, now):
    if not isinstance(value, str) or len(value) > 128:
        return None
    if re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,9})?", value.strip()):
        return _delay(float(value))
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            return None
        return _delay(max(0, (target - now).total_seconds()))
    except (ValueError, TypeError, OverflowError):
        return None


def _observed_message_quota(message: str):
    # Narrow compatibility for actual Interactions API responses that omit RPC
    # details. This is a scoped provider observation, never a global limit.
    match = re.search(
        r"Rate limit exceeded for model ([A-Za-z0-9._-]{1,160}) "
        r"\(limit: ([0-9]{1,9}) requests per (minute|day) on ([A-Za-z0-9 _-]{1,48})\)\."
        r"\s*Please retry in ((?:[0-9]{1,4}h)?(?:[0-9]{1,4}m)?(?:[0-9]{1,6}(?:\.[0-9]{1,9})?s)?)", message)
    if match is None:
        return None, None
    model, limit, window, tier, duration = match.groups()
    observation = {
        "kind": "minute" if window == "minute" else "daily", "unit": "requests",
        "limit": int(limit), "metric": "", "quota_id": "",
        "dimensions": {"model": model, "tier": tier}, "basis": "provider_message",
    }
    parsed = re.fullmatch(r"(?:(\d{1,4})h)?(?:(\d{1,4})m)?(?:(\d{1,6}(?:\.\d{1,9})?)s)?", duration)
    delay = None
    if parsed is not None and any(value is not None for value in parsed.groups()):
        hours, minutes, seconds = parsed.groups()
        delay = _delay(int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0))
    return observation, delay


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject(_value):
    raise ValueError


def parse_error(body: bytes, http_status: int, *, key: str | None = None,
                retry_after: str | None = None, operation: str = "",
                model: str | None = None, now: datetime | None = None,
                body_truncated: bool = False) -> dict:
    """Extract a whitelisted diagnostic from one non-successful HTTP response."""
    current = now or datetime.now(timezone.utc)
    classification = (QUOTA_REJECTED if http_status == 429 else
                      DISPATCHED_UNCERTAIN if http_status >= 500 else PROVIDER_REJECTED)
    secrets = (key,) if key else ()
    clean = lambda value, limit=1024: sanitize(value, secrets=secrets, limit=limit)
    result = {
        "schema_version": 1, "provider": "gemini", "classification": classification,
        "http_status": http_status, "code": None, "status": None,
        "message": "", "operation": clean(operation, 80), "model": clean(model, 160) or None,
        "observed_at": current.isoformat(), "field_violations": [], "error_info": [],
        "quota_violations": [], "quota_observations": [],
        "retry_after_seconds": None, "quota_kind": "unknown",
        "details_available": False, "details_truncated": bool(body_truncated or type(body) is bytes and len(body) > MAX_ERROR_BYTES),
        "scope_notice": "Observed for this provider response; not a universal RPM/RPD configuration.",
    }
    delays = []
    structured_retry = False
    header_delay = _retry_header(retry_after, current)
    if header_delay is not None:
        delays.append(header_delay)
    try:
        if type(body) is not bytes or len(body) > MAX_ERROR_BYTES or body_truncated:
            raise ValueError
        parsed = json.loads(body.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject)
        error = parsed.get("error") if type(parsed) is dict else None
        if type(error) is not dict:
            raise ValueError
        result["details_available"] = True
        code = error.get("code")
        result["code"] = _number(code) if type(code) is int else clean(code, 80) or None
        result["status"] = clean(error.get("status"), 80) or None
        result["message"] = clean(error.get("message"))
        details = error.get("details", [])
        if type(details) is not list:
            details = []
        if len(details) > 24:
            result["details_truncated"] = True
        for detail in details[:24]:
            if type(detail) is not dict:
                continue
            kind = detail.get("@type", "")
            if kind == "type.googleapis.com/google.rpc.RetryInfo":
                delay = _delay(detail.get("retryDelay", detail.get("retry_delay")))
                if delay is not None:
                    delays.append(delay)
                    structured_retry = True
            elif kind == "type.googleapis.com/google.rpc.BadRequest":
                fields = detail.get("fieldViolations", detail.get("field_violations", []))
                if type(fields) is list:
                    for field in fields[:8]:
                        if type(field) is dict:
                            result["field_violations"].append({
                                "field": clean(field.get("field"), 256),
                                "description": clean(field.get("description"), 512),
                            })
            elif kind == "type.googleapis.com/google.rpc.ErrorInfo":
                allowed = {"quota_metric", "quota_limit", "quota_limit_value", "quota_location",
                           "model", "location", "service", "consumer", "project", "unit"}
                metadata = detail.get("metadata", {})
                safe = {name: clean(value, 128) for name, value in metadata.items()
                        if name in allowed and isinstance(value, str)} if type(metadata) is dict else {}
                item = {"reason": clean(detail.get("reason"), 128),
                        "domain": clean(detail.get("domain"), 128), "metadata": safe}
                result["error_info"].append(item)
                if "quota_limit" in safe or "quota_metric" in safe:
                    result["quota_observations"].append({
                        "kind": _kind(safe.get("quota_limit", "")),
                        "unit": "tokens" if "token" in safe.get("quota_metric", "").lower()
                                else "requests" if "request" in safe.get("quota_metric", "").lower() else "unknown",
                        "limit": _number(safe.get("quota_limit_value")),
                        "metric": safe.get("quota_metric", ""), "quota_id": safe.get("quota_limit", ""),
                        "dimensions": {name: safe[name] for name in ("model", "location", "quota_location", "consumer", "project")
                                       if name in safe},
                    })
            elif kind == "type.googleapis.com/google.rpc.QuotaFailure":
                violations = detail.get("violations", [])
                if type(violations) is not list:
                    continue
                for violation in violations[:8]:
                    if type(violation) is not dict:
                        continue
                    dimensions = violation.get("quotaDimensions", violation.get("quota_dimensions", {}))
                    safe_dimensions = {}
                    if type(dimensions) is dict:
                        for name, value in list(dimensions.items())[:8]:
                            safe_name = clean(name, 64)
                            if safe_name.lower() in {"key", "api_key", "api-key", "x-goog-api-key", "authorization", "headers"}:
                                continue
                            if isinstance(value, str):
                                safe_dimensions[safe_name] = clean(value, 128)
                    metric = clean(violation.get("quotaMetric", violation.get("quota_metric")), 256)
                    quota_id = clean(violation.get("quotaId", violation.get("quota_id")), 256)
                    item = {
                        "metric": metric, "quota_id": quota_id, "dimensions": safe_dimensions,
                        "limit": _number(violation.get("quotaValue", violation.get("quota_value"))),
                        "subject": clean(violation.get("subject"), 256),
                        "description": clean(violation.get("description"), 256),
                    }
                    result["quota_violations"].append(item)
                    result["quota_observations"].append({
                        "kind": _kind(quota_id), "unit": "tokens" if "token" in (metric + quota_id).lower()
                        else "requests" if "request" in (metric + quota_id).lower() else "unknown",
                        "limit": item["limit"], "metric": metric, "quota_id": quota_id,
                        "dimensions": safe_dimensions,
                    })
    except (ValueError, UnicodeError, RecursionError):
        pass
    if http_status == 429:
        observation, message_delay = _observed_message_quota(result["message"])
        if observation is not None and not result["quota_observations"]:
            result["quota_observations"].append(observation)
        if message_delay is not None and not structured_retry:
            delays.append(message_delay)
    for name in ("field_violations", "error_info", "quota_violations", "quota_observations"):
        if len(result[name]) > 8:
            result[name] = result[name][:8]
            result["details_truncated"] = True
    result["retry_after_seconds"] = max(delays) if delays else None
    observed_kinds = {item["kind"] for item in result["quota_observations"]}
    result["quota_kind"] = "daily" if "daily" in observed_kinds else "minute" if "minute" in observed_kinds else "unknown"
    while len(json.dumps(result, ensure_ascii=False).encode()) > MAX_DIAGNOSTIC_BYTES:
        result["details_truncated"] = True
        longest = max(("field_violations", "error_info", "quota_violations", "quota_observations"),
                      key=lambda name: len(json.dumps(result[name])))
        if not result[longest]:
            break
        result[longest].pop()
    return result


def retry_after_seconds(diagnostic: dict) -> float | None:
    return _delay(diagnostic.get("retry_after_seconds")) if type(diagnostic) is dict else None


def quota_kind(diagnostic: dict) -> str:
    value = diagnostic.get("quota_kind") if type(diagnostic) is dict else None
    return value if value in ("minute", "daily") else "unknown"


def next_daily_reset(observed_unix: float) -> float | None:
    """Next documented RPD reset, midnight Pacific, strictly after observation.

    https://ai.google.dev/gemini-api/docs/rate-limits#how-rate-limits-work
    Use the next local calendar midnight, not observed time plus 24 hours:
    Pacific days around DST changes can contain 23 or 25 hours. An explicit
    provider retry hint takes precedence at the caller. If timezone data cannot
    establish the boundary, return None rather than inventing a reset time.
    """
    if type(observed_unix) not in (int, float):
        return None
    try:
        if not math.isfinite(observed_unix):
            return None
        pacific = ZoneInfo("America/Los_Angeles")
        observed = datetime.fromtimestamp(observed_unix, timezone.utc).astimezone(pacific)
        next_date = observed.date() + timedelta(days=1)
        return datetime.combine(next_date, datetime_time.min, tzinfo=pacific).timestamp()
    except (ZoneInfoNotFoundError, ValueError, OverflowError, OSError):
        return None
