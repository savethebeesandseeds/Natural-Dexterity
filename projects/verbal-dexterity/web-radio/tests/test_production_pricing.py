"""Versioned pricing separates estimates, allocations, and invoice attestations."""
from datetime import date

import pytest

from web_radio import gemini, production_pricing as pricing
from web_radio.budget import BudgetError

MODEL = "gemini-3.8-flash-tts"
LITE = "gemini-3.8-flash-lite-tts"
TODAY = date(2026, 10, 22)


def unary():
    script = gemini.Script(
        programme_id="fixture", series_id="fixture", kind="programme", title="Fixture", language="en",
        speakers=[{"name": "Moss", "voice": "Charon"}],
        turns=[{"speaker": "Moss", "text": "Hello."}],
    )
    return gemini.build_request(script, MODEL, max_output_tokens=1024)


def test_expected_cost_is_small_but_never_claims_to_be_allocation_or_invoice():
    expected = pricing.expected_estimate(MODEL, input_tokens=100, audio_seconds=10, today=TODAY)
    assert expected["estimated_microdollars"] == 2300
    assert expected["kind"] == "expected-token-cost-estimate" and not expected["is_invoice_charge"]
    maximum = pricing.conservative_allocation(MODEL, max_output_tokens=1024, today=TODAY)
    assert maximum == 13312
    assert maximum > expected["estimated_microdollars"]


def test_full_serving_ceiling_and_short_line_ceiling_are_distinct():
    assert pricing.conservative_allocation(MODEL, today=TODAY) == 151552
    assert pricing.conservative_allocation(LITE, today=TODAY) == 102400
    assert pricing.conservative_allocation(LITE, input_token_upper_bound=500,
                                           max_output_tokens=1024, today=TODAY) == 6394


def test_request_allocation_prices_exact_validated_input_and_explicit_output_limit():
    request = unary()
    result = pricing.request_allocation(request, today=TODAY)
    assert result["max_output_tokens"] == 1024
    assert result["input_token_upper_bound"] == gemini.validate_synthesis_request(request)
    assert result["reservation_microdollars"] == pricing.conservative_allocation(
        MODEL, input_token_upper_bound=result["input_token_upper_bound"], max_output_tokens=1024, today=TODAY)
    assert result["pricing_version"] == pricing.PRICING_VERSION
    assert not result["is_invoice_charge"] and result["evidence"]["tier"] == "standard"
    request["generation_config"]["max_output_tokens"] = True
    with pytest.raises(ValueError):
        pricing.request_allocation(request, today=TODAY)


@pytest.mark.parametrize("changes", [
    {"input_token_upper_bound": True}, {"input_token_upper_bound": 8193},
    {"input_token_upper_bound": -1}, {"max_output_tokens": True},
    {"max_output_tokens": 0}, {"max_output_tokens": 16385},
])
def test_invalid_allocation_bounds_rejected(changes):
    with pytest.raises(BudgetError):
        pricing.conservative_allocation(MODEL, today=TODAY, **changes)


@pytest.mark.parametrize("current", [date(2026, 10, 21), date(2027, 1, 1)])
def test_pricing_evidence_cannot_silently_cross_its_verified_period(current):
    with pytest.raises(BudgetError, match="verified date range"):
        pricing.conservative_allocation(MODEL, today=current)


@pytest.mark.parametrize("duration", [True, float("nan"), float("inf"), -1, 10000])
def test_expected_duration_must_be_bounded_finite_and_not_boolean(duration):
    with pytest.raises(BudgetError):
        pricing.expected_estimate(MODEL, input_tokens=5, audio_seconds=duration, today=TODAY)


def test_voice_design_preset_is_explicit_local_policy_not_a_fabricated_provider_quote():
    preset = pricing.voice_design_preset(LITE, today=TODAY)
    assert preset["allocation_microdollars"] == 250000 and preset["quoted_cost_usd"] == "0.250000"
    assert preset["basis"] == "versioned-local-allocation-policy"
    assert not preset["is_provider_quote"] and not preset["is_invoice_charge"]
    assert "not a provider quote" in preset["quote_evidence"]
    assert "usage" in preset["quote_evidence"]
    assert "2026-12-31" in preset["quote_valid_until"]


def test_reported_usage_prices_estimate_separately_from_invoice():
    estimate = pricing.provider_usage_estimate(
        MODEL, {"total_input_tokens": 20, "total_output_tokens": 25}, today=TODAY)
    assert estimate["estimated_microdollars"] == 235
    assert estimate["kind"] == "provider-token-usage-estimate" and not estimate["is_invoice_charge"]
    assert pricing.provider_usage_estimate(MODEL, {"total_output_tokens": 25}, today=TODAY) is None
    assert pricing.provider_usage_estimate(
        MODEL, {"total_input_tokens": True, "total_output_tokens": 25}, today=TODAY) is None


def test_invoice_charge_requires_explicit_external_attestation():
    with pytest.raises(BudgetError, match="Confirm"):
        pricing.verified_invoice_charge("0.000235", "External invoice row fixture")
    invoice = pricing.verified_invoice_charge("0.000235", "External invoice row fixture", confirmed=True)
    assert invoice["charged_microdollars"] == 235 and invoice["is_invoice_charge"]


@pytest.mark.parametrize("amount", ["NaN", "-1", "0.0000001", "invalid", "1e1000000"])
def test_invoice_amount_cannot_be_nonfinite_negative_or_rounded(amount):
    with pytest.raises(BudgetError):
        pricing.verified_invoice_charge(amount, "External invoice row fixture", confirmed=True)


def test_class_namespace_matches_module_api():
    assert pricing.ProductionPricing().request_allocation(unary(), today=TODAY)["max_output_tokens"] == 1024
