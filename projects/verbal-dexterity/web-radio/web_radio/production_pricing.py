"""Versioned production estimates, reservations, and separately attested invoices.

Evidence checked 2026-10-22 UTC (2026-10-23 in the project timezone):
https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash-tts
https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts
https://ai.google.dev/api/interactions-api (generation_config.max_output_tokens)
https://ai.google.dev/api/voices (prompted CreateVoice usage and sample_audio)

A token-priced estimate is not an invoice. A reservation is not a charge. Voice
creation exposes token usage but no standalone fixed fee/output cap is published;
its preset is therefore an explicit local allocation policy, not a billing cap.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING

from .budget import BudgetError, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS

PRICING_VERSION = "gemini-production-standard-2026-10-23-v1"
CHECKED_ON = date(2026, 10, 22)
VALID_THROUGH = date(2026, 12, 31)
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash-tts"
MODEL_SOURCE = "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash-tts"
CONFIG_SOURCE = "https://ai.google.dev/api/interactions-api"
DESIGN_SOURCE = "https://ai.google.dev/api/voices"
_RATES = {
    "gemini-3.8-flash-tts": (Decimal("0.50"), Decimal("9.00")),
    "gemini-3.8-flash-lite-tts": (Decimal("0.50"), Decimal("6.00")),
}
AUDIO_TOKENS_PER_SECOND = Decimal(25)


def _rate(model: str, today: date | None = None):
    if model not in _RATES:
        raise BudgetError("Choose a supported production TTS model.")
    current = today or date.today()
    if current < CHECKED_ON or current > VALID_THROUGH:
        raise BudgetError("Production pricing evidence is outside its verified date range; review it before dispatch.")
    return _RATES[model]


def _count(value, maximum, name):
    if type(value) is not int or not 0 <= value <= maximum:
        raise BudgetError(f"{name} must be a bounded nonnegative integer.")
    return value


def _cost(input_tokens: Decimal | int, output_tokens: Decimal | int, rates) -> int:
    amount = Decimal(input_tokens) * rates[0] + Decimal(output_tokens) * rates[1]
    # USD per million tokens numerically equals microdollars per token.
    return int(amount.to_integral_value(rounding=ROUND_CEILING))


def evidence(model: str, *, today: date | None = None) -> dict:
    input_rate, output_rate = _rate(model, today)
    return {
        "pricing_version": PRICING_VERSION, "checked_on": CHECKED_ON.isoformat(),
        "valid_through": VALID_THROUGH.isoformat(), "model": model, "tier": "standard",
        "text_input_usd_per_million_tokens": str(input_rate),
        "audio_output_usd_per_million_tokens": str(output_rate),
        "audio_output_tokens_per_second": str(AUDIO_TOKENS_PER_SECOND),
        "sources": [PRICE_SOURCE, MODEL_SOURCE, CONFIG_SOURCE, DESIGN_SOURCE],
    }


def conservative_allocation(model: str, *, input_token_upper_bound: int = MAX_INPUT_TOKENS,
                            max_output_tokens: int = MAX_OUTPUT_TOKENS,
                            today: date | None = None) -> int:
    """Allocate against validated input and the exact requested output ceiling."""
    rates = _rate(model, today)
    _count(input_token_upper_bound, MAX_INPUT_TOKENS, "Input token upper bound")
    _count(max_output_tokens, MAX_OUTPUT_TOKENS, "Maximum output tokens")
    if max_output_tokens == 0:
        raise BudgetError("Maximum output tokens must be positive.")
    return _cost(input_token_upper_bound, max_output_tokens, rates)


def request_allocation(request: dict, *, today: date | None = None) -> dict:
    from . import gemini
    bound = gemini.validate_synthesis_request(request)
    model = request["model"]
    ceiling = request["generation_config"]["max_output_tokens"]
    return {
        "reservation_microdollars": conservative_allocation(
            model, input_token_upper_bound=bound, max_output_tokens=ceiling, today=today),
        "input_token_upper_bound": bound, "max_output_tokens": ceiling,
        "pricing_version": PRICING_VERSION, "evidence": evidence(model, today=today),
        "basis": "validated-input-upper-bound-and-explicit-output-token-ceiling",
        "is_invoice_charge": False,
    }


def expected_estimate(model: str, *, input_tokens: int, audio_seconds,
                      today: date | None = None) -> dict:
    """Plan likely token cost from caller estimates; never use as a reservation."""
    rates = _rate(model, today)
    _count(input_tokens, MAX_INPUT_TOKENS, "Estimated input tokens")
    try:
        if type(audio_seconds) is bool:
            raise InvalidOperation
        seconds = Decimal(str(audio_seconds))
        if not seconds.is_finite() or not 0 <= seconds <= Decimal(MAX_OUTPUT_TOKENS) / AUDIO_TOKENS_PER_SECOND:
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        raise BudgetError("Estimated audio duration must fit one model response.") from None
    output_tokens = seconds * AUDIO_TOKENS_PER_SECOND
    amount = _cost(input_tokens, output_tokens, rates)
    return {
        "kind": "expected-token-cost-estimate", "estimated_microdollars": amount,
        "input_tokens": input_tokens, "audio_seconds": str(seconds),
        "estimated_output_tokens": str(output_tokens), "pricing_version": PRICING_VERSION,
        "evidence": evidence(model, today=today), "is_invoice_charge": False,
        "notice": "Duration and input tokens are estimates; not a reservation or verified invoice.",
    }


def provider_usage_estimate(model: str, usage: dict, *, today: date | None = None) -> dict | None:
    """Price complete, plausible reported token counts without claiming an invoice."""
    rates = _rate(model, today)
    if type(usage) is not dict:
        return None
    incoming, outgoing = usage.get("total_input_tokens"), usage.get("total_output_tokens")
    if (type(incoming) is not int or not 0 <= incoming <= MAX_INPUT_TOKENS
            or type(outgoing) is not int or not 0 < outgoing <= MAX_OUTPUT_TOKENS
            or usage.get("total_thought_tokens", 0) != 0 or usage.get("total_tool_use_tokens", 0) != 0):
        return None
    return {
        "kind": "provider-token-usage-estimate", "estimated_microdollars": _cost(incoming, outgoing, rates),
        "input_tokens": incoming, "output_tokens": outgoing, "pricing_version": PRICING_VERSION,
        "is_invoice_charge": False, "evidence": evidence(model, today=today),
    }


def voice_design_preset(model: str, preset: str = "standard", *, today: date | None = None) -> dict:
    """Provide an honest local allowance without requiring fabricated quote JSON."""
    proof = evidence(model, today=today)
    if preset != "standard":
        raise BudgetError("Choose the standard voice-design allocation preset.")
    return {
        "preset": "standard", "allocation_microdollars": 250000, "quoted_cost_usd": "0.250000",
        "quote_evidence": (
            f"Local voice-design standard allocation policy {PRICING_VERSION}. "
            "CreateVoice reports prompted token usage; standalone fixed price/output cap is not published. "
            "This USD0.25 allowance is not a provider quote or guaranteed billing ceiling. Sources: "
            + DESIGN_SOURCE + " " + PRICE_SOURCE),
        "quote_valid_until": VALID_THROUGH.isoformat() + "T23:59:59+00:00",
        "basis": "versioned-local-allocation-policy", "pricing_version": PRICING_VERSION,
        "evidence": proof, "is_provider_quote": False, "is_invoice_charge": False,
    }


def verified_invoice_charge(charged_usd: str, evidence: str, *, confirmed: bool = False) -> dict:
    """Record explicit operator attestation; do not infer verification from usage."""
    if confirmed is not True or not isinstance(evidence, str) or not 12 <= len(evidence) <= 2000:
        raise BudgetError("Confirm an external invoice and record its billing evidence.")
    try:
        if type(charged_usd) is not str or not 1 <= len(charged_usd) <= 32:
            raise InvalidOperation
        amount = Decimal(charged_usd)
        if not amount.is_finite() or not 0 <= amount <= 10000:
            raise InvalidOperation
        units = amount * 1_000_000
        if units != units.to_integral_value():
            raise InvalidOperation
    except (InvalidOperation, ValueError, TypeError):
        raise BudgetError("Invoice amount must be USD0–10000 with up to six decimal places.") from None
    return {
        "kind": "operator-confirmed-invoice-charge", "charged_microdollars": int(units),
        "evidence": evidence, "verification": "external-invoice-confirmed-by-operator",
        "is_invoice_charge": True,
    }


class ProductionPricing:
    """Convenience namespace; module functions are the canonical integration API."""
    expected_estimate = staticmethod(expected_estimate)
    conservative_allocation = staticmethod(conservative_allocation)
    request_allocation = staticmethod(request_allocation)
    provider_usage_estimate = staticmethod(provider_usage_estimate)
    voice_design_preset = staticmethod(voice_design_preset)
    verified_invoice_charge = staticmethod(verified_invoice_charge)
