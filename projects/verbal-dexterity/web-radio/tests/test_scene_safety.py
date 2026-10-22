"""Regression checks for selected scene dispatch, quotas, and local recovery."""
from __future__ import annotations

import copy
from datetime import date

import httpx
import pytest

from test_scene_service import Clock, quota, reply, scene
from web_radio import gemini, production_pricing
from web_radio.budget import BudgetError
from web_radio.characters import CharacterError

MODEL = "gemini-3.8-flash-tts"


@pytest.fixture(autouse=True)
def no_live_transport_or_environment_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Scene safety fixture attempted a live provider request")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)


def single_line(script):
    value = copy.deepcopy(script)
    value["turns"] = value["turns"][:1]
    return value


def test_crash_retained_running_attempt_blocks_equivalent_new_job_before_http(scene, monkeypatch):
    service, script, _ = scene
    script = single_line(script)
    first = service.create(script, "0.25")
    replacement = service.create(script, "0.25")

    class SimulatedProcessDeath(BaseException):
        pass

    generate = gemini.generate

    def die_after_dispatch_intent(*args, **kwargs):
        raise SimulatedProcessDeath

    monkeypatch.setattr(gemini, "generate", die_after_dispatch_intent)
    with pytest.raises(SimulatedProcessDeath):
        service.run(first["id"])
    assert service.status(first["id"])["lines"][0]["attempts"][0]["state"] == "running"
    monkeypatch.setattr(gemini, "generate", generate)
    calls = []

    def forbidden(request):
        calls.append(request)
        pytest.fail("Equivalent crash-retained work was dispatched again")

    result = service.run(replacement["id"], transport=httpx.MockTransport(forbidden))
    assert not calls and result["status"] == "blocked"
    assert result["lines"][0]["attempts"][0]["state"] == "pending"
    assert service.status(first["id"])["lines"][0]["attempts"][0]["state"] == "running"


def observed_policy(rpm, rpd):
    return {
        "max_wait_seconds": 0,
        "quota_limits": {
            MODEL: {
                "requests_per_minute": rpm, "requests_per_day": rpd,
                "observed_at": "2026-10-23T00:00:00Z",
                "evidence": "Explicit isolated fixture quota observation.",
            },
        },
    }


def test_stricter_new_policy_controls_pacing_after_an_existing_quota_observation(scene):
    service, script, clock = scene
    earlier = service.create(single_line(script), "0.25", policy=observed_policy(10, 100))
    assert service.run(earlier["id"], transport=httpx.MockTransport(reply))["status"] == "ready"
    clock.advance(9)
    current = service.create(script, "0.25", policy=observed_policy(1, 10))
    calls = []

    def respond(request):
        calls.append(request)
        return reply(request)

    result = service.run(current["id"], transport=httpx.MockTransport(respond))
    assert len(calls) == 1 and result["status"] == "waiting_quota"
    assert result["wait"]["retry_at_unix"] - clock() >= 62
    assert not clock.waits
    with service.budget.connect() as db:
        observation = db.execute("SELECT rpm,rpd FROM scene_quotas WHERE model=?", (MODEL,)).fetchone()
    assert tuple(observation) == (1, 10)


@pytest.mark.parametrize("invoice", [None, "0.000010"])
def test_saved_provider_response_recovers_after_price_expiry_without_http(scene, monkeypatch, invoice):
    service, script, _ = scene
    job = service.create(single_line(script), "0.25")
    complete = service._complete

    def fail_local_completion(*args, **kwargs):
        raise OSError("Isolated persistence interruption after response retention")

    monkeypatch.setattr(service, "_complete", fail_local_completion)
    calls = []

    def respond(request):
        calls.append(request)
        return reply(request)

    interrupted = service.run(job["id"], transport=httpx.MockTransport(respond))
    assert len(calls) == 1 and interrupted["status"] == "uncertain"
    attempt = interrupted["lines"][0]["attempts"][0]
    assert attempt["audio_sha256"] and attempt["response_sha256"]
    if invoice is not None:
        service.reconcile(job["id"], attempt["id"], invoice, "Verified isolated fixture invoice evidence.")
    monkeypatch.setattr(service, "_complete", complete)

    class ExpiredDate(date):
        @classmethod
        def today(cls):
            return date(2027, 1, 1)

    monkeypatch.setattr(production_pricing, "date", ExpiredDate)
    with pytest.raises(BudgetError):
        production_pricing.evidence(MODEL)

    def forbidden(*args, **kwargs):
        pytest.fail("Local response recovery attempted provider generation")

    monkeypatch.setattr(gemini, "generate", forbidden)
    recovered = service.recover(job["id"])
    assert recovered["status"] == "ready"
    assert recovered["lines"][0]["state"] == "completed"
    assert len(calls) == 1
    if invoice is None:
        assert recovered["budget"]["spent_microdollars"] == interrupted["budget"]["reserved_microdollars"]
        assert recovered["lines"][0]["attempts"][0]["accounting"] == "conservative-scene-allocation"
    else:
        assert recovered["budget"]["spent_microdollars"] == 10
        assert recovered["accounting"]["verified_invoice_microdollars"] == 10


def test_valid_replacement_recipe_revision_preserves_completed_sources_after_old_recipe_expiry(scene, monkeypatch):
    service, script, _ = scene
    job = service.create(script, "0.25")
    original = service.run(job["id"], transport=httpx.MockTransport(reply))
    assert original["status"] == "ready"
    old_recipe = original["lines"][0]["data"]["voice_recipe"]["recipe_id"]
    replacement = service.characters.add_prebuilt("alpha", "Puck", MODEL)
    bind_cast = service.characters.bind_cast

    def reject_expired_baseline(value):
        if any(speaker.voice_recipe_id == old_recipe for speaker in value.speakers):
            raise CharacterError("Isolated fixture baseline recipe expired")
        return bind_cast(value)

    monkeypatch.setattr(service.characters, "bind_cast", reject_expired_baseline)
    revised = service.revise(
        job["id"], 0, recipe_id=replacement["recipe_id"],
        style="A newly chosen voice and direction.", budget_usd="0.25",
    )
    assert revised["parent_job"] == job["id"]
    assert revised["lines"][0]["data"]["voice_recipe"]["recipe_id"] == replacement["recipe_id"]
    assert revised["lines"][0]["state"] == "pending"
    for index in (1, 2):
        assert revised["lines"][index]["state"] == "completed"
        assert revised["lines"][index]["source"] == original["lines"][index]["source"]
    calls = []

    def respond(request):
        calls.append(request)
        return reply(request, level=2000)

    rendered = service.run(revised["id"], transport=httpx.MockTransport(respond))
    assert rendered["status"] == "ready" and len(calls) == 1
    assert service.status(job["id"])["outputs"] == original["outputs"]


def test_actual_adapter_stop_observation_without_usage_requires_review_and_never_auto_retries(scene):
    service, script, _ = scene
    job = service.create(script, "0.25")
    calls = []

    def respond(request):
        calls.append(request)
        payload = reply(request, usage=False).json()
        # Defensive compatibility observation; documented incomplete responses
        # remain a separate adapter rejection path.
        payload["steps"][0]["stop_reason"] = "MAX_TOKENS"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(respond)
    result = service.run(job["id"], transport=transport)
    assert result["status"] == "review_required" and len(calls) == 1
    assert result["lines"][0]["state"] == "review_required"
    assert result["lines"][0]["source"]["source_attempt"]["output_limit_reached"] is True
    assert all(line["state"] == "pending" for line in result["lines"][1:])
    resumed = service.run(job["id"], resume=True, transport=transport)
    assert resumed["status"] == "review_required" and len(calls) == 1
