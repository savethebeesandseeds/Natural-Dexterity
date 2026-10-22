"""Backend speech planning and hostile/ambiguous provider responses use fixtures only."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import wave

import httpx
import pytest
from pydantic import ValidationError

from web_radio import gemini


def script(*voices, turns=None):
    voices = voices or ("Charon", "Kore")
    speakers = [{
        "name": f"Character {index}", "voice": voice, "recipe_version": 1,
        "style": "calm", "design_prompt": "Prebuilt creative brief, not a designed voice.",
    } for index, voice in enumerate(voices)]
    turns = turns or [{"speaker": speaker["name"], "text": f"Line {index}.", "style": ""}
                      for index, speaker in enumerate(speakers)]
    return gemini.Script(
        programme_id="fixture", series_id="fixture", kind="programme", title="Fixture",
        language="en", speakers=speakers, turns=turns,
    )


def wav_bytes(*, channels=1, width=2, rate=24000, frames=24):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(b"\0" * channels * width * frames)
    return stream.getvalue()


def audio_block(audio=None):
    return {"type": "audio", "mime_type": "audio/wav",
            "data": base64.b64encode(wav_bytes() if audio is None else audio).decode("ascii")}


def result(block=None):
    return {
        "id": "fixture-response", "status": "completed", "model": "gemini-3.8-flash-tts",
        "steps": [{"type": "model_output", "content": [audio_block() if block is None else block]}],
        "usage": {"total_input_tokens": 23, "total_output_tokens": 25},
    }


def request():
    return gemini.build_request(script(), "gemini-3.8-flash-tts")


@pytest.mark.parametrize("voice,approved", [("Charon", ()), ("voice_OriginalFixture", ("voice_OriginalFixture",))])
def test_single_speaker_uses_native_array_shape(voice, approved):
    payload = gemini.build_request(script(voice), "gemini-3.8-flash-tts",
                                   approved_designed_voices=approved)
    assert payload["generation_config"]["speech_config"] == [{"voice": voice, "language": "en"}]
    assert payload["input"][0]["content"][0]["annotations"][0]["speaker"] == "Character 0"
    assert payload["response_format"] == {"type": "audio", "mime_type": "audio/wav"}


def test_two_prebuilt_voices_use_one_native_request_and_exact_directions():
    scene = script()
    scene.speakers[0].pronunciation_notes = "Name sounds like ren."
    scene.turns[0].style = "whispered"
    plan = gemini.build_plan(scene, "gemini-3.8-flash-tts")
    assert plan["strategy"] == "native" and plan["request_count"] == 1
    payload = plan["requests"][0]["request"]
    assert payload == gemini.build_request(scene, "gemini-3.8-flash-tts")
    speakers = payload["generation_config"]["speech_config"]["speakers"]
    assert [speaker["voice"] for speaker in speakers] == ["Charon", "Kore"]
    first = payload["input"][0]["content"][0]
    assert first["text"] == "Line 0."
    assert first["annotations"] == [{"type": "speech_metadata", "speaker": "Character 0",
                                     "style": "calm; Name sounds like ren.; whispered"}]
    assert "Prebuilt creative brief" not in json.dumps(payload)


@pytest.mark.parametrize("voices,approved", [
    (("Charon", "Kore", "Puck"), ()),
    (("voice_OriginalFixture", "Kore"), ("voice_OriginalFixture",)),
])
def test_larger_or_designed_cast_uses_ordered_unary_turns(voices, approved):
    turns = [{"speaker": f"Character {index}", "text": text, "style": "quiet"}
             for index, text in [(1, "First."), (0, "Second."), (1, "Third.")]]
    scene = script(*voices, turns=turns)
    plan = gemini.build_plan(scene, "gemini-3.8-flash-lite-tts",
                             approved_designed_voices=approved)
    assert plan["strategy"] == "turns" and plan["request_count"] == len(turns)
    for index, item in enumerate(plan["requests"]):
        assert item["turn_index"] == index and item["turn_indices"] == [index]
        assert item["speaker"] == turns[index]["speaker"]
        payload = item["request"]
        assert payload["input"][0]["content"][0]["text"] == turns[index]["text"]
        cast_index = int(turns[index]["speaker"].split()[-1])
        assert payload["generation_config"]["speech_config"] == [{"voice": voices[cast_index], "language": "en"}]
        assert item["input_token_upper_bound"] <= gemini.MAX_INPUT_TOKENS
    with pytest.raises(ValueError, match="build_plan"):
        gemini.build_request(scene, "gemini-3.8-flash-lite-tts", approved_designed_voices=approved)


@pytest.mark.parametrize("voice,approved", [
    ("UnknownVoice", ()), ("voice_OriginalFixture", ()),
    ("voicekey_replication", ("voicekey_replication",)), ("replicated", ("replicated",)),
])
def test_unknown_or_replication_voice_is_rejected_before_request_planning(voice, approved):
    with pytest.raises(ValueError, match="character registry"):
        gemini.build_plan(script(voice), "gemini-3.8-flash-tts",
                           approved_designed_voices=approved)


def test_cast_references_are_retained_and_supported_cast_is_bounded():
    scene = script()
    scene.speakers[0].character_id = "moss"
    scene.speakers[0].character_revision = 2
    scene.speakers[0].voice_recipe_id = "a" * 32
    assert scene.model_dump()["speakers"][0]["character_revision"] == 2
    assert len(script(*(["Charon"] * 16)).speakers) == 16
    with pytest.raises(ValidationError):
        script(*(["Charon"] * 17))


def test_prebuilt_oversize_native_dialogue_chunks_at_turn_boundaries_without_loss():
    turns = [{"speaker": f"Character {index % 2}", "text": str(index) + "." + "w" * 900,
              "style": "unhurried"} for index in range(12)]
    scene = script(turns=turns)
    with pytest.raises(ValueError, match="conservative input-token bound"):
        gemini.build_request(scene, "gemini-3.8-flash-tts")
    plan = gemini.build_plan(scene, "gemini-3.8-flash-tts")
    assert plan["strategy"] == "chunks"
    assert 1 < plan["request_count"] < len(turns)
    indices = [index for item in plan["requests"] for index in item["turn_indices"]]
    rendered = [part["text"] for item in plan["requests"] for part in item["request"]["input"][0]["content"]]
    assert indices == list(range(len(turns)))
    assert rendered == [turn["text"] for turn in turns]
    assert all(item["input_token_upper_bound"] <= gemini.MAX_INPUT_TOKENS for item in plan["requests"])


def test_full_bound_counts_repeated_directions_voice_config_and_utf8():
    scene = script("Charon")
    scene.speakers[0].style = "s" * 4000
    scene.speakers[0].pronunciation_notes = "p" * 1900
    scene.turns[0].style = "t" * 1900
    scene.turns[0].text = "Short spoken text."
    with pytest.raises(ValueError, match="input-token bound"):
        gemini.build_plan(scene, "gemini-3.8-flash-tts")
    scene = script("Charon")
    scene.turns[0].text = "\U0001f333" * 2000
    with pytest.raises(ValueError, match="input-token bound"):
        gemini.build_plan(scene, "gemini-3.8-flash-tts")


def test_direct_oversize_dispatch_never_contacts_provider():
    payload = request()
    payload["input"][0]["content"][0]["annotations"][0]["style"] = "x" * 10000
    def forbidden(_request):
        pytest.fail("Oversized request reached dispatch")
    with pytest.raises(ValueError, match="input-token bound"):
        gemini.generate(payload, "fixture-not-real", httpx.MockTransport(forbidden))


def test_one_audio_result_returns_exact_original_with_safe_metadata():
    seen = []
    def response(sent):
        seen.append(sent)
        assert sent.headers["x-goog-api-key"] == "fixture-not-real"
        assert sent.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json=result())
    audio, metadata = gemini.generate(request(), "fixture-not-real", httpx.MockTransport(response))
    assert audio == wav_bytes()
    assert metadata["audio_sha256"] == hashlib.sha256(audio).hexdigest()
    assert metadata["usage"]["total_output_tokens"] == 25
    assert "fixture-not-real" not in json.dumps(metadata)
    assert len(seen) == 1


@pytest.mark.parametrize("bad", [
    None, [], True, {"status": "completed"}, {"status": "failed", "steps": []},
    {"status": "completed", "steps": None},
    {"status": "completed", "steps": [None]},
    {"status": "completed", "steps": [{"type": "model_output", "content": {}}]},
    {"status": "completed", "steps": [{"type": "model_output", "content": [None]}]},
    {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Unexpected."}]}]},
])
def test_malformed_or_unconfirmed_result_is_explicitly_ambiguous(bad):
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=bad))
    with pytest.raises(gemini.GeminiFailure) as failure:
        gemini.generate(request(), "fixture-not-real", transport)
    assert failure.value.ambiguous is True


def test_multiple_audio_blocks_are_never_silently_truncated_to_last_block():
    response = result()
    response["steps"].append({"type": "model_output", "content": [audio_block(wav_bytes(frames=48))]})
    with pytest.raises(gemini.GeminiFailure, match="exactly one") as failure:
        gemini.generate(request(), "fixture-not-real",
                         httpx.MockTransport(lambda _request: httpx.Response(200, json=response)))
    assert failure.value.ambiguous is True


@pytest.mark.parametrize("mime", [None, "audio/x-wav", "audio/l16", "audio/mpeg"])
def test_audio_declared_format_must_match_requested_wav(mime):
    block = audio_block()
    block["mime_type"] = mime
    with pytest.raises(gemini.GeminiFailure, match="MIME"):
        gemini.parse_audio_block(block)


@pytest.mark.parametrize("audio", [
    b"", b"\0" * 44, b"RIFF" + b"\0" * 4 + b"AVI " + b"\0" * 32,
    wav_bytes()[:-2], wav_bytes() + b"trailing",
    wav_bytes(width=3), wav_bytes(channels=3), wav_bytes(rate=1000), wav_bytes(frames=0),
])
def test_wav_must_be_complete_bounded_pcm(audio):
    with pytest.raises(gemini.GeminiFailure) as failure:
        gemini.parse_audio_block(audio_block(audio))
    assert failure.value.ambiguous is True


@pytest.mark.parametrize("data", [None, 123, "%invalid%", "\U0001f333"])
def test_audio_base64_must_be_valid_ascii_data(data):
    block = audio_block()
    block["data"] = data
    with pytest.raises(gemini.GeminiFailure):
        gemini.parse_audio_block(block)


def test_decoded_audio_size_is_bounded(monkeypatch):
    monkeypatch.setattr(gemini, "MAX_AUDIO_BYTES", 32)
    with pytest.raises(gemini.GeminiFailure, match="oversized"):
        gemini.parse_audio_block(audio_block())


class Chunks(httpx.SyncByteStream):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values


@pytest.mark.parametrize("failure", ["declared", "streamed", "encoding", "duplicate", "nonfinite", "overflow"])
def test_provider_json_is_bounded_and_unambiguous(failure, monkeypatch):
    monkeypatch.setattr(gemini, "MAX_RESPONSE_BYTES", 64)
    def response(_request):
        if failure == "declared":
            return httpx.Response(200, stream=Chunks([b"{}"]),
                                  headers={"content-length": "65"})
        if failure == "streamed":
            return httpx.Response(200, stream=Chunks([b"x" * 40, b"x" * 40]))
        if failure == "encoding":
            return httpx.Response(200, stream=Chunks([b"not-gzip"]),
                                  headers={"content-encoding": "gzip"})
        if failure == "duplicate":
            return httpx.Response(200, content=b'{"status":"completed","status":"failed"}')
        if failure == "nonfinite":
            return httpx.Response(200, content=b'{"status":NaN}')
        return httpx.Response(200, content=b'{"status":"completed","steps":[],"usage":{"x":1e999}}')
    with pytest.raises(gemini.GeminiFailure) as error:
        gemini.generate(request(), "fixture-not-real", httpx.MockTransport(response))
    assert error.value.ambiguous is True


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_failures_never_echo_provider_body_or_retry(status):
    seen = []
    def response(sent):
        seen.append(sent)
        return httpx.Response(status, text="fixture-not-real sensitive debug",
                              headers={"location": "https://example.invalid"})
    with pytest.raises(gemini.GeminiFailure) as error:
        gemini.generate(request(), "fixture-not-real", httpx.MockTransport(response))
    assert "fixture-not-real" not in str(error.value)
    assert "example.invalid" not in str(error.value)
    assert len(seen) == 1


def test_transport_timeout_is_ambiguous_without_automatic_retry():
    seen = []
    def timeout(sent):
        seen.append(sent)
        raise httpx.ReadTimeout("fixture-not-real private diagnostics", request=sent)
    with pytest.raises(gemini.GeminiFailure) as error:
        gemini.generate(request(), "fixture-not-real", httpx.MockTransport(timeout))
    assert error.value.ambiguous is True
    assert "fixture-not-real" not in str(error.value) and len(seen) == 1


def test_original_voice_design_helper_accepts_prompted_only_without_audio_inputs():
    payload = {
        "store": True,
        "voice": {"model": "gemini-3.8-flash-tts", "type": "prompted",
                  "display_name": "Original fixture", "language_code": "en",
                  "prompted": {"input": "An original gentle fictional creature."}},
    }
    calls = []
    def response(sent):
        calls.append(sent)
        assert sent.url.path == "/v1beta/voices"
        return httpx.Response(200, json={"id": "voice_fixture", "type": "prompted",
                                         "sample_audio": audio_block()})
    returned = gemini.post_json("/v1beta/voices", payload, "fixture-not-real",
                                transport=httpx.MockTransport(response))
    assert gemini.parse_audio_block(returned["sample_audio"]) == wav_bytes()
    assert len(calls) == 1
    payload["voice"]["type"] = "replicated"
    with pytest.raises(ValueError, match="original voices"):
        gemini.post_json("/v1beta/voices", payload, "fixture-not-real",
                         transport=httpx.MockTransport(response))
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["tools", "audio-input", "output-cap", "reference-audio"])
def test_unsupported_payload_fields_fail_before_network(mutation):
    payload = request()
    path = "/v1beta/interactions"
    if mutation == "tools":
        payload["tools"] = [{"type": "search"}]
    elif mutation == "audio-input":
        payload["input"][0]["content"][0]["type"] = "audio"
    elif mutation == "output-cap":
        payload["generation_config"]["max_output_tokens"] = gemini.MAX_OUTPUT_TOKENS + 1
    else:
        path = "/v1beta/voices"
        payload = {"store": True, "voice": {
            "model": "gemini-3.8-flash-tts", "type": "prompted", "display_name": "Fixture",
            "prompted": {"input": "A fictional voice.", "reference_audio": "not-supported"},
        }}
    with pytest.raises(ValueError):
        gemini.post_json(path, payload, "fixture-not-real",
                         transport=httpx.MockTransport(lambda _: pytest.fail("Unsupported payload dispatched")))


@pytest.mark.parametrize("mutation", [
    "replication-key", "three-native", "native-designed", "unknown-speaker",
    "unary-multiple-speakers", "wrong-language",
])
def test_retained_synthesis_request_has_strict_native_shapes(mutation):
    payload = request()
    config = payload["generation_config"]["speech_config"]
    if mutation == "replication-key":
        payload = gemini.build_request(script("Charon"), "gemini-3.8-flash-tts")
        payload["generation_config"]["speech_config"][0]["voice"] = "voicekey_person"
    elif mutation == "three-native":
        config["speakers"].append({"speaker": "Third", "voice": "Puck", "language": "en"})
    elif mutation == "native-designed":
        config["speakers"][0]["voice"] = "voice_original"
    elif mutation == "unknown-speaker":
        payload["input"][0]["content"][0]["annotations"][0]["speaker"] = "Missing"
    elif mutation == "unary-multiple-speakers":
        payload["generation_config"]["speech_config"] = [{"voice": "Charon", "language": "en"}]
    else:
        config["speakers"][0]["language"] = ""
    with pytest.raises(ValueError):
        gemini.validate_synthesis_request(payload)


@pytest.mark.parametrize("field,value", [
    ("usage", []), ("usage", "unexpected"), ("usage", {"count": float("inf")}),
    ("id", {}), ("model", 123),
])
def test_unexpected_metadata_never_becomes_a_completed_take(field, value):
    response = result()
    response[field] = value
    if field == "usage" and isinstance(value, dict):
        encoded = json.dumps(response).replace("Infinity", "1e999")
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=encoded))
    else:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    with pytest.raises(gemini.GeminiFailure) as failure:
        gemini.generate(request(), "fixture-not-real", transport)
    assert failure.value.ambiguous is True


def test_retained_request_validation_returns_the_documented_conservative_bound():
    payload = request()
    expected = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + gemini.REQUEST_TOKEN_OVERHEAD
    assert gemini.validate_synthesis_request(payload) == expected
    assert expected <= gemini.MAX_INPUT_TOKENS


def test_minimal_character_reference_needs_no_dummy_voice_fields():
    recipe = gemini.SpeakerRecipe(name="Moss", character_id="moss")
    assert recipe.voice == "" and recipe.recipe_version == 1
    assert recipe.style == "" and recipe.design_prompt == ""
    data = script("Charon").model_dump()
    data["speakers"] = [recipe.model_dump()]
    data["turns"] = [{"speaker": "Moss", "text": "A short line.", "style": ""}]
    scene = gemini.Script.model_validate(data)
    with pytest.raises(ValueError, match="character registry"):
        gemini.build_plan(scene, "gemini-3.8-flash-tts")


@pytest.mark.parametrize("values", [
    {"name": "No identity"},
    {"name": "Blank identity", "voice": "   "},
    {"name": "Orphan recipe", "voice": "Charon", "voice_recipe_id": "a" * 32},
    {"name": "Orphan revision", "voice": "Charon", "character_revision": 1},
])
def test_speaker_needs_real_or_resolvable_identity(values):
    with pytest.raises(ValidationError):
        gemini.SpeakerRecipe.model_validate(values)


@pytest.mark.parametrize("maximum", [1, 1024, 16384])
def test_explicit_short_line_output_ceiling_is_retained_in_native_and_turn_plans(maximum):
    native = gemini.build_request(script("Charon"), "gemini-3.8-flash-tts", max_output_tokens=maximum)
    assert native["generation_config"]["max_output_tokens"] == maximum
    gemini.validate_synthesis_request(native)
    plan = gemini.build_plan(script("Charon", "Kore", "Puck"), "gemini-3.8-flash-tts",
                             max_output_tokens=maximum)
    assert all(item["request"]["generation_config"]["max_output_tokens"] == maximum for item in plan["requests"])


@pytest.mark.parametrize("maximum", [True, 0, -1, 16385, "1024"])
def test_output_ceiling_cannot_be_invalid_or_boolean(maximum):
    with pytest.raises(ValueError):
        gemini.build_request(script("Charon"), "gemini-3.8-flash-tts", max_output_tokens=maximum)


def test_structured_provider_quota_error_is_sanitized_and_never_retried():
    key = "fixture-private-diagnostic-key"
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota rejected " + key,
                     "details": [
                         {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"},
                         {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                          "violations": [{"quotaMetric": "generate_requests",
                                          "quotaId": "RequestsPerMinutePerModel", "quotaValue": "2",
                                          "quotaDimensions": {"model": "fixture-model", "location": "global"}}]},
                     ]}}
    calls = []
    def provider(sent):
        calls.append(sent)
        return httpx.Response(429, json=body)
    with pytest.raises(gemini.GeminiFailure) as failure:
        gemini.generate(request(), key, httpx.MockTransport(provider))
    assert len(calls) == 1
    assert failure.value.classification == "quota-rejected" and not failure.value.ambiguous
    diagnostic = failure.value.diagnostic
    assert diagnostic["quota_kind"] == "minute" and diagnostic["retry_after_seconds"] == 31
    assert diagnostic["quota_observations"][0]["limit"] == 2
    assert key not in json.dumps(diagnostic) and key not in str(failure.value)


def test_non_success_error_body_is_bounded_without_partial_details():
    from web_radio.provider_diagnostics import MAX_ERROR_BYTES
    class ErrorChunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (MAX_ERROR_BYTES + 1)
            pytest.fail("Oversized provider diagnostic body continued reading")
    with pytest.raises(gemini.GeminiFailure) as failure:
        gemini.generate(request(), "fixture-key",
                        httpx.MockTransport(lambda _: httpx.Response(429, stream=ErrorChunks())))
    assert failure.value.diagnostic["details_truncated"]
    assert not failure.value.diagnostic["details_available"]


def test_provider_reported_tokens_cannot_exceed_requested_line_ceiling():
    payload = gemini.build_request(script("Charon"), "gemini-3.8-flash-tts", max_output_tokens=10)
    with pytest.raises(gemini.GeminiFailure, match="requested token ceiling") as failure:
        gemini.generate(payload, "fixture-key", httpx.MockTransport(lambda _: httpx.Response(200, json=result())))
    assert failure.value.ambiguous and failure.value.classification == "dispatched_uncertain"


def test_legacy_gemini_failure_constructor_is_backward_compatible():
    failure = gemini.GeminiFailure("Safe local message", True)
    assert failure.ambiguous and str(failure) == "Safe local message"
    assert failure.diagnostic["classification"] == "dispatched_uncertain"


@pytest.mark.parametrize("origin", ["interaction", "model_output"])
@pytest.mark.parametrize("field", ["finish_reason", "stop_reason", "finishReason", "stopReason"])
def test_explicit_completion_observation_survives_real_adapter_without_usage(origin, field):
    fixture = result()
    fixture.pop("usage")
    container = fixture if origin == "interaction" else fixture["steps"][0]
    container[field] = "MAX_TOKENS"
    original = copy.deepcopy(fixture)
    audio, metadata = gemini.generate(
        request(), "fixture-not-real", httpx.MockTransport(lambda _: httpx.Response(200, json=fixture)))
    assert audio == wav_bytes()
    assert fixture == original
    assert metadata["usage"] == {}
    assert metadata["finish_reason"] == "MAX_TOKENS"
    assert metadata["completion_observations"] == [{
        "source": "interaction" if origin == "interaction" else "steps[0]",
        "field": field, "value": "MAX_TOKENS"}]
    assert "not guaranteed" in metadata["completion_observations_notice"]


def test_documented_incomplete_status_rejects_truncation_even_without_usage():
    fixture = result()
    fixture.pop("usage")
    fixture["status"] = "incomplete"
    with pytest.raises(gemini.GeminiFailure, match="completed performance") as failure:
        gemini.generate(request(), "fixture-not-real",
                        httpx.MockTransport(lambda _: httpx.Response(200, json=fixture)))
    assert failure.value.ambiguous


def test_completion_observations_preserve_conflicting_sources_and_prioritize_explicit_token_stop():
    fixture = result()
    fixture["finish_reason"] = "STOP"
    fixture["steps"][0]["stop_reason"] = "MAX_OUTPUT_TOKENS"
    _, metadata = gemini.generate(request(), "fixture-not-real",
                                  httpx.MockTransport(lambda _: httpx.Response(200, json=fixture)))
    assert metadata["finish_reason"] == "MAX_OUTPUT_TOKENS"
    assert [item["value"] for item in metadata["completion_observations"]] == ["STOP", "MAX_OUTPUT_TOKENS"]


@pytest.mark.parametrize("value", [True, {"reason": "MAX_TOKENS"}, "x" * 81, "private\nvalue"])
def test_completion_observations_are_bounded_primitive_tokens(value):
    fixture = result()
    fixture["finish_reason"] = value
    with pytest.raises(gemini.GeminiFailure, match="completion metadata"):
        gemini.generate(request(), "fixture-not-real",
                        httpx.MockTransport(lambda _: httpx.Response(200, json=fixture)))


def test_absent_completion_fields_leave_default_metadata_unchanged():
    _, metadata = gemini.generate(request(), "fixture-not-real",
                                  httpx.MockTransport(lambda _: httpx.Response(200, json=result())))
    assert "completion_observations" not in metadata
    assert "finish_reason" not in metadata
