"""Embedding adapter fixtures; no credentials, network, or model-generated samples."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import struct
import wave

import httpx
import pytest

from web_radio import embedding_provider as provider


def wav(*, frames=800, rate=8000, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(width)
        recording.setframerate(rate)
        recording.writeframes(b"\0" * frames * channels * width)
    return output.getvalue()


def response(dimensions=128, **extra):
    return {"embedding": {"values": [0.25] * dimensions}, **extra}


def request():
    return provider.build_request(text="A quiet forest.", dimensions=128)


def embed_result(result=None, *, payload=None, status=200, headers=None):
    calls = []
    def handler(sent):
        calls.append(sent)
        return httpx.Response(status, json=response() if result is None else result, headers=headers)
    output = provider.embed(request() if payload is None else payload, "fixture-not-a-credential",
                            transport=httpx.MockTransport(handler))
    return output, calls


@pytest.mark.parametrize("text,audio,modalities", [
    ("A quiet forest.", None, ["text"]),
    (None, wav(), ["audio"]),
    ("A quiet forest.", wav(), ["text", "audio"]),
])
def test_modalities_use_one_content_and_identical_shared_space_config(text, audio, modalities):
    payload = provider.build_request(text=text, audio=audio)
    assert payload["model"] == "models/gemini-embedding-2"
    assert payload["embedContentConfig"] == {"outputDimensionality": 3072, "autoTruncate": False}
    assert set(payload["content"]) == {"parts"}
    assert len(payload["content"]["parts"]) == len(modalities)
    if text is not None:
        assert payload["content"]["parts"][0] == {"text": text}
    if audio is not None:
        part = payload["content"]["parts"][-1]["inlineData"]
        assert part["mimeType"] == "audio/wav"
        assert base64.b64decode(part["data"]) == audio
    assert not {"taskType", "title", "task_type"} & set(payload["embedContentConfig"])
    info = provider.validate_request(payload)
    assert info["modalities"] == modalities
    assert info["provider_input_token_limit"] == info["billing_token_ceiling"] == 8192
    assert info["billing_basis"] == "provider-input-token-ceiling"
    if audio is not None:
        assert info["audio_seconds"] == 0.1
        assert info["audio"]["frames"] == 800
        assert info["input_token_upper_bound"] is None
        assert info["token_bound_basis"] == "audio-tokenization-not-inferred"
    else:
        assert info["input_token_upper_bound"] == len(text.encode()) + 256


@pytest.mark.parametrize("dimensions", [128, 129, 768, 1536, 3072])
def test_all_documented_dimension_range_is_supported(dimensions):
    payload = provider.build_request("A forest.", dimensions=dimensions)
    vector, _ = provider.parse_response(response(dimensions), dimensions)
    assert len(vector) == dimensions
    assert payload["embedContentConfig"]["outputDimensionality"] == dimensions


@pytest.mark.parametrize("dimensions", [True, False, 127, 3073, 1.5, "768", None])
def test_invalid_dimensions_rejected_before_dispatch(dimensions):
    with pytest.raises(ValueError):
        provider.build_request("A forest.", dimensions=dimensions)


@pytest.mark.parametrize("text,audio", [
    (None, None), ("", None), (" \t\n", None), (123, None),
    ("\ud800", None), ("", wav()), (None, b""),
    (None, bytearray(wav())), (None, b"not a WAV"),
])
def test_missing_or_invalid_input_rejected(text, audio):
    with pytest.raises(ValueError):
        provider.build_request(text, audio)


def test_text_admission_is_utf8_bytes_and_never_truncates():
    exact = "🌳" * 1024
    assert provider.build_request(exact)["content"]["parts"][0]["text"] == exact
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        provider.build_request(exact + "🌳")


def test_full_audio_window_bound_is_measured_from_pcm_frames():
    exact = wav(frames=30 * 8000)
    info = provider.validate_request(provider.build_request(audio=exact))
    assert info["audio_seconds"] == 30
    assert info["audio"]["sha256"] == hashlib.sha256(exact).hexdigest()
    with pytest.raises(ValueError, match="30 seconds"):
        provider.build_request(audio=wav(frames=30 * 8000 + 1))


@pytest.mark.parametrize("audio", [
    wav(frames=0), wav(width=1), wav(width=3), wav(channels=3),
    wav(rate=96000), wav()[:-2], wav() + b"tail",
])
def test_malformed_or_unprepared_wav_rejected(audio):
    with pytest.raises(ValueError):
        provider.build_request(audio=audio)


def test_wav_checks_real_format_alignment_and_chunk_layout():
    valid = wav()
    floating = bytearray(valid)
    struct.pack_into("<H", floating, 20, 3)
    with pytest.raises(ValueError, match="PCM16"):
        provider.build_request(audio=bytes(floating))
    broken_rate = bytearray(valid)
    struct.pack_into("<I", broken_rate, 28, 1)
    with pytest.raises(ValueError, match="PCM16"):
        provider.build_request(audio=bytes(broken_rate))
    partial_frame = bytearray(valid + b"\0\0")
    struct.pack_into("<I", partial_frame, 4, len(partial_frame) - 8)
    struct.pack_into("<I", partial_frame, 40, 1601)
    with pytest.raises(ValueError, match="PCM16"):
        provider.build_request(audio=bytes(partial_frame))
    duplicate_data = bytearray(valid + b"data" + struct.pack("<I", 2) + b"\0\0")
    struct.pack_into("<I", duplicate_data, 4, len(duplicate_data) - 8)
    with pytest.raises(ValueError, match="exactly one"):
        provider.build_request(audio=bytes(duplicate_data))


@pytest.mark.parametrize("mutate", [
    lambda item: item.update(model="models/gemini-embedding-2-preview"),
    lambda item: item.update(endpoint="https://invalid.example"),
    lambda item: item["embedContentConfig"].update(autoTruncate=True),
    lambda item: item["embedContentConfig"].update(taskType="CLASSIFICATION"),
    lambda item: item.update(outputDimensionality=128),
    lambda item: item.update(content=[item["content"]]),
    lambda item: item["content"].update(role="user"),
    lambda item: item["content"]["parts"].append({"text": "Another text."}),
    lambda item: item["content"]["parts"].append({"fileData": {"fileUri": "file:///etc/passwd"}}),
    lambda item: item["content"]["parts"].append({"inlineData": {"mimeType": "audio/mpeg", "data": "AAAA"}}),
    lambda item: item["content"]["parts"].append({"inlineData": {"mimeType": "audio/wav", "data": "%%%bad"}}),
])
def test_retained_request_cannot_change_model_config_modality_or_envelope(mutate):
    payload = request()
    mutate(payload)
    with pytest.raises(ValueError):
        provider.validate_request(payload)


def test_request_bytes_limit_rechecked_before_http(monkeypatch):
    payload = request()
    monkeypatch.setattr(provider, "MAX_REQUEST_BYTES", 10)
    called = []
    transport = httpx.MockTransport(lambda item: called.append(item))
    with pytest.raises(ValueError, match="byte limit"):
        provider.embed(payload, "fixture-not-a-credential", transport=transport)
    assert not called


def test_embed_uses_fixed_endpoint_once_and_retains_model_and_usage_evidence():
    usage = {
        "promptTokenCount": 50,
        "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 8}, {"modality": "AUDIO", "tokenCount": 42}],
        "futureNumericField": 5,
    }
    result = response(128, usageMetadata=usage, model="models/gemini-embedding-2",
                      modelVersion="gemini-embedding-2-2026-04-22", providerFutureField={"value": 7})
    (vector, metadata), calls = embed_result(result)
    assert len(calls) == 1
    assert str(calls[0].url) == provider.ENDPOINT
    assert calls[0].method == "POST"
    assert calls[0].headers["x-goog-api-key"] == "fixture-not-a-credential"
    assert calls[0].headers["accept-encoding"] == "identity"
    assert json.loads(calls[0].content) == request()
    assert vector == [0.25] * 128
    assert metadata["usage"] == usage
    assert metadata["response"] == result
    assert metadata["returned_model"] == {
        "model": "models/gemini-embedding-2", "modelVersion": "gemini-embedding-2-2026-04-22",
    }
    canonical = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    assert metadata["response_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert metadata["request_sha256"] == hashlib.sha256(calls[0].content).hexdigest()
    assert "fixture-not-a-credential" not in json.dumps(metadata)
    assert "headers" not in metadata


def test_parse_response_is_local_preserves_raw_vector_and_detaches_evidence():
    result = response()
    vector, metadata = provider.parse_response(result, 128)
    assert vector == [0.25] * 128  # no normalization, truncation, or task-specific transform
    result["embedding"]["values"][0] = 9
    assert metadata["response"]["embedding"]["values"][0] == 0.25
    assert metadata["usage"] == {} and metadata["returned_model"] == {}


@pytest.mark.parametrize("result", [
    [], {}, {"embeddings": [response()["embedding"]]},
    {"error": {"message": "private body"}, **response()},
    {"embedding": {"values": [0.2] * 127}},
    {"embedding": {"values": [0.2] * 129}},
    {"embedding": {"values": [True] + [0.2] * 127}},
    {"embedding": {"values": ["0.2"] + [0.2] * 127}},
    {"embedding": {"values": [None] + [0.2] * 127}},
    {"embedding": {"values": [float("inf")] + [0.2] * 127}},
    {"embedding": {"values": [10**500] + [0.2] * 127}},
    response(model="gemini-embedding-001"),
    response(modelVersion=[]),
    response(usageMetadata=[]),
    response(usageMetadata={"promptTokenCount": True}),
    response(usageMetadata={"promptTokenCount": 8193}),
    response(usageMetadata={"promptTokenDetails": [{"modality": "AUDIO", "tokenCount": True}]}),
    response(usageMetadata={"promptTokenDetails": [{"modality": "AUDIO", "tokenCount": -1}]}),
])
def test_invalid_provider_vectors_models_or_usage_are_uncertain(result):
    with pytest.raises(provider.EmbeddingFailure) as failure:
        provider.parse_response(result, 128)
    assert failure.value.ambiguous


@pytest.mark.parametrize("location", ["root", "embedding", "statistics"])
def test_reported_truncation_is_rejected(location):
    result = response()
    target = result if location == "root" else result["embedding"]
    if location == "statistics":
        target = target.setdefault("statistics", {})
    target["truncated"] = True
    with pytest.raises(provider.EmbeddingFailure, match="truncation"):
        provider.parse_response(result, 128)


@pytest.mark.parametrize("status", [302, 400, 401, 429, 500, 503])
def test_http_errors_never_retry_redirect_or_repeat_private_error_body(status):
    calls = []
    def handler(sent):
        calls.append(sent)
        return httpx.Response(status, headers={"Location": "https://invalid.example/credential"},
                              json={"error": {"message": "private-error-body"}})
    with pytest.raises(provider.EmbeddingFailure) as failure:
        provider.embed(request(), "fixture-not-a-credential", transport=httpx.MockTransport(handler))
    assert len(calls) == 1
    assert str(calls[0].url) == provider.ENDPOINT
    assert "private-error-body" not in str(failure.value)
    assert "fixture-not-a-credential" not in str(failure.value)
    assert failure.value.ambiguous is (status >= 500)


def test_transport_failure_has_no_implicit_retry_or_exception_secret():
    calls = []
    def handler(sent):
        calls.append(sent)
        raise httpx.ReadTimeout("fixture-not-a-credential private transport trace", request=sent)
    with pytest.raises(provider.EmbeddingFailure) as failure:
        provider.embed(request(), "fixture-not-a-credential", transport=httpx.MockTransport(handler))
    assert len(calls) == 1 and failure.value.ambiguous
    assert "fixture-not-a-credential" not in str(failure.value)
    assert failure.value.__cause__ is None


class Chunks(httpx.SyncByteStream):
    def __init__(self, pieces):
        self.pieces = pieces
    def __iter__(self):
        yield from self.pieces


def streamed(pieces, **headers):
    return httpx.MockTransport(lambda sent: httpx.Response(
        200, stream=Chunks(pieces), headers={"Content-Type": "application/json", **headers}))


@pytest.mark.parametrize("raw", [
    b"[]", b'{"embedding":{},"embedding":{}}',
    b'{"embedding":{"values":[NaN]}}', b'{"embedding":{"values":[Infinity]}}',
    b'{"embedding":', b"\xff",
])
def test_bounded_json_parser_rejects_malformed_duplicate_and_nonfinite_json(raw):
    with pytest.raises(provider.EmbeddingFailure) as failure:
        provider.embed(request(), "fixture-not-a-credential", transport=streamed([raw]))
    assert failure.value.ambiguous


def test_streamed_response_without_content_length_is_bounded(monkeypatch):
    monkeypatch.setattr(provider, "MAX_RESPONSE_BYTES", 100)
    with pytest.raises(provider.EmbeddingFailure, match="byte limit"):
        provider.embed(request(), "fixture-not-a-credential", transport=streamed([b" " * 60, b" " * 60]))


@pytest.mark.parametrize("headers", [
    {"Content-Length": "999999999"}, {"Content-Length": "9" * 5000}, {"Content-Length": "not-a-length"},
    {"Content-Encoding": "gzip"}, {"Content-Type": "text/html"},
    {"Content-Length": "1"},
])
def test_response_headers_cannot_bypass_bounds_or_encoding(headers):
    raw = json.dumps(response()).encode()
    with pytest.raises(provider.EmbeddingFailure):
        provider.embed(request(), "fixture-not-a-credential", transport=streamed([raw], **headers))


def test_response_time_bound_is_checked_during_stream(monkeypatch):
    clock = iter([0, provider.REQUEST_SECONDS + 1])
    monkeypatch.setattr(provider.time, "monotonic", lambda: next(clock))
    with pytest.raises(provider.EmbeddingFailure, match="time limit"):
        provider.embed(request(), "fixture-not-a-credential", transport=streamed([b"{}"]))


def test_optional_usage_metadata_size_is_bounded(monkeypatch):
    monkeypatch.setattr(provider, "MAX_USAGE_BYTES", 10)
    with pytest.raises(provider.EmbeddingFailure, match="usage metadata"):
        provider.parse_response(response(usageMetadata={"futureField": "x" * 100}), 128)


@pytest.mark.parametrize("key", ["", "private\nheader", "private header", "é", None, "a" * 513])
def test_invalid_credential_shape_is_known_unsent(key):
    called = []
    with pytest.raises(ValueError, match="credential"):
        provider.embed(request(), key, transport=httpx.MockTransport(lambda sent: called.append(sent)))
    assert called == []


def test_finite_float32_overflow_is_uncertain_before_cache_or_settlement():
    result = response()
    result["embedding"]["values"][0] = 1e100
    with pytest.raises(provider.EmbeddingFailure) as failure:
        provider.parse_response(result, 128)
    assert failure.value.ambiguous
    assert result["embedding"]["values"][0] == 1e100
