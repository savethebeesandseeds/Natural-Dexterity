"""Bounded Gemini Embedding 2 REST adapter for one shared text/audio space.

Checked 2026-10-22:
https://ai.google.dev/gemini-api/docs/embeddings
https://ai.google.dev/api/embeddings
https://ai.google.dev/gemini-api/docs/pricing#gemini-embedding-2

The provider permits 8192 input tokens and 180-second WAV/MP3 audio. This
application deliberately accepts only complete PCM16 WAV windows up to 30s
and 4096 UTF-8 text bytes. These are admission limits, NOT an audio tokenizer.
No text task prefix, task type, title, modality-specific model, or vector
renormalization is added. All parts share one Content and return one vector.

Pricing is by modality input tokens. The published audio per-second price is
rounded; it does not document a precise embedding audio tokenization rule.
For accepted requests the provider's 8192-token ceiling can bound billing at
the highest enabled modality rate; it cannot predict a supplied WAV's tokens.
The caller owns current prices, reservations, and uncertain-outcome accounting.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import struct
import time
import wave

import httpx

MODEL = "gemini-embedding-2"
MODEL_RESOURCE = f"models/{MODEL}"
API_ORIGIN = "https://generativelanguage.googleapis.com"
ENDPOINT = f"{API_ORIGIN}/v1beta/models/{MODEL}:embedContent"
MIN_DIMENSIONS = 128
MAX_DIMENSIONS = DEFAULT_DIMENSIONS = 3072
MAX_INPUT_TOKENS = 8192
PROVIDER_MAX_AUDIO_SECONDS = 180
MAX_AUDIO_SECONDS = 30
MAX_TEXT_BYTES = 4096
TEXT_TOKEN_OVERHEAD = 256
MAX_AUDIO_BYTES = 6 * 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024 + 64 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_USAGE_BYTES = 32 * 1024
REQUEST_SECONDS = 60
DOCUMENTATION_CHECKED = "2026-10-22"


class EmbeddingFailure(RuntimeError):
    """Provider outcome; ambiguous outcomes must retain their allocation."""

    def __init__(self, message: str, ambiguous: bool = False):
        super().__init__(message)
        self.ambiguous = ambiguous


def _encoded(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("Embedding data must be finite UTF-8 JSON.") from None


def _wav_info(audio: bytes) -> dict:
    if type(audio) is not bytes or not 44 <= len(audio) <= MAX_AUDIO_BYTES:
        raise ValueError("Embedding audio must be a bounded PCM16 WAV window.")
    if (audio[:4] != b"RIFF" or audio[8:12] != b"WAVE"
            or struct.unpack_from("<I", audio, 4)[0] + 8 != len(audio)):
        raise ValueError("Embedding audio must be a complete RIFF WAV file.")

    # Inspect chunk boundaries too: wave.readframes alone rounds away an
    # incomplete PCM frame and can otherwise overlook inconsistent WAV tails.
    offset = 12
    formats = []
    data_sizes = []
    chunk_count = 0
    while offset < len(audio):
        chunk_count += 1
        if chunk_count > 64 or offset + 8 > len(audio):
            raise ValueError("Embedding WAV has invalid chunk boundaries.")
        kind, size = struct.unpack_from("<4sI", audio, offset)
        end = offset + 8 + size
        padded = end + size % 2
        if padded > len(audio):
            raise ValueError("Embedding WAV contains an incomplete chunk.")
        if kind == b"fmt ":
            formats.append(audio[offset + 8:end])
        elif kind == b"data":
            data_sizes.append(size)
        offset = padded
    if (len(formats) != 1 or len(data_sizes) != 1 or len(formats[0]) < 16
            or not data_sizes[0]):
        raise ValueError("Embedding WAV needs exactly one PCM format and audio chunk.")
    code, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", formats[0])
    if (code != 1 or channels not in (1, 2) or bits != 16
            or not 8000 <= rate <= 48000 or alignment != channels * 2
            or byte_rate != rate * alignment or data_sizes[0] % alignment):
        raise ValueError("Embedding WAV must contain PCM16 audio at 8–48 kHz in one or two channels.")
    frames = data_sizes[0] // alignment
    if frames > MAX_AUDIO_SECONDS * rate:
        raise ValueError("Embedding audio windows must be no longer than 30 seconds.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as recording:
            if (recording.getcomptype() != "NONE" or recording.getnframes() != frames
                    or recording.getnchannels() != channels or recording.getframerate() != rate
                    or recording.getsampwidth() != 2):
                raise ValueError("Embedding WAV has inconsistent PCM format information.")
            decoded = sum(len(chunk) for chunk in iter(lambda: recording.readframes(65536), b""))
            if decoded != data_sizes[0]:
                raise ValueError("Embedding WAV audio is truncated.")
    except (wave.Error, EOFError, struct.error):
        raise ValueError("Embedding audio must be a valid complete PCM16 WAV file.") from None
    return {
        "bytes": len(audio), "sha256": hashlib.sha256(audio).hexdigest(),
        "seconds": frames / rate, "frames": frames, "sample_rate": rate,
        "channels": channels, "sample_width": 2,
    }


def build_request(text: str | None = None, audio: bytes | None = None,
                  dimensions: int = DEFAULT_DIMENSIONS) -> dict:
    """Return one canonical embedContent body; do not reserve or send anything."""
    parts = []
    if text is not None:
        parts.append({"text": text})
    if audio is not None:
        _wav_info(audio)
        parts.append({"inlineData": {
            "mimeType": "audio/wav", "data": base64.b64encode(audio).decode("ascii"),
        }})
    request = {
        "model": MODEL_RESOURCE, "content": {"parts": parts},
        "embedContentConfig": {"outputDimensionality": dimensions, "autoTruncate": False},
    }
    validate_request(request)
    return request


def validate_request(request: dict) -> dict:
    """Preflight exact retained inputs, without estimating audio tokenization.

    input_token_upper_bound is deliberately null for audio. billing_token_ceiling
    describes the documented provider serving limit, not a measured input count.
    autoTruncate=False requires the provider to reject inputs exceeding its limit.
    """
    if type(request) is not dict or set(request) != {"model", "content", "embedContentConfig"}:
        raise ValueError("Unsupported embedding request envelope.")
    if request["model"] != MODEL_RESOURCE:
        raise ValueError("Only the stable Gemini Embedding 2 model is supported.")
    config = request["embedContentConfig"]
    if (type(config) is not dict or set(config) != {"outputDimensionality", "autoTruncate"}
            or type(config.get("outputDimensionality")) is not int
            or not MIN_DIMENSIONS <= config["outputDimensionality"] <= MAX_DIMENSIONS
            or config.get("autoTruncate") is not False):
        raise ValueError("Embedding dimensions must be 128–3072 and input truncation must be disabled.")
    content = request["content"]
    if type(content) is not dict or set(content) != {"parts"}:
        raise ValueError("Embedding requests need one Content object.")
    parts = content["parts"]
    if type(parts) is not list or not 1 <= len(parts) <= 2:
        raise ValueError("Embed one text part, one audio part, or both in a single Content.")
    modalities = []
    text_bytes = 0
    audio_info = None
    for part in parts:
        if type(part) is not dict:
            raise ValueError("Unsupported embedding content part.")
        if set(part) == {"text"}:
            text = part["text"]
            if type(text) is not str or not text.strip():
                raise ValueError("Embedding text must contain non-whitespace text.")
            try:
                text_bytes = len(text.encode("utf-8"))
            except UnicodeError:
                raise ValueError("Embedding text must be valid UTF-8.") from None
            if text_bytes > MAX_TEXT_BYTES:
                raise ValueError("Embedding text windows must fit within 4096 UTF-8 bytes.")
            modalities.append("text")
        elif set(part) == {"inlineData"}:
            inline = part["inlineData"]
            if (type(inline) is not dict or set(inline) != {"mimeType", "data"}
                    or inline.get("mimeType") != "audio/wav"
                    or type(inline.get("data")) is not str
                    or len(inline["data"]) > 4 * ((MAX_AUDIO_BYTES + 2) // 3)):
                raise ValueError("Embedding audio must be bounded inline WAV data.")
            try:
                audio = base64.b64decode(inline["data"], validate=True)
            except (ValueError, binascii.Error):
                raise ValueError("Embedding audio needs valid base64 data.") from None
            audio_info = _wav_info(audio)
            modalities.append("audio")
        else:
            raise ValueError("Only text and inline WAV embedding parts are supported.")
    if modalities not in (["text"], ["audio"], ["text", "audio"]):
        raise ValueError("Embedding Content needs distinct parts in text-then-audio order.")
    raw = _encoded(request)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Embedding request exceeds its byte limit.")
    text_bound = text_bytes + TEXT_TOKEN_OVERHEAD if text_bytes else 0
    return {
        "model": MODEL, "dimensions": config["outputDimensionality"],
        "modalities": modalities, "request_bytes": len(raw),
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "text_bytes": text_bytes, "text_token_upper_bound": text_bound,
        "audio": audio_info, "audio_seconds": audio_info["seconds"] if audio_info else 0,
        "input_token_upper_bound": None if audio_info else text_bound,
        "provider_input_token_limit": MAX_INPUT_TOKENS,
        "billing_token_ceiling": MAX_INPUT_TOKENS,
        "billing_basis": "provider-input-token-ceiling",
        "token_bound_basis": "audio-tokenization-not-inferred" if audio_info else "UTF-8-bytes-plus-framing",
    }


def _reject_constant(_value):
    raise ValueError("Non-finite embedding response JSON.")


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate embedding response field.")
        result[key] = value
    return result


def _post(request: dict, key: str, *, transport=None) -> tuple[dict, str]:
    if (type(key) is not str or not 1 <= len(key) <= 512 or not key.isascii()
            or any(character.isspace() or ord(character) < 33 or ord(character) > 126 for character in key)):
        raise ValueError("A valid private Gemini credential is required.")
    raw_request = _encoded(request)
    started = time.monotonic()
    declared = None
    try:
        with httpx.Client(timeout=httpx.Timeout(REQUEST_SECONDS, connect=15),
                          transport=transport, follow_redirects=False, trust_env=False) as client:
            with client.stream(
                "POST", ENDPOINT, content=raw_request,
                headers={"x-goog-api-key": key, "Content-Type": "application/json",
                         "Accept": "application/json", "Accept-Encoding": "identity"},
            ) as response:
                if response.status_code != 200:
                    raise EmbeddingFailure(
                        f"Gemini rejected the embedding request (HTTP {response.status_code}); allocation remains reserved.",
                        response.status_code >= 500,
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise EmbeddingFailure("Embedding response encoding is unsupported; allocation remains reserved.", True)
                if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                    raise EmbeddingFailure("Embedding response is not JSON; allocation remains reserved.", True)
                header = response.headers.get("content-length")
                if header is not None:
                    if not header.isdecimal() or len(header) > 20 or int(header) > MAX_RESPONSE_BYTES:
                        raise EmbeddingFailure("Embedding response exceeds its byte limit; allocation remains reserved.", True)
                    declared = int(header)
                body = bytearray()
                for chunk in ([response.content] if response.is_stream_consumed else response.iter_raw()):
                    if time.monotonic() - started > REQUEST_SECONDS:
                        raise EmbeddingFailure("Embedding response exceeded its time limit; allocation remains reserved.", True)
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise EmbeddingFailure("Embedding response exceeds its byte limit; allocation remains reserved.", True)
                    body.extend(chunk)
                if declared is not None and declared != len(body):
                    raise EmbeddingFailure("Embedding response was incomplete; allocation remains reserved.", True)
    except httpx.HTTPError:
        # Transport exception messages may include requests, credentials, or bodies.
        raise EmbeddingFailure("Embedding connection ended without a confirmed outcome; allocation remains reserved.", True) from None
    try:
        result = json.loads(body.decode("utf-8"), object_pairs_hook=_unique,
                            parse_constant=_reject_constant)
        if type(result) is not dict:
            raise ValueError
        _encoded(result)  # Includes overflowed numeric literals in optional metadata.
    except (ValueError, UnicodeError, RecursionError):
        raise EmbeddingFailure("Gemini returned unreadable embedding JSON; allocation remains reserved.", True) from None
    return result, hashlib.sha256(body).hexdigest()


def parse_response(response: dict, dimensions: int) -> tuple[list[float], dict]:
    """Validate retained provider JSON locally; never dispatch or alter a vector.

    response_sha256 uses compact UTF-8 JSON, ensure_ascii=False and allow_nan=False.
    The complete parsed response is retained for authoritative recovery evidence.
    """
    if type(dimensions) is not int or not MIN_DIMENSIONS <= dimensions <= MAX_DIMENSIONS:
        raise ValueError("Embedding dimensions must be 128–3072.")
    failed = "Gemini returned an invalid embedding result; allocation remains reserved."
    try:
        if type(response) is not dict:
            raise ValueError
        canonical = _encoded(response)
        if len(canonical) > MAX_RESPONSE_BYTES:
            raise ValueError
        # Detach metadata from a caller's mutable response object.
        result = json.loads(canonical)
    except (ValueError, RecursionError):
        raise EmbeddingFailure(failed, True) from None
    if "error" in result or "embeddings" in result:
        raise EmbeddingFailure(failed, True)
    embedding = result.get("embedding")
    if type(embedding) is not dict or type(embedding.get("values")) is not list:
        raise EmbeddingFailure(failed, True)
    if len(embedding["values"]) != dimensions:
        raise EmbeddingFailure("Gemini returned the wrong embedding dimensions; allocation remains reserved.", True)
    vector = []
    for value in embedding["values"]:
        if type(value) not in (int, float):
            raise EmbeddingFailure(failed, True)
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise EmbeddingFailure(failed, True) from None
        if not math.isfinite(number):
            raise EmbeddingFailure(failed, True)
        # Sidecars store float32. Reject provider values that cannot be safely
        # serialized before the caller commits a cache entry or settles billing.
        # This is a validation probe only; preserve the original vector value.
        try:
            if not math.isfinite(struct.unpack("<f", struct.pack("<f", number))[0]):
                raise OverflowError
        except (OverflowError, struct.error):
            raise EmbeddingFailure(failed, True) from None
        vector.append(number)
    # No silent input/vector truncation is accepted, even if a future response
    # adds a statistics flag used by other Google embedding APIs.
    for container in (result, embedding, embedding.get("statistics", {})):
        if type(container) is dict and "truncated" in container and container["truncated"] is not False:
            raise EmbeddingFailure("Gemini reported embedding truncation; allocation remains reserved.", True)
    usage = result.get("usageMetadata", {})
    if type(usage) is not dict or len(_encoded(usage)) > MAX_USAGE_BYTES:
        raise EmbeddingFailure("Gemini returned invalid embedding usage metadata; allocation remains reserved.", True)
    if "promptTokenCount" in usage and (
        type(usage["promptTokenCount"]) is not int or not 0 <= usage["promptTokenCount"] <= MAX_INPUT_TOKENS
    ):
        raise EmbeddingFailure("Gemini returned invalid embedding token usage; allocation remains reserved.", True)
    if "promptTokenDetails" in usage:
        details = usage["promptTokenDetails"]
        if (type(details) is not list or len(details) > 16
                or any(type(item) is not dict or type(item.get("modality")) is not str
                       or type(item.get("tokenCount")) is not int
                       or not 0 <= item["tokenCount"] <= MAX_INPUT_TOKENS for item in details)):
            raise EmbeddingFailure("Gemini returned invalid embedding modality usage; allocation remains reserved.", True)
    returned_model = {}
    for name in ("model", "modelVersion", "model_version"):
        if name in result:
            value = result[name]
            if type(value) is not str or not 1 <= len(value) <= 160:
                raise EmbeddingFailure("Gemini returned invalid embedding model metadata; allocation remains reserved.", True)
            if name == "model" and value not in (MODEL, MODEL_RESOURCE):
                raise EmbeddingFailure("Gemini returned a different embedding model; allocation remains reserved.", True)
            returned_model[name] = value
    return vector, {
        "provider": "google-gemini", "model": MODEL, "dimensions": dimensions,
        "usage": usage, "returned_model": returned_model,
        "response_sha256": hashlib.sha256(canonical).hexdigest(), "response": result,
    }


def embed(request: dict, key: str, *, transport=None) -> tuple[list[float], dict]:
    """Send exactly one validated request, with no redirect, retry, or paid fallback."""
    preflight = validate_request(request)
    result, digest = _post(request, key, transport=transport)
    vector, metadata = parse_response(result, preflight["dimensions"])
    metadata.update({"request_sha256": preflight["request_sha256"],
                     "modalities": preflight["modalities"], "wire_response_sha256": digest})
    return vector, metadata
