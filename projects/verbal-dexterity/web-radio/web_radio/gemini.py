"""Gemini speech planning and bounded REST responses; no implicit retries or cloning."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
import wave
from collections.abc import Collection
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, RATES
from . import provider_diagnostics as diagnostics


# Featured prebuilt voices, checked 2026-10-22:
# https://ai.google.dev/gemini-api/docs/speech-generation#prebuilt-voices
PREBUILT_VOICES = frozenset({
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
})
REQUEST_TOKEN_OVERHEAD = 256
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_AUDIO_SECONDS = 1800
MAX_USAGE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
REQUEST_SECONDS = 240
API_ORIGIN = "https://generativelanguage.googleapis.com"
APPROVED_PATHS = frozenset({"/v1beta/interactions", "/v1beta/voices"})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SpeakerRecipe(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    voice: str = Field(default="", max_length=160)
    recipe_version: int = Field(default=1, ge=1)
    style: str = Field(default="", max_length=4000)
    design_prompt: str = Field(default="", max_length=6000)
    pronunciation_notes: str = Field(default="", max_length=2000)
    reference_recording: str | None = Field(default=None, max_length=300)
    character_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    voice_recipe_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    character_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def usable_identity(self):
        if not self.voice.strip() and not self.character_id:
            raise ValueError("A speaker needs a voice or a registered character reference.")
        if not self.character_id and (self.voice_recipe_id is not None or self.character_revision is not None):
            raise ValueError("Pinned voice and character revisions need a character identity.")
        return self


class Turn(StrictModel):
    speaker: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=8000)
    style: str = Field(default="", max_length=2000)


class Script(StrictModel):
    programme_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    series_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    kind: Literal["programme", "ident"]
    title: str = Field(min_length=1, max_length=240)
    language: str = Field(min_length=2, max_length=35, pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")
    speakers: list[SpeakerRecipe] = Field(min_length=1, max_length=16)
    turns: list[Turn] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def matching_speakers(self):
        names = [speaker.name for speaker in self.speakers]
        if len(set(names)) != len(names):
            raise ValueError("Each speaker must have a unique name.")
        if any(turn.speaker not in names for turn in self.turns):
            raise ValueError("Each turn must name one of the configured speakers.")
        if sum(len(turn.text) for turn in self.turns) > 28000:
            raise ValueError("Keep this programme below 28,000 characters; split longer works into episodes.")
        return self

    def transcript(self) -> str:
        return "\n\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns) + "\n"


class GeminiFailure(RuntimeError):
    def __init__(self, message: str, ambiguous: bool = False, *, diagnostic: dict | None = None):
        super().__init__(message)
        self.ambiguous = ambiguous
        self.diagnostic = diagnostic or {
            "schema_version": 1, "provider": "gemini",
            "classification": diagnostics.DISPATCHED_UNCERTAIN if ambiguous else diagnostics.PROVIDER_REJECTED,
            "quota_kind": "unknown", "retry_after_seconds": None,
        }
        self.classification = self.diagnostic["classification"]


def _encoded(value: dict) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("The provider request must be finite UTF-8 JSON.") from exc


def request_token_upper_bound(request: dict) -> int:
    """Conservative UTF-8-byte bound plus framing allowance, not an exact tokenizer."""
    return len(_encoded(request)) + REQUEST_TOKEN_OVERHEAD


def validate_request(request: dict) -> int:
    """Reject oversized complete requests before any reservation or network dispatch."""
    if not isinstance(request, dict):
        raise ValueError("The provider request must be a JSON object.")
    bound = request_token_upper_bound(request)
    if bound > MAX_INPUT_TOKENS:
        raise ValueError("The complete request exceeds the conservative input-token bound; shorten text or directions.")
    return bound


def _voices(script: Script, model: str, approved_designed_voices: Collection[str]):
    if model not in RATES:
        raise ValueError("Unsupported TTS model.")
    approved = frozenset(approved_designed_voices)
    for speaker in script.speakers:
        if speaker.voice in PREBUILT_VOICES:
            continue
        if (not re.fullmatch(r"voice_[A-Za-z0-9_-]{1,150}", speaker.voice)
                or speaker.voice not in approved):
            raise ValueError(
                "Generation supports documented prebuilt Gemini voices only, unless an original design "
                "is approved in the character registry."
            )


def _payload(script: Script, model: str, turns: list[Turn], max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError("Maximum speech output tokens must be an integer from 1 to 16384.")
    recipes = {speaker.name: speaker for speaker in script.speakers}
    content = []
    for turn in turns:
        recipe = recipes[turn.speaker]
        style = "; ".join(part for part in (recipe.style, recipe.pronunciation_notes, turn.style) if part)
        content.append({
            "type": "text", "text": turn.text,
            "annotations": [{"type": "speech_metadata", "speaker": turn.speaker, "style": style}],
        })
    if len(script.speakers) == 1:
        speaker = script.speakers[0]
        speech = [{"voice": speaker.voice, "language": script.language}]
    else:
        speech = {"speakers": [
            {"speaker": speaker.name, "voice": speaker.voice, "language": script.language}
            for speaker in script.speakers
        ]}
    return {
        "model": model, "store": False,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio", "mime_type": "audio/wav"},
        "generation_config": {"max_output_tokens": max_output_tokens, "speech_config": speech},
    }


def build_request(script: Script, model: str, *, approved_designed_voices: Collection[str] = (),
                  max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Build one supported native request; use build_plan for longer or larger casts."""
    _voices(script, model, approved_designed_voices)
    if len(script.speakers) > 2 or (len(script.speakers) > 1
            and any(s.voice not in PREBUILT_VOICES for s in script.speakers)):
        raise ValueError("This cast requires ordered unary requests; use build_plan.")
    request = _payload(script, model, script.turns, max_output_tokens)
    validate_request(request)
    return request


def _plan_item(request: dict, indices: list[int], speaker: str | None, *, turn_index: int | None) -> dict:
    return {
        "turn_index": turn_index, "turn_indices": indices, "speaker": speaker,
        "request": request, "input_token_upper_bound": validate_request(request),
    }


def build_plan(script: Script, model: str, *, approved_designed_voices: Collection[str] = (),
               max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Prefer native dialogue; keep provider-required turn rendering inside the adapter.

    Native dialogue is currently documented for up to two prebuilt voices.
    Designed dialogue and larger casts use unary turns. Oversized prebuilt dialogue
    is greedily divided at existing turn boundaries, never reordered or truncated.
    """
    _voices(script, model, approved_designed_voices)
    requests = []
    if len(script.speakers) > 2 or (len(script.speakers) > 1
            and any(s.voice not in PREBUILT_VOICES for s in script.speakers)):
        strategy = "turns"
        reason = "cast-size" if len(script.speakers) > 2 else "designed-dialogue"
        recipes = {speaker.name: speaker for speaker in script.speakers}
        for index, turn in enumerate(script.turns):
            unary = script.model_copy(update={"speakers": [recipes[turn.speaker]], "turns": [turn]}, deep=True)
            request = _payload(unary, model, unary.turns, max_output_tokens)
            requests.append(_plan_item(request, [index], turn.speaker, turn_index=index))
    else:
        complete = _payload(script, model, script.turns, max_output_tokens)
        if request_token_upper_bound(complete) <= MAX_INPUT_TOKENS:
            strategy, reason = "native", "native-capability"
            requests.append(_plan_item(complete, list(range(len(script.turns))), None, turn_index=None))
        else:
            strategy, reason = "chunks", "request-size"
            pending = []
            indices = []
            for index, turn in enumerate(script.turns):
                candidate = _payload(script, model, [*pending, turn], max_output_tokens)
                if request_token_upper_bound(candidate) > MAX_INPUT_TOKENS:
                    if not pending:
                        raise ValueError("A turn with its directions exceeds the conservative input-token bound.")
                    requests.append(_plan_item(_payload(script, model, pending, max_output_tokens), indices, None, turn_index=indices[0]))
                    pending, indices = [], []
                    candidate = _payload(script, model, [turn], max_output_tokens)
                    validate_request(candidate)
                pending.append(turn)
                indices.append(index)
            if pending:
                requests.append(_plan_item(_payload(script, model, pending, max_output_tokens), indices, None, turn_index=indices[0]))
    return {
        "schema_version": 1, "strategy": strategy, "reason": reason,
        "requests": requests, "request_count": len(requests),
        "input_token_upper_bound": sum(item["input_token_upper_bound"] for item in requests),
    }


def _reject_constant(_value):
    raise ValueError("Non-finite provider JSON value.")


def _no_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate provider JSON key.")
        result[key] = value
    return result


def _validate_dispatch(path: str, request: dict):
    if path not in APPROVED_PATHS:
        raise ValueError("Unsupported Gemini endpoint.")
    if not isinstance(request, dict):
        raise ValueError("Provider request must be a JSON object.")
    if path == "/v1beta/interactions":
        validate_request(request)
        if set(request) != {"model", "store", "input", "response_format", "generation_config"}:
            raise ValueError("Unsupported speech request fields.")
        if request.get("model") not in RATES or request.get("store") is not False:
            raise ValueError("Unsupported speech request configuration.")
        if request.get("response_format") != {"type": "audio", "mime_type": "audio/wav"}:
            raise ValueError("Speech requests must explicitly request WAV audio.")
        generation = request.get("generation_config")
        if (not isinstance(generation, dict)
                or set(generation) != {"max_output_tokens", "speech_config"}
                or type(generation.get("max_output_tokens")) is not int
                or not 0 < generation["max_output_tokens"] <= MAX_OUTPUT_TOKENS):
            raise ValueError("Unsupported speech generation limits.")
        speech = generation.get("speech_config")
        if isinstance(speech, list):
            if len(speech) != 1 or not isinstance(speech[0], dict) or set(speech[0]) != {"voice", "language"}:
                raise ValueError("Unary speech requires one configured voice.")
            voices = speech
            cast_names = None
        elif isinstance(speech, dict) and set(speech) == {"speakers"}:
            voices = speech["speakers"]
            if (not isinstance(voices, list) or len(voices) != 2
                    or any(not isinstance(voice, dict) or set(voice) != {"speaker", "voice", "language"}
                           for voice in voices)):
                raise ValueError("Native dialogue requires two configured prebuilt voices.")
            names = [voice["speaker"] for voice in voices]
            if (any(not isinstance(name, str) or not 1 <= len(name) <= 80 for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("Native dialogue needs distinct configured speaker names.")
            cast_names = set(names)
        else:
            raise ValueError("Unsupported speech configuration.")
        for configured in voices:
            voice = configured.get("voice")
            if (not isinstance(voice, str)
                    or voice not in PREBUILT_VOICES and (
                        cast_names is not None or not re.fullmatch(r"voice_[A-Za-z0-9_-]{1,150}", voice)
                    )):
                raise ValueError("Only prebuilt or registered original-design speech voices are supported.")
            language = configured.get("language")
            if not isinstance(language, str) or not re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*", language):
                raise ValueError("Speech language must be a supported language-code shape.")
        inputs = request.get("input")
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError("Speech generation accepts one structured text input.")
        if set(inputs[0]) != {"type", "content"} or inputs[0].get("type") != "user_input":
            raise ValueError("Speech generation accepts text input only.")
        content = inputs[0].get("content")
        if not isinstance(content, list) or not 1 <= len(content) <= 200:
            raise ValueError("Speech generation needs at least one text turn.")
        for part in content:
            if (not isinstance(part, dict) or set(part) != {"type", "text", "annotations"}
                    or part.get("type") != "text" or not isinstance(part.get("text"), str)
                    or not part["text"]):
                raise ValueError("Speech generation accepts structured text turns only.")
            annotations = part.get("annotations")
            if (not isinstance(annotations, list) or len(annotations) != 1
                    or not isinstance(annotations[0], dict)):
                raise ValueError("Speech generation needs structured speech metadata.")
            annotation = annotations[0]
            if (set(annotation) != {"type", "speaker", "style"}
                    or annotation.get("type") != "speech_metadata"
                    or not isinstance(annotation.get("speaker"), str)
                    or not isinstance(annotation.get("style"), str)):
                raise ValueError("Unsupported speech metadata.")
            name = annotation["speaker"]
            if not 1 <= len(name) <= 80:
                raise ValueError("Speech metadata needs a configured speaker name.")
            if cast_names is not None and name not in cast_names:
                raise ValueError("Speech metadata names an unconfigured speaker.")
        if cast_names is None and len({part["annotations"][0]["speaker"] for part in content}) != 1:
            raise ValueError("Unary speech accepts turns from one speaker only.")
    else:
        if set(request) != {"store", "voice"} or request.get("store") is not True:
            raise ValueError("Original voice design must use persistent prompted voices.")
        voice = request.get("voice")
        if (not isinstance(voice, dict) or voice.get("type") != "prompted"
                or set(voice) - {"model", "type", "display_name", "gender", "language_code", "prompted"}
                or voice.get("model") not in RATES):
            raise ValueError("Only original voices designed from text descriptions are supported.")
        prompted = voice.get("prompted")
        if (not isinstance(prompted, dict) or set(prompted) != {"input"}
                or not isinstance(prompted.get("input"), str) or not prompted["input"].strip()):
            raise ValueError("Voice design accepts a text description only.")
        if request_token_upper_bound(request) > MAX_INPUT_TOKENS:
            raise ValueError("Voice design description exceeds its conservative input-token bound.")


def validate_synthesis_request(request: dict) -> int:
    """Validate a retained synthesis request before marking its dispatch in flight."""
    _validate_dispatch("/v1beta/interactions", request)
    return request_token_upper_bound(request)


def validate_design_request(request: dict) -> int:
    """Validate an original design before recording its dispatch in flight."""
    _validate_dispatch("/v1beta/voices", request)
    return request_token_upper_bound(request)


def post_json(path: str, request: dict, key: str, *, transport=None) -> dict:
    """POST only approved Gemini operations with bounded, credential-free error messages."""
    _validate_dispatch(path, request)
    raw_request = _encoded(request)
    if len(raw_request) > MAX_REQUEST_BYTES:
        raise ValueError("Provider request exceeds its byte limit.")
    started = time.monotonic()
    try:
        with httpx.Client(
            timeout=httpx.Timeout(REQUEST_SECONDS, connect=15), transport=transport,
            follow_redirects=False, trust_env=False,
        ) as client:
            with client.stream(
                "POST", API_ORIGIN + path,
                headers={"x-goog-api-key": key, "Content-Type": "application/json", "Accept-Encoding": "identity"},
                content=raw_request,
            ) as response:
                if response.status_code != 200:
                    error_body = bytearray()
                    truncated = response.headers.get("content-encoding", "identity").lower() != "identity"
                    if not truncated:
                        try:
                            for chunk in ([response.content] if response.is_stream_consumed else response.iter_raw()):
                                remaining = diagnostics.MAX_ERROR_BYTES - len(error_body)
                                if time.monotonic() - started > REQUEST_SECONDS:
                                    truncated = True
                                    break
                                error_body.extend(chunk[:remaining])
                                if len(chunk) > remaining:
                                    truncated = True
                                    break
                        except httpx.HTTPError:
                            truncated = True
                    model = request.get("model") or (request.get("voice") or {}).get("model")
                    diagnostic = diagnostics.parse_error(
                        bytes(error_body), response.status_code, key=key,
                        retry_after=response.headers.get("retry-after"), operation=path,
                        model=model, body_truncated=truncated)
                    raise GeminiFailure(
                        f"Gemini rejected the request (HTTP {response.status_code}); allocation remains reserved.",
                        response.status_code >= 500, diagnostic=diagnostic,
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise GeminiFailure("Provider returned an unsupported response encoding; allocation remains reserved.", True)
                declared = response.headers.get("content-length")
                if declared is not None and (not declared.isdecimal() or len(declared) > 20 or int(declared) > MAX_RESPONSE_BYTES):
                    raise GeminiFailure("Provider response exceeds its byte limit; allocation remains reserved.", True)
                body = bytearray()
                for chunk in ([response.content] if response.is_stream_consumed else response.iter_raw()):
                    if time.monotonic() - started > REQUEST_SECONDS:
                        raise GeminiFailure("Provider response exceeded its time limit; allocation remains reserved.", True)
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise GeminiFailure("Provider response exceeds its byte limit; allocation remains reserved.", True)
                    body.extend(chunk)
    except httpx.HTTPError:
        raise GeminiFailure("Provider connection ended without a confirmed outcome; allocation remains reserved.", True) from None
    try:
        result = json.loads(body, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise GeminiFailure("Provider returned an unreadable result; allocation remains reserved.", True) from exc
    if not isinstance(result, dict):
        raise GeminiFailure("Provider returned an invalid result shape; allocation remains reserved.", True)
    return result


def parse_audio_block(block: dict) -> bytes:
    """Validate one base64 PCM WAV block; never infer a format or manufacture headers."""
    if not isinstance(block, dict) or block.get("mime_type") != "audio/wav":
        raise GeminiFailure("Expected the requested audio/wav response MIME type; allocation remains reserved.", True)
    data = block.get("data")
    if not isinstance(data, str) or len(data) > ((MAX_AUDIO_BYTES + 2) // 3) * 4:
        raise GeminiFailure("Provider returned invalid or oversized audio encoding; allocation remains reserved.", True)
    try:
        audio = base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise GeminiFailure("Provider returned invalid audio encoding; allocation remains reserved.", True) from exc
    if (len(audio) > MAX_AUDIO_BYTES or len(audio) < 44 or audio[:4] != b"RIFF"
            or audio[8:12] != b"WAVE" or int.from_bytes(audio[4:8], "little") + 8 != len(audio)):
        raise GeminiFailure("Expected the requested complete WAV response; no format conversion was guessed.", True)
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            channels, width, rate, frames = source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()
            if (source.getcomptype() != "NONE" or channels not in (1, 2) or width != 2
                    or not 8000 <= rate <= 96000 or frames <= 0
                    or frames > rate * MAX_AUDIO_SECONDS
                    or frames * channels * width > MAX_AUDIO_BYTES):
                raise ValueError("Unsupported WAV parameters.")
            expected = frames * channels * width
            if len(source.readframes(frames)) != expected or source.readframes(1):
                raise ValueError("Incomplete WAV samples.")
    except (ValueError, wave.Error, EOFError, OSError) as exc:
        raise GeminiFailure("Expected the requested complete PCM WAV response; allocation remains reserved.", True) from exc
    return audio


decode_wav_audio = parse_audio_block


def _completion_observations(result: dict, steps: list[dict], key: str) -> list[dict]:
    """Retain explicit optional fields without claiming an Interactions guarantee.

    The documented Interactions truncation signal is status='incomplete', which
    generate rejects independently. Some provider responses may additionally
    expose finish/stop fields familiar from other APIs. Preserve these bounded
    observations and their source; never synthesize a successful stop reason.
    """
    observations = []
    containers = [("interaction", result), *(
        (f"steps[{index}]", step) for index, step in enumerate(steps)
        if step.get("type") == "model_output")]
    for origin, container in containers:
        for field in ("finish_reason", "stop_reason", "finishReason", "stopReason"):
            value = container.get(field)
            if value is None:
                continue
            if (type(value) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", value) is None
                    or diagnostics.sanitize(value, secrets=(key,), limit=80) != value
                    or len(observations) >= 16):
                raise GeminiFailure("Provider returned invalid completion metadata; allocation remains reserved.", True)
            observations.append({"source": origin, "field": field, "value": value})
    return observations


def generate(request: dict, key: str, transport=None) -> tuple[bytes, dict]:
    result = post_json("/v1beta/interactions", request, key, transport=transport)
    if result.get("status") != "completed":
        raise GeminiFailure("Provider did not confirm a completed performance; allocation remains reserved.", True)
    steps = result.get("steps")
    if not isinstance(steps, list) or len(steps) > 100:
        raise GeminiFailure("Provider returned invalid output steps; allocation remains reserved.", True)
    blocks = []
    for step in steps:
        if not isinstance(step, dict):
            raise GeminiFailure("Provider returned invalid output steps; allocation remains reserved.", True)
        if step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list) or len(content) > 100:
            raise GeminiFailure("Provider returned invalid output content; allocation remains reserved.", True)
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "audio":
                raise GeminiFailure("Provider returned unexpected output content; allocation remains reserved.", True)
            blocks.append(block)
    if len(blocks) != 1:
        raise GeminiFailure("Provider must return exactly one completed audio block; allocation remains reserved.", True)
    audio = parse_audio_block(blocks[0])
    usage = result.get("usage", {})
    if usage is None:
        usage = {}
    try:
        usage_valid = isinstance(usage, dict) and len(_encoded(usage)) <= MAX_USAGE_BYTES
    except ValueError:
        usage_valid = False
    if not usage_valid:
        raise GeminiFailure("Provider returned invalid usage metadata; allocation remains reserved.", True)
    reported_output = usage.get("total_output_tokens")
    if type(reported_output) is int and reported_output > request["generation_config"]["max_output_tokens"]:
        raise GeminiFailure("Provider reported output beyond the requested token ceiling; allocation remains reserved.", True)
    for field, maximum in (("id", 256), ("model", 160)):
        value = result.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > maximum):
            raise GeminiFailure("Provider returned invalid response metadata; allocation remains reserved.", True)
    metadata = {
        "interaction_id": result.get("id"), "status": result.get("status"),
        "model": result.get("model"), "usage": usage,
        "audio_mime_type": blocks[0]["mime_type"],
        "audio_bytes": len(audio), "audio_sha256": hashlib.sha256(audio).hexdigest(),
    }
    observations = _completion_observations(result, steps, key)
    if observations:
        metadata["completion_observations"] = observations
        metadata["completion_observations_notice"] = (
            "Explicit provider fields retained defensively; not guaranteed by the Interactions contract.")
        token_stops = {"MAX_TOKENS", "max_output_tokens", "MAX_OUTPUT_TOKENS"}
        chosen = next((item for item in observations if item["value"] in token_stops), observations[0])
        metadata["finish_reason"] = chosen["value"]
    return audio, metadata
