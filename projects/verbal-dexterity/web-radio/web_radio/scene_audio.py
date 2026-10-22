"""Private scene assembly from separately retained, explicitly selected turn audio.

The result is an editorial PCM assembly, never an uninterrupted provider take.
No generation, publication, release acceptance, scheduling, or network occurs here.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import wave

import rfc8785

from .contract import HASH_PATTERN, MAX_AUDIO_BYTES, MAX_DURATION_MS, SLUG_PATTERN
from .gemini import Script

SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 256 * 1024 * 1024
MAX_EXPORT_BYTES = 512 * 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_PROCESS_SECONDS = 180
PIPELINE = "ordered-pcm48k-mono-static-gain-v1"
TARGET_LUFS = -16.0
TARGET_TRUE_PEAK_DB = -1.5
MAX_GAIN_DB = 20.0
_REQUIRED = {"index", "speaker", "character_id", "source", "source_sha256",
             "text", "style", "actual_model"}
_OPTIONAL = {"recipe_id", "character_revision", "voice_recipe_snapshot", "source_request_sha256",
             "source_response_sha256", "source_attempt", "source_job", "source_take",
             "source_request_index"}
_SENSITIVE = {"api_key", "authorization", "x-goog-api-key", "credential", "credentials", "access_token"}


class SceneAudioError(ValueError):
    """Scene assembly failed; all original recordings remain untouched."""


def _canonical(value) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise SceneAudioError("Scene metadata must be finite ordinary JSON.") from exc


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ordinary(path: Path):
    for component in (path, *path.parents):
        if component.is_symlink():
            raise SceneAudioError("Scene paths must not contain symbolic links.")


def _read(path: Path, limit: int) -> bytes:
    _ordinary(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise SceneAudioError("Scene inputs must be bounded regular files.")
            data = source.read(limit + 1)
    except OSError as exc:
        raise SceneAudioError("A scene file is unavailable.") from exc
    if len(data) > limit:
        raise SceneAudioError("Scene file exceeds its byte limit.")
    return data


def _wav(data: bytes, *, canonical=False) -> tuple[dict, bytes]:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SceneAudioError("Scene source must be a complete PCM WAV.")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise SceneAudioError("Scene WAV has a truncated or trailing RIFF payload.")
    offset = 12
    chunks = {}
    while offset < len(data):
        if offset + 8 > len(data):
            raise SceneAudioError("Scene WAV contains an incomplete chunk header.")
        label = data[offset:offset + 4]
        size = int.from_bytes(data[offset + 4:offset + 8], "little")
        end = offset + 8 + size
        if end + (size & 1) > len(data):
            raise SceneAudioError("Scene WAV contains a truncated chunk.")
        if label in (b"fmt ", b"data"):
            if label in chunks:
                raise SceneAudioError("Scene WAV contains duplicate format or audio chunks.")
            chunks[label] = data[offset + 8:end]
        offset = end + (size & 1)
    if set(chunks) != {b"fmt ", b"data"}:
        raise SceneAudioError("Scene WAV needs one format and one audio chunk.")
    fmt = chunks[b"fmt "]
    if len(fmt) < 16 or int.from_bytes(fmt[:2], "little") != 1:
        raise SceneAudioError("Scene source must use signed 16-bit integer PCM.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels, width, rate, frames = (audio.getnchannels(), audio.getsampwidth(),
                                             audio.getframerate(), audio.getnframes())
            if (channels not in (1, 2) or width != 2 or not 8000 <= rate <= 192000
                    or audio.getcomptype() != "NONE" or frames <= 0):
                raise SceneAudioError("Scene source has an unsupported PCM format.")
            pcm = audio.readframes(frames + 1)
    except (wave.Error, EOFError) as exc:
        raise SceneAudioError("Scene WAV could not be completely decoded.") from exc
    if len(pcm) != frames * channels * width or len(chunks[b"data"]) != len(pcm):
        raise SceneAudioError("Scene PCM frame count does not match its exact audio bytes.")
    if frames * 1000 > MAX_DURATION_MS * rate:
        raise SceneAudioError("Scene source exceeds the duration limit.")
    if canonical and (rate, channels, width) != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
        raise SceneAudioError("Scene normalized audio format is inconsistent.")
    return {"sample_rate": rate, "channels": channels, "sample_width": width, "frames": frames,
            "pcm_sha256": _digest(pcm)}, pcm


def _safe_metadata(value, depth=0):
    if depth > 16:
        raise SceneAudioError("Scene provenance nesting exceeds its limit.")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower() in _SENSITIVE:
                raise SceneAudioError("Scene provenance contains unsupported credential fields.")
            _safe_metadata(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _safe_metadata(child, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise SceneAudioError("Scene provenance must contain ordinary JSON values.")
    if isinstance(value, str) and "\0" in value:
        raise SceneAudioError("Scene provenance contains an invalid text value.")
    if type(value) is float and not math.isfinite(value):
        raise SceneAudioError("Scene provenance must be finite.")


def _text(value, *, blank=False, limit=8000):
    if (not isinstance(value, str) or len(value) > limit or "\0" in value
            or (not blank and not value.strip())):
        raise SceneAudioError("Scene line metadata contains invalid text.")
    return value


def _recipe_override(line: dict, baseline, index: int) -> bool:
    snapshot = line.get("voice_recipe_snapshot")
    override = snapshot.get("_scene_override") if isinstance(snapshot, dict) else None
    if not isinstance(override, dict) or set(override) != {"baseline_recipe_id", "parent_job", "line_index"}:
        return False
    return (override["baseline_recipe_id"] == baseline.voice_recipe_id
            and type(override["line_index"]) is int and override["line_index"] == index
            and isinstance(override["parent_job"], str) and bool(override["parent_job"].strip())
            and len(override["parent_job"]) <= 200
            and snapshot.get("character_id") == line["character_id"]
            and snapshot.get("recipe_id", snapshot.get("id")) == line.get("recipe_id"))


def _inputs(script: Script | dict, lines: list[dict], stems: bool):
    try:
        script = Script.model_validate(script)
    except ValueError as exc:
        raise SceneAudioError("Scene script is invalid.") from exc
    if type(stems) is not bool or not isinstance(lines, list) or len(lines) != len(script.turns):
        raise SceneAudioError("Scene needs exactly one selected line per original script turn.")
    speakers = {speaker.name: speaker for speaker in script.speakers}
    frozen, sources = [], []
    source_bytes, duration_seconds = 0, 0.0
    character_speakers = {}
    for index, (turn, line) in enumerate(zip(script.turns, lines, strict=True)):
        if not isinstance(line, dict) or not _REQUIRED <= set(line) or set(line) - _REQUIRED - _OPTIONAL:
            raise SceneAudioError("Scene line fields are incomplete or unsupported.")
        if type(line["index"]) is not int or line["index"] != index:
            raise SceneAudioError("Scene selections must match original turn order exactly.")
        if (line["speaker"], line["text"], line["style"]) != (turn.speaker, turn.text, turn.style):
            raise SceneAudioError("Scene text, speaker, and delivery must match the frozen script.")
        _text(line["speaker"], limit=80)
        _text(line["text"])
        _text(line["style"], blank=True, limit=2000)
        _text(line["actual_model"], limit=160)
        character = line["character_id"]
        if not isinstance(character, str) or re.fullmatch(SLUG_PATTERN, character) is None:
            raise SceneAudioError("Scene lines require an explicit stable character identity.")
        old = character_speakers.setdefault(character, line["speaker"])
        if old != line["speaker"]:
            raise SceneAudioError("One character identity cannot silently merge different speaker recipes.")
        recipe = speakers[turn.speaker]
        if recipe.character_id is not None and character != recipe.character_id:
            raise SceneAudioError("Scene character does not match the frozen cast.")
        if (recipe.voice_recipe_id is not None and line.get("recipe_id") != recipe.voice_recipe_id
                and not _recipe_override(line, recipe, index)):
            raise SceneAudioError("Scene voice recipe does not match the frozen cast or an explicit scene override.")
        if recipe.character_revision is not None and line.get("character_revision") != recipe.character_revision:
            raise SceneAudioError("Scene character revision does not match the frozen cast.")
        if line.get("recipe_id") is not None:
            _text(line["recipe_id"], limit=160)
        if line.get("character_revision") is not None and (
                type(line["character_revision"]) is not int or line["character_revision"] <= 0):
            raise SceneAudioError("Scene character revision must be a positive integer.")
        if line.get("source_request_index") is not None and (
                type(line["source_request_index"]) is not int or line["source_request_index"] < 0):
            raise SceneAudioError("Scene source request index must be a nonnegative integer.")
        for name in ("source_job", "source_take"):
            if line.get(name) is not None:
                _text(line[name], limit=200)
        for name in ("source_sha256", "source_request_sha256", "source_response_sha256"):
            if line.get(name) is not None and (
                    not isinstance(line[name], str) or re.fullmatch(HASH_PATTERN, line[name]) is None):
                raise SceneAudioError("Scene source digest must be a lowercase SHA-256.")
        if not isinstance(line.get("voice_recipe_snapshot"), (dict, type(None))):
            raise SceneAudioError("Scene voice recipe snapshot must be a JSON object.")
        if not isinstance(line.get("source_attempt"), (str, dict, type(None))):
            raise SceneAudioError("Scene source attempt must be a reference or JSON object.")
        source = line["source"]
        if not isinstance(source, (str, Path)):
            raise SceneAudioError("Scene source must be an explicit local file path.")
        source = Path(source)
        _ordinary(source)
        source = source.absolute()
        data = _read(source, MAX_SOURCE_BYTES)
        if _digest(data) != line["source_sha256"]:
            raise SceneAudioError("Original scene source checksum changed.")
        native, _ = _wav(data)
        source_bytes += len(data)
        duration_seconds += native["frames"] / native["sample_rate"]
        if source_bytes > MAX_TOTAL_SOURCE_BYTES or duration_seconds * 1000 > MAX_DURATION_MS:
            raise SceneAudioError("Selected scene sources exceed aggregate size or duration limits.")
        row = {**line, "source": str(source), "native": native}
        _safe_metadata(row)
        if len(_canonical(row)) > 64 * 1024:
            raise SceneAudioError("Scene line provenance exceeds its byte limit.")
        frozen.append(row)
        sources.append(data)
    identity = {"schema_version": 1, "pipeline": PIPELINE, "stems": stems,
                "script": script.model_dump(mode="json"), "lines": frozen}
    if len(_canonical(identity)) > MAX_RECEIPT_BYTES // 2:
        raise SceneAudioError("Scene frozen inputs exceed their byte limit.")
    return script, identity, sources


def _run(arguments, *, timeout=MAX_PROCESS_SECONDS):
    try:
        result = subprocess.run(arguments, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SceneAudioError("Local scene audio processing failed; original takes are retained.") from exc
    if result.returncode:
        raise SceneAudioError("Local scene audio processing failed; original takes are retained.")
    return result


def _toolchain() -> dict:
    result = _run(["ffmpeg", "-version"], timeout=15)
    if len(result.stdout) > 64 * 1024:
        raise SceneAudioError("Audio toolchain identity exceeds its limit.")
    return {"ffmpeg_version": result.stdout.decode("utf-8", "strict").strip()}


def _ffmpeg(source: Path, destination: Path, *, extra=()):
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
          "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(source),
          "-map", "0:a:0", "-vn", "-map_metadata", "-1", *extra, str(destination)])


def _write_wav(path: Path, chunks):
    with path.open("xb") as output:
        with wave.open(output, "wb") as audio:
            audio.setnchannels(CHANNELS)
            audio.setsampwidth(SAMPLE_WIDTH)
            audio.setframerate(SAMPLE_RATE)
            for chunk in chunks:
                audio.writeframesraw(chunk)
        output.flush()
        os.fsync(output.fileno())


def _measure(path: Path) -> dict:
    result = _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
                   "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(path),
                   "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                   "-f", "null", "-"])
    if len(result.stderr) > 128 * 1024:
        raise SceneAudioError("Audio loudness analysis exceeds its limit.")
    try:
        text = result.stderr.decode("utf-8")
        report, _ = json.JSONDecoder().raw_decode(text[text.rindex("{"):])
        def finite(name):
            number = float(report[name])
            return number if math.isfinite(number) else None
        integrated, true_peak = finite("input_i"), finite("input_tp")
    except (ValueError, KeyError, UnicodeError) as exc:
        raise SceneAudioError("Audio loudness analysis is unavailable.") from exc
    desired = TARGET_LUFS - integrated if integrated is not None else 0.0
    peak_ceiling = TARGET_TRUE_PEAK_DB - true_peak if true_peak is not None else 0.0
    gain_db = max(-120.0, min(desired, peak_ceiling, MAX_GAIN_DB))
    return {"method": "one-static-gain-after-concatenation",
            "measurement": "ffmpeg-loudnorm-input-metrics-only",
            "input_lufs": integrated, "input_true_peak_db": true_peak,
            "target_lufs": TARGET_LUFS, "target_true_peak_db": TARGET_TRUE_PEAK_DB,
            "max_gain_db": MAX_GAIN_DB, "gain_db": gain_db, "target_guaranteed": False,
            "note": "One fixed gain preserves exact timing; peak/boost limits may prevent the loudness target."}


def _descriptor(root: Path, relative: str, mime: str) -> dict:
    data = _read(root / relative, MAX_AUDIO_BYTES if mime.startswith("audio/") else MAX_RECEIPT_BYTES)
    return {"path": relative, "mime_type": mime, "size_bytes": len(data), "sha256": _digest(data)}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SceneAudioError("Duplicate scene receipt JSON key.")
        result[key] = value
    return result


def _receipt_digest(receipt: dict) -> str:
    value = dict(receipt)
    value.pop("receipt_sha256", None)
    return _digest(_canonical(value))


def _verify_existing(destination: Path, assembly_id: str) -> dict:
    _ordinary(destination)
    if not destination.is_dir():
        raise SceneAudioError("Existing scene output must be an ordinary directory.")
    try:
        data = _read(destination / "assembly.json", MAX_RECEIPT_BYTES)
        receipt = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(SceneAudioError("Non-finite receipt.")))
        if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
                or receipt["schema_version"] != 1 or receipt.get("kind") != "assembled_scene"
                or receipt.get("assembly_id") != assembly_id
                or _digest(_canonical(receipt.get("frozen_inputs"))) != assembly_id
                or receipt.get("receipt_sha256") != _receipt_digest(receipt)):
            raise SceneAudioError("Existing scene output has conflicting or invalid provenance.")
        files = receipt["files"]
        expected, total = {"assembly.json"}, len(data)
        for item in files:
            relative = item["path"]
            if (not isinstance(relative, str)
                    or re.fullmatch(r"(?:scene\.(?:wav|mp3)|transcript\.txt|script\.json|stems/[a-z0-9][a-z0-9-]{0,127}\.wav)", relative) is None
                    or relative in expected):
                raise SceneAudioError("Existing scene output has an unsafe file inventory.")
            expected.add(relative)
            actual = _descriptor(destination, relative, item["mime_type"])
            if actual != item:
                raise SceneAudioError("Existing scene output checksum changed.")
            total += actual["size_bytes"]
        mandatory = {"scene.wav", "scene.mp3", "transcript.txt", "script.json"}
        if not mandatory <= expected or not isinstance(files, list) or not 4 <= len(files) <= 20:
            raise SceneAudioError("Existing scene output omits mandatory exports.")
        if total > MAX_EXPORT_BYTES:
            raise SceneAudioError("Existing scene output exceeds its aggregate limit.")
        inventory = set()
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise SceneAudioError("Existing scene output contains a symbolic link.")
            if path.is_file():
                inventory.add(path.relative_to(destination).as_posix())
            elif not path.is_dir() or path.relative_to(destination).as_posix() != "stems":
                raise SceneAudioError("Existing scene output inventory is invalid.")
        if inventory != expected:
            raise SceneAudioError("Existing scene output inventory changed.")
        info, _ = _wav(_read(destination / "scene.wav", MAX_AUDIO_BYTES), canonical=True)
        if info["frames"] != receipt["total_frames"]:
            raise SceneAudioError("Existing scene frame count changed.")
        return receipt
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise SceneAudioError("Existing scene output is incomplete or invalid.") from exc


@contextmanager
def _publication_lock(parent: Path):
    _ordinary(parent)
    descriptor = os.open(parent / ".scene-assembly.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise SceneAudioError("Scene publication lock must be ordinary.")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def assemble_scene(destination: Path, script: Script | dict, lines: list[dict], *, stems: bool = True) -> dict:
    """Atomically export selected original turns in script order at 48k mono PCM16.

    Sample intervals describe returned turns including source silence, not word
    timing or simultaneous voices. MP3 has encoder delay and is a listening
    derivative; WAV alone is the exact timing authority.
    """
    destination = Path(destination).absolute()
    _ordinary(destination)
    script, frozen, sources = _inputs(script, lines, stems)
    frozen["toolchain"] = _toolchain()
    assembly_id = _digest(_canonical(frozen))
    if destination.exists():
        return _verify_existing(destination, assembly_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".scene-assembly-", dir=destination.parent))
    try:
        work = staging / ".work"
        work.mkdir()
        canonical_chunks, assembled_lines = [], []
        frames = 0
        characters = list(dict.fromkeys(row["character_id"] for row in frozen["lines"]))
        for index, (row, original) in enumerate(zip(frozen["lines"], sources, strict=True)):
            native_path = work / f"{index:04d}-native.wav"
            native_path.write_bytes(original)
            normalized_path = work / f"{index:04d}-normalized.wav"
            _ffmpeg(native_path, normalized_path,
                    extra=("-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le"))
            normalized, pcm = _wav(_read(normalized_path, MAX_AUDIO_BYTES), canonical=True)
            count = normalized["frames"]
            if count < SAMPLE_RATE // 1000:
                raise SceneAudioError("Scene turn is shorter than one millisecond after normalization.")
            end = frames + count
            if end * 1000 > MAX_DURATION_MS * SAMPLE_RATE or end * 2 + 44 > MAX_AUDIO_BYTES:
                raise SceneAudioError("Scene master exceeds release size or duration limits.")
            projected_bytes = (end * 2 + 44) * (1 + (len(characters) if stems else 0))
            if projected_bytes + MAX_RECEIPT_BYTES + end // 2 > MAX_EXPORT_BYTES:
                raise SceneAudioError("Scene master and stems exceed aggregate export limits.")
            assembled_lines.append({**row, "canonical_frames": count,
                                    "canonical_pcm_sha256_before_gain": normalized["pcm_sha256"],
                                    "start_frame": frames, "end_frame": end,
                                    "start_ms": frames * 1000 // SAMPLE_RATE,
                                    "end_ms": end * 1000 // SAMPLE_RATE})
            canonical_chunks.append(pcm)
            sources[index] = b""
            frames = end
        raw_master = work / "concatenated.wav"
        _write_wav(raw_master, canonical_chunks)
        canonical_chunks.clear()
        loudness = _measure(raw_master)
        master = staging / "scene.wav"
        _ffmpeg(raw_master, master,
                extra=("-af", f"volume={loudness['gain_db']:.12f}dB:precision=double",
                       "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le"))
        master_data = _read(master, MAX_AUDIO_BYTES)
        master_info, master_pcm = _wav(master_data, canonical=True)
        if master_info["frames"] != frames:
            raise SceneAudioError("Static scene normalization changed the exact frame count.")
        audio_sha = _digest(master_data)
        del master_data
        _ffmpeg(master, staging / "scene.mp3",
                extra=("-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "1", "-id3v2_version", "3"))
        (staging / "transcript.txt").write_text(script.transcript(), encoding="utf-8")
        (staging / "script.json").write_bytes(_canonical(script.model_dump(mode="json")) + b"\n")
        for row in assembled_lines:
            row["final_pcm_sha256"] = _digest(master_pcm[row["start_frame"] * 2:row["end_frame"] * 2])
        bindings = []
        if stems:
            (staging / "stems").mkdir()
            silence = b"\0" * 131072
            for character in characters:
                relative = f"stems/{character}.wav"
                def stem_chunks():
                    for row in assembled_lines:
                        start, end = row["start_frame"] * 2, row["end_frame"] * 2
                        if row["character_id"] == character:
                            yield master_pcm[start:end]
                        else:
                            remaining = end - start
                            while remaining:
                                count = min(remaining, len(silence))
                                yield silence[:count]
                                remaining -= count
                _write_wav(staging / relative, stem_chunks())
                descriptor = _descriptor(staging, relative, "audio/wav")
                bindings.append({"character_id": character, "path": relative,
                                 "source_sha256": descriptor["sha256"], "timeline_audio_sha256": audio_sha,
                                 "sample_rate": SAMPLE_RATE, "total_frames": frames,
                                 "alignment_evidence": "Exact ordered turn concatenation; final master PCM copied only during this character's retained line intervals."})
        shutil.rmtree(work)
        files = [_descriptor(staging, path, mime) for path, mime in (
            ("scene.wav", "audio/wav"), ("scene.mp3", "audio/mpeg"),
            ("transcript.txt", "text/plain"), ("script.json", "application/json"))]
        files.extend(_descriptor(staging, binding["path"], "audio/wav") for binding in bindings)
        duration_ms, remainder = divmod(frames, SAMPLE_RATE // 1000)
        if remainder > 24 or (remainder == 24 and duration_ms % 2):
            duration_ms += 1
        receipt = {"schema_version": 1, "kind": "assembled_scene", "assembly_id": assembly_id,
                   "provenance": "ordered selected turn recordings; not one uninterrupted provider take",
                   "frozen_inputs": frozen, "script_sha256": _digest(_canonical(script.model_dump(mode="json"))),
                   "sample_rate": SAMPLE_RATE, "channels": CHANNELS, "sample_width": SAMPLE_WIDTH,
                   "total_frames": frames, "duration_ms": duration_ms, "audio_sha256": audio_sha,
                   "processing": {"pipeline": PIPELINE, "normalization": "ffmpeg PCM16 mono48k; no trims, gaps, crossfades, or time stretching",
                                  "loudness": loudness},
                   "lines": assembled_lines, "files": files,
                   "performance_candidate": {
                       "timeline_audio_sha256": audio_sha, "sample_rate": SAMPLE_RATE, "total_frames": frames,
                       "duration_ms": duration_ms,
                       "cues": [{"speaker_id": row["character_id"], "start_ms": row["start_ms"],
                                 "end_ms": row["end_ms"], "start_frame": row["start_frame"],
                                 "end_frame": row["end_frame"]} for row in assembled_lines],
                       "stems": bindings,
                       "release_binding": "Bind to an accepted release/package digest later; never install this candidate directly as performance v1."}}
        receipt["receipt_sha256"] = _receipt_digest(receipt)
        encoded = _canonical(receipt) + b"\n"
        if len(encoded) > MAX_RECEIPT_BYTES or len(encoded) + sum(item["size_bytes"] for item in files) > MAX_EXPORT_BYTES:
            raise SceneAudioError("Scene receipt or final exports exceed their byte limit.")
        (staging / "assembly.json").write_bytes(encoded)
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as output:
                    os.fsync(output.fileno())
        for directory in ([staging / "stems"] if stems else []) + [staging]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _verify_existing(staging, assembly_id)
        with _publication_lock(destination.parent):
            if destination.exists() or destination.is_symlink():
                return _verify_existing(destination, assembly_id)
            os.rename(staging, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)

