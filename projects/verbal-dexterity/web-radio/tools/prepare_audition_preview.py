#!/usr/bin/env python3
"""Prepare an isolated, sample-verified audition visualization without provider calls.

Run inside the project's managed development container. The source audition is
never published or scheduled; only the preview's private preparation journal is
allowed to accept its duplicate immutable release. No credentials are read.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web_radio.contract import create_release, validate_package
from web_radio.embedding_service import PreparationSpec
from web_radio.performance import Performance
from web_radio.playout import Journal

DEFAULT_CAST = [
    {"id": "nell", "color": "#508b82"},
    {"id": "rusk", "color": "#b28a4e"},
    {"id": "orro", "color": "#937794"},
]
BANDS = {
    "energy": None,
    "low": "highpass=f=80,lowpass=f=300",
    "mid": "highpass=f=300,lowpass=f=2500",
    "high": "highpass=f=2500,lowpass=f=8000",
}
NORMALIZATION = "loudnorm=I=-16:TP=-1.5:LRA=11"
HOP_MS = 50
MAX_AUDIO_BYTES = 128 * 1024 * 1024
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,159}\Z")


def ordinary(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Source and output paths must not traverse symbolic links.")
    return path


def under(path: Path, parent: Path) -> Path:
    path = ordinary(path)
    if not path.is_relative_to(ordinary(parent)) or path == parent:
        raise ValueError("Source/output path is outside its documented private subtree.")
    return path


def read(path: Path, limit: int = MAX_AUDIO_BYTES) -> bytes:
    path = ordinary(path)
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Expected a bounded ordinary source file: " + str(path))
    data = path.read_bytes()
    if len(data) > limit:
        raise ValueError("Source file grew beyond its bound.")
    return data


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, data: bytes) -> None:
    path = ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path, max(len(data), 1)) != data:
            raise ValueError("Existing immutable preview artifact conflicts: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def wav(path: Path):
    read(path)
    with wave.open(str(path), "rb") as recording:
        params = recording.getparams()
        data = recording.readframes(params.nframes)
    if params.comptype != "NONE" or len(data) != params.nframes * params.nchannels * params.sampwidth:
        raise ValueError("Expected complete uncompressed PCM WAV.")
    return params, data


def ffmpeg_pcm(path: Path, filters: str | None = None, *, mono_float=False) -> bytes:
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(path), "-vn"]
    if mono_float:
        filters = "pan=mono|c0=0.5*c0+0.5*c1" + (("," + filters) if filters else "")
    if filters:
        args += ["-af", filters]
    args += ["-ar", "48000", "-ac", "1" if mono_float else "2",
             "-c:a", "pcm_f32le" if mono_float else "pcm_s16le",
             "-f", "f32le" if mono_float else "s16le", "pipe:1"]
    result = subprocess.run(args, capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise ValueError("Read-only audio extraction failed: " + result.stderr.decode(errors="replace"))
    return result.stdout


def feature_tracks(audio: Path, params, cues, cast):
    count = math.ceil(params.nframes / params.framerate * 1000 / HOP_MS) + 1
    rate = params.framerate
    half_window_frames = rate * HOP_MS // 2000
    owners = []
    bounds = []
    for index in range(count):
        stamp = index * HOP_MS
        cue = next((cue for cue in cues if cue["start_ms"] <= stamp < cue["end_ms"]), None)
        owners.append(cue["speaker_id"] if cue else None)
        if cue:
            center = stamp * rate // 1000
            bounds.append((max(cue["start_ms"] * rate // 1000, center - half_window_frames),
                           min(cue["end_ms"] * rate // 1000, center + half_window_frames, params.nframes)))
        else:
            bounds.append((0, 0))
    tracks = {item["id"]: {band: [0.0] * count for band in BANDS} for item in cast}
    scales = {}
    for band, filters in BANDS.items():
        data = ffmpeg_pcm(audio, filters, mono_float=True)
        values = array.array("f")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != params.nframes:
            raise ValueError("Frequency extraction changed the verified sample clock.")
        rms = [math.sqrt(math.fsum(v * v for v in values[start:end]) / (end - start))
               if end > start else 0.0 for start, end in bounds]
        positive = sorted(value for value in rms if value > 0)
        scale = positive[min(len(positive) - 1, math.ceil(len(positive) * 0.95) - 1)] if positive else 1.0
        scales[band] = scale
        previous_owner = None
        previous = 0.0
        for index, (owner, value) in enumerate(zip(owners, rms)):
            if owner is None:
                previous_owner, previous = None, 0.0
                continue
            if owner != previous_owner:
                previous = 0.0
            target = min(1.0, max(0.0, value / scale))
            time_constant_ms = 25 if target > previous else 80
            coefficient = 1 - math.exp(-HOP_MS / time_constant_ms)
            previous += coefficient * (target - previous)
            tracks[owner][band][index] = round(min(1.0, max(0.0, previous)), 6)
            previous_owner = owner
    for character, track in tracks.items():
        for band, values in track.items():
            if len(values) != count or any(not 0 <= value <= 1 or not math.isfinite(value) for value in values):
                raise ValueError("Invalid measured feature track.")
            if any(value != 0 for index, value in enumerate(values) if owners[index] != character):
                raise ValueError("Character features leaked outside their authored source cues.")
    receipt = {"hop_ms": HOP_MS, "sample_count_per_track": count,
               "mono_mix": "0.5*left+0.5*right", "sample_rate": rate,
               "filters": BANDS, "rms_window_ms": HOP_MS,
               "window": "centered, clipped to authored cue and audio bounds",
               "normalization": "shared across cast per band; positive measured RMS 95th percentile; clipped to [0,1]",
               "normalization_scales_rms": scales, "smoothing": {"attack_ms": 25, "release_ms": 80},
               "silence": "measured PCM RMS; no synthetic speech activity",
               "gating": "exactly zero outside each character's half-open cues; state reset on owner change"}
    return {"hop_ms": HOP_MS, "tracks": tracks}, receipt


def prepare(audition: Path, output: Path, preview_id: str, release_id: str, cast):
    audition = under(audition, ROOT / ".runtime/creative")
    output = under(output, ROOT / ".runtime/visual-previews")
    if not SLUG.fullmatch(preview_id) or not SLUG.fullmatch(release_id):
        raise ValueError("Use ordinary preview and release slugs.")
    provenance_path = audition / "provenance.json"
    provenance_bytes = read(provenance_path, 1024 * 1024)
    provenance = json.loads(provenance_bytes)
    if provenance.get("publication") != "not_published_or_scheduled":
        raise ValueError("This preparer requires an unpublished, unscheduled audition.")
    sources = provenance["sources"]
    if [source["turn_index"] for source in sources] != list(range(len(sources))):
        raise ValueError("Source turns must form one exact ordered sequence.")
    cast_ids = {item["id"] for item in cast}
    if len(cast_ids) != len(cast):
        raise ValueError("Cast identities must be unique.")
    originals = {provenance_path: sha(provenance_bytes)}
    transcript_path = audition / "transcript.txt"
    transcript = read(transcript_path, 1024 * 1024)
    transcript.decode("utf-8")
    if sha(transcript) != provenance["files"]["transcript.txt"]["sha256"]:
        raise ValueError("Transcript differs from retained audition provenance.")
    originals[transcript_path] = sha(transcript)
    parts, cues, alignment, windows = [], [], [], []
    cursor_frames = 0
    transcript_cursor = 0
    for index, source in enumerate(sources):
        relative = Path(source["audio_path"])
        if relative.is_absolute() or ".." in relative.parts or chr(92) in str(relative):
            raise ValueError("Source WAV must be an ordinary relative project path.")
        source_path = under(ROOT / relative, ROOT / ".runtime/studio/takes")
        raw = read(source_path)
        if sha(raw) != source["audio_sha256"]:
            raise ValueError("Source audio does not match its provenance hash.")
        originals[source_path] = sha(raw)
        params, data = wav(source_path)
        if (params.nchannels, params.sampwidth, params.framerate) != (1, 2, 24000):
            raise ValueError("Audition sources must be unmodified 24 kHz mono s16 PCM.")
        character = source["speaker"].lower()
        if character not in cast_ids:
            raise ValueError("Source speaker needs an explicit cast identity/color.")
        end_frame = cursor_frames + params.nframes
        if cursor_frames * 1000 % 24000 or end_frame * 1000 % 24000:
            raise ValueError("Source frame edges must map exactly to integer milliseconds.")
        start_ms, end_ms = cursor_frames * 1000 // 24000, end_frame * 1000 // 24000
        cues.append({"speaker_id": character, "start_ms": start_ms, "end_ms": end_ms})
        text_bytes = source["text"].encode("utf-8")
        prefix = (source["speaker"] + ": ").encode("utf-8")
        block = prefix + text_bytes
        location = transcript.find(block, transcript_cursor)
        if location < 0 or transcript[transcript_cursor:location].strip():
            raise ValueError("Transcript does not match the exact ordered source text.")
        text_start = location + len(prefix)
        text_end = text_start + len(text_bytes)
        if transcript[text_start:text_end] != text_bytes:
            raise ValueError("UTF-8 transcript provenance failed.")
        transcript_cursor = text_end
        row = {"turn_index": index, "speaker_id": character, "text": source["text"],
               "source_audio_path": str(relative), "source_audio_sha256": sha(raw),
               "source_frames": params.nframes, "source_sample_rate": 24000,
               "start_frame_24000": cursor_frames, "end_frame_24000": end_frame,
               "start_frame_48000": cursor_frames * 2, "end_frame_48000": end_frame * 2,
               "start_ms": start_ms, "end_ms": end_ms,
               "text_start_byte": text_start, "text_end_byte": text_end}
        alignment.append(row)
        evidence = (f"Exact retained source {relative}, SHA256 {sha(raw)}, {params.nframes} frames at 24000 Hz. "
                    f"Verified PCM concatenation with zero added gaps maps source interval [{cursor_frames},{end_frame}) "
                    f"at 24000 Hz to [{cursor_frames*2},{end_frame*2}) at 48000 Hz after {NORMALIZATION}. "
                    f"See private evidence/alignment.json; provenance SHA256 {sha(provenance_bytes)}.")
        windows.append({"window_id": f"turn-{index:03d}", "passage_id": f"turn-{index:03d}",
                        "character_id": character, "start_ms": start_ms, "end_ms": end_ms,
                        "timing_evidence": evidence, "text": source["text"],
                        "text_start_byte": text_start, "modalities": ["text"]})
        parts.append(data)
        cursor_frames = end_frame
    if transcript[transcript_cursor:].strip():
        raise ValueError("Transcript contains text outside the retained source turns.")
    source_path = audition / "scene-source.wav"
    source_params, scene_pcm = wav(source_path)
    originals[source_path] = sha(read(source_path))
    if (source_params.nchannels, source_params.sampwidth, source_params.framerate) != (1, 2, 24000):
        raise ValueError("Scene source format differs from original source turns.")
    if scene_pcm != b"".join(parts) or source_params.nframes != cursor_frames:
        raise ValueError("Scene source is not the exact source concatenation.")
    audio_names = [name for name in provenance["files"] if Path(name).suffix.lower() == ".wav"]
    if len(audio_names) != 1 or Path(audio_names[0]).name != audio_names[0]:
        raise ValueError("Expected one explicit final WAV in audition provenance.")
    final_path = audition / audio_names[0]
    final_bytes = read(final_path)
    audio_sha256 = sha(final_bytes)
    if audio_sha256 != provenance["files"][audio_names[0]]["sha256"]:
        raise ValueError("Final WAV differs from its retained provenance.")
    originals[final_path] = audio_sha256
    final_params, final_pcm = wav(final_path)
    if (final_params.nchannels, final_params.sampwidth, final_params.framerate, final_params.nframes) != (2, 2, 48000, cursor_frames * 2):
        raise ValueError("Final WAV does not preserve the verified source sample clock.")
    if ffmpeg_pcm(source_path, NORMALIZATION) != final_pcm:
        raise ValueError("Final WAV PCM differs from the verified normalization recipe.")
    final_take = under(ROOT / ".runtime/studio/takes" / provenance["final_review_take_id"], ROOT / ".runtime/studio/takes")
    for name in ("original.wav", "mix.wav"):
        path = final_take / name
        if read(path) != final_bytes:
            raise ValueError("Final retained take differs from audition WAV.")
        originals[path] = audio_sha256
    duration_ms = cursor_frames * 1000 // 24000
    if duration_ms != round(provenance["duration_seconds"] * 1000):
        raise ValueError("Measured duration differs from audition provenance.")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "audio.wav", final_bytes)
    preparation = output / "preparation"
    release = preparation / ".runtime/station/releases" / release_id
    if release.exists():
        manifest = validate_package(release)
        roles = {asset.role: asset for asset in manifest.assets}
        if roles["audio"].sha256 != audio_sha256 or roles["transcript"].sha256 != sha(transcript):
            raise ValueError("Existing isolated release differs from this audition.")
    else:
        manifest = create_release(final_path, transcript.decode("utf-8"), release,
                                  programme_id=preview_id, title=provenance["title"], release_id=release_id)
    journal = Journal(preparation / ".runtime/station/journal.sqlite3")
    accepted = {row["release_id"]: row["digest"] for row in journal.accepted()}
    if release_id in accepted and accepted[release_id] != manifest.package_sha256:
        raise ValueError("Conflicting isolated accepted release is preserved.")
    journal.remember(release_id, manifest.package_sha256)
    if journal.schedule() or journal.control() is not None:
        raise ValueError("Preparation root must remain unscheduled and free of playout control.")
    spec = PreparationSpec.model_validate({"schema_version": 1, "release_id": release_id,
            "package_sha256": manifest.package_sha256, "duration_ms": duration_ms,
            "dimensions": 3072, "windows": windows})
    write(preparation / "text-embedding-spec.json", json_bytes(spec.model_dump(mode="json", exclude_none=True)))
    features, feature_receipt = feature_tracks(final_path, final_params, cues, cast)
    Performance.model_validate({"schema_version": 1, "release_id": manifest.release_id,
            "package_sha256": manifest.package_sha256, "duration_ms": duration_ms,
            "cast": cast, "cues": cues, "features": features})
    performance = {"schema_version": 1, "preview_id": preview_id, "audio_sha256": audio_sha256,
                   "duration_ms": duration_ms, "cast": cast, "cues": cues, "features": features}
    performance_bytes = json_bytes(performance)
    write(output / "performance.json", performance_bytes)
    preview_manifest = {"schema_version": 1, "preview_id": preview_id, "title": provenance["title"],
            "duration_ms": duration_ms,
            "audio": {"file": "audio.wav", "sha256": audio_sha256, "size_bytes": len(final_bytes)},
            "performance": {"file": "performance.json", "sha256": sha(performance_bytes), "size_bytes": len(performance_bytes)},
            "semantic": None}
    write(output / "manifest.json", json_bytes(preview_manifest))
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    receipt = {"schema_version": 1, "preview_id": preview_id, "audio_sha256": audio_sha256,
            "final_pcm_sha256": sha(final_pcm), "source_pcm_sha256": sha(scene_pcm),
            "source_provenance_sha256": sha(provenance_bytes), "transcript_sha256": sha(transcript),
            "source_sample_rate": 24000, "source_frames": cursor_frames,
            "final_sample_rate": 48000, "final_frames": final_params.nframes,
            "duration_ms": duration_ms, "inserted_gap_frames": 0,
            "concatenation_verified_byte_exact": True, "normalization_pcm_verified_byte_exact": True,
            "normalization_filter": NORMALIZATION, "ffmpeg_version": version,
            "timing_kind": "known source boundaries, including natural per-turn silence; not word alignment",
            "publication": "not_published_or_scheduled", "source_status": provenance.get("status"),
            "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": digest} for path, digest in originals.items()],
            "turns": alignment,
            "isolated_release": {"release_id": manifest.release_id, "package_sha256": manifest.package_sha256,
                                 "path": str(release.relative_to(ROOT)), "actual_station_modified": False}}
    write(output / "evidence/source-provenance.json", provenance_bytes)
    write(output / "evidence/alignment.json", json_bytes(receipt))
    write(output / "evidence/feature-extraction.json", json_bytes(feature_receipt))
    for path, digest in originals.items():
        if sha(read(path)) != digest:
            raise ValueError("Source changed while preparing preview; retained output requires review.")
    return {"preview_root": str(output), "manifest": str(output / "manifest.json"),
            "embedding_preparation_root": str(preparation),
            "text_embedding_spec": str(preparation / "text-embedding-spec.json"),
            "release_id": release_id, "package_sha256": manifest.package_sha256,
            "audio_sha256": audio_sha256, "duration_ms": duration_ms, "cue_count": len(cues),
            "feature_samples_per_track": math.ceil(duration_ms / HOP_MS) + 1,
            "embedding_dimensions": 3072, "embedding_modalities": ["text"],
            "provider_calls": 0, "credentials_read_or_copied": False,
            "actual_station_modified": False, "source_publication": provenance["publication"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audition", type=Path, default=ROOT / ".runtime/creative/empty-chair-first-guest-v1")
    parser.add_argument("--preview-id", default="empty-chair-first-guest-v1")
    parser.add_argument("--release-id", default="empty-chair-first-guest-visual-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cast-json", type=Path, help="Explicit [{id,color}] file for another known cast")
    args = parser.parse_args()
    cast = json.loads(read(args.cast_json, 65536)) if args.cast_json else DEFAULT_CAST
    output = args.output or ROOT / ".runtime/visual-previews" / args.preview_id
    summary = prepare(args.audition, output, args.preview_id, args.release_id, cast)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
