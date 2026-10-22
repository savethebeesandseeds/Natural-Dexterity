"""Scene exports use isolated synthetic PCM fixtures; no speech generation or keys."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import wave

import pytest

from web_radio import scene_audio
from web_radio.gemini import Script


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wav_bytes(*, rate=24000, frames=2400, channels=1, amplitude=6000, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        if width == 2:
            output.writeframes(b"".join(struct.pack("<h", round(amplitude * math.sin(2 * math.pi * 330 * i / rate)))
                                       * channels for i in range(frames)))
        else:
            output.writeframes(b"\x80" * frames * channels * width)
    return buffer.getvalue()


@pytest.fixture
def scene(tmp_path):
    script = Script.model_validate({
        "programme_id": "scene-fixture", "series_id": "fixture", "kind": "programme",
        "title": "ISOLATED SYNTHETIC SCENE FIXTURE", "language": "en",
        "speakers": [
            {"name": "Moss", "voice": "Kore", "character_id": "moss",
             "voice_recipe_id": "1" * 32, "character_revision": 1},
            {"name": "Wren", "voice": "Puck", "character_id": "wren",
             "voice_recipe_id": "2" * 32, "character_revision": 1},
        ],
        "turns": [{"speaker": "Moss", "text": "This is the first fixture turn.", "style": "quiet"},
                  {"speaker": "Wren", "text": "The second fixture follows.", "style": "warm"},
                  {"speaker": "Moss", "text": "Only synthetic tones are recorded.", "style": ""}],
    })
    lines = []
    for index, (turn, rate, count, channels) in enumerate(zip(
            script.turns, [24000, 44100, 48000], [2400, 8820, 14400], [1, 2, 1], strict=True)):
        source = tmp_path / f"source-{index}.wav"
        source.write_bytes(wav_bytes(rate=rate, frames=count, channels=channels))
        character = turn.speaker.lower()
        lines.append({
            "index": index, "speaker": turn.speaker, "character_id": character,
            "source": source, "source_sha256": digest(source.read_bytes()),
            "text": turn.text, "style": turn.style, "actual_model": "fixture-model",
            "recipe_id": ("1" if character == "moss" else "2") * 32, "character_revision": 1,
            "voice_recipe_snapshot": {"id": ("1" if character == "moss" else "2") * 32,
                                      "character_id": character, "kind": "prebuilt"},
            "source_request_sha256": "3" * 64, "source_response_sha256": "4" * 64,
            "source_attempt": {"take_id": "isolated-fixture", "request_index": index},
        })
    return script, lines


def pcm(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getparams(), audio.readframes(audio.getnframes())


def test_export_preserves_originals_and_measures_exact_ordered_samples(scene, tmp_path):
    script, lines = scene
    before = {line["source"]: line["source"].read_bytes() for line in lines}
    destination = tmp_path / "assembled"
    receipt = scene_audio.assemble_scene(destination, script, lines)
    assert receipt["provenance"] == "ordered selected turn recordings; not one uninterrupted provider take"
    assert receipt["sample_rate"] == 48000 and receipt["channels"] == 1
    assert receipt["total_frames"] == 28800 and receipt["duration_ms"] == 600
    assert [(row["start_frame"], row["end_frame"]) for row in receipt["lines"]] == [
        (0, 4800), (4800, 14400), (14400, 28800)]
    assert [(row["start_ms"], row["end_ms"]) for row in receipt["lines"]] == [(0, 100), (100, 300), (300, 600)]
    assert [row["native"]["frames"] for row in receipt["lines"]] == [2400, 8820, 14400]
    assert [row["native"]["sample_rate"] for row in receipt["lines"]] == [24000, 44100, 48000]
    assert all(row["actual_model"] == "fixture-model" for row in receipt["lines"])
    assert receipt["lines"][1]["source_attempt"]["request_index"] == 1
    assert all(source.read_bytes() == original for source, original in before.items())
    params, master = pcm(destination / "scene.wav")
    assert (params.framerate, params.nchannels, params.sampwidth, params.nframes) == (48000, 1, 2, 28800)
    assert receipt["audio_sha256"] == digest((destination / "scene.wav").read_bytes())
    assert (destination / "transcript.txt").read_text() == script.transcript()
    assert Script.model_validate_json((destination / "script.json").read_bytes()) == script
    assert json.loads((destination / "assembly.json").read_bytes()) == receipt
    assert not (destination / ".work").exists()
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                            "-of", "json", str(destination / "scene.mp3")],
                           capture_output=True, check=True, text=True)
    assert json.loads(probe.stdout)["streams"][0]["codec_name"] == "mp3"
    assert receipt["processing"]["loudness"]["method"] == "one-static-gain-after-concatenation"
    assert receipt["processing"]["loudness"]["target_guaranteed"] is False
    for row in receipt["lines"]:
        assert digest(master[row["start_frame"] * 2:row["end_frame"] * 2]) == row["final_pcm_sha256"]


def test_character_stems_copy_final_master_only_at_verified_turn_intervals(scene, tmp_path):
    script, lines = scene
    destination = tmp_path / "stemmed"
    receipt = scene_audio.assemble_scene(destination, script, lines)
    _, master = pcm(destination / "scene.wav")
    stems = {character: pcm(destination / f"stems/{character}.wav")[1] for character in ("moss", "wren")}
    assert all(len(stem) == len(master) for stem in stems.values())
    for row in receipt["lines"]:
        first, last = row["start_frame"] * 2, row["end_frame"] * 2
        for character, stem in stems.items():
            assert stem[first:last] == (master[first:last] if character == row["character_id"] else b"\0" * (last - first))
    summed = b"".join(struct.pack("<h", one[0] + two[0]) for one, two in zip(
        struct.iter_unpack("<h", stems["moss"]), struct.iter_unpack("<h", stems["wren"]), strict=True))
    assert summed == master
    candidate = receipt["performance_candidate"]
    assert candidate["timeline_audio_sha256"] == receipt["audio_sha256"]
    assert "release_id" not in candidate and "package_sha256" not in candidate
    assert all(binding["timeline_audio_sha256"] == receipt["audio_sha256"] for binding in candidate["stems"])
    assert [cue["speaker_id"] for cue in candidate["cues"]] == ["moss", "wren", "moss"]


def test_no_stems_and_idempotent_recovery_preserve_all_output_bytes(scene, tmp_path, monkeypatch):
    script, lines = scene
    destination = tmp_path / "export"
    first = scene_audio.assemble_scene(destination, script.model_dump(), lines, stems=False)
    assert not (destination / "stems").exists()
    before = {path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    # Checking an existing completed output needs toolchain identity, not rerendering.
    monkeypatch.setattr(scene_audio, "_ffmpeg", lambda *args, **kwargs: pytest.fail("rerendered paid source assembly"))
    second = scene_audio.assemble_scene(destination, script, lines, stems=False)
    assert first == second
    assert {path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(destination, script, lines, stems=True)


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.pop(),
    lambda rows: rows.reverse(),
    lambda rows: rows[0].update(index=True),
    lambda rows: rows[0].update(speaker="Wren"),
    lambda rows: rows[0].update(text="rewritten"),
    lambda rows: rows[0].update(style="different"),
    lambda rows: rows[0].update(character_id="other"),
    lambda rows: rows[0].update(character_id="../private"),
    lambda rows: rows[0].update(recipe_id="5" * 32),
    lambda rows: rows[0].update(character_revision=2),
    lambda rows: rows[0].update(character_revision=True),
    lambda rows: rows[0].update(source_sha256="0" * 64),
    lambda rows: rows[0].update(actual_model=""),
    lambda rows: rows[0].update(source_request_sha256="not-a-hash"),
    lambda rows: rows[0].update(unknown_field="silently lost"),
    lambda rows: rows[0].update(voice_recipe_snapshot={"api_key": "fixture-fake-key"}),
    lambda rows: rows[0].update(source_attempt={"invalid": float("nan")}),
])
def test_invalid_selection_is_rejected_before_audio_processing(scene, tmp_path, monkeypatch, mutate):
    script, lines = scene
    mutate(lines)
    monkeypatch.setattr(scene_audio, "_toolchain", lambda: pytest.fail("processing started before validation"))
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(tmp_path / "invalid", script, lines)
    assert not (tmp_path / "invalid").exists()


def test_explicit_targeted_recipe_override_is_frozen_without_rewriting_baseline(scene, tmp_path):
    script, lines = scene
    baseline = script.speakers[0].voice_recipe_id
    lines[0]["recipe_id"] = "8" * 32
    lines[0]["voice_recipe_snapshot"] = {
        "recipe_id": "8" * 32, "character_id": "moss",
        "_scene_override": {"baseline_recipe_id": baseline, "parent_job": "fixture-parent", "line_index": 0},
    }
    receipt = scene_audio.assemble_scene(tmp_path / "overridden", script, lines, stems=False)
    assert receipt["frozen_inputs"]["script"]["speakers"][0]["voice_recipe_id"] == baseline
    assert receipt["lines"][0]["recipe_id"] == "8" * 32
    assert receipt["lines"][2]["recipe_id"] == baseline
    assert receipt["lines"][0]["voice_recipe_snapshot"]["_scene_override"]["parent_job"] == "fixture-parent"


@pytest.mark.parametrize("mutate", [
    lambda snapshot: snapshot.update(character_id="wren"),
    lambda snapshot: snapshot.update(recipe_id="7" * 32),
    lambda snapshot: snapshot["_scene_override"].update(baseline_recipe_id="7" * 32),
    lambda snapshot: snapshot["_scene_override"].update(line_index=1),
    lambda snapshot: snapshot["_scene_override"].update(parent_job=""),
    lambda snapshot: snapshot.pop("_scene_override"),
])
def test_targeted_recipe_override_needs_explicit_coherent_evidence(scene, tmp_path, mutate):
    script, lines = scene
    lines[0]["recipe_id"] = "8" * 32
    lines[0]["voice_recipe_snapshot"] = {
        "recipe_id": "8" * 32, "character_id": "moss",
        "_scene_override": {"baseline_recipe_id": "1" * 32, "parent_job": "fixture-parent", "line_index": 0},
    }
    mutate(lines[0]["voice_recipe_snapshot"])
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(tmp_path / "bad-override", script, lines)


@pytest.mark.parametrize("data", [
    b"not a WAV",
    wav_bytes()[:-2],
    wav_bytes() + b"trailing",
    wav_bytes(width=1),
    wav_bytes(frames=0),
])
def test_original_wav_must_be_complete_pcm16(scene, tmp_path, data):
    script, lines = scene
    source = lines[0]["source"]
    source.write_bytes(data)
    lines[0]["source_sha256"] = digest(data)
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(tmp_path / "bad-wav", script, lines)
    assert source.read_bytes() == data


def test_silent_source_is_finite_and_frame_preserving(scene, tmp_path):
    script, lines = scene
    for line in lines:
        line["source"].write_bytes(wav_bytes(amplitude=0))
        line["source_sha256"] = digest(line["source"].read_bytes())
    receipt = scene_audio.assemble_scene(tmp_path / "silent", script, lines)
    assert receipt["total_frames"] == 14400
    assert receipt["processing"]["loudness"]["input_lufs"] is None
    assert receipt["processing"]["loudness"]["gain_db"] == 0
    assert pcm(tmp_path / "silent/scene.wav")[1] == b"\0" * 28800


def test_source_destination_and_output_symlinks_are_rejected(scene, tmp_path):
    script, lines = scene
    original = lines[0]["source"]
    alias = tmp_path / "source-alias.wav"
    alias.symlink_to(original)
    lines[0]["source"] = alias
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(tmp_path / "invalid", script, lines)
    lines[0]["source"] = original
    alias_dir = tmp_path / "output-alias"
    alias_dir.symlink_to(tmp_path / "untouched", target_is_directory=True)
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(alias_dir / "scene", script, lines)
    assert not (tmp_path / "untouched").exists()
    destination = tmp_path / "valid"
    scene_audio.assemble_scene(destination, script, lines, stems=False)
    (destination / "scene.mp3").unlink()
    (destination / "scene.mp3").symlink_to(original)
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(destination, script, lines, stems=False)


@pytest.mark.parametrize("corruption", ["audio", "receipt", "extra", "missing"])
def test_existing_immutable_outputs_are_never_repaired_overwritten_or_deleted(scene, tmp_path, corruption):
    script, lines = scene
    destination = tmp_path / "existing"
    scene_audio.assemble_scene(destination, script, lines, stems=False)
    if corruption == "audio":
        (destination / "scene.wav").write_bytes(b"corrupted")
    elif corruption == "receipt":
        (destination / "assembly.json").write_text('{"incomplete":true}')
    elif corruption == "extra":
        (destination / "keep-private.txt").write_text("Preserve this existing file.")
    else:
        (destination / "scene.mp3").unlink()
    before = {path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(destination, script, lines, stems=False)
    assert {path.relative_to(destination).as_posix(): path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before


def test_media_budgets_fail_safely_and_originals_survive(scene, tmp_path, monkeypatch):
    script, lines = scene
    before = [line["source"].read_bytes() for line in lines]
    with monkeypatch.context() as cap:
        cap.setattr(scene_audio, "MAX_SOURCE_BYTES", 64)
        with pytest.raises(scene_audio.SceneAudioError):
            scene_audio.assemble_scene(tmp_path / "source-limit", script, lines)
    with monkeypatch.context() as cap:
        cap.setattr(scene_audio, "MAX_EXPORT_BYTES", 1000)
        with pytest.raises(scene_audio.SceneAudioError):
            scene_audio.assemble_scene(tmp_path / "export-limit", script, lines)
    assert [line["source"].read_bytes() for line in lines] == before
    assert not (tmp_path / "export-limit").exists()
    assert not list(tmp_path.glob(".scene-assembly-*"))


def test_processing_failure_keeps_originals_and_cleans_only_own_staging(scene, tmp_path, monkeypatch):
    script, lines = scene
    preserved = tmp_path / ".scene-assembly-unrelated"
    preserved.mkdir()
    (preserved / "keep.txt").write_text("Unrelated retained evidence.")
    before = [line["source"].read_bytes() for line in lines]
    def failure(*args, **kwargs):
        raise scene_audio.SceneAudioError("Fixture processor failure.")
    monkeypatch.setattr(scene_audio, "_ffmpeg", failure)
    with pytest.raises(scene_audio.SceneAudioError):
        scene_audio.assemble_scene(tmp_path / "not-published", script, lines)
    assert [line["source"].read_bytes() for line in lines] == before
    assert not (tmp_path / "not-published").exists()
    assert list(tmp_path.glob(".scene-assembly-*")) == [preserved]


def test_rounding_matches_release_half_even_while_frames_remain_authoritative(scene, tmp_path):
    script, lines = scene
    # Every line is valid >=1ms. Three x 2.5ms =7.5ms -> nearest-even8ms.
    for line in lines:
        line["source"].write_bytes(wav_bytes(rate=48000, frames=120))
        line["source_sha256"] = digest(line["source"].read_bytes())
    receipt = scene_audio.assemble_scene(tmp_path / "fractional", script, lines, stems=False)
    assert receipt["total_frames"] == 360
    assert receipt["duration_ms"] == 8
    assert [(row["start_ms"], row["end_ms"]) for row in receipt["lines"]] == [(0, 2), (2, 5), (5, 7)]
    assert receipt["performance_candidate"]["cues"][-1]["end_frame"] == 360

