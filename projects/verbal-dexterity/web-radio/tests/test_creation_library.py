"""Retained local creation listening never publishes, prepares or writes assets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from web_radio import creation_library as module
from web_radio.performance import _fallback_character_color


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wav(path, frames=48030, rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\0\0" * frames)


def identity(root, project):
    return "creation-" + hashlib.sha256(root.relative_to(project).as_posix().encode()).hexdigest()[:24]


def seal(root):
    rows = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    write_json(root / "manifest.json", {"schema_version": 2, "files": rows})


def episode(root, *, legacy=False):
    root.mkdir(parents=True)
    audio = root / "audio/master.wav"
    wav(audio)
    characters = [{"character_id": "aster", "spiral_color": "#508b82"},
                  {"character_id": "bramble", "spiral_color": "#937794"}]
    if legacy:
        rows = []
        for character in characters:
            path = root / ("provenance/character-revisions/" + character["character_id"] + ".json")
            write_json(path, character)
            rows.append({"character_id": character["character_id"],
                         "character_snapshot": {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}})
        voices = {"schema_version": 2, "voices": rows}
    else:
        voices = {"schema_version": 1, "speakers": [
            {"speaker": character["character_id"].title(), "character": character,
             "voice_recipe": {"private": "private-provider-data"}}
            for character in characters]}
    write_json(root / "voices.json", voices)
    cues = [
        {"character_id": "aster", "start_frame": 4800, "end_frame": 14400,
         "start_ms": 100, "end_ms": 300, "text": "Private transcript."},
        {"character_id": "bramble", "start_frame": 24000, "end_frame": 48030,
         "start_ms": 500, "end_ms": 1001, "text": "More private transcript."},
    ]
    write_json(root / "performance.json", {"schema_version": 1, "sample_rate": 48000,
               "total_frames": 48030, "duration_ms": 1001, "audio_sha256": sha(audio), "cues": cues})
    role, timing = ("lossless_master", "timing") if legacy else ("master", "performance")
    production = {"schema_version": 1, "kind": "complete_creative_production" if legacy else "podcast_production",
                  "title": "Fixture " + root.parent.name, "creative_version": root.name,
                  "duration_ms": 1001, "publication": "Private review only, not published.",
                  "provider_credentials": "must-never-be-public",
                  "deliverables": {role: {"path": "audio/master.wav", "sha256": sha(audio)},
                                   timing: {"path": "performance.json", "sha256": sha(root / "performance.json")}}}
    if legacy:
        production["casting"] = {"voices": {"path": "voices.json", "sha256": sha(root / "voices.json")}}
    else:
        production["voices"] = {"path": "voices.json"}
    write_json(root / "production.json", production)
    write_json(root / "script.json", {"title": production["title"], "text": "Frozen fixture dialogue."})
    write_json(root / "provider-responses/private.json", {"secret": "must-never-be-public"})
    seal(root)
    return root


def audition(root, character="aster"):
    wav(root / "audio/provider-sample.wav", 12000, 16000)
    write_json(root / "production.json", {
        "schema_version": 1, "kind": "voice_audition", "title": "Unapproved voice audition",
        "character_id": character, "voice_accepted_by_listener": False,
        "native_sample": {"path": "audio/provider-sample.wav", "sha256": sha(root / "audio/provider-sample.wav")},
        "request": "private-request.json",
    })
    write_json(root / "private-request.json", {"key": "must-never-be-public"})
    return root


def prepared_fixture(root, project, runtime, name, dimensions=5):
    library = module._Library(project, runtime)
    entry = library.entry(root, verify=True)
    performance_bytes = library.performance(entry)
    performance = json.loads(performance_bytes)
    preview = runtime / "visual-previews" / name
    preview.mkdir(parents=True)
    shutil.copyfile(root / "audio/master.wav", preview / "audio.wav")
    performance["preview_id"] = name
    write_json(preview / "performance.json", performance)
    semantic = {
        "schema_version": 1, "preview_id": name, "audio_sha256": entry["audio_sha"],
        "duration_ms": entry["measured"]["duration_ms"], "model": "gemini-embedding-2",
        "space_id": "1" * 64,
        "projection": {"method": "pca", "dimensions": dimensions,
                       "fitted_count": len(performance["cues"]), "basis_sha256": "2" * 64},
        "windows": [{"id": "turn-" + str(index), "character_id": cue["speaker_id"],
                     "start_ms": cue["start_ms"], "end_ms": cue["end_ms"], "position": [0.0, 0.0, 0.0],
                     **({"rotation": 0.0} if dimensions >= 4 else {}),
                     **({"depth_rotation": 0.0} if dimensions == 5 else {})}
                    for index, cue in enumerate(performance["cues"])],
    }
    write_json(preview / "semantic.json", semantic)
    manifest = {"schema_version": 1, "preview_id": name, "title": entry["title"],
                "duration_ms": entry["measured"]["duration_ms"]}
    for role, name in [("audio", "audio.wav"), ("performance", "performance.json"), ("semantic", "semantic.json")]:
        path = preview / name
        manifest[role] = {"file": name, "sha256": sha(path), "size_bytes": path.stat().st_size}
    write_json(preview / "manifest.json", manifest)
    write_json(preview / "evidence/creation-source-v1.json", {
        "creation_root": root.relative_to(project).as_posix(),
        "source_manifest_sha256": sha(root / "manifest.json"), "audio_sha256": entry["audio_sha"],
        "script_sha256": sha(root / "script.json"), "timing_sha256": sha(root / "performance.json"),
    })
    return preview


@pytest.fixture
def rig(tmp_path):
    project = tmp_path / "project"
    runtime = project / ".runtime"
    first = episode(project / "creations/series/project/first/v1")
    legacy = episode(project / "creations/series/project/second/v2", legacy=True)
    sample = audition(project / "creations/voice-auditions/aster/sample-v1")
    prepared_fixture(first, project, runtime, "zz-fixture-first")
    prepared_fixture(legacy, project, runtime, "zz-fixture-legacy")
    app = FastAPI()
    module.register_creation_library(app, project, runtime)
    with TestClient(app) as client:
        yield SimpleNamespace(project=project, runtime=runtime, first=first,
                              legacy=legacy, sample=sample, client=client)


def snapshot(project):
    return {str(path.relative_to(project)): path.read_bytes()
            for path in project.rglob("*") if path.is_file()}


def test_get_catalog_is_read_only_lists_private_demos_and_uses_only_public_fields(rig):
    before = snapshot(rig.project)
    result = rig.client.get("/api/creation-library")
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    rows = result.json()["items"]
    assert result.json()["schema_version"] == 1 and len(rows) == 2
    assert {row["kind"] for row in rows} == {"episode", "voice_audition"}
    for row in rows:
        assert set(row) == {"id", "title", "version", "duration_ms", "kind", "playback_url", "semantic_ready", "playback_ready"}
        selected_root = next(root for root in (rig.first, rig.legacy, rig.sample)
                             if identity(root, rig.project) == row["id"])
        selected_id = (module._project_identity(selected_root, rig.project)
                       if row["kind"] == "episode" else row["id"])
        assert row["playback_url"] == "/?creation=" + selected_id
    for row in rows:
        metadata = rig.client.get("/api/creation-library/" + row["id"])
        assert metadata.status_code == 200
        assert row["playback_ready"] is True
        assert row["semantic_ready"] is (row["kind"] == "episode")
        assert (metadata.json()["semantic"] is not None) is row["semantic_ready"]
        assert metadata.json()["preview_id"] == row["id"]
        assert metadata.json()["local_preview"] is True
        for role in ("audio", "performance"):
            asset = metadata.json()[role]
            response = rig.client.get(asset["url"])
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == asset["sha256"]
            assert len(response.content) == asset["size_bytes"]
    payload = result.text + metadata.text + rig.client.get(metadata.json()["performance"]["url"]).text
    assert "must-never-be-public" not in payload and "Private transcript" not in payload
    assert str(rig.project) not in payload and "provider" not in payload
    assert snapshot(rig.project) == before
    assert not (rig.project / ".runtime/station").exists()


@pytest.mark.parametrize("name", ["first", "legacy"])
def test_frozen_colors_and_exact_cue_gaps_are_preserved(rig, name):
    identifier = identity(getattr(rig, name), rig.project)
    document = rig.client.get("/creation-library/" + identifier + "/performance.json").json()
    assert document["cast"] == [{"id": "aster", "color": "#508b82"},
                                {"id": "bramble", "color": "#937794"}]
    assert document["cues"] == [{"speaker_id": "aster", "start_ms": 100, "end_ms": 300},
                                {"speaker_id": "bramble", "start_ms": 500, "end_ms": 1000}]
    assert document["duration_ms"] == 1001 and document["features"] is None
    assert set(document) == {"schema_version", "preview_id", "audio_sha256", "duration_ms", "cast", "cues", "features"}


def test_audition_full_native_sample_has_one_known_character_and_stable_fallback(rig):
    identifier = identity(rig.sample, rig.project)
    metadata = rig.client.get("/api/creation-library/" + identifier).json()
    assert metadata["duration_ms"] == 750
    assert rig.client.get(metadata["audio"]["url"]).content == (rig.sample / "audio/provider-sample.wav").read_bytes()
    document = rig.client.get(metadata["performance"]["url"]).json()
    assert document["cast"] == [{"id": "aster", "color": _fallback_character_color("aster")}]
    assert document["cues"] == [{"speaker_id": "aster", "start_ms": 0, "end_ms": 750}]


def test_audio_ranges_and_fixed_asset_boundary(rig):
    identifier = identity(rig.first, rig.project)
    url = "/creation-library/" + identifier + "/audio.wav"
    response = rig.client.get(url, headers={"Range": "bytes=0-43"})
    assert response.status_code == 206
    assert response.content == (rig.first / "audio/master.wav").read_bytes()[:44]
    assert response.headers["content-range"].startswith("bytes 0-43/")
    for path in ["production.json", "voices.json", "private-request.json",
                 "provider-responses/private.json", "%2e%2e%2fproduction.json"]:
        assert rig.client.get("/creation-library/" + identifier + "/" + path).status_code == 404
    assert rig.client.get("/api/creation-library/not-a-real-id").status_code == 404
    assert rig.client.post("/api/creation-library").status_code == 405
    assert rig.client.get(url + "?path=production.json").status_code == 422
    assert rig.client.get("/api/creation-library?refresh=1").status_code == 422


def test_new_canonical_folder_appears_but_nested_checkpoint_does_not(rig):
    before = len(rig.client.get("/api/creation-library").json()["items"])
    new = episode(rig.project / "creations/series/new-project/new-work/v1")
    episode(rig.first / "provenance/checkpoints/ten/v1")
    items = rig.client.get("/api/creation-library").json()["items"]
    assert len(items) == before + 1
    assert identity(new, rig.project) in {item["id"] for item in items}


def test_audio_hash_cache_is_invalidated_even_when_size_and_mtime_are_restored(rig):
    identifier = identity(rig.first, rig.project)
    url = "/api/creation-library/" + identifier
    assert rig.client.get(url).status_code == 200
    path = rig.first / "audio/master.wav"
    before = path.stat()
    with path.open("r+b") as output:
        output.seek(-1, 2)
        output.write(b"\x01")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert rig.client.get(url).status_code == 409
    assert rig.client.get("/creation-library/" + identifier + "/audio.wav").status_code == 409


@pytest.mark.parametrize("mutation", ["audio-binding", "unknown-character", "cue-frame", "snapshot-binding"])
def test_changed_bindings_cannot_be_served_as_verified_audio(rig, mutation):
    root = rig.legacy if mutation == "snapshot-binding" else rig.first
    if mutation == "audio-binding":
        path = root / "production.json"
        value = json.loads(path.read_text())
        value["deliverables"]["master"]["sha256"] = "f" * 64
    elif mutation == "snapshot-binding":
        path = root / "provenance/character-revisions/aster.json"
        value = json.loads(path.read_text())
        value["spiral_color"] = "#123456"
    else:
        path = root / "performance.json"
        value = json.loads(path.read_text())
        value["cues"][0]["character_id" if mutation == "unknown-character" else "end_frame"] = "unknown" if mutation == "unknown-character" else 14401
    write_json(path, value)
    seal(root)
    identifier = identity(root, rig.project)
    assert rig.client.get("/api/creation-library/" + identifier).status_code == 409
    assert rig.client.get("/creation-library/" + identifier + "/audio.wav").status_code == 409


def test_symbolic_links_never_expose_external_sources(rig, tmp_path):
    outside = tmp_path / "outside.wav"
    wav(outside)
    audio = rig.first / "audio/master.wav"
    audio.unlink()
    audio.symlink_to(outside)
    linked = rig.project / "creations/series/project/linked/v1"
    linked.parent.mkdir()
    linked.symlink_to(rig.legacy, target_is_directory=True)
    rows = rig.client.get("/api/creation-library").json()["items"]
    assert identity(linked, rig.project) not in {row["id"] for row in rows}
    assert rig.client.get("/api/creation-library/" + identity(rig.first, rig.project)).status_code == 409


def test_prepared_matching_creation_reuses_existing_semantic_preview(rig, monkeypatch):
    root = episode(rig.project / "creations/series/prepared-project/prepared-test/v1")
    identifier = identity(root, rig.project)
    library = module._Library(rig.project, rig.runtime)
    entry = library.entry(root, verify=True)
    performance_bytes = library.performance(entry)
    metadata = library.metadata(entry, performance_bytes)
    performance = json.loads(performance_bytes)
    preview = rig.runtime / "visual-previews/prepared-fixture"
    preview.mkdir(parents=True)
    shutil.copyfile(root / "audio/master.wav", preview / "audio.wav")
    performance["preview_id"] = preview.name
    write_json(preview / "performance.json", performance)
    semantic = {"schema_version": 1, "preview_id": preview.name,
                "audio_sha256": sha(preview / "audio.wav"), "duration_ms": 1001,
                "model": "gemini-embedding-2", "space_id": "1" * 64,
                "projection": {"method": "pca", "dimensions": 5, "fitted_count": 2, "basis_sha256": "2" * 64},
                "windows": [{"id": "turn-" + str(index), "character_id": cue["speaker_id"],
                             "start_ms": cue["start_ms"], "end_ms": cue["end_ms"],
                             "position": [0.0, 0.0, 0.0], "rotation": 0.0, "depth_rotation": 0.0}
                            for index, cue in enumerate(performance["cues"])]}
    write_json(preview / "semantic.json", semantic)
    manifest = {key: value for key, value in metadata.items() if key not in ("kind", "local_preview")}
    manifest["preview_id"] = preview.name
    for role, name in [("audio", "audio.wav"), ("performance", "performance.json"), ("semantic", "semantic.json")]:
        path = preview / name
        manifest[role] = {"file": name, "sha256": sha(path), "size_bytes": path.stat().st_size}
    write_json(preview / "manifest.json", manifest)
    receipt = {"creation_root": root.relative_to(rig.project).as_posix(),
               "source_manifest_sha256": sha(root / "manifest.json"), "audio_sha256": sha(preview / "audio.wav")}
    write_json(preview / "evidence/creation-source-v1.json", receipt)
    before = snapshot(rig.project)
    rows = rig.client.get("/api/creation-library").json()["items"]
    assert next(row for row in rows if row["id"] == identifier)["playback_url"] == "/?creation=" + module._project_identity(root, rig.project)
    prepared_metadata = rig.client.get("/api/creation-library/" + identifier).json()
    assert prepared_metadata["semantic"] is not None
    for role in ("audio", "performance", "semantic"):
        asset = prepared_metadata[role]
        response = rig.client.get(asset["url"])
        assert response.status_code == 200
        assert len(response.content) == asset["size_bytes"]
        assert hashlib.sha256(response.content).hexdigest() == asset["sha256"]
        if role != "audio":
            assert response.json()["preview_id"] == identifier
    adapted = rig.client.get(prepared_metadata["semantic"]["url"]).json()
    assert adapted["projection"]["dimensions"] == 5
    assert all("rotation" in row and "depth_rotation" in row for row in adapted["windows"])
    assert rig.client.get(prepared_metadata["audio"]["url"]).content == (root / "audio/master.wav").read_bytes()
    assert snapshot(rig.project) == before
    calls = []
    original_reader = module.read_preview

    def tracked_reader(*args):
        calls.append(args)
        return original_reader(*args)

    monkeypatch.setattr(module, "read_preview", tracked_reader)
    rig.client.get("/api/creation-library/" + identifier)
    assert calls == []  # Unchanged fixed assets reuse verified semantic data.
    semantic_path = preview / "semantic.json"
    original_semantic = semantic_path.read_bytes()
    semantic_path.write_bytes(b"broken semantic payload")
    invalid = rig.client.get("/api/creation-library/" + identifier)
    assert invalid.status_code == 409 and "Visual preparation is required" in invalid.text
    assert len(calls) == 1  # A changed asset invalidates the prepared-preview cache.
    semantic_path.write_bytes(original_semantic)
    receipt["audio_sha256"] = "f" * 64
    write_json(preview / "evidence/creation-source-v1.json", receipt)
    mismatched = rig.client.get("/api/creation-library/" + identifier)
    assert mismatched.status_code == 409
    assert rig.client.get("/creation-library/" + identifier + "/semantic.json").status_code == 409



def test_audition_uses_authored_canonical_color_when_available(rig):
    write_json(rig.project / "characters/aster/character.json",
               {"character_id": "aster", "name": "Aster", "spiral_color": "#123456"})
    identifier = identity(rig.sample, rig.project)
    document = rig.client.get("/creation-library/" + identifier + "/performance.json").json()
    assert document["cast"] == [{"id": "aster", "color": "#123456"}]


def test_unprepared_programme_is_listed_but_cannot_silently_drop_its_visuals(rig):
    root = episode(rig.project / "creations/non-verba/unprepared/v1")
    identifier = identity(root, rig.project)
    before = snapshot(rig.project)
    row = next(row for row in rig.client.get("/api/creation-library").json()["items"] if row["id"] == identifier)
    assert row["semantic_ready"] is False and row["playback_ready"] is False
    for url in ["/api/creation-library/" + identifier,
                "/creation-library/" + identifier + "/audio.wav",
                "/creation-library/" + identifier + "/performance.json"]:
        response = rig.client.get(url)
        assert response.status_code == 409 and "Visual preparation is required" in response.text
    assert snapshot(rig.project) == before


def test_three_dimensional_preparation_is_not_mislabelled_as_ready_five_dimensions(rig):
    root = episode(rig.project / "creations/non-verba/older-projection/v1")
    prepared_fixture(root, rig.project, rig.runtime, "older-projection", dimensions=3)
    identifier = identity(root, rig.project)
    row = next(row for row in rig.client.get("/api/creation-library").json()["items"] if row["id"] == identifier)
    assert row["semantic_ready"] is False and row["playback_ready"] is False
    assert rig.client.get("/api/creation-library/" + identifier).status_code == 409


def test_curator_move_reuses_unchanged_preparation_without_reviving_the_old_path(rig):
    original = rig.first
    old_id = identity(original, rig.project)
    receipt_path = rig.runtime / "visual-previews/zz-fixture-first/evidence/creation-source-v1.json"
    retained_receipt = receipt_path.read_bytes()
    target = rig.project / "creations/non-verba/first/v1"
    target.parent.mkdir(parents=True)
    original.rename(target)
    before = snapshot(rig.project)
    identifier = identity(target, rig.project)
    rows = rig.client.get("/api/creation-library").json()["items"]
    assert old_id not in {row["id"] for row in rows}
    row = next(row for row in rows if row["id"] == identifier)
    assert row["semantic_ready"] is True and row["playback_ready"] is True
    metadata = rig.client.get("/api/creation-library/" + identifier)
    assert metadata.status_code == 200
    semantic = rig.client.get(metadata.json()["semantic"]["url"]).json()
    assert semantic["preview_id"] == identifier and semantic["projection"]["dimensions"] == 5
    assert receipt_path.read_bytes() == retained_receipt
    assert not original.exists()
    assert snapshot(rig.project) == before


@pytest.mark.parametrize("blocked", ["old-source-exists", "ambiguous-copy", "script-binding", "timing-binding"])
def test_relocation_reuse_requires_missing_old_source_and_unambiguous_exact_bindings(rig, blocked):
    target = rig.project / "creations/non-verba/first/v1"
    target.parent.mkdir(parents=True)
    if blocked == "old-source-exists":
        shutil.copytree(rig.first, target)
    else:
        rig.first.rename(target)
    if blocked == "ambiguous-copy":
        shutil.copytree(target, rig.project / "creations/non-verba/duplicate/v1")
    if blocked in ("script-binding", "timing-binding"):
        path = rig.runtime / "visual-previews/zz-fixture-first/evidence/creation-source-v1.json"
        value = json.loads(path.read_text())
        value["script_sha256" if blocked == "script-binding" else "timing_sha256"] = "f" * 64
        write_json(path, value)
    identifier = identity(target, rig.project)
    rows = rig.client.get("/api/creation-library").json()["items"]
    if blocked == "ambiguous-copy":
        assert identifier not in {row["id"] for row in rows}
        alias = module._project_identity(target, rig.project)
        assert rig.client.get("/api/creation-library/" + alias).status_code == 409
    else:
        row = next(row for row in rows if row["id"] == identifier)
        assert row["semantic_ready"] is False and row["playback_ready"] is False
    assert rig.client.get("/api/creation-library/" + identifier).status_code == 409


@pytest.mark.parametrize("hierarchy", [
    ("non-verba", "curated-story", "v1"),
    ("forest-series", "non-verba", "legacy-story", "v1"),
], ids=["curated-project-work-version", "legacy-series-project-work-version"])
def test_episode_tool_name_agrees_in_catalogue_and_player_without_rewriting_sources(rig, hierarchy):
    root = episode(rig.project / "creations" / Path(*hierarchy))
    preview = prepared_fixture(root, rig.project, rig.runtime, "naming-fixture")
    # The prepared title may retain an independently approved historical name.
    manifest_path = preview / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["title"] = "Preserved prepared programme title"
    write_json(manifest_path, manifest)
    identifier = identity(root, rig.project)
    before = snapshot(rig.project)

    catalog = rig.client.get("/api/creation-library")
    assert catalog.status_code == 200
    row = next(item for item in catalog.json()["items"] if item["id"] == identifier)
    metadata = rig.client.get("/api/creation-library/" + identifier)
    assert metadata.status_code == 200
    assert row["title"] == metadata.json()["title"] == "non-verba"
    audition_row = next(item for item in catalog.json()["items"] if item["kind"] == "voice_audition")
    assert audition_row["title"] == "Unapproved voice audition"
    assert json.loads((root / "production.json").read_text())["title"] != "non-verba"
    assert json.loads(manifest_path.read_text())["title"] == "Preserved prepared programme title"
    assert snapshot(rig.project) == before


def test_complete_podcast_production_uses_existing_bound_master_and_five_dimensional_protocol(rig):
    root = episode(rig.project / "creations/cuwacunu/what-should-capital-help-achieve/v2")
    production_path = root / "production.json"
    production = json.loads(production_path.read_text())
    production.update(kind="complete_podcast_production", complete=True, curated=False,
                      full_episode_audio_available=True, completed_turns=2, total_turns=2, remaining_turns=0)
    production["voices"]["sha256"] = sha(root / "voices.json")
    write_json(production_path, production)
    seal(root)
    identifier = identity(root, rig.project)

    # Curating a retained creation lists it without publishing it, and playback
    # still requires real prepared five-dimensional semantic data.
    unprepared = next(row for row in rig.client.get("/api/creation-library").json()["items"]
                      if row["id"] == identifier)
    assert unprepared["title"] == "cuwacunu"
    assert unprepared["version"] == "v2" and unprepared["kind"] == "episode"
    assert unprepared["semantic_ready"] is False and unprepared["playback_ready"] is False
    assert rig.client.get("/api/creation-library/" + identifier).status_code == 409

    prepared_fixture(root, rig.project, rig.runtime, "cuwacunu-fixture")
    before = snapshot(rig.project)
    row = next(row for row in rig.client.get("/api/creation-library").json()["items"]
               if row["id"] == identifier)
    assert row["semantic_ready"] is True and row["playback_ready"] is True
    metadata = rig.client.get("/api/creation-library/" + identifier)
    assert metadata.status_code == 200 and metadata.json()["title"] == "cuwacunu"
    assets = {role: rig.client.get(metadata.json()[role]["url"])
              for role in ("audio", "performance", "semantic")}
    assert all(response.status_code == 200 for response in assets.values())
    assert assets["audio"].content == (root / "audio/master.wav").read_bytes()
    assert assets["performance"].json()["cast"] == [{"id": "aster", "color": "#508b82"},
                                                    {"id": "bramble", "color": "#937794"}]
    assert assets["performance"].json()["cues"][-1]["end_ms"] == 1000
    assert assets["semantic"].json()["projection"]["dimensions"] == 5
    for role, response in assets.items():
        assert hashlib.sha256(response.content).hexdigest() == metadata.json()[role]["sha256"]
    assert json.loads(production_path.read_text())["curated"] is False
    assert snapshot(rig.project) == before


@pytest.mark.parametrize("name, expected", [("v0", 0), ("v2", 2), ("v10", 10), ("v02", 2),
                                           ("draft", None), ("v2-extra", None), ("2", None)])
def test_version_folder_numbers_are_numeric(name, expected):
    assert module._version_number(name) == expected
    assert module._version_number(Path("/fixture/work") / name) == expected


def test_highest_project_version_wins_across_work_folders_and_archives_are_excluded(rig):
    older = episode(rig.project / "creations/forest-series/numeric-tool/older-story/v2")
    newest = episode(rig.project / "creations/numeric-tool/different-story/v10")
    episode(rig.project / "archive/creations/numeric-tool/archived-story/v99")
    prepared_fixture(older, rig.project, rig.runtime, "numeric-v2")
    prepared_fixture(newest, rig.project, rig.runtime, "numeric-v10")
    old_id, new_id = identity(older, rig.project), identity(newest, rig.project)
    alias = module._project_identity(older, rig.project)
    assert alias == module._project_identity(newest, rig.project)
    assert alias.startswith("project-")
    assert module.latest_project_creation(older, rig.project) == newest
    rows = [row for row in rig.client.get("/api/creation-library").json()["items"]
            if row["title"] == "numeric-tool"]
    assert len(rows) == 1 and rows[0]["id"] == new_id and rows[0]["version"] == "v10"
    assert rows[0]["playback_url"] == "/?creation=" + alias
    metadata = rig.client.get("/api/creation-library/" + alias)
    assert metadata.status_code == 200 and metadata.json()["preview_id"] == new_id
    for role in ("audio", "performance", "semantic"):
        assert metadata.json()[role]["url"] == "/creation-library/" + new_id + "/" + metadata.json()[role]["file"]
        assert rig.client.get("/creation-library/" + alias + "/" + metadata.json()[role]["file"]).status_code == 404
    assert rig.client.get("/api/creation-library/" + old_id).status_code == 200


def test_latest_unprepared_version_is_pending_instead_of_using_older_ready_audio(rig):
    older = episode(rig.project / "creations/pending-tool/old-story/v2")
    newest = episode(rig.project / "creations/pending-tool/new-story/v10")
    prepared_fixture(older, rig.project, rig.runtime, "pending-tool-v2")
    alias = module._project_identity(newest, rig.project)
    rows = [row for row in rig.client.get("/api/creation-library").json()["items"]
            if row["title"] == "pending-tool"]
    assert len(rows) == 1 and rows[0]["id"] == identity(newest, rig.project)
    assert rows[0]["semantic_ready"] is False and rows[0]["playback_ready"] is False
    assert rig.client.get("/api/creation-library/" + alias).status_code == 409
    assert rig.client.get("/api/creation-library/" + identity(older, rig.project)).status_code == 200


def test_latest_invalid_version_cannot_expose_an_older_project_version(rig):
    older = episode(rig.project / "creations/invalid-tool/old-story/v2")
    newest = episode(rig.project / "creations/invalid-tool/new-story/v10")
    prepared_fixture(older, rig.project, rig.runtime, "invalid-tool-v2")
    (newest / "production.json").write_bytes(b"{")
    rows = rig.client.get("/api/creation-library").json()["items"]
    assert not any(row["title"] == "invalid-tool" for row in rows)
    assert any(row["id"] == identity(rig.legacy, rig.project) for row in rows)
    assert rig.client.get("/api/creation-library/" + module._project_identity(newest, rig.project)).status_code == 409
    assert rig.client.get("/api/creation-library/" + identity(older, rig.project)).status_code == 200


def test_equal_highest_versions_are_ambiguous_only_for_their_project(rig):
    first = episode(rig.project / "creations/tied-tool/first-story/v2")
    second = episode(rig.project / "creations/forest-series/tied-tool/second-story/v02")
    alias = module._project_identity(first, rig.project)
    rows = rig.client.get("/api/creation-library").json()["items"]
    assert not any(row["title"] == "tied-tool" for row in rows)
    assert any(row["id"] == identity(rig.legacy, rig.project) for row in rows)
    assert rig.client.get("/api/creation-library/" + alias).status_code == 409
    with pytest.raises(module.CreationLibraryError, match="ambiguous"):
        module.latest_project_creation(second, rig.project)


def test_nonversion_episode_folders_remain_exactly_addressable_but_are_not_catalogued(rig):
    draft = episode(rig.project / "creations/draft-tool/a-story/draft")
    prepared_fixture(draft, rig.project, rig.runtime, "draft-exact")
    identifier = identity(draft, rig.project)
    assert identifier not in {row["id"] for row in rig.client.get("/api/creation-library").json()["items"]}
    assert rig.client.get("/api/creation-library/" + identifier).status_code == 200
    assert rig.client.get("/api/creation-library/" + module._project_identity(draft, rig.project)).status_code == 404


def test_project_metadata_pins_assets_when_a_new_version_appears_later(rig):
    older = episode(rig.project / "creations/pinned-tool/old-story/v2")
    prepared_fixture(older, rig.project, rig.runtime, "pinned-v2")
    alias = module._project_identity(older, rig.project)
    old_metadata = rig.client.get("/api/creation-library/" + alias).json()
    newer = episode(rig.project / "creations/pinned-tool/new-story/v10")
    audio = newer / "audio/master.wav"
    with audio.open("r+b") as output:
        output.seek(-1, 2)
        output.write(b"\x01")
    timing_path = newer / "performance.json"
    timing = json.loads(timing_path.read_text())
    timing["audio_sha256"] = sha(audio)
    write_json(timing_path, timing)
    production_path = newer / "production.json"
    production = json.loads(production_path.read_text())
    production["deliverables"]["master"]["sha256"] = sha(audio)
    production["deliverables"]["performance"]["sha256"] = sha(timing_path)
    write_json(production_path, production)
    seal(newer)
    prepared_fixture(newer, rig.project, rig.runtime, "pinned-v10")
    before = snapshot(rig.project)

    new_metadata = rig.client.get("/api/creation-library/" + alias).json()
    assert old_metadata["preview_id"] == identity(older, rig.project)
    assert new_metadata["preview_id"] == identity(newer, rig.project)
    assert old_metadata["audio"]["sha256"] != new_metadata["audio"]["sha256"]
    for metadata in (old_metadata, new_metadata):
        for role in ("audio", "performance", "semantic"):
            response = rig.client.get(metadata[role]["url"])
            assert response.status_code == 200
            assert hashlib.sha256(response.content).hexdigest() == metadata[role]["sha256"]
    assert snapshot(rig.project) == before


def test_public_version_comes_from_folder_not_retained_metadata(rig):
    root = episode(rig.project / "creations/folder-tool/a-story/v10")
    production_path = root / "production.json"
    production = json.loads(production_path.read_text())
    production["creative_version"] = "v2"
    write_json(production_path, production)
    seal(root)
    row = next(row for row in rig.client.get("/api/creation-library").json()["items"]
               if row["id"] == identity(root, rig.project))
    assert row["version"] == "v10"
    assert json.loads(production_path.read_text())["creative_version"] == "v2"
