from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from uuid import uuid4
import wave

import pytest

from web_radio.characters import CharacterStore, CharacterError, VoiceRecipe
from web_radio.gemini import Script, parse_audio_block

SOURCE = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.8-flash-lite-tts"


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\x00\x00" * 2400)
    return output.getvalue()


def original_response(voice="voice_original_fixture"):
    return {"id": voice, "type": "prompted", "model": MODEL,
            "sample_audio": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes()).decode()}}


def original_request():
    return {"store": True, "voice": {
        "model": MODEL, "type": "prompted", "display_name": "Moss",
        "language_code": "en", "prompted": {"input": "A gentle woodland creature with a low textured voice."},
    }}


@pytest.fixture
def store(tmp_path):
    service = CharacterStore(tmp_path)
    service.create({"character_id": "moss", "name": "Moss", "personality": "Patient", "pronunciation_notes": "Wren is ren."})
    return service


def script():
    return Script.model_validate_json((SOURCE / "content/series/first-light/programme.json").read_text())


def register_original(store):
    response = original_response()
    return store._register_designed("moss", request=original_request(), response=response,
                                    audio=parse_audio_block(response["sample_audio"]), job_id=uuid4().hex)


def test_character_history_and_optimistic_updates_are_immutable(store):
    first = store.get("moss")
    second = store.update("moss", {"personality": "Patient and curious"}, expected_revision=1)
    assert second["revision"] == 2
    assert store.get("moss", 1) == first
    assert store.list() == [second]
    with pytest.raises(CharacterError, match="current revision"):
        store.update("moss", {"personality": "Stale overwrite"}, expected_revision=1)
    with pytest.raises(CharacterError, match="already exists"):
        store.create({"character_id": "moss", "name": "Another Moss"})
    assert store.get("moss") == second
    assert len(list((store.state / "moss/revisions").glob("*.json"))) == 2


def test_prebuilt_recipes_and_cast_pin_identity_separately_from_scene_alias(store):
    first = store.add_prebuilt("moss", "Charon", MODEL)
    pinned_character = store.get("moss")
    document = script()
    document.speakers[0].character_id = "moss"
    document.speakers[0].name = "The Old One"
    document.turns = [turn.model_copy(update={"speaker": "The Old One"}) if turn.speaker == "Moss" else turn
                      for turn in document.turns]
    bound, snapshot = store.bind_cast(document)
    assert bound.speakers[0].voice == "Charon"
    assert bound.speakers[0].name == "The Old One"
    assert bound.speakers[0].character_revision == pinned_character["revision"]
    assert bound.speakers[0].voice_recipe_id == first["recipe_id"]
    assert bound.speakers[0].pronunciation_notes == "Wren is ren."
    store.add_prebuilt("moss", "Kore", MODEL)
    historical, historical_snapshot = store.bind_cast(bound)
    assert historical.speakers[0].voice == "Charon"
    assert historical_snapshot == snapshot
    assert store.get("moss")["preferred_voice_recipe_id"] != first["recipe_id"]
    assert store.get_recipe("moss", first["recipe_id"]) == first


@pytest.mark.parametrize("voice", ["voice_unproven", "voicekey_replica", "UnknownVoice"])
def test_prebuilt_registration_rejects_custom_or_unknown_ids(store, voice):
    with pytest.raises(ValueError):
        store.add_prebuilt("moss", voice, MODEL)
    assert store.recipes("moss") == []
    assert store.get("moss")["revision"] == 1


@pytest.mark.parametrize("identity", ["../escape", "Moss", "/tmp/moss", "moss/other", ""])
def test_character_paths_require_stable_safe_ids(store, identity):
    with pytest.raises(ValueError):
        store.get(identity)


def test_designed_recipe_preserves_prompt_response_audition_and_no_arbitrary_origin(store):
    recipe = register_original(store)
    assert recipe["origin"] == "original_design"
    assert recipe["design_prompt"] == original_request()["voice"]["prompted"]["input"]
    assert recipe["expiry_basis"] == "documented_ttl_conservative"
    assert store.approved_designed_voices() == {"voice_original_fixture"}
    document = script()
    document.speakers[0].character_id = "moss"
    bound, snapshot = store.bind_cast(document)
    assert bound.speakers[0].voice == "voice_original_fixture"
    assert snapshot["speakers"][0]["voice_recipe"] == recipe
    assert store.artifact(recipe["audition_path"]).read_bytes() == wav_bytes()


def test_tampered_audition_blocks_binding_and_approval(store):
    recipe = register_original(store)
    store.artifact(recipe["audition_path"]).write_bytes(b"changed")
    assert store.approved_designed_voices() == set()
    document = script()
    document.speakers[0].character_id = "moss"
    with pytest.raises(CharacterError, match="verification"):
        store.bind_cast(document)


def test_replication_response_is_never_registered(store):
    response = original_response()
    response["type"] = "replicated"
    with pytest.raises(CharacterError, match="description-designed"):
        store._register_designed("moss", request=original_request(), response=response,
                                 audio=wav_bytes(), job_id=uuid4().hex)
    assert store.recipes("moss") == []


def test_raw_designed_voice_without_character_reference_is_rejected(store):
    register_original(store)
    document = script()
    document.speakers[0].voice = "voice_original_fixture"
    with pytest.raises(CharacterError, match="registered character"):
        store.bind_cast(document)


def test_audition_paths_cannot_escape_or_follow_symlinks(store, tmp_path):
    for path in ("../secret", "/tmp/secret", "moss\\outside"):
        with pytest.raises(CharacterError):
            store.artifact(path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.state / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CharacterError, match="symlink"):
        store.write_artifact("link/unsafe.json", b"{}")


def test_expired_recipe_remains_inspectable_but_cannot_synthesize(store):
    response = original_response()
    response["expire_time"] = "2099-01-01T00:00:00+00:00"
    recipe = store._register_designed("moss", request=original_request(), response=response,
                                     audio=wav_bytes(), job_id=uuid4().hex)
    with store.connect() as db:
        expired = {**recipe, "expires_at": "2000-01-01T00:00:00+00:00"}
        db.execute("UPDATE recipes SET payload=? WHERE recipe_id=?",
                   (json.dumps(expired), recipe["recipe_id"]))
    assert store.get_recipe("moss", recipe["recipe_id"])["expires_at"].startswith("2000")
    assert store.approved_designed_voices() == set()
    document = script()
    document.speakers[0].character_id = "moss"
    with pytest.raises(CharacterError, match="expired"):
        store.bind_cast(document)
