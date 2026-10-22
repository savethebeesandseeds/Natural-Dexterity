"""Private station controls: digest-bound, revisioned, boundary-applied, and offline durable."""
import json
import shutil
import time

import pytest

from web_radio.playout import Playout
from web_radio.rundown import (ControlError, Rundown, parse_rundown, CONTROL_LIMIT,
                               CATALOG_LIMIT, CATALOG_ENTRIES)
from test_station import publish, wait_until


def ref(manifest):
    return {"release_id": manifest.release_id, "package_sha256": manifest.package_sha256}


def command(runtime, programmes, fallback=None, *, revision=1, mode="programmes"):
    data = {"schema_version": 1, "revision": revision, "mode": mode,
            "programme_releases": [ref(item) for item in programmes],
            "fallback_release": ref(fallback) if fallback else None}
    path = runtime / "library" / "station-control.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def ack(runtime):
    return json.loads((runtime / "library" / "station-control-status.json").read_text())


@pytest.mark.parametrize("patch", [
    {"revision": True}, {"revision": 0}, {"schema_version": True},
    {"mode": "paused"}, {"programme_releases": []},
    {"unexpected": "field"},
])
def test_rundown_rejects_ambiguous_or_coerced_controls(patch):
    good = {"schema_version": 1, "revision": 1, "mode": "programmes",
            "programme_releases": [{"release_id": "fixture", "package_sha256": "a" * 64}],
            "fallback_release": None}
    with pytest.raises(ControlError):
        parse_rundown(json.dumps(good | patch).encode())


def test_rundown_rejects_duplicate_json_keys_and_size_overflow():
    with pytest.raises(ControlError):
        parse_rundown(b'{"schema_version":1,"revision":1,"revision":2}')
    with pytest.raises(ControlError, match="64 KiB"):
        parse_rundown(b" " * (CONTROL_LIMIT + 1))


def test_fallback_mode_requires_explicit_ident_and_programmes_are_unique():
    base = {"schema_version": 1, "revision": 1, "mode": "fallback",
            "programme_releases": [], "fallback_release": None}
    with pytest.raises(ControlError):
        parse_rundown(json.dumps(base).encode())
    entry = {"release_id": "fixture", "package_sha256": "a" * 64}
    with pytest.raises(ControlError):
        parse_rundown(json.dumps(base | {"mode": "programmes", "programme_releases": [entry, entry]}).encode())


def test_order_idempotence_and_catalogue_updates_do_not_override_operator_rundown(tmp_path):
    runtime = tmp_path / "runtime"
    first, _ = publish(runtime, tmp_path, release_id="first-v1")
    second, _ = publish(runtime, tmp_path, release_id="second-v1")
    ident, _ = publish(runtime, tmp_path, ident=True)
    station = Playout(runtime)
    station.refresh()
    requested = command(runtime, [second, first], ident)
    station.refresh()
    assert ack(runtime)["status"] == "accepted"
    assert station.schedule_ids == [second.release_id, first.release_id]
    assert station._select()[0].release_id == second.release_id
    # Formatting-only re-delivery of an accepted revision must not reset its cursor.
    (runtime / "library" / "station-control.json").write_text(json.dumps(requested, indent=2))
    station.refresh()
    assert station._select()[0].release_id == first.release_id
    assert len([e for e in station.journal.events() if e["event"] == "control_accepted"]) == 1
    third, _ = publish(runtime, tmp_path, release_id="third-v1")
    station.refresh()
    assert third.release_id in station.known
    assert station.schedule_ids == [second.release_id, first.release_id]
    assert station._select()[0].release_id == second.release_id


def test_invalid_unknown_digest_role_and_conflicting_revisions_preserve_accepted_config(tmp_path):
    runtime = tmp_path / "runtime"
    first, _ = publish(runtime, tmp_path, release_id="first-v1")
    second, _ = publish(runtime, tmp_path, release_id="second-v1")
    ident, _ = publish(runtime, tmp_path, ident=True)
    station = Playout(runtime)
    command(runtime, [first], ident, revision=3)
    station.refresh()
    original = station.control.model_dump(mode="json")
    conflicting = command(runtime, [second], ident, revision=3)
    station.refresh()
    assert "conflicts" in ack(runtime)["reason"]
    assert ack(runtime)["accepted_revision"] == 3
    assert station.control.model_dump(mode="json") == original
    command(runtime, [second], ident, revision=2)
    station.refresh()
    assert "older" in ack(runtime)["reason"]
    for invalid in [
        {"release_id": "unknown-v1", "package_sha256": "a" * 64},
        {"release_id": second.release_id, "package_sha256": "b" * 64},
        ref(ident),
    ]:
        requested = conflicting | {"revision": 4, "programme_releases": [invalid]}
        (runtime / "library" / "station-control.json").write_text(json.dumps(requested))
        station.refresh()
        assert ack(runtime)["status"] == "rejected"
        assert ack(runtime)["accepted_revision"] == 3
        assert station.control.model_dump(mode="json") == original
        assert station.schedule_ids == [first.release_id]


def test_fallback_selection_and_control_survive_missing_studio_and_station_restart(tmp_path):
    runtime = tmp_path / "runtime"
    programme, _ = publish(runtime, tmp_path)
    first_ident, _ = publish(runtime, tmp_path, ident=True, release_id="first-ident")
    selected_ident, _ = publish(runtime, tmp_path, ident=True, release_id="selected-ident")
    station = Playout(runtime)
    command(runtime, [programme], selected_ident, revision=7, mode="fallback")
    station.refresh()
    assert station._select()[0].release_id == selected_ident.release_id
    assert station.selection_reason == "operator_selected_fallback"
    # A task-owned library disappears; the station has its own verified copies + config.
    shutil.rmtree(runtime / "library")
    restarted = Playout(runtime)
    restarted.refresh()
    assert restarted.control.revision == 7
    assert restarted.control.mode == "fallback"
    assert restarted._select()[0].release_id == selected_ident.release_id
    # A different cached ident is not silently substituted for a corrupt explicit choice.
    selected_package = restarted.known[selected_ident.release_id][1]
    audio = next(a for a in selected_ident.assets if a.role == "audio")
    (selected_package / audio.path).write_bytes(b"corrupt task-created fixture")
    assert first_ident.release_id in restarted.known
    assert restarted._select() is None
    assert restarted.selection_reason == "fallback_unavailable"


def test_metadata_resource_limits_preserve_the_accepted_rundown(tmp_path):
    runtime = tmp_path / "runtime"
    programme, _ = publish(runtime, tmp_path)
    station = Playout(runtime)
    command(runtime, [programme], revision=1)
    station.refresh()
    accepted = station.control.model_dump(mode="json")
    catalog = runtime / "library" / "catalog.json"
    catalog.write_bytes(b" " * (CATALOG_LIMIT + 1))
    station.refresh()
    assert station.schedule_ids == [programme.release_id]
    entries = [{"release_id": f"r-{n}", "package_sha256": "a" * 64}
               for n in range(CATALOG_ENTRIES + 1)]
    catalog.write_text(json.dumps({"schema_version": 1, "releases": entries}))
    station.refresh()
    assert station.schedule_ids == [programme.release_id]
    (runtime / "library" / "station-control.json").write_bytes(b" " * (CONTROL_LIMIT + 1))
    station.refresh()
    assert ack(runtime)["status"] == "rejected"
    assert station.control.model_dump(mode="json") == accepted
    assert station._select()[0].release_id == programme.release_id


def test_live_control_acknowledges_during_programme_and_switches_at_boundary(tmp_path):
    runtime = tmp_path / "runtime"
    programme, _ = publish(runtime, tmp_path, seconds=5)
    ident, _ = publish(runtime, tmp_path, ident=True, seconds=0.5)
    station = Playout(runtime)
    station.start()
    try:
        wait_until(lambda: any(e["event"] == "started" for e in station.journal.events()))
        session_id = station.session_id
        command(runtime, [programme], ident, revision=1, mode="fallback")
        wait_until(lambda: (runtime / "library" / "station-control-status.json").exists())
        assert ack(runtime)["status"] == "accepted"
        # Acknowledgement must not interrupt the programme already being emitted.
        assert station.playback_kind == "programme"
        assert station.state == "broadcasting"
        wait_until(lambda: station.state == "fallback", timeout=8)
        assert station.session_id == session_id
        assert station.snapshot()["mode"] == "fallback"
        assert station.snapshot()["fallback_reason"] == "operator_selected_fallback"
        completed = station.journal.last_completed()
        assert completed["release_id"] == programme.release_id
        assert not any(e["event"] == "interrupted" for e in station.journal.events())
        # Accepted control continues when the studio's publication files disappear.
        shutil.rmtree(runtime / "library")
        prior = len([e for e in station.journal.events() if e["event"] == "fallback"])
        wait_until(lambda: len([e for e in station.journal.events() if e["event"] == "fallback"]) > prior)
        assert station.state == "fallback"
        assert station.session_id == session_id
    finally:
        station.stop()
    assert not station._updates_thread.is_alive()


def test_public_station_has_no_mutating_control_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from web_radio.station import create_app
    with TestClient(create_app(tmp_path / "runtime", start_engine=False)) as client:
        response = client.put("/api/station-control", json={"mode": "fallback"})
        assert response.status_code in (404, 405)
        state = client.get("/api/status").json()
        assert state["state"] == "waiting"
        assert state["waiting_reason"] == "no_released_programmes"
