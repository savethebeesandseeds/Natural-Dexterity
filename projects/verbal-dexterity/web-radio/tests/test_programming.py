"""Private programming enforces trusted releases and optimistic updates."""
import json
import math
import struct
import wave
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from web_radio.contract import create_release
from web_radio.station_controls import make_router

def setup_catalog(root):
    library = root / ".runtime" / "library"
    releases = []
    for kind in ("programme", "ident"):
        source = root / (kind + ".wav")
        with wave.open(str(source), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000)
            output.writeframes(b"".join(struct.pack("<h",int(1200*math.sin(i/20))) for i in range(6000)))
        manifest = create_release(source, "Explicit synthetic test recording.", library / "releases" / kind,
                                  programme_id=kind, release_id=kind, title="Test " + kind, kind=kind)
        releases.append({"release_id":manifest.release_id,"package_sha256":manifest.package_sha256})
    (library / "catalog.json").write_text(json.dumps({"schema_version":1,"releases":releases}))
    return releases

def client(root):
    app = FastAPI()
    app.include_router(make_router(root))
    return TestClient(app)

def change(refs, revision=0):
    return {"expected_revision": revision, "mode":"programmes",
            "programme_releases":[refs[0]],"fallback_release":refs[1]}

def test_submission_requires_valid_roles_and_bytes(tmp_path):
    refs = setup_catalog(tmp_path)
    with client(tmp_path) as http:
        listing = http.get("/api/programming").json()
        assert {r["kind"] for r in listing["releases"]} == {"programme","ident"}
        wrong = change(refs); wrong["fallback_release"] = refs[0]
        assert http.put("/api/programming",json=wrong).status_code == 422
        assert not (tmp_path / ".runtime/library/station-control.json").exists()
        accepted = http.put("/api/programming",json=change(refs))
        assert accepted.status_code == 200
        assert accepted.json()["applies"] == "next-airing-boundary"
        original = (tmp_path / ".runtime/library/station-control.json").read_bytes()
        audio = tmp_path / ".runtime/library/releases/programme/audio.wav"
        audio.write_bytes(audio.read_bytes() + b"tampered")
        assert http.put("/api/programming",json=change(refs,1)).status_code == 422
        assert (tmp_path / ".runtime/library/station-control.json").read_bytes() == original

def test_stale_window_cannot_overwrite_newer_rundown(tmp_path):
    refs=setup_catalog(tmp_path)
    with client(tmp_path) as http:
        assert http.put("/api/programming",json=change(refs)).status_code==200
        assert http.put("/api/programming",json=change(refs)).status_code==409
        assert http.get("/api/programming").json()["revision"]==1

def test_fallback_mode_needs_verified_ident(tmp_path):
    refs=setup_catalog(tmp_path)
    with client(tmp_path) as http:
        body={"expected_revision":0,"mode":"fallback","programme_releases":[],"fallback_release":None}
        assert http.put("/api/programming",json=body).status_code==422
        body["fallback_release"]=refs[1]
        assert http.put("/api/programming",json=body).status_code==200

def test_catalogue_and_command_limits_fail_closed(tmp_path,monkeypatch):
    import web_radio.station_controls as controls
    refs=setup_catalog(tmp_path)
    with client(tmp_path) as http:
        command=tmp_path / ".runtime/library/station-control.json"
        command.write_text('{"schema_version":1,"schema_version":1}')
        assert http.put("/api/programming",json=change(refs)).status_code==409
        assert len(command.read_bytes())>0
        monkeypatch.setattr(controls,"CATALOG_LIMIT",8)
        listing=http.get("/api/programming").json()
        assert listing["releases"]==[]
        assert listing["problems"]

def test_duplicate_catalogue_identity_cannot_hide_behind_bad_digest(tmp_path):
    refs=setup_catalog(tmp_path)
    bad=dict(refs[0],package_sha256="0"*64)
    catalogue=tmp_path / ".runtime/library/catalog.json"
    catalogue.write_text(json.dumps({"schema_version":1,"releases":[bad,*refs]}))
    with client(tmp_path) as http:
        state=http.get("/api/programming").json()
        assert state["releases"]==[]
        assert state["problems"]
        assert http.put("/api/programming",json=change(refs)).status_code==422

def test_public_station_has_no_programming_mutation_route(tmp_path):
    from web_radio.station import create_app
    with TestClient(create_app(tmp_path / ".runtime",start_engine=False)) as http:
        assert http.get("/api/programming").status_code==404
        assert http.put("/api/programming",json={}).status_code==404

def test_lost_command_recovers_above_station_accepted_revision(tmp_path):
    refs=setup_catalog(tmp_path)
    acknowledgement=tmp_path / ".runtime/library/station-control-status.json"
    acknowledgement.write_text(json.dumps({"schema_version":1,"requested_revision":5,"accepted_revision":5,
                                           "status":"accepted","reason":None,"updated_at":"2026-10-22T00:00:00Z"}))
    with client(tmp_path) as http:
        assert http.get("/api/programming").json()["revision"]==5
        assert http.put("/api/programming",json=change(refs,0)).status_code==409
        response=http.put("/api/programming",json=change(refs,5))
        assert response.status_code==200
        assert response.json()["revision"]==6
