"""A stale PID record must never authorize signalling another process."""
import json
import pytest
from web_radio import control

def test_reused_pid_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CONTROL", tmp_path)
    (tmp_path / "studio.json").write_text(json.dumps({"pid": 9001, "start_ticks": 100}))
    monkeypatch.setattr(control, "_process_identity", lambda pid: ([b"python", b"-m", b"uvicorn", b"web_radio.studio:app"], 200))
    signals = []
    monkeypatch.setattr(control.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="stale PID"):
        control.stop("studio")
    assert signals == []

def test_unmanaged_command_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "CONTROL", tmp_path)
    (tmp_path / "station.json").write_text(json.dumps({"pid": 9002, "start_ticks": 100}))
    monkeypatch.setattr(control, "_process_identity", lambda pid: ([b"python", b"other.py"], 100))
    signals = []
    monkeypatch.setattr(control.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="does not match"):
        control.stop("station")
    assert signals == []
