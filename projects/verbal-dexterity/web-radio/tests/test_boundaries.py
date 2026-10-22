"""Resource and trust boundary regressions; all media and servers are fixtures."""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import wave

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from web_radio import contract


@pytest.fixture
def short_wav(tmp_path):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 4800)
    return path


@pytest.fixture
def manifest(tmp_path, short_wav):
    return contract.create_release(
        short_wav, "TEST FIXTURE transcript.", tmp_path / "valid-release",
        "boundary-fixture", "Synthetic boundary fixture",
        release_id="boundary-fixture-v1",
    )


@pytest.mark.parametrize(("field", "value"), [
    ("audio_size", contract.MAX_AUDIO_BYTES + 1),
    ("transcript_size", contract.MAX_TRANSCRIPT_BYTES + 1),
    ("duration", contract.MAX_DURATION_MS + 1),
])
def test_manifest_resource_caps_agree_in_python_and_json_schema(manifest, field, value):
    data = manifest.model_dump(mode="json")
    if field == "duration":
        data["duration_ms"] = value
    else:
        role = "audio" if field == "audio_size" else "transcript"
        next(asset for asset in data["assets"] if asset["role"] == role)["size_bytes"] = value
    with pytest.raises(ValidationError):
        contract.Manifest.model_validate(data)
    schema_path = Path(__file__).resolve().parents[1] / "packages/media-contract/manifest.schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    assert list(validator.iter_errors(data))


def test_oversized_source_is_rejected_before_copy_or_decode(tmp_path, monkeypatch):
    source = tmp_path / "oversized.wav"
    # Sparse fixture: large logical size without allocating or reading 128 MiB.
    with source.open("wb") as output:
        output.truncate(contract.MAX_AUDIO_BYTES + 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized input reached copying or decoding")
    monkeypatch.setattr(contract, "_copy_bounded", forbidden)
    monkeypatch.setattr(contract.subprocess, "run", forbidden)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(contract.ContractError, match="128 MiB"):
        contract.create_release(source, "TEST FIXTURE", destination, "fixture", "Fixture")
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_transcript_cap_counts_utf8_bytes_before_publication(tmp_path, short_wav):
    # Character count is below 1 MiB, but encoded bytes exceed the 1 MiB limit.
    transcript = "\U0001f98a" * (contract.MAX_TRANSCRIPT_BYTES // 4 + 1)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(contract.ContractError, match="1 MiB"):
        contract.create_release(short_wav, transcript, destination, "fixture", "Fixture")
    assert not destination.exists()


def test_transcript_read_rejects_growth_beyond_limit(tmp_path):
    transcript = tmp_path / "transcript.txt"
    transcript.write_bytes(b"x" * (contract.MAX_TRANSCRIPT_BYTES + 1))
    with pytest.raises(contract.ContractError, match="1 MiB"):
        contract._read_transcript(transcript)


def test_copy_and_hash_limits_cover_changed_files(tmp_path):
    source = tmp_path / "growing.bin"
    source.write_bytes(b"x" * 33)
    destination = tmp_path / "copy.bin"
    with pytest.raises(contract.ContractError, match="byte limit"):
        contract._copy_bounded(source, destination, 32)
    assert destination.stat().st_size <= 32
    with pytest.raises(contract.ContractError, match="byte limit"):
        contract._sha256_file(source, max_bytes=32)


def test_duration_is_bounded_before_full_decode(short_wav, monkeypatch):
    calls = []
    def probe_only(command, **kwargs):
        calls.append(command[0])
        if command[0] != "ffprobe":
            raise AssertionError("Overlong media reached full decoding")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le",
                         "channels": 1, "sample_rate": "24000"}],
            "format": {"format_name": "wav", "duration": "1800.001"},
        }), stderr="")
    monkeypatch.setattr(contract.subprocess, "run", probe_only)
    with pytest.raises(contract.ContractError, match="30 minute"):
        contract._audio_duration_ms(short_wav)
    assert calls == ["ffprobe"]


def test_disguised_playlist_never_fetches_external_media(tmp_path):
    requests = []
    class SegmentHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not valid transport-stream data")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), SegmentHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    source = tmp_path / "disguised.wav"
    source.write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n"
        "#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:2,\n"
        f"http://127.0.0.1:{server.server_port}/segment.ts\n#EXT-X-ENDLIST\n"
    )
    try:
        with pytest.raises(contract.ContractError, match="audio"):
            contract.create_release(
                source, "TEST FIXTURE transcript.", tmp_path / "rejected",
                "playlist-fixture", "Disguised playlist fixture",
            )
        assert requests == []
        assert not (tmp_path / "rejected").exists()
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
