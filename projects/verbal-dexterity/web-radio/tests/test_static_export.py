"""Static publication preserves approved media and excludes private production data."""
import hashlib
import io
import tarfile
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_creation_library import episode, prepared_fixture, snapshot, write_json
from web_radio import creation_library
from web_radio import static_export as module


@pytest.fixture
def rig(tmp_path):
    project = tmp_path / "project"
    runtime = project / ".runtime"
    old = episode(project / "creations/non-verba/conversation/v1")
    current = episode(project / "creations/non-verba/conversation/v2")
    prepared_fixture(old, project, runtime, "retained-old-preview")
    prepared_fixture(current, project, runtime, "current-preview")
    # An unrelated retained preview and secret must never enter the public build.
    write_json(runtime / "visual-previews/private-preview/provider-response.json",
               {"secret": "must-never-be-public"})
    (runtime / "secrets").mkdir()
    (runtime / "secrets/provider-key").write_text("must-never-be-public")
    alias = creation_library._project_identity(current, project)
    player = project / "web/player"
    player.mkdir(parents=True)
    (player / "index.html").write_text(
        '<!doctype html><html><head><title>Fixture</title><link rel="icon" href="/favicon.ico"></head><body><main data-default-recording="' + alias + '"></main></body></html>')
    for name in (*module.PLAYER_ASSETS, *module.ROOT_PLAYER_ASSETS):
        (player / name).write_bytes(("public fixture " + name).encode())
    (player / "private-debug.txt").write_text("must-never-be-public")
    for name in module.PUBLIC_NOTICES:
        (project / name).write_text("public notice " + name)
    (project / "web/static").mkdir()
    (project / "web/static/_headers").write_text("/data/*\n  Cache-Control: no-cache\n")
    (project / "web/static/private-config.json").write_text("must-never-be-public")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "hls.min.js").write_bytes(b"/* pinned fixture HLS */")
    (vendor / "HLS-LICENSE.txt").write_text("public vendor license")
    (vendor / "hls.sha256").write_text(hashlib.sha256((vendor / "hls.min.js").read_bytes()).hexdigest())
    write_json(project / "creations/catalogue.json", {
        "schema_version": 2, "creations": [{"curated": True, "complete": True,
            "listener_approval": "approved", "version": "v2", "bundle": "non-verba/conversation/v2"}]})
    return SimpleNamespace(project=project, runtime=runtime, old=old, current=current,
                           alias=alias, vendor=vendor, output=tmp_path / "export")


def export(rig, output=None):
    return module.export_static(output or rig.output, project=rig.project,
                                runtime=rig.runtime, vendor=rig.vendor)


def test_export_is_deterministic_uses_latest_curated_audio_and_contains_only_public_data(rig):
    before = snapshot(rig.project)
    receipt = export(rig)
    second = rig.output.with_name("second-export")
    export(rig, second)
    assert snapshot(rig.output) == snapshot(second)
    assert snapshot(rig.project) == before
    assert receipt["ok"] and receipt["recordings"] == 1
    catalogue = json.loads((rig.output / "data/library.json").read_bytes())
    item, = catalogue["items"]
    assert item["version"] == "v2" and item["playback_ready"] is True
    assert item["playback_url"] == "/?creation=" + rig.alias
    canonical = item["id"]
    metadata_path = rig.output / ("data/recordings/" + canonical + ".json")
    alias_path = rig.output / ("data/recordings/" + rig.alias + ".json")
    assert alias_path.read_bytes() == metadata_path.read_bytes()
    metadata = json.loads(metadata_path.read_bytes())
    assert metadata["preview_id"] == canonical
    assert (rig.output / metadata["audio"]["url"].lstrip("/")).read_bytes() == (rig.current / "audio/master.wav").read_bytes()
    for role in ("audio", "performance", "semantic"):
        asset = metadata[role]
        data = (rig.output / asset["url"].lstrip("/")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]
        assert len(data) == asset["size_bytes"]
    html = (rig.output / "index.html").read_text()
    assert 'data-static-catalogue="/data/library.json"' in html
    assert (rig.project / "web/player/index.html").read_text().count("data-static-catalogue") == 0
    payload = b"".join(snapshot(rig.output).values())
    for private in (b"must-never-be-public", b"Private transcript", b"private-provider-data", b"retained-old-preview"):
        assert private not in payload
    assert b"provider-responses" not in payload and str(rig.project).encode() not in payload
    assert not (rig.output / "assets/private-debug.txt").exists()
    assert not (rig.output / "private-config.json").exists()
    assert module.verify_export(rig.output) == receipt


@pytest.mark.parametrize("corruption", ["unready", "changed-audio", "stale-catalogue", "unapproved"])
def test_export_fails_before_creating_output_for_unverified_or_uncurated_programmes(rig, corruption):
    if corruption == "unready":
        (rig.runtime / "visual-previews/current-preview/semantic.json").write_text("{}")
    elif corruption == "changed-audio":
        with (rig.current / "audio/master.wav").open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x01")
    else:
        path = rig.project / "creations/catalogue.json"
        catalogue = json.loads(path.read_bytes())
        if corruption == "stale-catalogue":
            catalogue["creations"][0].update(bundle="non-verba/conversation/v1", version="v1")
        else:
            catalogue["creations"][0]["listener_approval"] = "pending"
        write_json(path, catalogue)
    before = snapshot(rig.project)
    with pytest.raises(module.StaticExportError):
        export(rig)
    assert not rig.output.exists()
    assert snapshot(rig.project) == before


def test_export_refuses_oversized_audio_and_preserves_existing_output(rig, monkeypatch):
    monkeypatch.setattr(module, "MAX_STATIC_FILE_BYTES", 50_000)
    with pytest.raises(module.StaticExportError, match="25 MiB"):
        export(rig)
    assert not rig.output.exists()
    rig.output.mkdir()
    sentinel = rig.output / "keep.txt"
    sentinel.write_text("existing user data")
    with pytest.raises(module.StaticExportError, match="already exists"):
        export(rig)
    assert sentinel.read_text() == "existing user data"


@pytest.mark.parametrize("tamper", ["audio", "extra", "symlink"])
def test_verify_rejects_changed_or_unlisted_files(rig, tamper):
    export(rig)
    if tamper == "audio":
        audio, = rig.output.glob("creation-library/*/audio.wav")
        with audio.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x01")
    elif tamper == "extra":
        (rig.output / "secret.txt").write_text("must-never-be-public")
    else:
        (rig.output / "linked-private").symlink_to(rig.runtime / "secrets", target_is_directory=True)
    with pytest.raises(module.StaticExportError):
        module.verify_export(rig.output)


def test_write_failure_preserves_partial_output_and_sources_without_completion_inventory(rig, monkeypatch):
    before = snapshot(rig.project)
    original = module._write
    calls = 0
    def fail_after_first(output, item):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture interrupted write")
        original(output, item)
    monkeypatch.setattr(module, "_write", fail_after_first)
    with pytest.raises(module.StaticExportError, match="preserved"):
        export(rig)
    assert rig.output.is_dir() and any(rig.output.iterdir())
    assert not (rig.output / module.MANIFEST).exists()
    assert snapshot(rig.project) == before

def test_verify_rejects_private_path_even_if_inserted_into_inventory(rig):
    export(rig)
    private = rig.output / "private-production.json"
    private.write_bytes(b"must-never-be-public")
    inventory_path = rig.output / module.MANIFEST
    inventory = json.loads(inventory_path.read_bytes())
    inventory["files"].append({"path": private.name, "bytes": private.stat().st_size,
        "sha256": hashlib.sha256(private.read_bytes()).hexdigest()})
    write_json(inventory_path, inventory)
    with pytest.raises(module.StaticExportError, match="inventory entry"):
        module.verify_export(rig.output)

def rewrite_exported_json(output, relative, change):
    """Model an inventory-consistent edit; verification must check playback roles too."""
    path = output / relative
    document = json.loads(path.read_bytes())
    change(document)
    write_json(path, document)
    inventory_path = output / module.MANIFEST
    inventory = json.loads(inventory_path.read_bytes())
    row = next(row for row in inventory["files"] if row["path"] == relative)
    row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(inventory_path, inventory)


@pytest.mark.parametrize("role", ["index", "favicon", "catalogue", "script", "notice", "vendor", "metadata",
                                 "alias", "audio", "performance", "semantic"])
def test_verify_requires_complete_playback_even_when_missing_file_is_removed_from_inventory(rig, role):
    export(rig)
    item, = json.loads((rig.output / "data/library.json").read_bytes())["items"]
    identifier = item["id"]
    paths = {"index": "index.html", "favicon": "favicon.ico", "catalogue": "data/library.json", "script": "assets/player.js",
             "notice": "NOTICE", "vendor": "vendor/hls.min.js",
             "metadata": "data/recordings/" + identifier + ".json",
             "alias": "data/recordings/" + rig.alias + ".json",
             "audio": "creation-library/" + identifier + "/audio.wav",
             "performance": "creation-library/" + identifier + "/performance.json",
             "semantic": "creation-library/" + identifier + "/semantic.json"}
    relative = paths[role]
    (rig.output / relative).unlink()
    inventory_path = rig.output / module.MANIFEST
    inventory = json.loads(inventory_path.read_bytes())
    inventory["files"] = [row for row in inventory["files"] if row["path"] != relative]
    write_json(inventory_path, inventory)
    with pytest.raises(module.StaticExportError):
        module.verify_export(rig.output)


@pytest.mark.parametrize("change", ["empty-recordings", "duplicate-recordings", "invalid-recording", "missing-default"])
def test_verify_rejects_invalid_recording_declarations_and_unresolved_default(rig, change):
    export(rig)
    path = rig.output / module.MANIFEST
    inventory = json.loads(path.read_bytes())
    if change == "empty-recordings":
        inventory["recordings"] = []
    elif change == "duplicate-recordings":
        inventory["recordings"] *= 2
    elif change == "invalid-recording":
        inventory["recordings"] = ["../private"]
    else:
        inventory["default_recording"] = "project-" + "0" * 24
    write_json(path, inventory)
    with pytest.raises(module.StaticExportError):
        module.verify_export(rig.output)


@pytest.mark.parametrize("change", ["catalogue-alias", "metadata-url", "metadata-size", "semantic-clock"])
def test_verify_checks_catalogue_and_asset_bindings_beyond_inventory_hashes(rig, change):
    export(rig)
    item, = json.loads((rig.output / "data/library.json").read_bytes())["items"]
    identifier = item["id"]
    if change == "catalogue-alias":
        rewrite_exported_json(rig.output, "data/library.json",
            lambda document: document["items"][0].update(playback_url="/?creation=project-" + "0" * 24))
    elif change == "semantic-clock":
        rewrite_exported_json(rig.output, "creation-library/" + identifier + "/semantic.json",
            lambda document: document.update(duration_ms=document["duration_ms"] + 1))
    else:
        def alter(document):
            if change == "metadata-url":
                document["audio"]["url"] = "/api/private-audio"
            else:
                document["audio"]["size_bytes"] += 1
        for identity in (identifier, rig.alias):
            rewrite_exported_json(rig.output, "data/recordings/" + identity + ".json", alter)
    with pytest.raises(module.StaticExportError):
        module.verify_export(rig.output)

def public_archive(rig, tmp_path, *, extra=None):
    source = tmp_path / "saved-public-export"
    export(rig, source)
    archive = tmp_path / "public-media.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        if extra is not None:
            info, data = extra
            bundle.addfile(info, io.BytesIO(data))
    return SimpleNamespace(path=archive, source=source,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((source / module.MANIFEST).read_bytes()).hexdigest())


def assemble(rig, bundle, **kwargs):
    return module.assemble_public_archive(bundle.path, rig.output, sha256=bundle.sha256,
        manifest_sha256=bundle.manifest_sha256, project=rig.project, **kwargs)


def test_github_pages_mode_adds_early_strict_meta_policy_and_domain_without_cloudflare_headers(rig):
    receipt = module.export_static(rig.output, project=rig.project, runtime=rig.runtime,
                                  vendor=rig.vendor, platform="github-pages")
    assert receipt["ok"]
    inventory = json.loads((rig.output / module.MANIFEST).read_bytes())
    assert inventory["platform"] == "github-pages"
    assert inventory["custom_domain"] == "radio.waajacu.com"
    assert (rig.output / "CNAME").read_text() == "radio.waajacu.com\n"
    assert not (rig.output / "_headers").exists()
    assert not (rig.output / "_redirects").exists()
    assert not (rig.output / ".nojekyll").exists()
    html = (rig.output / "index.html").read_text()
    assert html.count(module.PAGES_CSP_META) == 1
    assert html.index(module.PAGES_CSP_META) < html.index("</head>")
    assert "frame-ancestors" not in html and "'unsafe-inline'" not in html
    assert "style-src-attr 'none'" in html and "script-src 'self'" in html


def test_release_assembly_uses_current_frontend_and_exact_pinned_media_without_private_sources(rig, tmp_path):
    bundle = public_archive(rig, tmp_path)
    original_audio = {path.relative_to(bundle.source): path.read_bytes()
                      for path in bundle.source.glob("creation-library/*/*")}
    # Only current public source files need to exist in an Actions checkout.
    checkout = tmp_path / "checkout"
    for name in ("index.html", *module.PLAYER_ASSETS, *module.ROOT_PLAYER_ASSETS):
        target = checkout / "web/player" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (rig.project / "web/player" / name).read_bytes()
        target.write_bytes(data + (b"\n/* current commit */" if name == "player.js" else b""))
    for name in module.PUBLIC_NOTICES:
        (checkout / name).write_bytes((rig.project / name).read_bytes())
    receipt = module.assemble_public_archive(bundle.path, rig.output, sha256=bundle.sha256,
        manifest_sha256=bundle.manifest_sha256, project=checkout)
    assert receipt["ok"]
    assert b"current commit" in (rig.output / "assets/player.js").read_bytes()
    for relative, data in original_audio.items():
        assert (rig.output / relative).read_bytes() == data
    assert not (checkout / ".runtime").exists() and not (checkout / "creations").exists()
    assert not (rig.output / "_headers").exists()
    assert module.PAGES_CSP_META in (rig.output / "index.html").read_text()
    assert module.verify_export(bundle.source)["ok"]  # Retained export remains compatible.


@pytest.mark.parametrize("pin", ["archive", "manifest"])
def test_release_assembly_rejects_wrong_pins_before_output(rig, tmp_path, pin):
    bundle = public_archive(rig, tmp_path)
    if pin == "archive":
        bundle.sha256 = "0" * 64
    else:
        bundle.manifest_sha256 = "0" * 64
    with pytest.raises(module.StaticExportError, match="pinned release"):
        assemble(rig, bundle)
    assert not rig.output.exists()


@pytest.mark.parametrize("attack", ["traversal", "symlink", "hardlink", "private", "duplicate"])
def test_release_assembly_rejects_unsafe_or_unlisted_archive_members(rig, tmp_path, attack):
    name = {"traversal": "../escape.txt", "private": ".runtime/secrets/key",
            "duplicate": "index.html"}.get(attack, "assets/player.js")
    info, data = tarfile.TarInfo(name), b"untrusted"
    info.size = len(data)
    if attack in {"symlink", "hardlink"}:
        info.type = tarfile.SYMTYPE if attack == "symlink" else tarfile.LNKTYPE
        info.linkname = "../outside"
        info.size = 0
        data = b""
    bundle = public_archive(rig, tmp_path, extra=(info, data))
    with pytest.raises(module.StaticExportError):
        assemble(rig, bundle)

    assert not rig.output.exists() and not (tmp_path / "escape.txt").exists()


def test_publication_download_requires_pinned_github_https_and_assembles_verified_bytes(rig, tmp_path, monkeypatch):
    bundle = public_archive(rig, tmp_path)
    url = "https://github.com/waajacu/Natural-Dexterity/releases/download/public-radio-v1/media.tar.gz"
    publication = tmp_path / "publication.json"
    write_json(publication, {"schema_version": 1, "platform": "github-pages",
        "custom_domain": "radio.waajacu.com", "media": {"url": url, "sha256": bundle.sha256,
        "manifest_sha256": bundle.manifest_sha256}})
    calls = []
    class Download(io.BytesIO):
        def geturl(self):
            return "https://release-assets.githubusercontent.com/public-media"
    def download(requested, timeout):
        calls.append(requested)
        return Download(bundle.path.read_bytes())
    monkeypatch.setattr(module, "urlopen", download)
    assert module.assemble_publication(publication, rig.output, project=rig.project)["ok"]
    assert calls == [url]
    with pytest.raises(module.StaticExportError, match="HTTPS GitHub Release"):
        module._download_public_archive("http://example.com/private", tmp_path / "forbidden.tar.gz")
    assert not (tmp_path / "forbidden.tar.gz").exists()


@pytest.mark.parametrize("tamper", ["csp", "domain"])
def test_pages_verifier_requires_csp_and_domain_even_if_inventory_is_updated(rig, tamper):
    module.export_static(rig.output, project=rig.project, runtime=rig.runtime,
                         vendor=rig.vendor, platform="github-pages")
    relative = "index.html" if tamper == "csp" else "CNAME"
    path = rig.output / relative
    path.write_text(path.read_text().replace(module.PAGES_CSP_META, "") if tamper == "csp" else "wrong.example.com\n")
    inventory_path = rig.output / module.MANIFEST
    inventory = json.loads(inventory_path.read_bytes())
    row = next(row for row in inventory["files"] if row["path"] == relative)
    row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(inventory_path, inventory)
    with pytest.raises(module.StaticExportError, match="strict HTML CSP|custom domain"):
        module.verify_export(rig.output)

def test_public_archive_expansion_enforces_file_size_limit_before_assembly(rig, tmp_path, monkeypatch):
    bundle = public_archive(rig, tmp_path)
    monkeypatch.setattr(module, "MAX_STATIC_FILE_BYTES", 50_000)
    with pytest.raises(module.StaticExportError, match="member exceeds"):
        assemble(rig, bundle)
    assert not rig.output.exists()

def test_frontend_icon_reference_requires_source_and_legacy_export_without_reference_still_verifies(rig):
    (rig.project / "web/player/favicon.ico").unlink()
    with pytest.raises(module.StaticExportError, match="missing root asset"):
        export(rig)
    assert not rig.output.exists()
    index = rig.project / "web/player/index.html"
    index.write_text(index.read_text().replace('<link rel="icon" href="/favicon.ico">', ""))
    assert export(rig)["ok"]
    assert not (rig.output / "favicon.ico").exists()
    assert module.verify_export(rig.output)["ok"]
