"""Publication configuration must keep media pinned and CI deployment narrowly scoped."""
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    configured = os.environ.get("RADIO_WORKFLOW_PATH")
    candidates = ([Path(configured)] if configured else
                  [parent / ".github/workflows/radio-pages.yml" for parent in (PROJECT, *PROJECT.parents)])
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        pytest.skip("Repository-level workflow is outside the local container's sole project bind.")
    return source.read_text(encoding="utf-8")


def test_publication_media_is_a_versioned_repository_release_with_both_hashes():
    document = json.loads((PROJECT / "publication.json").read_text(encoding="utf-8"))
    assert set(document) == {"schema_version", "platform", "custom_domain", "media"}
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert document["platform"] == "github-pages"
    assert document["custom_domain"] == "radio.waajacu.com"
    media = document["media"]
    assert set(media) == {"url", "sha256", "manifest_sha256"}
    url = urlsplit(media["url"])
    assert url.scheme == "https" and url.netloc == "github.com"
    assert not url.query and not url.fragment and not url.username and not url.password
    prefix = "/savethebeesandseeds/Natural-Dexterity/releases/download/"
    assert url.path.startswith(prefix)
    release, asset = url.path.removeprefix(prefix).split("/")
    assert release not in {"latest", "main", "master"} and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release)
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz", asset)
    for field in ("sha256", "manifest_sha256"):
        assert re.fullmatch(r"[a-f0-9]{64}", media[field]) and media[field] != "0" * 64


def test_ci_pins_all_actions_and_the_build_container(workflow):
    actions = re.findall(r"^\s+uses:\s*(\S+)", workflow, re.MULTILINE)
    assert len(actions) == 3
    expected = {"actions/checkout", "actions/upload-pages-artifact", "actions/deploy-pages"}
    assert {source.partition("@")[0] for source in actions} == expected
    assert all(re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", source) for source in actions)
    images = re.findall(r"^\s+image:\s*(\S+)", workflow, re.MULTILINE)
    assert len(images) == 1
    assert re.fullmatch(r"python:3\.13-slim-bookworm@sha256:[a-f0-9]{64}", images[0])


def test_ci_runs_project_commands_only_in_the_build_container(workflow):
    build = re.search(r"(?ms)^  build:\n(.*?)(?=^  deploy:\n)", workflow)
    deploy = re.search(r"(?ms)^  deploy:\n(.*)\Z", workflow)
    assert build and deploy
    assert re.search(r"^    container:\n", build[1], re.MULTILINE)
    assert "working-directory: projects/verbal-dexterity/web-radio" in build[1]
    assert "--require-hashes" in build[1]
    assert "--publication publication.json" in build[1] and "--platform github-pages" in build[1]
    assert not re.search(r"^\s+run:", deploy[1], re.MULTILINE)
    assert "needs: build" in deploy[1]
    assert "if: github.ref == 'refs/heads/main'" in deploy[1]
    assert not re.search(r"^\s+contents:\s*write", workflow, re.MULTILINE)
    assert "secrets." not in workflow and "pull_request_target" not in workflow


def test_only_main_radio_changes_or_manual_dispatch_trigger_publication(workflow):
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "branches: [main]" in trigger
    assert "'projects/verbal-dexterity/web-radio/**'" in trigger
    assert "'.github/workflows/radio-pages.yml'" in trigger
    assert "workflow_dispatch:" in trigger
    assert "pull_request" not in trigger and "schedule:" not in trigger
