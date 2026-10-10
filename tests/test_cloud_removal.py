"""Retired cloud paths stay disabled; documentation matches the local product."""

import importlib.util
import re
import socket
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

import cli
from api.app import APISettings, create_app
from core.service import AnalysisService
from registry import Registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["analyze", "scan"])
@pytest.mark.parametrize("flag", ["--ai", "--allow-ai-file-upload"])
def test_cli_rejects_retired_options_before_discovery(monkeypatch, capsys, command, flag):
    monkeypatch.setattr(cli, "_build_registry", lambda: pytest.fail("discovery must not run"))
    args = (
        ["--in", "not-opened.png"] if command == "analyze"
        else ["--dir", "not-opened", "--out", "not-created.json"]
    )
    assert cli.main(["--quiet", command, *args, flag]) == 2
    assert "removed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit:
        cli.main([command, "--help"])
    assert exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--ai" not in help_text and "--allow-ai-file-upload" not in help_text


def test_builtin_analysis_has_no_cloud_provider_even_with_legacy_environment(
    tmp_path, monkeypatch
):
    for name in ("NVIDIA_NIM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "unused-test-placeholder")
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("network"))
    assert importlib.util.find_spec("modules.ai_provider_nim") is None
    assert importlib.util.find_spec("modules.ai_triage") is None
    registry = Registry()
    registry.autodiscover()
    assert not any(a.name == "ai_triage" for a in registry.all_analyzers())
    source = tmp_path / "ordinary.txt"
    source.write_text("Ordinary harmless text.\n")
    report = AnalysisService(registry, ai_provider=None).analyze(source)
    assert not any(r.analyzer == "ai_triage" for r in report.results)


def test_legacy_server_environment_cannot_enable_cloud(tmp_path, monkeypatch):
    monkeypatch.setenv("STEGANO_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("STEGANO_ALLOW_AI_FILE_UPLOAD", "1")
    settings = APISettings.from_environment()
    assert not settings.allow_ai_file_upload
    with TestClient(create_app(settings)) as client:
        result = client.post(
            "/v1/analyze?ai=false&allow_ai_file_upload=false",
            files={"file": ("ordinary.txt", b"plain text", "text/plain")},
        )
        assert result.status_code == 200
        assert not any(r["analyzer"] == "ai_triage" for r in result.json()["results"])
        names = client.get("/v1/modules").json()["analyzers"]
        assert "ai_triage" not in names


def test_package_license_and_environment_contract():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert metadata["license"] == "MIT"
    assert metadata["license-files"] == ["LICENSE"]
    assert "ai" not in metadata["optional-dependencies"]
    assert not any(d.startswith("httpx") for d in metadata["dependencies"])
    license_text = (ROOT / "LICENSE").read_text()
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 erkanrzgc" in license_text
    settings = (ROOT / ".env.example").read_text()
    assert not any(name in settings for name in ("NVIDIA_NIM", "OPENAI", "ANTHROPIC", "ALLOW_AI"))


@pytest.mark.parametrize("name", [
    "README.md", "docs/README.md", "docs/LIMITATIONS.md", "docs/LICENSING.md",
    "docs/MIGRATION.md", "CONTRIBUTING.md", "SECURITY.md",
])
def test_documentation_links_resolve(name):
    document = ROOT / name
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text()):
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        resolved = (document.parent / url.path).resolve()
        assert resolved.is_relative_to(ROOT)
        assert resolved.is_file(), f"broken local link in {name}: {target}"


def test_readme_has_restrained_badges_and_honest_scope():
    readme = (ROOT / "README.md").read_text()
    assert 400 <= len(readme.split()) <= 900
    assert readme.count("[![") == 4
    assert "AI-powered" not in readme
    assert "No research model has" in readme
    assert "not validated detection accuracy" in readme
    assert "synthetic regression results are not an overall accuracy score" in readme
