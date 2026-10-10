"""Pinned inventory identities, portable output and adversarial filesystem limits."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import tool_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]


def lock():
    return inventory.read_lock(ROOT / "tools/full-image.lock.json")


@pytest.mark.parametrize(
    "fault", [None, "missing", "version", "probe", "license", "banner", "empty"]
)
def test_inventory_verification_is_bounded_portable_and_fail_closed(tmp_path, monkeypatch, fault):
    spec = lock()
    gem = tmp_path / "gem"
    gem.mkdir()
    (gem / "LICENSE").write_text("fixture license")
    by_name = {row["name"]: row for row in [*spec["tools"], *spec["gems"]]}
    by_package = {row["package"]: row for row in spec["tools"] if "package" in row}

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 10, "max_output": 16384}

        def run(self, tool, args, **kwargs):
            assert kwargs["cwd"] == tmp_path and kwargs["_include_version"] is False
            if tool == "ruby":
                row = by_name[args[-1]]
                declaration = "" if fault == "license" else row["license"]
                output = f"{row['version']}\n{declaration}\n{gem}\n"
                if fault == "empty":
                    output = ""
            elif tool == "dpkg-query":
                output = by_package[args[-1]]["version"]
                if fault == "version":
                    output = "999.0"
            else:
                assert tool == "stegseek" and args == ["--version"]
                output = "invalid" if fault == "banner" else "StegSeek 0.6 - upstream"
            return SimpleNamespace(
                status="failed" if fault == "probe" else "completed", output=output
            )

    monkeypatch.setattr(inventory, "ToolRunner", Runner)
    monkeypatch.setattr(
        inventory.shutil, "which", lambda n: None if fault == "missing" else str(gem)
    )
    monkeypatch.setattr(inventory, "digest_file", lambda path: "0" * 64)
    result = inventory.verify_inventory(spec, cwd=tmp_path)
    assert result["status"] == ("verified" if fault is None else "unavailable")
    assert len(result["entries"]) == 16
    assert str(tmp_path) not in json.dumps(result)
    assert not result["transitive_dependencies_fully_locked"]
    if fault is None:
        assert all(row["status"] == "verified" for row in result["entries"])
        assert all("version" in row for row in result["entries"])


@pytest.mark.parametrize(
    "fault", ["schema", "rows", "row", "name", "version", "duplicate", "sha", "package",
              "gem", "license"]
)
def test_lock_rejects_invalid_or_injected_metadata(tmp_path, fault):
    value = copy.deepcopy(lock())
    if fault == "schema":
        value["schema_version"] = "unknown"
    elif fault == "rows":
        value["gems"] = []
    elif fault == "row":
        value["tools"][0] = None
    elif fault == "duplicate":
        value["tools"].append(value["tools"][0])
    elif fault == "sha":
        value["tools"][0]["sha256"] = 123
    elif fault == "package":
        value["tools"][0]["package"] = "--malicious"
    elif fault == "gem":
        value["tools"][0]["gem"] = "injected"
    else:
        value["tools"][0][fault] = "../unsafe; command"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        inventory.read_lock(path)


def test_lock_byte_symlink_and_file_bounds(tmp_path):
    path = tmp_path / "lock"
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError):
        inventory.read_lock(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        inventory.read_lock(link)
    with pytest.raises(ValueError):
        inventory.read_lock(tmp_path)
    path.write_text("[]")
    with pytest.raises(ValueError):
        inventory.read_lock(path)


def test_installed_tree_identity_changes_and_rejects_symlinks_and_limits(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"first")
    assert inventory.digest_file(path) == hashlib.sha256(b"first").hexdigest()
    first = inventory.digest_tree(tmp_path)
    path.write_bytes(b"second")
    assert first != inventory.digest_tree(tmp_path)
    child = tmp_path / "nested"
    child.mkdir()
    (child / "LICENSE").write_text("fixture")
    assert first != inventory.digest_tree(tmp_path)
    link = child / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        inventory.digest_tree(tmp_path)
    with pytest.raises(ValueError):
        inventory.digest_tree(link)
    link.unlink()
    monkeypatch.setattr(inventory, "MAX_FILE_BYTES", 6)
    with pytest.raises(ValueError):
        inventory.digest_tree(tmp_path)
    path.write_bytes(b"too large")
    with pytest.raises(ValueError):
        inventory.digest_file(path)
    with pytest.raises(ValueError):
        inventory.digest_file(tmp_path)


def test_lock_matches_docker_download_checksums_and_exact_package_pins():
    dockerfile = (ROOT / "Dockerfile.full").read_text()
    spec = lock()
    for row in [*spec["tools"], *spec["gems"]]:
        if "artifact_arg" in row:
            assert f"ARG {row['artifact_arg']}_SHA256={row['sha256']}" in dockerfile
        if "package" in row and row["name"] != "openstego":
            assert f"{row['package']}={row['version']}" in dockerfile
    assert dockerfile.count("@sha256:") == 3
