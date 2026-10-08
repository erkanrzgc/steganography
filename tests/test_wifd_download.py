"""Pinned data/license, bounded network, reserved identities and fresh outputs."""

import hashlib
import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def module():
    spec = importlib.util.spec_from_file_location(
        "wifd_fetch", Path(__file__).parents[1] / "scripts/fetch-wifd.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = module()
    license_raw = b"MIT License: fixture"
    readme = b"data and code: MIT License"
    monkeypatch.setattr(m, "LICENSE_BLOB", m.blob_hash(license_raw))
    monkeypatch.setattr(m, "README_BLOB", m.blob_hash(readme))
    bodies = {}
    entries = []
    for i in range(4):
        camera = f"camera_{i // 2}"
        path = f"{camera}/sdr_image/jpg/{camera}_{i}.jpg"
        buffer = io.BytesIO()
        Image.fromarray(np.full((256, 256, 3), i * 30, dtype=np.uint8)).save(buffer, format="JPEG")
        data = buffer.getvalue()
        bodies[m.RAW + "dataset/" + path] = data
        entries.append(
            {
                "path": path,
                "type": "blob",
                "mode": "100644",
                "sha": m.blob_hash(data),
                "size": len(data),
            }
        )
    tree = {"sha": m.TREE, "truncated": False, "tree": entries}

    def get(self, url, limit):
        if url == m.RAW + "LICENSE":
            return license_raw
        if url == m.RAW + "README.md":
            return readme
        if url == m.API:
            return json.dumps(tree).encode()
        data = bodies[url]
        assert len(data) <= limit
        return data

    monkeypatch.setattr(m.Fetcher, "get", get)
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    return m, reserved, tree, bodies


def test_complete_acquisition_pinned_hashes_decode_no_secrets_fresh_and_origin_reserved(
    setup, tmp_path
):
    m, reserved, tree, _ = setup
    out = tmp_path / "images"
    r = m.acquire(out, [reserved])
    assert len(r["samples"]) == 4 and r["license"] == "MIT" and not r["scene_independence_verified"]
    assert {s["split"] for s in r["samples"]} == {"test"} and r[
        "accuracy_qualification"
    ] == "unavailable"
    assert {s["device"] for s in r["samples"]} == {"camera_0", "camera_1"}
    for row in r["samples"]:
        raw = (out / row["path"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["git_blob_sha1"] == m.blob_hash(raw)
        assert row["scene"] is None
    assert str(tmp_path) not in json.dumps(r)
    assert len(m.select(tree, 20, set())) == 4
    excluded = {tree["tree"][0]["path"]}
    assert len(m.select(tree, 120, excluded)) == 3
    with pytest.raises(FileExistsError):
        m.acquire(out, [reserved])
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        m.acquire(link / "bad", [reserved])


@pytest.mark.parametrize(
    "fault",
    ["truncated", "tree", "duplicate", "symlink", "size", "sha", "parent", "no_cameras", "entries"],
)
def test_malformed_selection_fail_closed(setup, fault):
    m, _, tree, _ = setup
    if fault == "truncated":
        tree["truncated"] = True
    if fault == "tree":
        tree["sha"] = "0" * 40
    if fault == "duplicate":
        tree["tree"].append(tree["tree"][0].copy())
    if fault == "symlink":
        tree["tree"][0]["mode"] = "120000"
    if fault == "size":
        tree["tree"][0]["size"] = m.MAX_FILE + 1
    if fault == "sha":
        tree["tree"][0]["sha"] = "invalid"
    if fault == "parent":
        tree["tree"][0]["path"] = "camera_0/sdr_image/../bad.jpg"
    if fault == "no_cameras":
        tree["tree"] = []
    if fault == "entries":
        tree["tree"] = [{}] * 30001
    with pytest.raises(ValueError):
        m.select(tree, 20, set())


@pytest.mark.parametrize(
    "fault",
    [
        "license",
        "readme",
        "identity",
        "overlap",
        "duplicate",
        "geometry",
        "format",
        "empty_reserved",
        "bad_reserved",
        "count",
    ],
)
def test_integrity_license_image_and_reserved_failures_never_write_success(
    setup, monkeypatch, tmp_path, fault
):
    m, reserved, tree, bodies = setup
    if fault == "license":
        monkeypatch.setattr(m, "LICENSE_BLOB", "0" * 40)
    if fault == "readme":
        monkeypatch.setattr(m, "README_BLOB", "0" * 40)
    if fault == "identity":
        tree["tree"][0]["sha"] = "0" * 40
    if fault == "overlap":
        reserved.write_text(
            json.dumps(
                {"samples": [{"sha256": hashlib.sha256(next(iter(bodies.values()))).hexdigest()}]}
            )
        )
    if fault == "duplicate":
        raw = next(iter(bodies.values()))
        for row in tree["tree"]:
            bodies[m.RAW + "dataset/" + row["path"]] = raw
            row.update(sha=m.blob_hash(raw), size=len(raw))
    if fault in ("geometry", "format"):
        raw = io.BytesIO()
        Image.new("RGB", (16, 16) if fault == "geometry" else (256, 256)).save(
            raw, format="JPEG" if fault == "geometry" else "PNG"
        )
        row = tree["tree"][0]
        bodies[m.RAW + "dataset/" + row["path"]] = raw.getvalue()
        row.update(sha=m.blob_hash(raw.getvalue()), size=len(raw.getvalue()))
    if fault == "empty_reserved":
        reserved.write_text('{"samples":[]}')
    if fault == "bad_reserved":
        reserved.write_text('{"samples":[{}]}')
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        m.acquire(out, [reserved], per_device=1 if fault == "count" else 20)
    assert not (out / "source.json").exists()
    with pytest.raises(ValueError):
        m.acquire(tmp_path / "no-reserved", [])


def test_reserved_member_exclusion_and_symlink_manifest(setup, tmp_path):
    m, reserved, tree, _ = setup
    reserved.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sha256": "a" * 64,
                        "source_group": "WIFD",
                        "upstream_member": tree["tree"][0]["path"],
                    }
                ]
            }
        )
    )
    r = m.acquire(tmp_path / "new", [reserved])
    assert len(r["samples"]) == 3
    link = tmp_path / "link.json"
    link.symlink_to(reserved)
    with pytest.raises(ValueError):
        m.acquire(tmp_path / "bad", [link])


def test_reserved_manifest_size_and_growth_are_bounded(setup, monkeypatch, tmp_path):
    m, reserved, _, _ = setup
    monkeypatch.setattr(m, "MAX_META", 1)
    with pytest.raises(ValueError, match="unbounded"):
        m.acquire(tmp_path / "large", [reserved])
    original = Path.stat

    def stat(path, *a, **kw):
        result = original(path, *a, **kw)
        return SimpleNamespace(st_size=1, st_mode=result.st_mode) if path == reserved else result

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(ValueError, match="grew"):
        m.acquire(tmp_path / "grew", [reserved])


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, headers=None):
        super().__init__(raw)
        self.headers = headers or {}


def test_network_bounds_fixed_host_no_redirect_retries_and_budget(monkeypatch):
    m = module()
    responses = [urllib.error.HTTPError(m.API, 500, "error", {}, None), Response(b"abc")]

    def open_request(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        m.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=open_request)
    )
    monkeypatch.setattr(m.time, "sleep", lambda *a: None)
    f = m.Fetcher()
    assert f.get(m.API, 3) == b"abc" and f.requests == 2 and f.budget_bytes == 6
    assert f.received_bytes == 3
    with pytest.raises(ValueError):
        f.get("https://private.example/secret", 3)
    with pytest.raises(ValueError):
        f.get(m.API, 0)
    for response in (Response(b"toolong"), Response(b"x", {"Content-Length": "9"})):
        responses[:] = [response]
        with pytest.raises(ValueError):
            f.get(m.API, 3)
    responses[:] = [urllib.error.HTTPError(m.API, 302, "redirect", {}, None)]
    with pytest.raises(ValueError):
        f.get(m.API, 3)
    assert m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other") is None
    f.budget_bytes = m.MAX_TOTAL
    with pytest.raises(ValueError):
        f.get(m.API, 1)
    f.budget_bytes = 0
    f.requests = 6100
    with pytest.raises(ValueError):
        f.get(m.API, 1)
    f.requests = 0
    f.started -= 1801
    with pytest.raises(ValueError):
        f.get(m.API, 1)


def test_network_retry_exhaustion_deadline_during_read_and_status(monkeypatch):
    m = module()
    monkeypatch.setattr(m.time, "sleep", lambda *a: None)

    def fail(*a, **kw):
        raise urllib.error.URLError("private token")

    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=fail))
    with pytest.raises(ValueError, match="unavailable"):
        m.Fetcher().get(m.API, 3)
    responses = [Response(b"abc")]
    responses[0].status = 206
    monkeypatch.setattr(
        m.urllib.request,
        "build_opener",
        lambda *a: SimpleNamespace(open=lambda *a, **kw: responses.pop(0)),
    )
    with pytest.raises(ValueError):
        m.Fetcher().get(m.API, 3)
    ticks = iter((0, 0, 1801))
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks))
    responses[:] = [Response(b"abc")]
    with pytest.raises(ValueError, match="time"):
        m.Fetcher().get(m.API, 3)


def test_cli_limits_success_and_secret_redaction(setup, monkeypatch, tmp_path, capsys):
    import resource

    m, reserved, _, _ = setup
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    args = ["--out", str(tmp_path / "out"), "--reserved-manifest", str(reserved)]
    assert m.main(args) == 0 and len(calls) == 4

    def fail(*a, **kw):
        raise ValueError("private/token=secret")

    monkeypatch.setattr(m, "acquire", fail)
    assert m.main(args) == 2 and "secret" not in capsys.readouterr().out


@pytest.mark.parametrize("frames", [2, 4, 5])
def test_native_mpo_requires_frozen_explicit_retry_and_bounded_frames(setup, tmp_path, frames):
    m, reserved, tree, bodies = setup
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "purple").save(
        buffer,
        format="MPO",
        save_all=True,
        append_images=[Image.new("RGB", (256, 256), "blue") for _ in range(frames - 1)],
    )
    raw = buffer.getvalue()
    row = tree["tree"][0]
    bodies[m.RAW + "dataset/" + row["path"]] = raw
    row.update(sha=m.blob_hash(raw), size=len(raw))
    with pytest.raises(ValueError):
        m.acquire(tmp_path / "strict", [reserved])
    if frames > 4:
        with pytest.raises(ValueError):
            m.acquire(tmp_path / "retry", [reserved], allow_bounded_mpo=True)
        assert not (tmp_path / "retry/source.json").exists()
    else:
        report = m.acquire(tmp_path / "retry", [reserved], allow_bounded_mpo=True)
        native = next(r for r in report["samples"] if r["format"] == "MPO")
        assert native["declared_frames"] == frames and native["decoded_frames"] == 1
        assert native["path"].endswith(".mpo")
        assert report["retry_protocol_sha256"] == m.RETRY_PROTOCOL_SHA
        assert (tmp_path / "retry" / native["path"]).read_bytes() == raw


def test_frozen_protocol_mutation_and_invalid_policy_fail_before_network(
    setup, monkeypatch, tmp_path
):
    m, reserved, _, _ = setup
    with pytest.raises(ValueError, match="policy"):
        m.acquire(tmp_path / "bad-policy", [reserved], allow_bounded_mpo=1)
    monkeypatch.setattr(m, "RETRY_PROTOCOL_SHA", "0" * 64)
    with pytest.raises(ValueError, match="retry protocol"):
        m.acquire(tmp_path / "bad-retry", [reserved], allow_bounded_mpo=True)
    monkeypatch.setattr(m, "PROTOCOL_SHA", "0" * 64)
    with pytest.raises(ValueError, match="acquisition protocol"):
        m.acquire(tmp_path / "bad-protocol", [reserved])
