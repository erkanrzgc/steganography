"""Candidate integrity regressions independent of the published Kodak images."""

import base64
import io
import os
import struct
import sys
import urllib.parse
from pathlib import Path

import pytest
from PIL import Image

from core.ctf import (
    CTFLimits,
    CTFService,
    _decoded_candidates,
    _JobState,
    _signature_offsets,
    _valid_bmp_header,
)
from core.ctf_types import ToolExecution
from core.tools import ToolRunner


def bmp_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (66, 77, 255)).save(output, format="BMP")
    return output.getvalue()


def job(tmp_path, *, limits=None):
    (tmp_path / "artifacts").mkdir()
    state = _JobState(tmp_path, limits or CTFLimits())
    source = tmp_path / "source.bmp"
    source.write_bytes(bmp_bytes())
    root = CTFService()._copy_input(state, source)
    return state, root


def test_binary_is_not_lossily_url_decoded():
    binary = bmp_bytes() + b"%41\xff%ZZ"
    assert not any(
        name == "decoded-url.bin" for _, name, _ in _decoded_candidates(binary, deep=False)
    )
    for text in (b"plain 100%", b"abc%ZZ\xff", b"abc%00\x00"):
        assert not any(
            name == "decoded-url.bin" for _, name, _ in _decoded_candidates(text, deep=False)
        )
    encoded = urllib.parse.quote_from_bytes(binary).encode("ascii")
    decoded = _decoded_candidates(encoded, deep=False)
    assert [value for value, name, _ in decoded if name == "decoded-url.bin"] == [binary]
    assert any(
        value == b"flag{url}" for value, _, _ in _decoded_candidates(b"flag%7Burl%7D", deep=False)
    )


@pytest.mark.parametrize(
    ("field", "value", "size"),
    [
        (2, 9999, 4),
        (6, 1, 2),
        (10, 0, 4),
        (14, 3, 4),
        (18, 0, 4),
        (22, 0, 4),
        (26, 2, 2),
        (28, 3, 2),
    ],
)
def test_bmp_carving_rejects_incoherent_headers(field, value, size):
    data = bytearray(bmp_bytes())
    data[field : field + size] = value.to_bytes(size, "little")
    assert not _valid_bmp_header(bytes(data), 0)
    assert list(_signature_offsets(b"prefix" + data)) == []


def test_bmp_carving_checks_header_and_preserves_exact_extent(tmp_path):
    bmp = bmp_bytes()
    decoy = b"BM" + b"not a header" * 3
    data = b"prefix" + decoy + bmp + b"unrelated suffix"
    offset = len(b"prefix" + decoy)
    assert list(_signature_offsets(data)) == [(offset, "bmp")]
    assert not _valid_bmp_header(b"BM", 0)
    assert list(_signature_offsets(b"x" + b"BM" * 10000)) == []
    # OS/2 core headers and top-down DIBs must not be rejected as accidental BM.
    core = struct.pack("<2sIHHIIHHHH", b"BM", 30, 0, 0, 26, 12, 1, 1, 1, 24) + bytes(4)
    assert _valid_bmp_header(core, 0)
    top_down = bytearray(bmp)
    top_down[22:26] = (-8).to_bytes(4, "little", signed=True)
    assert _valid_bmp_header(bytes(top_down), 0)
    state, root = job(tmp_path)
    embedded = CTFService()._store(state, data, "container.bin", root, 1, "test")
    carved = CTFService()._carve_and_decode(state, embedded, mode="balanced")
    assert len(carved) == 1
    assert carved[0].path.read_bytes() == bmp
    assert carved[0].name.endswith(".bmp")
    assert carved[0].parent_id == embedded.id
    assert f"offset {offset}" in carved[0].provenance


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_max_depth_prevents_all_child_generation(tmp_path, monkeypatch, depth):
    service = CTFService()
    visited = []

    def candidates(state, artifact, **kwargs):
        visited.append(artifact.depth)
        return [
            service._store(
                state,
                b"payload" + bytes([artifact.depth]),
                "child.png",
                artifact,
                artifact.depth + 1,
                "fixture",
            )
        ]

    monkeypatch.setattr(service, "_native_candidates", candidates)
    monkeypatch.setattr(service, "_analyze", lambda *args: None)
    source = tmp_path / "challenge.png"
    source.write_bytes(b"plain")
    report = service.solve(
        source, tmp_path / "out", mode="quick", limits=CTFLimits(max_depth=depth)
    )
    assert report.status == "completed"
    assert visited == list(range(depth))
    assert max(item.depth for item in report.artifacts) == depth
    state, root = job(tmp_path, limits=CTFLimits(max_depth=0))
    assert service._carve_and_decode(state, root, mode="balanced") == []


def test_boundary_artifacts_are_analyzed_without_decoding_further(tmp_path):
    source = tmp_path / "nested.txt"
    inner = base64.b64encode(b"flag{nested}")
    source.write_bytes(base64.b64encode(inner))
    report = CTFService().solve(
        source, tmp_path / "out", mode="quick", limits=CTFLimits(max_depth=1)
    )
    assert report.status == "completed"
    assert len(report.analyses) == len(report.artifacts) == 2
    assert report.artifacts[-1].path.read_bytes() == inner


@pytest.mark.parametrize("suffix", ["wav", "gif", "txt", "mp3", "pdf", "jpg"])
@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_native_extractors_propagate_limits_instead_of_reporting_completion(
    tmp_path, monkeypatch, suffix, limit
):
    import piexif

    from modules import audio_mp3, audio_wav, file_pdf, image_gif, image_jpeg_dct
    from modules.text_whitespace import TextWhitespace

    payload = b"flag{native-limit}" * 20
    monkeypatch.setattr(audio_wav, "extract_wav_payload", lambda *a: (payload, "fixture"))
    monkeypatch.setattr(image_gif, "parse_gif_stream", lambda *a: {"delays": []})
    monkeypatch.setattr(image_gif, "extract_gif_comments", lambda *a: payload)
    monkeypatch.setattr(TextWhitespace, "extract", lambda *a: payload)
    extracted = [("candidate.bin", payload, "fixture", True)]
    monkeypatch.setattr(audio_mp3, "extract_mp3_payloads", lambda *a: extracted)
    monkeypatch.setattr(file_pdf, "extract_pdf_payloads", lambda *a, **kw: extracted)
    monkeypatch.setattr(piexif, "load", lambda *a: {"0th": {1: payload}})

    def missing(*args, **kwargs):
        raise ValueError("fixture has no project/JSteg payload")

    monkeypatch.setattr(image_jpeg_dct, "extract_jsteg", missing)
    service = CTFService()
    monkeypatch.setattr(service.stego, "extract", missing)
    monkeypatch.setattr(service, "_analyze", lambda *args: None)
    source = tmp_path / f"fixture.{suffix}"
    source.write_bytes(b"fixture")
    limits = CTFLimits(max_artifacts=1) if limit == "count" else CTFLimits(max_bytes=16)
    report = service.solve(source, tmp_path / "out", mode="quick", limits=limits)
    assert report.status == "cancelled" and report.verdict == "inconclusive"
    assert report.error == f"artifact {'count' if limit == 'count' else 'byte'} limit reached"
    assert len(report.artifacts) == 1


class OutputRunner:
    def __init__(self, action, status="completed", exit_code=0):
        self.action, self.status, self.exit_code = action, status, exit_code
        self.calls = []

    def run(self, tool, args, **kwargs):
        self.calls.append((tool, args))
        self.action(tool, kwargs["cwd"])
        return ToolExecution(tool, "fixture", self.status, 1.0, self.exit_code, (tool,))


def external(state, root, runner):
    return CTFService(tool_runner=runner)._external_tools(
        state, root, mode="balanced", wordlist=None, password=None, should_cancel=None
    )


def test_only_the_current_extractors_successful_output_is_adopted(tmp_path):
    state, root = job(tmp_path)

    def action(tool, cwd):
        if tool == "zsteg":
            (cwd / "core.123").write_bytes(b"\x7fELFflag{crash-memory}")
            (cwd / "diagnostic.log").write_bytes(b"flag{not-payload}")
        if tool == "steghide":
            (cwd / "steghide.bin").write_bytes(b"flag{real-payload}")

    runner = OutputRunner(action)
    found = external(state, root, runner)
    assert len(found) == 1
    assert found[0].path.read_bytes() == b"flag{real-payload}"
    assert "via steghide" in found[0].provenance
    assert not (tmp_path / "steghide.bin").exists()
    assert state.confirmed
    assert ("stegseek", ["--seed", f"artifacts/{root.name}", "stegseek.bin"]) in runner.calls


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("failed", 1),
        ("timed_out", -9),
        ("cancelled", -9),
        ("unavailable", None),
        ("completed", 1),
    ],
)
def test_failed_partial_extraction_cannot_confirm_or_consume_budget(tmp_path, status, code):
    state, root = job(tmp_path)

    def action(tool, cwd):
        if tool == "steghide":
            (cwd / "steghide.bin").write_bytes(b"flag{partial-output}")

    assert external(state, root, OutputRunner(action, status, code)) == []
    assert not state.confirmed and state.output_bytes == 0


@pytest.mark.parametrize(
    "kind", ["symlink", "directory", "empty", "preexisting", "preexisting-link"]
)
def test_extraction_refuses_unsafe_empty_or_preexisting_output(tmp_path, kind):
    state, root = job(tmp_path)
    original = tmp_path / "original"
    original.write_bytes(b"flag{must-not-be-adopted}")
    target = tmp_path / "steghide.bin"
    if kind == "preexisting":
        target.write_bytes(b"preserved")
    elif kind == "preexisting-link":
        target.symlink_to(tmp_path / "missing")

    def action(tool, cwd):
        if tool != "steghide":
            return
        if kind == "symlink":
            target.symlink_to(original)
        elif kind == "directory":
            target.mkdir()
        elif kind == "empty":
            target.write_bytes(b"")
        else:
            pytest.fail("must not invoke extractor over an existing output")

    runner = OutputRunner(action)
    assert external(state, root, runner) == []
    assert not state.confirmed and original.read_bytes() == b"flag{must-not-be-adopted}"
    if kind.startswith("preexisting"):
        assert "steghide" not in [tool for tool, _ in runner.calls]
        assert any(
            item.component == "steghide" and item.status == "failed" for item in state.coverage
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_tool_core_dumps_disabled_only_in_child(tmp_path: Path):
    import resource

    previous = resource.getrlimit(resource.RLIMIT_CORE)
    execution = ToolRunner().run(
        sys.executable,
        ["-c", "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))"],
        cwd=tmp_path,
    )
    assert execution.status == "completed" and execution.output.strip() == "(0, 0)"
    assert resource.getrlimit(resource.RLIMIT_CORE) == previous
