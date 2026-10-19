"""Bounded AudioMNIST acquisition with generated responses only (no network)."""

import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from steganography import research_wav_prepare as prepare

DRAFT = Path(__file__).resolve().parents[1] / "benchmarks/wav-next-experiment-draft-20261016.json"


def wav_bytes(samples, *, channels=1, rate=48000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def generated(url):
    digest = hashlib.sha256(url.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return wav_bytes(rng.integers(-32768, 32768, size=160, dtype=np.int16))


def rebound(tmp_path, monkeypatch, mutate):
    doc = json.loads(DRAFT.read_text())
    mutate(doc)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(prepare, "DRAFT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def _schema(doc):
    doc["status"] = "executable"


def _count(doc):
    doc["proposed_files"] = doc["proposed_files"][:599]


def _speakers(doc):
    doc["speaker_roles"]["reserved"][0] = doc["speaker_roles"]["train"][0]


def _traversal(doc):
    row = doc["proposed_files"][0]
    row["relative_source_path"] = f"data/{row['speaker_group']}/../x.wav"


def _reserved_role(doc):
    doc["proposed_files"][0]["role"] = "reserved"


def _role_count(doc):
    group = doc["speaker_roles"]["development"][0]
    doc["proposed_files"][0].update(
        role="development", speaker_group=group, relative_source_path=f"data/{group}/moved.wav"
    )


def _group_count(doc):
    target = doc["speaker_roles"]["train"][1]
    for index, row in enumerate(doc["proposed_files"]):
        if row["speaker_group"] == doc["speaker_roles"]["train"][0]:
            row.update(speaker_group=target, relative_source_path=f"data/{target}/m{index}.wav")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_schema, "draft binding mismatch"),
        (_count, "600-file"),
        (_speakers, "speaker assignment"),
        (_traversal, "unsafe or inconsistent"),
        (_reserved_role, "unsafe or inconsistent"),
        (_role_count, "role counts changed"),
        (_group_count, "speaker role count changed"),
    ],
)
def test_read_manifest_rejects_rebound_changes(tmp_path, monkeypatch, mutate, message):
    with pytest.raises(ValueError, match=message):
        prepare._read_manifest(rebound(tmp_path, monkeypatch, mutate))


class Response:
    def __init__(self, url, raw, length="auto"):
        self.url, self.raw = url, raw
        self.headers = (
            {}
            if length is None
            else {"Content-Length": str(len(raw) if length == "auto" else length)}
        )

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.raw[:limit]


def test_fetch_is_bounded_to_pinned_source(monkeypatch):
    url = prepare.BASE + "data/42/0_42_0.wav"
    seen = []

    def urlopen(request, timeout):
        seen.append((request.full_url, request.get_header("User-agent"), timeout))
        return Response(url, b"wav")

    monkeypatch.setattr(prepare.urllib.request, "urlopen", urlopen)
    deadline = prepare.time.monotonic() + 100
    assert prepare._fetch(url, deadline) == b"wav"
    assert (
        seen[0][0] == url and seen[0][1].startswith("steganography-research") and seen[0][2] == 30
    )
    for bad in (
        "https://example.com/" + prepare.COMMIT,
        "https://raw.githubusercontent.com/soerenab/AudioMNIST/main/data/1.wav",
    ):
        with pytest.raises(ValueError, match="outside pinned"):
            prepare._fetch(bad, deadline)
    with pytest.raises(TimeoutError, match="deadline"):
        prepare._fetch(url, prepare.time.monotonic() - 1)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (lambda url: Response(url + "?redirected", b"x"), "redirect"),
        (lambda url: Response(url, b"x", length=prepare.MAX_FILE + 1), "per-file download bound"),
        (lambda url: Response(url, b"x" * (prepare.MAX_FILE + 1), length=None), "length bound"),
        (lambda url: Response(url, b"xy", length=3), "length bound/mismatch"),
    ],
)
def test_fetch_rejects_redirects_and_oversized_responses(monkeypatch, response, message):
    url = prepare.BASE + "data/42/0_42_0.wav"
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda _req, timeout: response(url))
    with pytest.raises(ValueError, match=message):
        prepare._fetch(url, prepare.time.monotonic() + 100)


def test_acquire_writes_only_frozen_rows_and_bounded_manifest(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        prepare, "_fetch", lambda url, _deadline: fetched.append(url) or generated(url)
    )
    out = tmp_path / "acquired"
    result = prepare.acquire(DRAFT, out)
    draft = json.loads(DRAFT.read_text())
    reserved = set(draft["speaker_roles"]["reserved"])
    assert len(fetched) == 600 and all(url.startswith(prepare.BASE) for url in fetched)
    assert not {url.split("/")[-2] for url in fetched} & reserved
    assert result["speaker_role_counts"] == {"train": 400, "development": 200}
    assert result["reserved_speakers_accessed"] is False and result["optimizer_updates"] == 0
    assert result["draft_sha256"] == prepare.DRAFT_SHA256
    assert json.loads((out / "manifest.json").read_text()) == result
    row = result["files"][0]
    assert (out / row["relative_source_path"]).read_bytes() == generated(
        prepare.BASE + row["relative_source_path"]
    )
    assert (out / "manifest.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="new non-symlink directory"):
        prepare.acquire(DRAFT, out)


def test_acquire_rejects_symlinked_output_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(ValueError, match="new non-symlink directory"):
        prepare.acquire(DRAFT, tmp_path / "link" / "out")


def test_acquire_rejects_symlink_planted_inside_output(tmp_path, monkeypatch):
    out = tmp_path / "out"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def fetch(url, _deadline):
        group = url[len(prepare.BASE) :].split("/")[1]
        (out / "data").mkdir(exist_ok=True)
        (out / "data" / group).symlink_to(elsewhere)
        return generated(url)

    monkeypatch.setattr(prepare, "_fetch", fetch)
    with pytest.raises(ValueError, match="symlink output path"):
        prepare.acquire(DRAFT, out)
    assert not list(elsewhere.iterdir())


@pytest.mark.parametrize(
    ("fetch", "patch", "error", "message"),
    [
        (lambda url: generated(url), {"MAX_TOTAL": 100}, ValueError, "raw-transfer ceiling"),
        (lambda _url: generated("same"), {}, ValueError, "duplicate source file"),
        (lambda _url: b"RIFF-not-a-wave", {}, ValueError, "invalid WAV source"),
        (lambda _url: wav_bytes(np.zeros(320), channels=2), {}, ValueError, "mono PCM16"),
        (lambda _url: wav_bytes(np.zeros(160), rate=44100), {}, ValueError, "mono PCM16"),
        (lambda _url: wav_bytes(np.zeros(159)), {}, ValueError, "mono PCM16"),
        (lambda url: generated(url), {"MAX_SECONDS": 0}, TimeoutError, "deadline"),
    ],
)
def test_acquire_fails_closed_on_bounds_and_native_format(
    tmp_path, monkeypatch, fetch, patch, error, message
):
    monkeypatch.setattr(prepare, "_fetch", lambda url, _deadline: fetch(url))
    for name, value in patch.items():
        monkeypatch.setattr(prepare, name, value)
    with pytest.raises(error, match=message):
        prepare.acquire(DRAFT, tmp_path / "out")


def test_main_prints_bounded_summary(tmp_path, monkeypatch, capsys):
    calls = []

    def acquire(draft, out):
        calls.append((draft, out))
        return {
            "status": "acquired_native_headers_verified_not_training_ready",
            "speaker_role_counts": {"train": 400, "development": 200},
            "raw_bytes": 1,
            "elapsed_seconds": 0.5,
            "files": [{"sha256": "secret-free"}],
        }

    monkeypatch.setattr(prepare, "acquire", acquire)
    monkeypatch.setattr(sys, "argv", ["research_wav_prepare", "--out", str(tmp_path / "o")])
    prepare.main()
    summary = json.loads(capsys.readouterr().out)
    assert set(summary) == {"status", "speaker_role_counts", "raw_bytes", "elapsed_seconds"}
    assert calls == [(Path("benchmarks/wav-next-experiment-draft-20261016.json"), tmp_path / "o")]
