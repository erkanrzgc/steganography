"""Generated fixtures exercise the native adapter without a real-source fit."""

import hashlib
import io
import json
import math
import struct
import time
import wave

import numpy as np
import pytest

from core import wav_pair_native as n
from steganography import research_wav_pair_native as cli


def save(path, value):
    path.write_text(json.dumps(value))


def bind(root):
    path = root / "protocol.json"
    save(
        path,
        {
            "schema_version": "wav-pair-native-protocol-v1",
            "protocol_id": n.PROTOCOL,
            "spec": n.SPEC,
            "real_fit_authorized": False,
            "preparation_authorized": True,
            "source_sha256": {name: n.base.old.gate.source_hash(root / name) for name in n.SOURCES},
        },
    )
    return path


def proof(root, path):
    qa = root / "qa.json"
    save(
        qa,
        {
            "protocol_sha256": n.base.digest(path),
            "full_passed": 3662,
            "coverage_percent": 96.0,
            "ruff": "passed",
            "mypy": "passed",
            "diff_check": "passed",
            "new_module_coverage_percent": {
                "core/wav_pair_native.py": 100,
                "steganography/research_wav_pair_native.py": 100,
            },
        },
    )
    return qa


def create(root):
    for name in n.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    groups = {
        "train": [f"{i:02}" for i in range(1, 21)],
        "development": [f"{i:02}" for i in range(21, 31)],
        "reserved": [f"{i:02}" for i in range(31, 61)],
    }
    save(root / n.ASSIGNMENT, {"speaker_roles": groups})
    save(root / n.CONTROL, {"arm_decisions": {"train_zscore": "passed_generated_only"}})
    media = root / "media"
    media.mkdir()
    (media / "manifest.json").write_text("generated-fixture-manifest")
    rng = np.random.default_rng(901)
    records = []
    for role in ("train", "development"):
        for group in groups[role]:
            for digit in range(10):
                for take in range(2):
                    pcm = rng.integers(-32000, 32000, 640, dtype=np.int16).astype("<i2").tobytes()
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as audio:
                        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                        audio.writeframes(pcm)
                    raw = buffer.getvalue()
                    rel = f"data/{group}/{digit}_{group}_{take}.wav"
                    target = media / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    records.append(
                        {
                            "role": role,
                            "speaker_group": group,
                            "relative_source_path": rel,
                            "sha256": n.base.wav_pilot.sha(raw),
                            "decoded_pcm_sha256": n.base.wav_pilot.sha(pcm),
                            "frames": 640,
                            "bytes": len(raw),
                            "channels": 1,
                            "sample_rate_hz": 48000,
                            "sample_width_bytes": 2,
                            "parent_recording_id": None,
                        }
                    )
    save(
        root / n.PREPARATION,
        {
            "files": records,
            "acquisition_manifest_sha256": n.base.digest(media / "manifest.json"),
            "source": {
                "commit": "630d7dab4c040882834de6fa21baf9a60372accd",
                "repository_license": "MIT",
            },
        },
    )
    protocol = bind(root)
    return protocol, proof(root, protocol), media


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("native-generated-adapter")
    protocol, qa, media = create(root)
    prior = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    opened = []
    materialize = n.base.wav_pilot.materialize

    def controlled(folder, source, recipe):
        opened.append(source["speaker_group"])
        return materialize(folder, source, recipe)

    def forbidden(*a, **kw):
        raise AssertionError("profile must never construct an optimizer")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(n.base, "resources", lambda: {"host_peak_bytes": 1024, "swap_bytes": 0})
        patch.setattr(n.base.old, "process_started", time.monotonic)
        patch.setattr(n.base.wav_pilot, "materialize", controlled)
        patch.setattr(torch.optim, "AdamW", forbidden)
        result = n.profile(root, protocol, media, root / "cache", qa)
    torch.set_num_threads(prior[0])
    torch.use_deterministic_algorithms(prior[1])
    return root, protocol, qa, media, result, opened


def test_profile_verifies_full_corpus_with_zero_updates_and_reserved_closed(prepared):
    root, protocol, qa, _, result, opened = prepared
    assert result["optimizer_updates"] == 0 and result["parameters_unchanged"] is True
    assert result["originals_verified"] == 600 and result["payload_oracles_passed"] == 4800
    assert result["backward_batches"] == 20 and result["reserved_media_opened"] is False
    assert len(opened) == 5400 and set(opened) == {f"{i:02}" for i in range(1, 31)}
    assert n.quality(root, protocol, qa) == result["protocol_sha256"]
    assert n.read_array(root / "cache/train.npy", 3600).shape == (3600, 10)
    state = json.loads((root / "cache/normalization.json").read_text())
    assert state == n.comparison.normalization(n.read_array(root / "cache/train.npy", 3600))
    assert result["real_fit_authorized"] is False and result["checkpoint_written"] is False


@pytest.mark.parametrize("fault", ["spec", "permission", "prepare", "schema", "sources", "hash"])
def test_protocol_changes_fail_before_media(prepared, tmp_path, fault):
    root, protocol, *_ = prepared
    value = json.loads(protocol.read_text())
    if fault == "spec":
        value["spec"]["threshold"] = 0.7
    if fault == "permission":
        value["real_fit_authorized"] = True
    if fault == "prepare":
        value["preparation_authorized"] = False
    if fault == "schema":
        value["schema_version"] = "old"
    if fault == "sources":
        value["source_sha256"].pop(n.SOURCES[0])
    if fault == "hash":
        value["source_sha256"][n.SOURCES[0]] = "0" * 64
    path = tmp_path / "bad.json"
    save(path, value)
    with pytest.raises(ValueError, match="native"):
        n.load(root, path)


@pytest.mark.parametrize(
    "fault",
    [
        "path",
        "role",
        "license",
        "hash",
        "pcm",
        "frame",
        "bytes",
        "parent",
        "stereo",
        "rate",
        "reserved",
        "overlap",
    ],
)
def test_metadata_rejects_wrong_domain_and_split(prepared, tmp_path, fault):
    root, *_ = prepared
    prep = json.loads((root / n.PREPARATION).read_text())
    draft = json.loads((root / n.ASSIGNMENT).read_text())
    row = prep["files"][0]
    if fault == "path":
        row["relative_source_path"] = "data/01/../../secret.wav"
    if fault == "role":
        row["role"] = "development"
    if fault == "license":
        prep["source"]["repository_license"] = "unknown"
    if fault == "hash":
        row["sha256"] = prep["files"][1]["sha256"]
    if fault == "pcm":
        row["decoded_pcm_sha256"] = "g" * 64
    if fault == "frame":
        row["frames"] = 220501
    if fault == "bytes":
        row["bytes"] = True
    if fault == "parent":
        row["parent_recording_id"] = "guessed"
    if fault == "stereo":
        row["channels"] = 2
    if fault == "rate":
        row["sample_rate_hz"] = 8000
    if fault == "reserved":
        draft["speaker_roles"]["reserved"][0] = "01"
    if fault == "overlap":
        draft["speaker_roles"]["development"][0] = "01"
    (tmp_path / "benchmarks").mkdir()
    save(tmp_path / n.PREPARATION, prep)
    save(tmp_path / n.ASSIGNMENT, draft)
    with pytest.raises(ValueError, match="native"):
        n.records(tmp_path)


@pytest.mark.parametrize(
    "fault", ["count", "coverage", "coverage_bool", "module", "ruff", "mypy", "diff", "binding"]
)
def test_quality_gate_is_required(prepared, tmp_path, fault):
    root, protocol, qa, *_ = prepared
    value = json.loads(qa.read_text())
    if fault == "count":
        value["full_passed"] = True
    if fault == "coverage":
        value["coverage_percent"] = 89.9
    if fault == "coverage_bool":
        value["coverage_percent"] = True
    if fault == "module":
        value["new_module_coverage_percent"]["core/wav_pair_native.py"] = 94.9
    if fault in ("ruff", "mypy"):
        value[fault] = "unavailable"
    if fault == "diff":
        value["diff_check"] = "failed"
    if fault == "binding":
        value["protocol_sha256"] = "0" * 64
    path = tmp_path / "qa.json"
    save(path, value)
    with pytest.raises(ValueError, match="software"):
        n.quality(root, protocol, path)


@pytest.mark.parametrize(
    "fault",
    ["version", "header", "huge", "dtype", "fortran", "truncate", "trailing", "nan", "range"],
)
def test_npy_rejected_before_allocation(tmp_path, monkeypatch, fault):
    header = {"descr": "<f4", "fortran_order": False, "shape": (2, 10)}
    if fault == "huge":
        header["shape"] = (10**100, 10)
    if fault == "dtype":
        header["descr"] = "|O"
    if fault == "fortran":
        header["fortran_order"] = 0
    encoded = (repr(header) + "\n").encode()
    values = np.zeros((2, 10), dtype="<f4")
    if fault == "nan":
        values[0, 0] = np.nan
    if fault == "range":
        values[0, 0] = 1.1
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + values.tobytes()
    if fault == "version":
        raw = b"bad" + raw[3:]
    if fault == "header":
        raw = raw[:8] + struct.pack("<H", 513) + raw[10:]
    if fault == "truncate":
        raw = raw[:-1]
    if fault == "trailing":
        raw += b"x"
    path = tmp_path / "values.npy"
    path.write_bytes(raw)
    if fault not in ("nan", "range"):
        monkeypatch.setattr(n.np, "frombuffer", lambda *a, **k: pytest.fail("allocated too early"))
    with pytest.raises(ValueError, match="native"):
        n.read_array(path, 2)


def approval(root, profile):
    path = root / "approval.json"
    save(
        path,
        {
            "schema_version": "wav-pair-native-approval-v1",
            "protocol_sha256": profile["protocol_sha256"],
            "profile_sha256": n.base.digest(root / "cache/profile.json"),
            "explicit_user_approval": True,
            "approval_reference": "generated fixture; no actual native approval",
            "fit_attempts_allowed": 1,
            "profile_debit_seconds": math.ceil(profile["profile_wall_seconds"]) + 1,
        },
    )
    return path


@pytest.mark.parametrize(
    "fault", ["permission", "reference", "bool_attempt", "hash", "profile", "debit", "nan"]
)
def test_explicit_approval_is_bound_to_exact_profile(prepared, tmp_path, fault):
    root, _, _, _, receipt, _ = prepared
    value = {
        "schema_version": "wav-pair-native-approval-v1",
        "protocol_sha256": receipt["protocol_sha256"],
        "profile_sha256": "b" * 64,
        "explicit_user_approval": True,
        "approval_reference": "fixture",
        "fit_attempts_allowed": 1,
        "profile_debit_seconds": 299,
    }
    if fault == "permission":
        value["explicit_user_approval"] = False
    if fault == "reference":
        value["approval_reference"] = ""
    if fault == "bool_attempt":
        value["fit_attempts_allowed"] = True
    if fault == "hash":
        value["protocol_sha256"] = "0" * 64
    if fault == "profile":
        value["profile_sha256"] = "0" * 64
    if fault == "debit":
        value["profile_debit_seconds"] = 0
    if fault == "nan":
        value["profile_debit_seconds"] = math.nan
    path = tmp_path / "approval.json"
    save(path, value)
    with pytest.raises(ValueError, match="approval"):
        n.authorize(path, receipt["protocol_sha256"], "b" * 64, receipt["profile_wall_seconds"])


def test_cells_match_independent_scalar_threshold_and_auc():
    rows = [{"group": f"speaker-{i // 2}"} for i in range(20) for _ in range(9)]
    scores = np.tile(np.arange(9) / 10, 20)
    scores[::9] = np.linspace(0.1, 0.9, 20)
    report = n.cell_report(rows, scores, scores.copy())
    for i, cell in enumerate(report, 1):
        neg, pos = scores[::9], scores[i::9]
        expected = sum(float(a > b) + 0.5 * float(a == b) for a in pos for b in neg) / 400
        recall, fpr = np.mean(pos >= 0.5), np.mean(neg >= 0.5)
        assert cell["metrics"]["roc_auc"] == pytest.approx(expected)
        assert cell["metrics"]["recall"] == recall
        assert cell["metrics"]["balanced_accuracy"] == pytest.approx((recall + 1 - fpr) / 2)
        assert cell["paired_delta_ci95"]["roc_auc"] == [0, 0]
        assert cell["speaker_groups"] == 10
    with pytest.raises(ValueError, match="score domain"):
        n.cell_report(rows, scores[:-1], scores)


def fake_fit(torch, values, rows, seed, arm, expires, on_update, on_epoch):
    # Adapter plumbing only: no optimizer or fit; counts are mocked fixture evidence.
    model = n.base.candidate.network()
    run = run_evidence(seed)
    for i in range(1, 21):
        for _ in range(57):
            on_update()
        on_epoch(run["epoch_telemetry"][i - 1])
    return model, run


def run_evidence(seed=20261016):
    rng = np.random.default_rng(seed ^ 0x4E4A5047)
    order = hashlib.sha256()
    epochs = []
    for i in range(1, 21):
        order.update(rng.permutation(3600).astype("<i4").tobytes())
        epochs.append(
            {
                "epoch": i,
                "optimizer_updates": 57 * i,
                "row_presentations": 3600 * i,
                "auc_all": 0.8,
                "cover_loss": 0.5,
                "unweighted_loss": 0.5,
                "weighted_objective": 0.5,
                "cells": [
                    {"method": m, "rate_percent": r, "plane": p, "roc_auc": 0.8}
                    for m, r, p in n.base.wav_pilot.RECIPES
                ],
            }
        )
    return {
        "seed": seed,
        "arm": "train_zscore",
        "domain": n.SPEC["domain"],
        "epochs": 20,
        "optimizer_updates": 1140,
        "row_presentations": 72000,
        "initial_parameter_sha256": "a" * 64,
        "final_parameter_sha256": "b" * 64,
        "order_sha256": order.hexdigest(),
        "parameter_change_norm": 1.0,
        "epoch_telemetry": epochs,
        "step_telemetry": [
            {"update": i, "objective": 0.5, "gradient_norm": 0.1, "parameter_step_norm": 0.01}
            for i in n.comparison.POINTS
        ],
    }


@pytest.mark.parametrize(
    "fault", ["updates", "domain", "order", "epochs", "epoch_count", "loss", "steps", "step"]
)
def test_partial_fit_evidence_is_unavailable(fault):
    run = run_evidence()
    if fault == "updates":
        run["optimizer_updates"] = 1139
    if fault == "domain":
        run["domain"] = n.base.candidate.DOMAIN
    if fault == "order":
        run["order_sha256"] = "0" * 64
    if fault == "epochs":
        run["epoch_telemetry"].pop()
    if fault == "epoch_count":
        run["epoch_telemetry"][0]["row_presentations"] = 3599
    if fault == "loss":
        run["epoch_telemetry"][0]["weighted_objective"] = math.nan
    if fault == "steps":
        run["step_telemetry"].pop()
    if fault == "step":
        run["step_telemetry"][0]["gradient_norm"] = math.inf
    with pytest.raises(ValueError, match="native"):
        n.verify_run(run, 20261016)


def test_independent_score_replay_detects_changed_auc():
    scores = np.array([0.1, 0.4, 0.3, 0.5, 0.6, 0.3, 0.8, 0.9, 0.5] * 4)
    rows = [{"recipe": recipe} for _ in range(4) for recipe in (None, *n.base.wav_pilot.RECIPES)]
    auc, cells = n.base.old.metrics(rows, scores)
    record = {"auc_all": auc, "cells": cells}
    assert n.replay_auc(scores, record) == 9
    cells[0]["roc_auc"] = 0.1
    with pytest.raises(ValueError, match="independent AUC"):
        n.replay_auc(scores, record)


def test_fit_adapter_runs_only_after_bound_approval_and_cannot_retry(prepared, monkeypatch):
    root, protocol, qa, _, receipt, _ = prepared
    path = approval(root, receipt)
    monkeypatch.setattr(n.base, "resources", lambda: {"host_peak_bytes": 1024, "swap_bytes": 0})
    monkeypatch.setattr(n.base.old, "process_started", time.monotonic)
    monkeypatch.setattr(n.comparison, "fit_one", fake_fit)
    monkeypatch.setattr(n, "cell_report", lambda *a: [{"fixture": "not native accuracy"}])
    result = n.fit(root, protocol, root / "cache", qa, path)
    assert result["status"] == "completed_native_exploratory_pilot"
    assert len(result["runs"]) == 2
    assert (root / "cache/fit/20261016/epoch-20.json").is_file()
    assert result["checkpoint_written"] is False and result["production_qualified"] is False
    with pytest.raises(FileExistsError):
        n.fit(root, protocol, root / "cache", qa, path)


@pytest.mark.parametrize("fault", ["profile", "cache", "normalizer", "order", "train_only"])
def test_fit_rejects_changed_cache_before_optimizer(prepared, tmp_path, monkeypatch, fault):
    import shutil

    root, protocol, qa, _, original, _ = prepared
    cache = tmp_path / "cache"
    cache.mkdir()
    for path in (root / "cache").iterdir():
        if path.is_file():
            shutil.copyfile(path, cache / path.name)
    receipt = json.loads((cache / "profile.json").read_text())
    if fault == "profile":
        receipt["optimizer_updates"] = True
    if fault == "cache":
        receipt["arrays_sha256"]["train"] = "0" * 64
    if fault == "normalizer":
        receipt["normalization_sha256"] = "0" * 64
    if fault == "order":
        receipt["row_order_sha256"]["development"] = "0" * 64
    if fault == "train_only":
        value = json.loads((cache / "normalization.json").read_text())
        value["mean"][0] += 0.01
        save(cache / "normalization.json", value)
        receipt["normalization_sha256"] = n.base.digest(cache / "normalization.json")
    save(cache / "profile.json", receipt)
    path = approval(tmp_path, receipt)
    monkeypatch.setattr(n.base, "runtime", lambda: pytest.fail("optimizer path reached"))
    with pytest.raises(ValueError, match="native"):
        n.fit(root, protocol, cache, qa, path)


def test_profile_exclusive_output_and_wrong_manifest(prepared, tmp_path, monkeypatch):
    root, protocol, qa, media, *_ = prepared
    monkeypatch.setattr(n.base, "resources", lambda: {})
    monkeypatch.setattr(n.base.old, "process_started", time.monotonic)
    with pytest.raises(FileExistsError):
        n.profile(root, protocol, media, root / "cache", qa)
    (tmp_path / "manifest.json").write_text("wrong")
    with pytest.raises(ValueError, match="manifest"):
        n.profile(root, protocol, tmp_path, tmp_path / "out", qa)
    (tmp_path / "manifest.json").unlink()
    (tmp_path / "manifest.json").symlink_to(media / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        n.profile(root, protocol, tmp_path, tmp_path / "out", qa)


@pytest.mark.parametrize("fault", ["header", "decoded", "deadline", "torch"])
def test_profile_bad_media_limits_and_missing_runtime(prepared, tmp_path, monkeypatch, fault):
    root, protocol, qa, media, *_ = prepared
    monkeypatch.setattr(n.base, "resources", lambda: {})
    monkeypatch.setattr(
        n.base.old,
        "process_started",
        lambda: time.monotonic() - (300 if fault == "deadline" else 0),
    )
    if fault == "header":
        original = n.base.wav_pilot.materialize

        def bad_header(*args):
            raw = bytearray(original(*args))
            raw[24:28] = (44100).to_bytes(4, "little")
            raw[28:32] = (88200).to_bytes(4, "little")
            return bytes(raw)

        monkeypatch.setattr(n.base.wav_pilot, "materialize", bad_header)
    if fault == "decoded":
        prep, rows = n.records(root)
        rows["train"][0]["source"]["decoded_pcm_sha256"] = "0" * 64
        monkeypatch.setattr(n, "records", lambda *a: (prep, rows))
    if fault == "torch":

        def missing():
            raise ImportError("optional Torch unavailable")

        monkeypatch.setattr(n.base, "runtime", missing)
    with pytest.raises((ValueError, TimeoutError, ImportError)):
        n.profile(root, protocol, media, tmp_path / "out", qa)
    assert not (tmp_path / "out/profile.json").exists()


def test_native_limits_and_symlinks(tmp_path):
    path = tmp_path / "link"
    path.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        n.storage(tmp_path)
    path.unlink()
    path.write_bytes(b"x" * (n.base.MIB + 1))
    with pytest.raises(ValueError, match="file size"):
        n.storage(tmp_path)
    path.unlink()
    for i in range(65):
        (tmp_path / str(i)).write_bytes(b"x" * n.base.MIB)
    with pytest.raises(ValueError, match="storage limit"):
        n.storage(tmp_path)


@pytest.mark.parametrize("phase", ["profile", "fit"])
def test_cli_delegates_and_requires_specific_arguments(monkeypatch, tmp_path, phase):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *a: None)
    args = [phase, "--protocol", "p", "--cache", "c", "--qa", "q"]
    assert cli.main(args) == 2

    def ready(*a):
        return {"status": "fixture-only"}

    monkeypatch.setattr(cli.service, phase, ready)
    assert (
        cli.main(args + (["--media", str(tmp_path)] if phase == "profile" else ["--approval", "a"]))
        == 0
    )
