"""Generated signal accounting; not physical data or model accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import jpeg_crop_signal as signal


def samples():
    return [
        {
            "source_group": source,
            "lineage": f"{source}-{i:02}",
            "quality_factor": q,
            "method": method,
            "label": "cover" if method is None else "stego",
            "split": "train",
        }
        for source, qualities in (("A", (None,)), ("B", (75, 95)), ("C", (75, 95)))
        for i in range(32)
        for q in qualities
        for method in (None, "JUNIWARD", "UERD")
    ]


def fetch(indices):
    values = np.zeros((4, 1, 256, 256), dtype="<f4")
    values[1, 0, 0, 0] = 1
    return values


def test_only_fit_pairs_are_read_and_difference_is_exact():
    rows = samples()
    seen = set()

    def tracked(indices):
        seen.update(indices.tolist())
        assert indices[0] == indices[2] and indices[1] == indices[3]
        return fetch(indices)

    result = signal.summarize(rows, tracked)
    assert len(result) == 10 and all(c["fit_pairs"] == 24 for c in result)
    assert all(c["identical_crops"] == 0 for c in result)
    assert all(c["minimum_rms_pixel_difference"] == 1 / 256 for c in result)
    assert len(seen) == 360
    assert all(int(rows[i]["lineage"].split("-")[1]) < 24 for i in seen)


@pytest.mark.parametrize("kind", ["size", "role", "groups", "cells", "nan", "shape", "dtype"])
def test_incomplete_or_invalid_signal_is_not_success(kind):
    rows = samples()
    if kind == "size":
        rows.pop()
    elif kind == "role":
        rows[0]["split"] = "test"
    elif kind == "groups":
        rows[0]["lineage"] = "extra"
    elif kind == "cells":
        for r in rows:
            if r["method"] == "UERD":
                r["label"] = "cover"

    def bad(indices):
        result = fetch(indices)
        if kind == "nan":
            result[0, 0, 0, 0] = np.nan
        elif kind == "shape":
            result = result[:2]
        elif kind == "dtype":
            result = result.astype(np.float64)
        return result

    with pytest.raises(ValueError):
        signal.summarize(rows, bad)


def test_reader_is_reverified_and_missing_signal_is_partial(monkeypatch):
    class Reader:
        samples = samples()
        verified = False

        def __init__(self, root, audit, *, deadline):
            assert root == Path("root") and audit == Path("audit")
            deadline()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def batch(self, indices):
            return np.zeros((4, 1, 256, 256), dtype="<f4")

        def verify(self):
            Reader.verified = True

    monkeypatch.setattr(signal, "TimingReader", Reader)
    report = signal.inspect_fit_signal(Path("root"), Path("audit"))
    assert Reader.verified and report["identical_crops"] == 240
    assert report["crop_signal_coverage"] == "partial"
    assert report["probe_pixels_read"] == 0 and report["optimizer_updates"] == 0
    assert report["accuracy_qualification"] == "unavailable" and report["deployed"] is False


def test_physical_crop_report_preserves_partial_signal_and_source_binding():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "benchmarks/jpeg-train-crop-signal-20261010.json").read_bytes())
    assert (
        report["source_sha256"]["core/jpeg_crop_signal.py"]
        == hashlib.sha256((root / "core/jpeg_crop_signal.py").read_bytes()).hexdigest()
    )
    assert report["manifest_sha256"] == signal.MANIFEST_SHA
    assert report["audit_sha256"] == signal.AUDIT_SHA
    assert report["fit_pairs"] == 240 and report["identical_crops"] == 3
    assert report["crop_signal_coverage"] == "partial"
    assert report["probe_pixels_read"] == 0 and report["optimizer_updates"] == 0
    assert len(report["cells"]) == 10
    affected = [r for r in report["cells"] if r["identical_crops"]]
    assert len(affected) == 1
    assert affected[0]["source_group"] == "ALASKA2" and affected[0]["method"] == "UERD"
