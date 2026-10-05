import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from core import spatial_parity as sp
from steganography import research_parity as rp
from steganography import research_spatial as rs
from steganography.research import ResearchManifestError
from steganography.research_features import feature_inputs
from tests.test_research_spatial import development  # noqa: F401 - fixture


def test_joint_histograms_independent_scalar():
    pixels = np.random.default_rng(4).integers(0, 256, (9, 10, 3), dtype=np.int16)
    for direction in sp.DIRECTIONS:
        for bit in range(3):
            counts = [0] * 18
            for y in range(9):
                for x in range(10):
                    yy = y + (direction != "h")
                    xx = x + (direction in ("h", "d")) - (direction == "a")
                    if not 0 <= yy < 9 or not 0 <= xx < 10:
                        continue
                    for c in range(3):
                        center = int(pixels[y, x, c])
                        neighbor = int(pixels[yy, xx, c])
                        index = ((center >> bit) & 1) * 9 + max(-4, min(4, neighbor - center)) + 4
                        counts[index] += 1
            np.testing.assert_allclose(
                sp.parity_histogram(pixels, direction, bit),
                np.array(counts) / sum(counts),
                rtol=0,
                atol=1e-12,
            )
    with pytest.raises(ValueError, match="plane"):
        sp.parity_histogram(pixels, "h", 3)
    with pytest.raises(ValueError, match="direction"):
        sp.parity_histogram(pixels, "invalid", 0)


def test_features_format_channel_pooling_limits_and_quantization(monkeypatch):
    gray = np.arange(12 * 13, dtype=np.uint8).reshape(12, 13)
    values = []
    for fmt in ("PNG", "BMP"):
        for pixels in (gray, np.repeat(gray[..., None], 3, axis=2)):
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format=fmt)
            values.append(sp.spatial_parity_features(stream.getvalue()))
    assert len(sp.FEATURE_NAMES) == len(values[0]) == 684
    assert len(set(sp.FEATURE_NAMES)) == 684
    for vector in values[1:]:
        np.testing.assert_array_equal(vector, values[0])
    assert np.isfinite(values).all() and np.min(values) >= 0 and np.max(values) <= 1
    assert all(round(v, 8) == v for v in values[0])
    np.testing.assert_allclose(np.array(values[0][468:]).reshape(12, 18).sum(axis=1), 1, atol=1e-7)
    for size, fmt in (((7, 8), "PNG"), ((8, 8), "JPEG")):
        stream = io.BytesIO()
        Image.new("L", size).save(stream, format=fmt)
        with pytest.raises(ValueError):
            sp.spatial_parity_features(stream.getvalue())
    stream = io.BytesIO()
    Image.new("L", (8, 8)).save(stream, format="PNG")
    monkeypatch.setattr(sp, "MAX_PIXELS", 1)
    with pytest.raises(ValueError, match="dimensions"):
        sp.spatial_parity_features(stream.getvalue())
    monkeypatch.setattr(sp, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="byte"):
        sp.spatial_parity_features(b"xx")


def test_comparison_end_to_end_and_reference_integrity(development, tmp_path, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    root, reserved, _ = development
    reference_dir = tmp_path / "reference"
    rs.run_development(
        root,
        reference_dir,
        source_sha256=hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        reserved_manifests=[reserved],
    )
    corpus = reference_dir / "corpus"
    reference = reference_dir / "report.json"
    kwargs = {
        "manifest_sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        "reference_report": reference,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }
    out = tmp_path / "new"
    result = rp.run_comparison(corpus, out, **kwargs)
    assert result["onnx_parity"]["passed"] and not result["deployed"]
    assert len(result["by_method_rate"]) == 6 and str(tmp_path) not in json.dumps(result)
    saved = torch.load(out / "baseline.pt", weights_only=True)
    assert saved["domain"] == "spatial-parity-residual-linear-v1"
    assert saved["preprocessing"]["inference_arithmetic"] == "float64"
    config = json.loads((out / "train-config.json").read_bytes())
    x, _, _ = feature_inputs(config, split="train")
    np.testing.assert_allclose(
        saved["preprocessing"]["normalization"]["mean"], x.mean(axis=0), atol=1e-7
    )
    with pytest.raises(FileExistsError):
        rp.run_comparison(corpus, out, **kwargs)
    with pytest.raises(ResearchManifestError, match="checksum"):
        rp.run_comparison(corpus, tmp_path / "bad", **{**kwargs, "manifest_sha256": "0" * 64})
    altered = json.loads(reference.read_bytes())
    altered["manifest_sha256"] = "0" * 64
    reference.write_text(json.dumps(altered))
    kwargs["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ResearchManifestError, match="corpus mismatch"):
        rp.run_comparison(corpus, tmp_path / "bad", **kwargs)
    altered["manifest_sha256"] = kwargs["manifest_sha256"]
    altered["experiments"]["spatial-cooccurrence-v1"]["prediction_sha256"] = "0" * 64
    reference.write_text(json.dumps(altered))
    kwargs["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ResearchManifestError, match="predictions checksum"):
        rp.run_comparison(corpus, tmp_path / "bad", **kwargs)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        rp.run_comparison(corpus, link / "bad", **kwargs)
    monkeypatch.setattr(rp, "run_comparison", lambda *a, **k: result)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            str(corpus),
            str(out),
            "--manifest-sha256",
            kwargs["manifest_sha256"],
            "--reference-report",
            str(reference),
            "--reference-sha256",
            kwargs["reference_sha256"],
        ],
    )
    rp.main()
