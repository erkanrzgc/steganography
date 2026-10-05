import hashlib
import json

import numpy as np
import pytest

from core import jpeg_context as jc
from core import jpeg_features as jf
from steganography.research import ResearchManifestError, export_onnx, train_model
from steganography.research_features import feature_contract, training_inputs
from steganography.research_weighting import JPEG_RECIPE, training_weights
from tests.test_research_features import setup_features


def summary():
    coefficients = np.zeros((2, 2, 8, 8), dtype=np.int16)
    coefficients[0, 0, 0, 1] = 7
    return jf.coefficient_features(coefficients, np.full((8, 8), 32))


def test_context_scalar_oracle_and_no_metadata_inputs(monkeypatch):
    original = summary()
    values = jc.summary_context_features(original)
    assert len(values) == len(jc.FEATURE_NAMES) == 1098
    np.testing.assert_allclose(values[:968], original, atol=5e-9, rtol=0)
    assert values[968] == 0.125  # clipped magnitude mean 1/4 times Q strength 1/2
    entropy = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)) / 3
    assert values[968 + 63] == pytest.approx(entropy * 0.5, abs=5e-9)
    assert values[-4:] == pytest.approx([0.5, 0.25 / 63, entropy / 63, 0.25 / 63], abs=5e-9)
    assert np.isfinite(values).all() and min(values) >= 0 and max(values) <= 1
    monkeypatch.setattr(jf, "jpeg_features", lambda data: original)
    assert jc.jpeg_context_features(b"image bytes") == values
    assert feature_contract(jc.FEATURE_VERSION)[0] == jc.FEATURE_NAMES
    doubled_q = original.copy()
    doubled_q[-64:] = [64 / 65535] * 64
    assert jc.summary_context_features(doubled_q)[968] == pytest.approx(1 / 6, abs=5e-9)
    assert jc.summary_context_features(doubled_q)[-4] == pytest.approx(2 / 3, abs=5e-9)


@pytest.mark.parametrize("mutation", ["shape", "nan", "negative", "range", "sum", "zero-q"])
def test_invalid_context_vectors(mutation):
    values = summary()
    if mutation == "shape":
        values.pop()
    elif mutation == "nan":
        values[0] = float("nan")
    elif mutation == "negative":
        values[0] = -1
    elif mutation == "range":
        values[0] = 2
    elif mutation == "sum":
        values[0] = 0.5
    else:
        values[-1] = 0
    with pytest.raises(ValueError):
        jc.summary_context_features(values)


@pytest.fixture
def multi_source(tmp_path):
    _, manifest, path = setup_features(tmp_path)
    samples = []
    for index, (source, label) in enumerate(
        [
            ("one", "cover"),
            ("one", "stego"),
            ("one", "stego"),
            ("two", "cover"),
            ("two", "stego"),
        ]
    ):
        samples.append(
            {
                **manifest["samples"][0],
                "path": f"{index}.jpg",
                "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "source_group": source,
                "lineage": f"lineage-{source}",
                "label": label,
                "method": None if label == "cover" else "UERD",
                "format": "JPEG",
            }
        )
    manifest["samples"] = samples + manifest["samples"][2:]
    manifest["catalog"] = {
        "source_records": {
            source: {
                "origin_manifest_sha256": str(i) * 64,
                "license": "fixture",
                "source_url": "https://example.org/fixture",
            }
            for i, source in enumerate(("one", "two"), 1)
        }
    }
    path.write_text(json.dumps(manifest))
    features = tmp_path / "context.json"
    features.write_text(
        json.dumps(
            {
                "schema_version": "research-features-1",
                "feature_version": jc.FEATURE_VERSION,
                "feature_names": list(jc.FEATURE_NAMES),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split": "train",
                "rows": [
                    {
                        **{k: s[k] for k in ("sha256", "lineage", "label")},
                        "values": jc.summary_context_features(summary()),
                    }
                    for s in samples
                ],
            }
        )
    )
    config = {
        "manifest": str(path),
        "features": str(features),
        "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        "sample_weighting": JPEG_RECIPE,
        "epochs": 2,
        "inference_arithmetic": "float64",
    }
    return config, path


def test_balanced_source_label_mass_and_context_training(multi_source, tmp_path):
    torch = pytest.importorskip("torch")
    config, _ = multi_source
    _, labels, provenance = training_inputs(config)
    weights = training_weights(config, provenance)
    assert weights.tolist() == pytest.approx([1.25, 0.625, 0.625, 1.25, 1.25])
    assert weights[labels == 0].sum() == weights[labels == 1].sum() == 2.5
    assert provenance["source_balance"]["declared_sources"] == 2
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    checkpoint_path = tmp_path / "model.pt"
    result = train_model(config_path, checkpoint_path)
    saved = torch.load(checkpoint_path, weights_only=True)
    assert saved["domain"] == "jpeg-context-summary-linear-v1"
    assert (
        result["training_provenance"]["training"]["sample_weighting"]["positive_class_weight"] == 1
    )
    pytest.importorskip("onnxruntime")
    card = export_onnx(checkpoint_path, tmp_path / "model.onnx")
    assert card["training_provenance"]["source_balance"]["declared_sources"] == 2


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("no-recipe", "explicit multi-source"),
        ("wrong-contract", "JPEG context contract"),
        ("one-source", "two declared"),
        ("unknown-source", "named source"),
        ("missing-records", "origin records"),
        ("bad-record", "origin/license"),
        ("renamed-origin", "renamed copies"),
        ("missing-label", "both labels"),
        ("missing-method", "must be declared"),
        ("source-method-shortcut", "training shortcut"),
    ],
)
def test_source_guards(multi_source, mutation, match):
    config, path = multi_source
    _, _, provenance = training_inputs(config)
    manifest = json.loads(path.read_bytes())
    if mutation == "no-recipe":
        config.pop("sample_weighting")
    elif mutation == "wrong-contract":
        provenance["feature_version"] = jf.FEATURE_VERSION
    elif mutation == "one-source":
        for s in manifest["samples"][:5]:
            s["source_group"] = "one"
    elif mutation == "unknown-source":
        manifest["samples"][0]["source_group"] = None
    elif mutation == "missing-records":
        manifest["catalog"] = None
    elif mutation == "bad-record":
        manifest["catalog"]["source_records"]["one"] = None
    elif mutation == "renamed-origin":
        manifest["catalog"]["source_records"]["two"]["origin_manifest_sha256"] = "1" * 64
    elif mutation == "missing-label":
        manifest["samples"][4]["label"] = "cover"
    elif mutation == "missing-method":
        manifest["samples"][4]["method"] = None
    else:
        manifest["samples"][4]["method"] = "JMiPOD"
    path.write_text(json.dumps(manifest))
    provenance["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ResearchManifestError, match=match):
        training_weights(config, provenance)
