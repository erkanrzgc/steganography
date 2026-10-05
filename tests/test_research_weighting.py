import copy
import hashlib
import json

import numpy as np
import pytest

from core.spatial_parity import FEATURE_NAMES
from steganography.research import ResearchManifestError, train_model
from steganography.research_features import training_inputs
from steganography.research_weighting import RECIPE, training_weights
from tests.test_research_features import setup_features


def weighted_fixture(tmp_path):
    _, manifest, path = setup_features(tmp_path)
    template = manifest["samples"][0]
    rows = []
    for index, (method, rate) in enumerate(
        [(None, 0), *((m, r) for m in ("sequential", "scattered") for r in (5, 20, 40))]
    ):
        rows.append(
            {
                **template,
                "path": f"variant-{index}.png",
                "sha256": hashlib.sha256(f"variant-{index}".encode()).hexdigest(),
                "label": "cover" if method is None else "stego",
                "method": method,
                "rate_percent": rate,
            }
        )
    manifest["samples"] = rows + manifest["samples"][2:]
    path.write_text(json.dumps(manifest))
    artifact = tmp_path / "weighted-features.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "research-features-1",
                "feature_version": "spatial-parity-residual-v1",
                "feature_names": list(FEATURE_NAMES),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "split": "train",
                "rows": [
                    {
                        **{k: row[k] for k in ("sha256", "lineage", "label")},
                        "values": [index / 10] * len(FEATURE_NAMES),
                    }
                    for index, row in enumerate(rows)
                ],
            }
        )
    )
    config = {
        "manifest": str(path),
        "features": str(artifact),
        "features_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "sample_weighting": RECIPE,
        "epochs": 1,
        "learning_rate": 0.01,
        "seed": 42,
        "standardize": False,
        "class_balanced": True,
    }
    return config, path


def test_train_weights_and_independent_optimizer_oracle(tmp_path):
    torch = pytest.importorskip("torch")
    config, _ = weighted_fixture(tmp_path)
    features, labels, provenance = training_inputs(config)
    weights = training_weights(config, provenance)
    assert weights.tolist() == [1, 4, 1, 1, 4, 1, 1]
    torch.manual_seed(42)
    model = torch.nn.Linear(684, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    logits = model(torch.from_numpy(features)).reshape(-1)
    y = torch.from_numpy(labels)
    # Independently express the stable BCE and weighted positive class mass.
    terms = (1 - y) * torch.nn.functional.softplus(logits)
    terms += y * torch.nn.functional.softplus(-logits) / 12
    expected = (terms * torch.tensor([1, 4, 1, 1, 4, 1, 1])).mean()
    expected.backward()
    optimizer.step()
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    checkpoint_path = tmp_path / "weighted.pt"
    report = train_model(config_path, checkpoint_path)
    assert report["loss"] == pytest.approx(float(expected.detach()), abs=1e-7)
    saved = torch.load(checkpoint_path, weights_only=True)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, saved["state_dict"][key], atol=1e-7, rtol=0)
    recipe = report["training_provenance"]["training"]["sample_weighting"]
    assert recipe["positive_mass"] == 12 and recipe["negative_mass"] == 1
    assert recipe["positive_class_weight"] == pytest.approx(1 / 12)
    assert str(tmp_path) not in json.dumps(report)
    config["class_balanced"] = False
    config_path.write_text(json.dumps(config))
    unbalanced = train_model(config_path, tmp_path / "unbalanced.pt")
    assert (
        unbalanced["training_provenance"]["training"]["sample_weighting"]["positive_class_weight"]
        == 1
    )


@pytest.mark.parametrize("value", [None, True, [], {}, "unknown"])
def test_unknown_recipe(tmp_path, value):
    config, _ = weighted_fixture(tmp_path)
    _, _, provenance = training_inputs(config)
    config["sample_weighting"] = value
    with pytest.raises(ResearchManifestError, match="unknown"):
        training_weights(config, provenance)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("feature_version", "spatial-summary-v1", "contract"),
        ("manifest_sha256", "0" * 64, "manifest"),
        ("split", "validation", "manifest"),
        ("split", "test", "manifest"),
        ("samples", 8, "count"),
    ],
)
def test_wrong_provenance(tmp_path, key, value, match):
    config, _ = weighted_fixture(tmp_path)
    _, _, provenance = training_inputs(config)
    provenance[key] = value
    with pytest.raises(ResearchManifestError, match=match):
        training_weights(config, provenance)


@pytest.mark.parametrize("mutation", ["missing", "recipe", "label", "duplicate"])
def test_incomplete_training_recipe(tmp_path, mutation):
    config, path = weighted_fixture(tmp_path)
    _, _, provenance = training_inputs(config)
    manifest = json.loads(path.read_text())
    if mutation == "missing":
        manifest["samples"].pop(1)
        provenance["samples"] -= 1
    elif mutation == "recipe":
        manifest["samples"][1]["rate_percent"] = 10
    elif mutation == "label":
        manifest["samples"][1]["label"] = "cover"
    else:
        manifest["samples"][1].update(method="scattered", rate_percent=20)
    path.write_text(json.dumps(manifest))
    provenance["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ResearchManifestError, match="complete controlled"):
        training_weights(config, provenance)


def test_default_does_not_read_manifest_and_validation_metadata_unused(tmp_path):
    assert training_weights({}, {}) is None
    config, path = weighted_fixture(tmp_path)
    _, _, provenance = training_inputs(config)
    expected = training_weights(config, provenance)
    manifest = json.loads(path.read_text())
    for sample in manifest["samples"][7:]:
        sample.update(method="unsupported", rate_percent=99)
    path.write_text(json.dumps(manifest))
    provenance = copy.deepcopy(provenance)
    provenance["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(training_weights(config, provenance), expected)
