import copy
import hashlib
import json

import numpy as np
import pytest

from core.feature_model import feature_model
from core.jpeg_features import FEATURE_VERSION
from steganography import research_jpeg as rj
from steganography.research import (
    ResearchManifestError,
    export_onnx,
    train_model,
    verify_dataset_manifest,
)
from steganography.research_features import extract_features, feature_inputs, training_inputs
from tests.test_alaska2_download import setup
from tests.test_research_features import config_for


@pytest.fixture
def development(tmp_path, monkeypatch):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch)
    root = tmp_path / "development"
    source = module.acquire(root, credential, [reserved], count=4, purpose="development", seed=2)
    return root, source, reserved


def import_args(root, reserved):
    return {
        "source_sha256": hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        "reserved_manifests": [reserved],
    }


def write_source(root, source):
    (root / "source.json").write_text(json.dumps(source))


def test_development_import_isolation_and_ambiguity_quarantine(development, tmp_path):
    root, source, reserved = development
    out = tmp_path / "manifest.json"
    result = rj.import_development(root, out, **import_args(root, reserved))
    assert len(result["samples"]) == 16 and not result["quarantined_lineages"]
    assert result["partition"]["test_sources"] == []
    assert {r["split"] for r in result["samples"]} == {"train", "validation"}
    verify_dataset_manifest(result, source=root)
    with pytest.raises(FileExistsError):
        rj.import_development(root, out, **import_args(root, reserved))
    broken = copy.deepcopy(result)
    broken["samples"][0]["split"] = "test"
    with pytest.raises(ResearchManifestError):
        verify_dataset_manifest(broken, verify_files=False)
    broken = copy.deepcopy(result)
    broken["partition"]["test_sources"] = ["ALASKA2"]
    with pytest.raises(ResearchManifestError):
        verify_dataset_manifest(broken, verify_files=False)
    cover, uerd = source["samples"][0], source["samples"][3]
    (root / uerd["path"]).write_bytes((root / cover["path"]).read_bytes())
    uerd.update(sha256=cover["sha256"], size=cover["size"])
    write_source(root, source)
    quarantine = rj.import_development(
        root, tmp_path / "quarantine.json", **import_args(root, reserved)
    )
    assert len(quarantine["samples"]) == 12
    assert quarantine["quarantined_lineages"] == [cover["sha256"]]
    assert all(s["lineage"] != cover["sha256"] for s in quarantine["samples"])
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rj.write_json(link / "new.json", {})


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda s: s.update(purpose="evaluation"), "development acquisition"),
        (lambda s: s.update(selection_sha256="0" * 64), "selection checksum"),
        (lambda s: s.update(samples=[]), "count"),
        (lambda s: s["samples"].pop(), "membership"),
        (lambda s: s["samples"][0].update(source_group="other"), "path/source"),
        (lambda s: s["samples"][0].update(split="train"), "split policy"),
        (lambda s: s["samples"][0].update(label="unknown"), "labels"),
        (lambda s: s["samples"][0].update(method="UERD"), "labels"),
        (lambda s: s["samples"][0].update(size=0), "integrity"),
        (lambda s: s["samples"][0].update(sha256="a" * 64), "frozen"),
        (lambda s: s["samples"][0].update(lineage="changed"), "lineage mismatch"),
        (lambda s: s.update(reserved_manifest_sha256=[]), "reserved acquisition"),
    ],
)
def test_development_refuses_modified_provenance(development, tmp_path, mutation, match):
    root, source, reserved = development
    mutation(source)
    write_source(root, source)
    with pytest.raises(ResearchManifestError, match=match):
        rj.import_development(root, tmp_path / "out", **import_args(root, reserved))


def test_development_symlinks_missing_reserved_and_checksums(development, tmp_path):
    root, source, reserved = development
    out = tmp_path / "out"
    with pytest.raises(ResearchManifestError, match="checksum"):
        rj.import_development(root, out, source_sha256="0" * 64, reserved_manifests=[reserved])
    options = import_args(root, reserved)
    with pytest.raises(ResearchManifestError, match="mandatory"):
        rj.import_development(root, out, **{**options, "reserved_manifests": []})
    for rows, error in (([], "empty"), ([{}], "identity")):
        reserved.write_text(json.dumps({"samples": rows}))
        with pytest.raises(ResearchManifestError, match=error):
            rj.import_development(root, out, **options)
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64, "lineage": "b" * 64}]}))
    image = root / source["samples"][0]["path"]
    data = image.read_bytes()
    target = tmp_path / "image.jpg"
    target.write_bytes(data)
    image.unlink()
    image.symlink_to(target)
    with pytest.raises(ResearchManifestError, match="regular nonsymlink"):
        rj.import_development(root, out, **options)


def test_jpeg_training_validation_export_and_no_test_access(development, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("jpeglib")
    root, _, reserved = development
    manifest = tmp_path / "manifest.json"
    rj.import_development(root, manifest, **import_args(root, reserved))
    configs = {}
    for split in ("train", "validation"):
        artifact = tmp_path / f"{split}-features.json"
        report = extract_features(
            manifest, artifact, source=root, split=split, feature_version=FEATURE_VERSION, workers=2
        )
        assert report["samples"] == 8
        config = {**config_for(manifest, artifact), "epochs": 3, "seed": 42, "learning_rate": 0.01}
        config_path = tmp_path / f"{split}.json"
        config_path.write_text(json.dumps(config))
        configs[split] = (config_path, config)
    with pytest.raises(ResearchManifestError, match="train or validation"):
        feature_inputs(configs["validation"][1], split="test")
    checkpoint = tmp_path / "model.pt"
    training = train_model(configs["train"][0], checkpoint)
    assert training["training_provenance"]["training"]["class_balanced"]
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["domain"] == "jpeg-dct-summary-linear-v1"
    x, _, _ = training_inputs(configs["train"][1])
    np.testing.assert_allclose(saved["preprocessing"]["normalization"]["mean"], x.mean(axis=0))
    assert saved["preprocessing"]["normalization"]["method"].startswith("train-only")
    output = tmp_path / "predictions.json"
    predictions = rj.predict_validation(configs["validation"][0], checkpoint, output)
    assert len(predictions["predictions"]) == 8 and not predictions["calibrated"]
    assert all(0 <= r["score"] <= 1 for r in predictions["predictions"])
    assert str(tmp_path) not in output.read_text()
    with pytest.raises(FileExistsError):
        rj.predict_validation(configs["validation"][0], checkpoint, output)
    with pytest.raises(ResearchManifestError, match="contract"):
        rj.predict_validation(configs["train"][0], checkpoint, tmp_path / "wrong-split")
    with pytest.raises(ResearchManifestError, match="bounded checkpoint"):
        rj.predict_validation(
            configs["validation"][0], tmp_path / "missing", tmp_path / "missing-out"
        )
    modified = copy.deepcopy(saved)
    modified["training_provenance"]["manifest_sha256"] = "0" * 64
    bad = tmp_path / "bad.pt"
    torch.save(modified, bad)
    with pytest.raises(ResearchManifestError, match="contract"):
        rj.predict_validation(configs["validation"][0], bad, tmp_path / "bad-out")
    modified = copy.deepcopy(saved)
    modified["state_dict"]["weight"][:] = float("nan")
    torch.save(modified, bad)
    with pytest.raises(ResearchManifestError, match="nonfinite"):
        rj.predict_validation(configs["validation"][0], bad, tmp_path / "nan-out")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    onnx_path = tmp_path / "jpeg.onnx"
    card = export_onnx(checkpoint, onnx_path)
    assert card["preprocessing"] == saved["preprocessing"]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = feature_model(saved)
    np.testing.assert_allclose(
        session.run(["logit"], {"input": x})[0],
        model(torch.from_numpy(x)).detach().numpy(),
        atol=1e-6,
    )


def test_research_jpeg_cli(development, tmp_path, monkeypatch, capsys):
    root, _, reserved = development
    args = import_args(root, reserved)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jpeg",
            "import-development",
            "--source",
            str(root),
            "--source-sha256",
            args["source_sha256"],
            "--reserved-manifest",
            str(reserved),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    rj.main()
    assert "development rows: 16" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        rj.main()
    assert exc.value.code == 2 and str(tmp_path) not in capsys.readouterr().err
    monkeypatch.setattr(
        "sys.argv",
        [
            "jpeg",
            "predict-validation",
            "--config",
            "fixture.json",
            "--checkpoint",
            "fixture.pt",
            "--out",
            "fixture-out",
        ],
    )
    monkeypatch.setattr(rj, "predict_validation", lambda *args: {"predictions": [1, 2]})
    rj.main()
    assert "validation predictions: 2" in capsys.readouterr().out


def test_fixed_development_experiment_and_cli(development, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("jpeglib")
    root, _, reserved = development
    out = tmp_path / "experiment"
    report = rj.run_development(root, out, **import_args(root, reserved))
    assert report["scope"] == "same-source validation only"
    assert report["features"]["train"]["samples"] == 8
    assert not report["deployed"] and not report["old_test_images_accessed"]
    assert report["training"]["epochs"] == 300
    for metrics in report["by_method"].values():
        assert metrics["samples"] == 4 and metrics["threshold"] == 50
        assert metrics["positives"] == metrics["negatives"] == 2
        assert "recommended_threshold" not in metrics
    with pytest.raises(FileExistsError):
        rj.run_development(root, out, **import_args(root, reserved))
    link = tmp_path / "experiment-link"
    link.symlink_to(out)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rj.run_development(root, link, **import_args(root, reserved))
    monkeypatch.setattr(rj, "run_development", lambda *a, **kw: report)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jpeg",
            "run-development",
            "--source",
            str(root),
            "--source-sha256",
            import_args(root, reserved)["source_sha256"],
            "--reserved-manifest",
            str(reserved),
            "--out",
            str(out),
        ],
    )
    rj.main()
    assert "development complete; deployed=False" in capsys.readouterr().out


def test_bad_feature_normalization_is_rejected():
    torch = pytest.importorskip("torch")
    linear = torch.nn.Linear(2, 1)
    for normalization in (
        {"method": "other", "mean": [0, 0], "scale": [1, 1]},
        {"method": "train-only-standardization", "mean": [0], "scale": [1, 1]},
        {"method": "train-only-standardization", "mean": [0, 0], "scale": [1, 0]},
        {"method": "train-only-standardization", "mean": [0, float("nan")], "scale": [1, 1]},
    ):
        with pytest.raises(ValueError, match="normalization"):
            feature_model(
                {
                    "features": 2,
                    "state_dict": linear.state_dict(),
                    "preprocessing": {"normalization": normalization},
                }
            )
