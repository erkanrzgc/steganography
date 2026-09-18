import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from cli import main
from core import features
from steganography import research_features as rf
from steganography.research import ResearchManifestError


def setup_features(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    samples = []
    for index, split in enumerate(("train", "validation", "test")):
        for label in ("cover", "stego"):
            path = root / f"{split}-{label}.png"
            Image.fromarray(
                np.full((4, 4, 3), index * 30 + (label == "stego"), dtype=np.uint8)
            ).save(path)
            data = path.read_bytes()
            samples.append(
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "label": label,
                    "split": split,
                    "lineage": f"original-{index}",
                    "source_group": "held-out" if split == "test" else "development",
                }
            )
    manifest = {
        "schema_version": "1.0",
        "source": str(root),
        "samples": samples,
        "partition": {
            "policy": "identity-camera-device-components-v1",
            "test_sources": ["held-out"],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, manifest, path


def config_for(path, artifact):
    return {
        "manifest": str(path),
        "features": str(artifact),
        "features_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def test_exact_features_and_dimensions(tmp_path, monkeypatch):
    stream = io.BytesIO()
    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(stream, format="PNG")
    result = features.spatial_features(stream.getvalue())
    assert len(result) == 12
    assert result[:4] == pytest.approx([1.5 / 255, 0.5 / 255, 0.5, 0.5])
    assert result[:4] == result[4:8] == result[8:]
    for image, fmt, error in (
        (Image.new("RGB", (1, 2)), "PNG", "dimensions"),
        (Image.new("RGB", (2, 2)), "JPEG", "PNG/BMP"),
    ):
        stream = io.BytesIO()
        image.save(stream, format=fmt)
        with pytest.raises(ValueError, match=error):
            features.spatial_features(stream.getvalue())
    stream = io.BytesIO()
    Image.new("RGB", (2, 2)).save(stream, format="BMP")
    assert features.spatial_features(stream.getvalue()) == [0, 0, 0, 1] * 3
    monkeypatch.setattr(features, "MAX_PIXELS", 3)
    with pytest.raises(ValueError, match="dimensions"):
        features.spatial_features(stream.getvalue())
    monkeypatch.setattr(features, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        features.spatial_features(b"xx")


def test_extract_train_only_roundtrip_cli(tmp_path, capsys):
    root, manifest, path = setup_features(tmp_path)
    # Test/validation images are unavailable: train extraction must not read them.
    for sample in manifest["samples"][2:]:
        (root / sample["path"]).rename(tmp_path / sample["path"])
    artifact = tmp_path / "features.json"
    assert (
        main(
            [
                "--quiet",
                "research",
                "features",
                "--manifest",
                str(path),
                "--source",
                str(root),
                "--out",
                str(artifact),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["samples"] == 2
    x, y, provenance = rf.training_inputs(config_for(path, artifact))
    assert x.shape == (2, 12) and x.dtype == np.float32
    assert y.tolist() == [0, 1]
    assert provenance["features_sha256"] == summary["sha256"]
    assert provenance["split"] == "train"
    assert str(tmp_path) not in artifact.read_text()
    before = artifact.read_bytes()
    with pytest.raises(FileExistsError):
        rf.extract_features(path, artifact)
    assert artifact.read_bytes() == before


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda a: a.update(split="validation"), "contract"),
        (lambda a: a.update(split="test"), "contract"),
        (lambda a: a.update(manifest_sha256="0" * 64), "contract"),
        (lambda a: a.update(feature_names=[]), "contract"),
        (lambda a: a.update(rows=[]), "complete training"),
        (lambda a: a["rows"].reverse(), "provenance"),
        (lambda a: a["rows"][0].update(label="stego"), "provenance"),
        (lambda a: a["rows"][0].update(lineage="other"), "provenance"),
        (lambda a: a["rows"][0].update(values=[0]), "vector"),
        (lambda a: a["rows"][0].update(values=["0"] * 12), "vector"),
        (lambda a: a["rows"][0].update(values=[float("nan")] * 12), "finite"),
        (lambda a: a["rows"][0].update(values=[2] * 12), "finite"),
    ],
)
def test_training_rejects_modified_artifacts(tmp_path, mutation, message):
    _, _, path = setup_features(tmp_path)
    artifact = tmp_path / "features.json"
    rf.extract_features(path, artifact)
    document = json.loads(artifact.read_text())
    mutation(document)
    artifact.write_text(json.dumps(document))
    with pytest.raises(ResearchManifestError, match=message):
        rf.training_inputs(config_for(path, artifact))


def test_contract_missing_tampered_and_validation_only(tmp_path):
    _, _, path = setup_features(tmp_path)
    with pytest.raises(ResearchManifestError, match="requires manifest"):
        rf.training_inputs({"features": "legacy.npz"})
    artifact = tmp_path / "features.json"
    rf.extract_features(path, artifact, split="validation")
    config = config_for(path, artifact)
    with pytest.raises(ResearchManifestError, match="contract"):
        rf.training_inputs(config)
    artifact.write_text(artifact.read_text() + " ")
    with pytest.raises(ResearchManifestError, match="checksum"):
        rf.training_inputs(config)


def test_extraction_guards(tmp_path, monkeypatch):
    root, manifest, path = setup_features(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ResearchManifestError, match="train or validation"):
        rf.extract_features(path, out, split="test")
    saved_partition = manifest.pop("partition")
    with pytest.raises(ResearchManifestError, match="partitioned"):
        rf.selected_samples(manifest, "train")
    manifest["partition"] = saved_partition
    monkeypatch.setattr(rf, "MAX_ROWS", 1)
    with pytest.raises(ResearchManifestError, match="row count"):
        rf.selected_samples(manifest, "train")
    monkeypatch.setattr(rf, "MAX_ROWS", 100)
    manifest["samples"][0]["label"] = "stego"
    with pytest.raises(ResearchManifestError, match="both labels"):
        rf.selected_samples(manifest, "train")
    manifest["samples"][0]["label"] = "cover"
    manifest["samples"][0]["lineage"] = None
    with pytest.raises(ResearchManifestError, match="lineage"):
        rf.selected_samples(manifest, "train")
    image = root / manifest["samples"][0]["path"]
    original = image.read_bytes()
    image.write_bytes(b"modified")
    with pytest.raises(ResearchManifestError, match="integrity"):
        rf.extract_features(path, out)
    image.write_bytes(original)
    monkeypatch.setattr(rf, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ResearchManifestError, match="byte limit"):
        rf.extract_features(path, out)
    monkeypatch.setattr(rf, "MAX_IMAGE_BYTES", 1024)
    image.rename(tmp_path / "target.png")
    image.symlink_to(tmp_path / "target.png")
    with pytest.raises(ResearchManifestError, match="symlink"):
        rf.extract_features(path, out)
    assert not out.exists()


def test_bounded_documents(tmp_path, monkeypatch):
    path = tmp_path / "document"
    path.write_text("[]")
    with pytest.raises(ResearchManifestError, match="object"):
        rf.read_document(path)
    monkeypatch.setattr(rf, "MAX_DOCUMENT_BYTES", 1)
    with pytest.raises(ResearchManifestError, match="byte limit"):
        rf.read_document(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rf.read_document(link)
    with pytest.raises(ResearchManifestError, match="regular file"):
        rf.read_document(tmp_path)


def test_output_symlinks_and_missing_selected_image(tmp_path):
    root, manifest, path = setup_features(tmp_path)
    link = tmp_path / "output-link"
    link.symlink_to(tmp_path / "not-created")
    with pytest.raises(FileExistsError):
        rf.extract_features(path, link)
    image = root / manifest["samples"][0]["path"]
    image.rename(tmp_path / "moved.png")
    with pytest.raises(ResearchManifestError, match="regular file"):
        rf.extract_features(path, tmp_path / "out")
