import hashlib
import json

import pytest
from PIL import Image

from core import jpeg_context as jc
from core import jpeg_features as jf
from steganography import research_jpeg_multisource as ms
from steganography.research import ResearchManifestError
from tests.test_jpeg_context import summary


@pytest.fixture
def inputs(tmp_path):
    manifests = []
    roots = []
    for source_index, group in enumerate(("ALASKA2", "BOSSbase-1.01")):
        root = tmp_path / f"source{source_index}"
        root.mkdir()
        rows = []
        for i, (split, label) in enumerate(
            [(s, label) for s in ("train", "validation") for label in ("cover", "stego")]
        ):
            path = root / f"{i}.jpg"
            Image.new("L", (32, 32), source_index * 100 + i * 20).save(
                path, format="JPEG", quality=75
            )
            rows.append(
                {
                    "path": path.name,
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "lineage": f"original-{split}",
                    "source_group": group,
                    "split": split,
                    "label": label,
                    "method": None if label == "cover" else "UERD",
                    "format": "JPEG",
                }
            )
        record = {
            "origin_manifest_sha256": str(source_index + 1) * 64,
            "license": "fixture",
            "source_url": "https://example.org/fixture",
        }
        manifest = {
            "schema_version": "1.0",
            "source": ".",
            "samples": rows,
            "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
            "source_sha256": record["origin_manifest_sha256"],
            "catalog": {**record, "source_records": {group: record}},
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        if source_index == 0:
            for split in ("train", "validation"):
                path = root / f"{split}-features.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "research-features-1",
                            "feature_version": jf.FEATURE_VERSION,
                            "feature_names": list(jf.FEATURE_NAMES),
                            "manifest_sha256": hashlib.sha256(
                                (root / "manifest.json").read_bytes()
                            ).hexdigest(),
                            "split": split,
                            "rows": [
                                {
                                    **{k: s[k] for k in ("sha256", "lineage", "label")},
                                    "values": summary(),
                                }
                                for s in rows
                                if s["split"] == split
                            ],
                        }
                    )
                )
                (root / f"{split}-config.json").write_text(
                    json.dumps(
                        {
                            "manifest": str(root / "manifest.json"),
                            "features": str(path),
                            "features_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                )
        else:
            path = root / "feature-rows.json"
            path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                **{k: s[k] for k in ("sha256", "lineage", "label")},
                                "values": jc.summary_context_features(summary()),
                            }
                            for s in rows
                        ]
                    }
                )
            )
            manifest["feature_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
        manifests.append(manifest)
        roots.append(root)
    return roots, manifests


def prepare(inputs, out, **overrides):
    roots, _ = inputs
    hashes = {
        "alaska_manifest_sha256": hashlib.sha256(
            (roots[0] / "manifest.json").read_bytes()
        ).hexdigest(),
        "boss_manifest_sha256": hashlib.sha256(
            (roots[1] / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    return ms.prepare_dataset(roots[0], roots[0], roots[1], out, **{**hashes, **overrides})


def test_two_origin_preparation_no_original_changes(inputs, tmp_path):
    roots, _ = inputs
    before = {p: p.read_bytes() for root in roots for p in root.iterdir()}
    result = prepare(inputs, tmp_path / "combined")
    assert result["sources"] == 2 and result["jpeg_files"] == 8
    assert result["feature_artifacts"]["train"]["samples"] == 4
    assert result["feature_version"] == jc.FEATURE_VERSION and not result["deployed"]
    assert all(p.read_bytes() == data for p, data in before.items())
    assert str(tmp_path) not in json.dumps(result)
    with pytest.raises(FileExistsError):
        prepare(inputs, tmp_path / "combined")
    with pytest.raises(ResearchManifestError, match="checksum"):
        prepare(inputs, tmp_path / "bad", alaska_manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "fault,match", [("hash", "checksum"), ("identity", "identity"), ("values", "feature vector")]
)
def test_boss_feature_failures(inputs, tmp_path, fault, match):
    roots, manifests = inputs
    path = roots[1] / "feature-rows.json"
    artifact = json.loads(path.read_bytes())
    if fault == "identity":
        artifact["rows"][0]["lineage"] = "different"
    elif fault == "values":
        artifact["rows"][0]["values"] = []
    else:
        artifact["rows"][0]["values"][0] = 0.9
    path.write_text(json.dumps(artifact))
    if fault != "hash":
        manifests[1]["feature_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (roots[1] / "manifest.json").write_text(json.dumps(manifests[1]))
    with pytest.raises(ResearchManifestError, match=match):
        prepare(inputs, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_image_corruption_and_limits(inputs, tmp_path, monkeypatch):
    roots, _ = inputs
    with monkeypatch.context() as patch:
        patch.setattr(ms, "MAX_DOCUMENT_BYTES", 1)
        with pytest.raises(ResearchManifestError, match="artifact exceeds"):
            prepare(inputs, tmp_path / "limit")
    image = roots[0] / "0.jpg"
    image.write_bytes(b"altered")
    with pytest.raises(ResearchManifestError, match="size/hash"):
        prepare(inputs, tmp_path / "bad-image")
    assert not (tmp_path / "bad-image/preparation.json").exists()


def rebind_original(inputs, mutate):
    roots, manifests = inputs
    manifest = manifests[0]
    mutate(manifest)
    (roots[0] / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((roots[0] / "manifest.json").read_bytes()).hexdigest()
    for split in ("train", "validation"):
        path = roots[0] / f"{split}-features.json"
        artifact = json.loads(path.read_bytes())
        artifact["manifest_sha256"] = digest
        artifact["rows"] = [
            {**{k: s[k] for k in ("sha256", "lineage", "label")}, "values": summary()}
            for s in manifest["samples"]
            if s["split"] == split
        ]
        path.write_text(json.dumps(artifact))
        config_path = roots[0] / f"{split}-config.json"
        config = json.loads(config_path.read_bytes())
        config["features_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config))


@pytest.mark.parametrize(
    "value,message", [(None, "must be declared"), ("other", "declaration mismatch")]
)
def test_source_declarations(inputs, tmp_path, value, message):
    rebind_original(inputs, lambda m: m["samples"][0].update(source_group=value))
    with pytest.raises(ResearchManifestError, match=message):
        prepare(inputs, tmp_path / "bad")


def test_large_metadata_budget_and_cached_contract(inputs, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        original = ms.feature_inputs

        def wrong_provenance(*a, **k):
            x, y, p = original(*a, **k)
            p["manifest_sha256"] = "0" * 64
            return x, y, p

        patch.setattr(ms, "feature_inputs", wrong_provenance)
        with pytest.raises(ResearchManifestError, match="cached feature contract"):
            prepare(inputs, tmp_path / "contract")
    rebind_original(inputs, lambda m: m["samples"][0].update(size=3 * 1024**3))
    with pytest.raises(ResearchManifestError, match="byte limit"):
        prepare(inputs, tmp_path / "budget")
    assert not (tmp_path / "budget").exists()


def test_matched_family_filter_never_reads_excluded_images(inputs, tmp_path):
    def mutate(manifest):
        manifest["samples"].append(
            {
                **manifest["samples"][1],
                "path": "absent-JMiPOD.jpg",
                "sha256": "3" * 64,
                "method": "JMiPOD",
            }
        )

    rebind_original(inputs, mutate)
    result = prepare(inputs, tmp_path / "matched")
    assert result["jpeg_files"] == 8
    manifest = json.loads((tmp_path / "matched/manifest.json").read_bytes())
    assert {s["method"] for s in manifest["samples"]} == {None, "UERD"}


def test_source_and_target_symlink_rejection(inputs, tmp_path, monkeypatch):
    roots, _ = inputs
    image = roots[0] / "0.jpg"
    data = image.read_bytes()
    image.unlink()
    image.symlink_to(roots[1] / "0.jpg")
    with pytest.raises(ResearchManifestError, match="nonsymlink"):
        prepare(inputs, tmp_path / "source-link")
    image.unlink()
    image.write_bytes(data)
    original_mkdir = type(tmp_path).mkdir

    def injected_mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path.name == "origin0":
            path.rmdir()
            path.symlink_to(roots[1], target_is_directory=True)
        return result

    monkeypatch.setattr(type(tmp_path), "mkdir", injected_mkdir)
    with pytest.raises(ResearchManifestError, match="artifact uses a symlink"):
        prepare(inputs, tmp_path / "target-link")
