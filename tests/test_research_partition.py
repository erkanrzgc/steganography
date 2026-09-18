import copy
import hashlib
import json

import pytest

from cli import main
from core.dataset import grouped_indices, identity_keys
from steganography.research import (
    ResearchManifestError,
    partition_dataset,
    verify_dataset_manifest,
    verify_split_isolation,
)


def fixture_manifest(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    samples = []
    for index in range(40):
        for label in ("cover", "stego"):
            data = f"{index}-{label}".encode()
            name = f"{index}-{label}.png"
            (root / name).write_bytes(data)
            samples.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "lineage": hashlib.sha256(f"original-{index}".encode()).hexdigest(),
                    "source_group": "held-out" if index >= 30 else "development",
                    "label": label,
                    "size": len(data),
                    "split": "test",
                    "camera": None,
                    "device": None,
                }
            )
    manifest = {"schema_version": "1.0", "source": str(root), "samples": samples}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    return manifest, path, reserved


def test_identity_and_transitive_components():
    assert identity_keys({}) == set()
    assert identity_keys({"lineage": "A" * 64}) == {("hash", "a" * 64)}
    samples = [
        {"lineage": "same", "source_group": "a", "camera": "one"},
        {"lineage": "same", "source_group": "a", "device": "two"},
        {"lineage": "third", "source_group": "a", "device": "two"},
        {"lineage": "fourth", "source_group": "a", "camera": "one"},
        {"lineage": "same", "source_group": "b"},
    ]
    assert grouped_indices(samples) == [[0, 1, 2, 3], [4]]
    samples[0]["split"] = "train"
    samples[1]["split"] = "test"
    assert not verify_split_isolation({"samples": samples[:2]})
    samples[1]["split"] = "train"
    assert verify_split_isolation({"samples": samples[:2]})
    assert grouped_indices([]) == []


def test_partition_reproducible_and_cli(tmp_path, capsys):
    manifest, path, reserved = fixture_manifest(tmp_path)
    original = path.read_bytes()
    options = {"test_sources": ["held-out"], "reserved_manifests": [reserved]}
    out = tmp_path / "partition.json"
    result = partition_dataset(path, out, source=tmp_path / "corpus", **options)
    assert verify_dataset_manifest(result)["split_isolation"]
    assert all(s["split"] == "test" for s in result["samples"] if s["source_group"] == "held-out")
    assert all(
        s["split"] != "test" for s in result["samples"] if s["source_group"] == "development"
    )
    assert all(result["splits"].values())
    assert result["partition"]["group_count"] == 40
    assert not result["partition"]["camera_device_metadata_complete"]
    assert path.read_bytes() == original
    frozen_output = out.read_bytes()
    with pytest.raises(FileExistsError):
        partition_dataset(path, out, **options)
    assert out.read_bytes() == frozen_output
    with pytest.raises(FileExistsError):
        partition_dataset(path, path, **options)
    assert path.read_bytes() == original
    manifest["samples"].reverse()
    path.write_text(json.dumps(manifest))
    reordered = partition_dataset(path, tmp_path / "reordered.json", **options)
    assert {s["sha256"]: s["split"] for s in result["samples"]} == {
        s["sha256"]: s["split"] for s in reordered["samples"]
    }
    assert (
        main(
            [
                "--quiet",
                "research",
                "partition",
                "--manifest",
                str(path),
                "--out",
                str(tmp_path / "cli.json"),
                "--reserved-manifest",
                str(reserved),
                "--test-source",
                "held-out",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["test_sources"] == ["held-out"]


@pytest.mark.parametrize(
    "reserved_record",
    [
        lambda s: {"sha256": s["sha256"]},
        lambda s: {"sha256": s["lineage"]},
        lambda s: {"lineage": s["lineage"], "source_group": "renamed", "device": "changed"},
    ],
)
def test_frozen_overlap_refused(tmp_path, reserved_record):
    manifest, path, reserved = fixture_manifest(tmp_path)
    reserved.write_text(json.dumps({"samples": [reserved_record(manifest["samples"][0])]}))
    out = tmp_path / "out.json"
    with pytest.raises(ResearchManifestError, match="overlaps frozen"):
        partition_dataset(path, out, test_sources=["held-out"], reserved_manifests=[reserved])
    assert not out.exists()


@pytest.mark.parametrize(
    "field,value",
    [("lineage", None), ("lineage", " "), ("source_group", "unspecified"), ("label", "unknown")],
)
def test_partition_requires_explicit_metadata(tmp_path, field, value):
    manifest, path, reserved = fixture_manifest(tmp_path)
    manifest["samples"][0][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ResearchManifestError, match="requires explicit"):
        partition_dataset(
            path, tmp_path / "out", test_sources=["held-out"], reserved_manifests=[reserved]
        )


def test_partition_invalid_sources_reserved_and_small_groups(tmp_path):
    manifest, path, reserved = fixture_manifest(tmp_path)
    out = tmp_path / "out"
    for sources in ([], ["missing"], ["development", "held-out"]):
        with pytest.raises(ResearchManifestError, match="proper subset"):
            partition_dataset(path, out, test_sources=sources, reserved_manifests=[reserved])
    with pytest.raises(ResearchManifestError, match="at least one"):
        partition_dataset(path, out, test_sources=["held-out"], reserved_manifests=[])
    for document in ({"samples": []}, {"samples": [{}]}, {"samples": ["bad"]}):
        reserved.write_text(json.dumps(document))
        with pytest.raises(ResearchManifestError, match="reserved"):
            partition_dataset(path, out, test_sources=["held-out"], reserved_manifests=[reserved])
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    for sample in manifest["samples"]:
        sample["camera"] = "same-camera"
        sample["device"] = "same-device"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ResearchManifestError, match="more independent groups"):
        partition_dataset(path, out, test_sources=["held-out"], reserved_manifests=[reserved])
    manifest["samples"][60]["lineage"] = manifest["samples"][0]["lineage"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ResearchManifestError, match="source boundary"):
        partition_dataset(path, out, test_sources=["held-out"], reserved_manifests=[reserved])
    assert not out.exists()


def test_verifier_rejects_lineage_metadata_bypass_and_symlinks(tmp_path):
    manifest, _, _ = fixture_manifest(tmp_path)
    leaked = copy.deepcopy(manifest)
    leaked["samples"][0].update(
        split="train", camera="changed", device="changed", source_group="renamed"
    )
    with pytest.raises(ResearchManifestError, match="lineage split leakage"):
        verify_dataset_manifest(leaked)
    assert not verify_split_isolation(leaked)
    path = tmp_path / "corpus" / manifest["samples"][0]["path"]
    target = tmp_path / "outside.png"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ResearchManifestError, match="symlinks"):
        verify_dataset_manifest(manifest)


def test_partition_policy_remains_enforced_after_edit(tmp_path):
    _, path, reserved = fixture_manifest(tmp_path)
    result = partition_dataset(
        path, tmp_path / "out", test_sources=["held-out"], reserved_manifests=[reserved]
    )
    for policy in (False, {"policy": "unknown"}):
        edited = copy.deepcopy(result)
        edited["partition"] = policy
        with pytest.raises(ResearchManifestError, match="unsupported partition"):
            verify_dataset_manifest(edited, verify_files=False)
    for sources in ([], "held-out", [None]):
        edited = copy.deepcopy(result)
        edited["partition"]["test_sources"] = sources
        with pytest.raises(ResearchManifestError, match="test source names"):
            verify_dataset_manifest(edited, verify_files=False)
    edited = copy.deepcopy(result)
    edited["partition"]["test_sources"] = ["development"]
    with pytest.raises(ResearchManifestError, match="source isolation"):
        verify_dataset_manifest(edited, verify_files=False)
    for sample in result["samples"]:
        if sample["split"] != "test":
            sample["device"] = "now-shared"
    with pytest.raises(ResearchManifestError, match="device split leakage"):
        verify_dataset_manifest(result, verify_files=False)
