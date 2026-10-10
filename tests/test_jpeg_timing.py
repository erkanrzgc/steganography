"""Independent selection/limit fixtures; generated tensors are not accuracy."""

import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from core import jpeg_timing as kit


def digest(data):
    return hashlib.sha256(data).hexdigest()


def samples(source, count=33):
    result = []
    for i in range(count):
        lineage = digest(f"original-{source}-{i}".encode())
        qualities = (None,) if source == "ALASKA2" else (75, 95)
        for quality in qualities:
            for method in (None, "JUNIWARD", "UERD"):
                data = f"{source}-{i}-{quality}-{method}".encode()
                result.append(
                    {
                        "sha256": digest(data),
                        "lineage": lineage,
                        "source_group": source,
                        "format": "JPEG",
                        "split": "train",
                        "quality_factor": quality,
                        "method": method,
                        "label": "cover" if method is None else "stego",
                        "size": len(data),
                        "path": data.decode(),
                    }
                )
    return result


def test_selection_independent_order_and_complete_families():
    for source in kit.ORIGINS[:2]:
        rows = samples(source)
        before = copy.deepcopy(rows)
        selected = kit.select(rows, source, 32)
        groups = sorted(
            {r["lineage"] for r in rows},
            key=lambda key: digest(f"timing-kit:20261010:{source}:{key}".encode()),
        )[:32]
        assert list(dict.fromkeys(r["lineage"] for r in selected)) == groups
        assert (
            rows == before
            and kit.select(list(reversed(rows)), source, 32)[0]["lineage"] == groups[0]
        )
        assert Counter(r["lineage"] for r in selected) == dict.fromkeys(
            groups, 3 if source == "ALASKA2" else 6
        )


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "count",
        "limit",
        "object",
        "digest",
        "role",
        "duplicate",
        "cross-role",
        "family",
        "quality",
        "short",
        "bows",
    ],
)
def test_selection_rejections(fault):
    rows, source, count = samples("ALASKA2", 2), "ALASKA2", 2
    if fault == "source":
        source = "unknown"
    elif fault == "count":
        count = True
    elif fault == "limit":
        rows *= 3000
    elif fault == "object":
        rows[0] = None
    elif fault == "digest":
        rows[0]["sha256"] = "bad"
    elif fault == "role":
        rows[0]["split"] = "test"
    elif fault == "duplicate":
        rows[1]["sha256"] = rows[0]["sha256"]
    elif fault == "cross-role":
        rows[1]["split"] = "validation"
    elif fault == "family":
        rows.pop()
    elif fault == "quality":
        for r in rows:
            r["quality_factor"] = 75
    elif fault == "short":
        rows = rows[:3]
    else:
        source = "BOWS2"
        for r in rows:
            r["source_group"] = source
    with pytest.raises(ValueError):
        kit.select(rows, source, count)


def test_validation_originals_never_selected():
    rows = samples("ALASKA2", 3)
    for row in rows[:3]:
        row["split"] = "validation"
    assert len(kit.select(rows, "ALASKA2", 2)) == 6
    assert all(r["split"] == "train" for r in kit.select(rows, "ALASKA2", 2))


def test_read_write_identity_symlink_size_and_no_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"hello")
    assert kit.read_bytes(path, digest(b"hello"), 5) == b"hello"
    for sha, size in (
        (digest(b"bad"), 5),
        (digest(b"hello"), 0),
        (digest(b"hello"), True),
        (digest(b"hello"), 4),
    ):
        with pytest.raises(ValueError):
            kit.read_bytes(path, sha, size)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        kit.read_bytes(link, digest(b"hello"), 5)
    output = tmp_path / "json"
    sha = kit.write_document(output, {"a": 1})
    assert sha == digest(output.read_bytes())
    with pytest.raises(FileExistsError):
        kit.write_document(output, {})
    with pytest.raises(ValueError, match="metadata"):
        kit.write_document(tmp_path / "big", {"a": "a" * (16 * 1024**2)})


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    root, bows, out = (tmp_path / name for name in ("root", "bows", "out"))
    root.mkdir()
    bows.mkdir()
    rows = samples("ALASKA2", 32) + samples("BOSSbase-1.01", 32)
    for row in rows:
        (root / row["path"]).write_bytes(row["path"].encode())
    originals = []
    for i in range(32):
        data = f"bows-{i}".encode()
        path = bows / f"{i}.pgm"
        path.write_bytes(data)
        originals.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": digest(data),
                "lineage": digest(data),
                "split": "train",
                "source_group": "BOWS2",
            }
        )
    source = {
        "schema_version": "bows2-original-acquisition-v1",
        "status": "completed",
        "samples": originals,
        "reserved_manifest_sha256": ["1" * 64],
    }
    source_sha = kit.write_document(bows / "source.json", source)
    proof = {
        "schema_version": "bows2-independent-original-audit-v1",
        "status": "completed",
        "source_manifest_sha256": source_sha,
        "exact_identity_overlap": 0,
        "decoded_boss_pixel_overlap": 0,
        "reserved_manifest_sha256": source["reserved_manifest_sha256"],
    }
    proof_path = tmp_path / "proof.json"
    proof_sha = kit.write_document(proof_path, proof)
    manifest_sha = kit.write_document(root / "manifest.json", {"samples": rows})
    index_sha = kit.write_document(
        root / "index.json",
        {"blocks": [{"path": "manifest.json", "manifest_sha256": manifest_sha}]},
    )
    binding = {
        "index_sha256": index_sha,
        "audit_sha256": "a" * 64,
        "bows_sha256": source_sha,
        "bows_audit_sha256": proof_sha,
    }
    protocol = tmp_path / "protocol.json"
    protocol_sha = kit.write_document(
        protocol,
        {
            "schema_version": "jpeg-real-timing-protocol-v1",
            "inputs": binding,
            "seed": kit.SEED,
            "originals_per_source": 32,
        },
    )

    class Reader:
        def __init__(self, *a, **kw):
            self.samples = rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(kit, "TrainBlocks", Reader)
    monkeypatch.setattr(
        kit.jpeg_float256,
        "pixel_batch",
        lambda files: np.zeros((len(files), 1, 256, 256), dtype="<f4"),
    )
    monkeypatch.setattr(
        kit.jpeg_float256,
        "region",
        lambda data: {"image_size": [512, 512], "region": [128, 128, 256, 256]},
    )
    monkeypatch.setattr(kit.jpeg_float256, "decoder_contract", lambda: {"fixture": True})

    def generate(data, directory, lineage):
        generated = samples("BOWS2", 1)
        directory.mkdir(parents=True)
        for i, row in enumerate(generated):
            encoded = data + str(i).encode()
            path = directory / f"{i}.jpg"
            path.write_bytes(encoded)
            row.update(
                sha256=digest(encoded),
                size=len(encoded),
                lineage=lineage,
                path=f"{lineage}/{path.name}",
                payload_unit="bpnzAC",
                payload_rate=None if row["label"] == "cover" else 0.2,
                coefficient_changes=0 if row["label"] == "cover" else 1,
            )
        return generated

    args = dict(
        root=root,
        bows=bows,
        out=out,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        audit=tmp_path / "unused-audit",
        bows_audit=proof_path,
        deadline=lambda: None,
        generator=generate,
        **binding,
    )
    return SimpleNamespace(
        args=args, rows=rows, proof=proof, originals=originals, generate=generate
    )


def test_complete_kit_and_distinct_full_epoch_accounting(preparation):
    result = kit.prepare(**preparation.args)
    out = preparation.args["out"]
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert result["jpeg_rows"] == 480 and result["timing_updates"] == 192
    assert manifest["full_corpus_optimizer_updates_per_epoch"] == 4932
    assert (out / "pixels.f32").stat().st_size == 125829120
    assert manifest["tensor_sha256"] == digest((out / "pixels.f32").read_bytes())
    assert Counter(r["source_group"] for r in manifest["samples"]) == {
        "ALASKA2": 96,
        "BOSSbase-1.01": 192,
        "BOWS2": 192,
    }
    assert str(out.parent) not in json.dumps(manifest)
    assert not manifest["model_trained"] and manifest["accuracy_qualification"] == "unavailable"
    with pytest.raises(FileExistsError):
        kit.prepare(**preparation.args)


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "proof",
        "overlap",
        "path",
        "budget",
        "worker-count",
        "worker-source",
        "worker-path",
        "worker-quality",
        "schedule",
        "decoder",
        "deadline",
    ],
)
def test_preparation_fail_closed(preparation, monkeypatch, fault):
    args = preparation.args
    if fault == "protocol":
        args["index_sha256"] = "0" * 64
    elif fault == "proof":
        preparation.proof["exact_identity_overlap"] = 1
        args["bows_audit"].write_text(json.dumps(preparation.proof))
    elif fault == "overlap":
        for r in preparation.rows[:3]:
            r["lineage"] = preparation.originals[0]["lineage"]
    elif fault == "path":
        index = json.loads((args["root"] / "index.json").read_bytes())
        index["blocks"][0]["path"] = "../escape"
        original = kit.document
        monkeypatch.setattr(
            kit, "document", lambda p, s: index if p.name == "index.json" else original(p, s)
        )
    elif fault == "budget":
        monkeypatch.setattr(kit, "MAX_OUTPUT", 1)
    elif fault.startswith("worker-"):

        def bad(data, directory, lineage):
            rows = preparation.generate(data, directory, lineage)
            if fault == "worker-count":
                return []
            if fault == "worker-source":
                rows[0]["source_group"] = "BOSSbase-1.01"
            elif fault == "worker-path":
                rows[0]["path"] = "other/file.jpg"
            else:
                for r in rows:
                    r["quality_factor"] = 90 if r["quality_factor"] == 95 else 75
            return rows

        args["generator"] = bad
    elif fault == "schedule":
        monkeypatch.setattr(
            kit.srnet_diversity_sampling, "epoch_batches", lambda *a, **kw: ([], {})
        )
    elif fault == "decoder":
        monkeypatch.setattr(
            kit.jpeg_float256, "pixel_batch", lambda files: np.zeros(1, dtype="<f4")
        )
    else:

        def expired():
            raise ValueError("deadline")

        args["deadline"] = expired
    with pytest.raises((ValueError, FileNotFoundError)):
        kit.prepare(**args)
