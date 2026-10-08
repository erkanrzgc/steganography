"""Independent fixture builder and byte/scalar replay/adversarial audit tests."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core import jpeg_float256


def module():
    spec = importlib.util.spec_from_file_location(
        "scale_audit", Path(__file__).parents[1] / "scripts/audit-jpeg-scale.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    pytest.importorskip("jpeglib")
    m = module()
    root = tmp_path / "prepared"
    root.mkdir()
    aroot = tmp_path / "a"
    aroot.mkdir()
    broot = tmp_path / "b"
    broot.mkdir()
    sources = [[], []]
    blocks = []
    for group_no, group in enumerate(("ALASKA2", "BOSSbase-1.01")):
        folder = root / f"block-{group_no:03d}"
        folder.mkdir()
        rows = []
        keys = []
        for i in range(2):
            pixels = np.random.default_rng(group_no * 100 + i).integers(
                0, 256, (512, 512), dtype=np.uint8
            )
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(
                buffer, format="JPEG" if group_no == 0 else "PPM", quality=85
            )
            original_raw = buffer.getvalue()
            lineage = hashlib.sha256(original_raw).hexdigest()
            keys.append(lineage)
            origin_name = f"original-{i}.{'jpg' if group_no == 0 else 'pgm'}"
            ((aroot if group_no == 0 else broot) / origin_name).write_bytes(original_raw)
            split = "train" if i == 0 else "validation"
            base = {
                "sha256": lineage,
                "lineage": lineage,
                "size": len(original_raw),
                "path": origin_name,
                "source_group": group,
                "split": split,
                "label": "cover",
                "method": None,
                "format": "JPEG" if group_no == 0 else "PGM",
            }
            if group_no:
                sources[group_no].append(base)
            for quality in (None,) if group_no == 0 else (75, 95):
                for method_no, method in enumerate((None, "JUNIWARD", "UERD")):
                    if group_no == 0 and method is None:
                        body = original_raw
                    else:
                        buffer = io.BytesIO()
                        Image.fromarray(
                            pixels
                            if method is None
                            else (pixels.astype(np.int16) + method_no).clip(0, 255).astype(np.uint8)
                        ).save(
                            buffer,
                            format="JPEG",
                            quality=85 if quality is None else quality,
                            optimize=False,
                        )
                        body = buffer.getvalue()
                    sha = hashlib.sha256(body).hexdigest()
                    name = f"{i}-{quality}-{method}.jpg"
                    (folder / name).write_bytes(body)
                    r = {
                        **base,
                        "sha256": sha,
                        "size": len(body),
                        "path": name,
                        "quality_factor": quality,
                        "label": "cover" if method is None else "stego",
                        "method": method,
                        "format": "JPEG",
                    }
                    rows.append(r)
                    if group_no == 0:
                        sources[0].append(r)
        manifest_sha = save(folder / "manifest.json", {"schema_version": "1.0", "samples": rows})
        caches = {}
        for split in ("train", "validation"):
            selected = [r for r in rows if r["split"] == split]
            cache_folder = folder / split
            cache_folder.mkdir()
            raw = b"".join(
                jpeg_float256.decode((folder / r["path"]).read_bytes()) for r in selected
            )
            (cache_folder / "pixels.f32").write_bytes(raw)
            data_sha = hashlib.sha256(raw).hexdigest()
            doc = {
                "schema_version": "research-float256-cache-v1",
                "manifest_sha256": manifest_sha,
                "feature_version": "jpeg-y-idct-center256-phase0-f32-v1",
                "split": split,
                "dtype": "<f4",
                "shape": [len(selected), 1, 256, 256],
                "decoder": jpeg_float256.decoder_contract(),
                "data_sha256": data_sha,
                "rows": [
                    {
                        **{k: r[k] for k in ("sha256", "lineage", "label")},
                        "image_size": [512, 512],
                        "region": [128, 128, 256, 256],
                    }
                    for r in selected
                ],
            }
            cache_sha = save(cache_folder / "cache.json", doc)
            caches[split] = {
                "path": f"{folder.name}/{split}/cache.json",
                "sha256": cache_sha,
                "rows": len(selected),
                "data_sha256": data_sha,
            }
        blocks.append(
            {
                "path": f"{folder.name}/manifest.json",
                "manifest_sha256": manifest_sha,
                "source_group": group,
                "lineages": keys,
                "caches": caches,
            }
        )
    source_shas = [
        save(p / "source.json", {"samples": r})
        for p, r in zip((aroot, broot), sources, strict=True)
    ]
    audit = tmp_path / "acquisition.json"
    audit_sha = save(
        audit,
        {
            "status": "completed",
            "models_trained": False,
            "sources": [{"source_group": "ALASKA2", "quarantined_lineages": []}],
        },
    )
    reserved = tmp_path / "reserved.json"
    reserved_sha = save(reserved, {"samples": [{"sha256": "a" * 64}]})
    index = {
        "schema_version": "jpeg-scale-blocks-v1",
        "status": "completed",
        "protocol_sha256": m.PROTOCOL_SHA,
        "acquisition_audit_sha256": audit_sha,
        "source_manifest_sha256": source_shas,
        "quarantined_lineages": [],
        "reserved_manifest_sha256": [reserved_sha],
        "blocks": blocks,
        "jpeg_rows": 18,
        "splits": {"train": 9, "validation": 9},
        "original_lineages": 4,
    }
    save(root / "index.json", index)
    return m, (
        root,
        aroot / "source.json",
        broot / "source.json",
        audit,
        [reserved],
        tmp_path / "evidence.json",
    )


def test_all_real_jpeg_bytes_tensor_hashes_and_nine_scalar_contexts(setup):
    m, args = setup
    result = m.audit(*args)
    assert result["jpeg_rows"] == 18 and len(result["scalar_oracles"]) == 9
    assert all(r["passed"] and r["max_scalar_difference"] < 2e-4 for r in result["scalar_oracles"])
    assert result["tensor_bytes"] == 18 * 256 * 256 * 4
    assert str(args[0].parent) not in args[-1].read_text()
    with pytest.raises(FileExistsError):
        m.audit(*args)


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "protocol",
        "quarantine",
        "reserved",
        "blocks",
        "block_path",
        "block_hash",
        "family",
        "split",
        "source",
        "lineages",
        "native_hash",
        "geometry",
        "jpeg_size",
        "jpeg_hash",
        "crop",
        "cache_hash",
        "cache_contract",
        "cache_rows",
        "decoder",
        "tensor_size",
        "tensor_hash",
        "tensor_nan",
        "scalar",
        "counts",
        "missing_cache",
        "missing_family",
        "boss_ancestry",
        "boss_reencode",
    ],
)
def test_forgery_truncation_and_invalid_numeric_data_never_pass(setup, monkeypatch, fault):
    m, args = setup
    root, ap, bp, audit, reserved, out = args
    index = json.loads((root / "index.json").read_bytes())
    block = index["blocks"][0]
    path = root / block["path"]
    manifest = json.loads(path.read_bytes())
    row = manifest["samples"][0]
    cache_record = block["caches"]["train"]
    cp = root / cache_record["path"]
    cache = json.loads(cp.read_bytes())
    if fault == "status":
        index["status"] = "partial"
    if fault == "protocol":
        index["protocol_sha256"] = "0" * 64
    if fault == "quarantine":
        index["quarantined_lineages"] = ["f" * 64]
    if fault == "reserved":
        index["reserved_manifest_sha256"] = []
    if fault == "blocks":
        index["blocks"] = []
    if fault == "block_path":
        block["path"] = "../escape.json"
    if fault == "block_hash":
        block["manifest_sha256"] = "0" * 64
    if fault == "family":
        row["method"] = "JMiPOD"
    if fault == "split":
        row["split"] = "test"
    if fault == "source":
        row["source_group"] = "WIFD"
    if fault == "lineages":
        block["lineages"].pop()
    if fault in ("native_hash", "geometry"):
        if fault == "native_hash":
            body = (path.parent / manifest["samples"][1]["path"]).read_bytes()
        else:
            buffer = io.BytesIO()
            Image.new("L", (32, 32)).save(buffer, format="JPEG")
            body = buffer.getvalue()
        (path.parent / row["path"]).write_bytes(body)
        row.update(sha256=hashlib.sha256(body).hexdigest(), size=len(body))
    if fault == "jpeg_size":
        row["size"] += 1
    if fault == "jpeg_hash":
        (path.parent / row["path"]).write_bytes(b"modified")
    if fault == "crop":
        cache["rows"][0]["region"][0] = 0
    if fault == "cache_hash":
        cache_record["sha256"] = "0" * 64
    if fault == "cache_contract":
        cache["dtype"] = "uint8"
    if fault == "cache_rows":
        cache["rows"][0]["sha256"] = "f" * 64
    if fault == "decoder":
        cache["decoder"] = {"changed": True}
    if fault == "tensor_size":
        (cp.parent / "pixels.f32").write_bytes(b"small")
    if fault in ("tensor_hash", "tensor_nan", "scalar"):
        tensor = cp.parent / "pixels.f32"
        values = np.frombuffer(tensor.read_bytes(), dtype="<f4").copy()
        values[0] = float("nan") if fault == "tensor_nan" else values[0] + 10
        tensor.write_bytes(values.tobytes())
        if fault != "tensor_hash":
            cache["data_sha256"] = cache_record["data_sha256"] = hashlib.sha256(
                tensor.read_bytes()
            ).hexdigest()
    if fault == "counts":
        index["jpeg_rows"] += 1
    if fault == "missing_cache":
        block["caches"].pop("validation")
    if fault == "missing_family":
        manifest["samples"].pop()
    if fault in ("boss_ancestry", "boss_reencode"):
        doc = json.loads(bp.read_bytes())
        original = bp.parent / doc["samples"][0]["path"]
        if fault == "boss_ancestry":
            original.write_bytes(b"wrong")
        else:
            p = root / index["blocks"][1]["path"]
            d = json.loads(p.read_bytes())
            r = d["samples"][0]
            body = (p.parent / d["samples"][1]["path"]).read_bytes()
            (p.parent / r["path"]).write_bytes(body)
            r.update(sha256=hashlib.sha256(body).hexdigest(), size=len(body))
            index["blocks"][1]["manifest_sha256"] = save(p, d)
    if fault not in ("block_path", "block_hash"):
        block["manifest_sha256"] = save(path, manifest)
        cache["manifest_sha256"] = block["manifest_sha256"]
    if fault != "cache_hash":
        cache_record["sha256"] = save(cp, cache)
    save(root / "index.json", index)
    with pytest.raises(ValueError):
        m.audit(*args)
    assert not out.exists()


def test_path_symlink_byte_bounds_and_cli_secret_redaction(setup, monkeypatch, capsys):
    import resource

    m, args = setup
    with pytest.raises(ValueError):
        m.relative(args[0], "a\\b")
    link = args[0] / "link"
    link.symlink_to(args[1])
    with pytest.raises(ValueError):
        m.read(link, 100)
    with pytest.raises(ValueError):
        m.read(args[1], 1)
    original = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda p, *a, **kw: SimpleNamespace(st_size=0, st_mode=original(p, *a, **kw).st_mode),
    )
    with pytest.raises(ValueError):
        m.read(args[1], 1)
    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    cli = [
        "--root",
        str(args[0]),
        "--alaska-source",
        str(args[1]),
        "--boss-source",
        str(args[2]),
        "--acquisition-audit",
        str(args[3]),
        "--reserved",
        str(args[4][0]),
        "--out",
        str(args[5]),
    ]
    assert m.main(cli) == 0 and len(calls) == 4
    assert m.main(cli) == 2 and str(args[0].parent) not in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault", ["acquisition", "source_rows", "reserved_empty", "reserved_missing", "global_missing"]
)
def test_external_scope_and_acquisition_validation(setup, fault):
    m, args = setup
    root, ap, bp, acquisition, reserved, out = args
    index = json.loads((root / "index.json").read_bytes())
    if fault == "acquisition":
        doc = json.loads(acquisition.read_bytes())
        doc["status"] = "partial"
        index["acquisition_audit_sha256"] = save(acquisition, doc)
    if fault == "source_rows":
        doc = json.loads(ap.read_bytes())
        doc["samples"][0]["source_group"] = "WIFD"
        index["source_manifest_sha256"][0] = save(ap, doc)
    if fault == "reserved_empty":
        index["reserved_manifest_sha256"] = [save(reserved[0], {"samples": []})]
    if fault == "reserved_missing":
        reserved.clear()
    if fault == "global_missing":
        index["blocks"].pop()
    save(root / "index.json", index)
    with pytest.raises(ValueError):
        m.audit(*args)
    assert not out.exists()


def test_reserved_lineages_and_short_scalar_tensor(setup):
    m, args = setup
    root, _, _, _, reserved, _ = args
    index = json.loads((root / "index.json").read_bytes())
    index["reserved_manifest_sha256"] = [
        save(reserved[0], {"samples": [{"sha256": "a" * 64, "lineage": "b" * 64}]})
    ]
    save(root / "index.json", index)
    assert m.audit(*args)["reserved_identity_overlap"] == 0
    jpeg = next((root / "block-000").glob("*.jpg"))
    short = root / "short.f32"
    short.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        m.scalar_replay(jpeg, short, 0)
