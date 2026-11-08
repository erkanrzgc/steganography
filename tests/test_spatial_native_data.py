"""Generated-only native adapter fixtures and independent ancestry/LSB checks."""

import dataclasses
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

from core import colab_artifacts as a
from core import spatial_native_data as d


def source(index=0, mode="L", **changes):
    values = {
        "id": f"source-{index}",
        "mode": mode,
        "group": f"group-{index}",
        "author": f"Fixture author {index}",
        "page": f"https://commons.wikimedia.org/wiki/File:Fixture_{index}.png",
        "original_sha256": a.digest(f"original-{index}".encode()),
        "crop_sha256": a.digest(f"crop-{index}".encode()),
        "license": "CC-BY-4.0",
        "license_url": d.native.LICENSES["CC-BY-4.0"],
        "evidence_sha256": a.digest(f"evidence-{index}".encode()),
    }
    return d.Source(**{**values, **changes})


def reference(root, index=0, mode="L", extension="png", size=(256, 256)):
    shape = (*size, *((3,) if mode == "RGB" else ()))
    values = np.random.default_rng(index + 7).integers(0, 256, shape, dtype="u1")
    buf = io.BytesIO()
    Image.fromarray(values).save(buf, format=extension.upper())
    raw = buf.getvalue()
    path = root / f"source-{index}-cover.{extension}"
    path.write_bytes(raw)
    row = source(index, mode, crop_sha256=a.digest(values.tobytes()))
    return d.Input(row, path, a.digest(raw)), values


def cache_from(root, refs):
    out = root / "cache"
    value = d.prepare(tuple(refs), out)
    checksum = a.digest((out / "cache.json").read_bytes())
    return d.load(out, checksum), value


@pytest.mark.parametrize(
    "mode,extension", [("L", "png"), ("L", "bmp"), ("RGB", "png"), ("RGB", "bmp")]
)
def test_native_cover_roundtrip_full_alphabet_lsb_and_fp32(tmp_path, mode, extension):
    ref, original = reference(tmp_path, mode=mode, extension=extension)
    before = ref.cover.read_bytes()
    cache, manifest = cache_from(tmp_path, [ref])
    assert manifest["optimizer_updates"] == 0 and manifest["roles"] == "unassigned"
    assert manifest["pixel_bytes"] == original.size
    assert str(tmp_path) not in json.dumps(manifest)
    assert cache.rows == (ref.source,)
    cover, control, receipt = d.pair(cache, 0, 123)
    assert np.array_equal(cover.reshape(original.shape), original)
    assert (cover & 1).any() and ((cover & 1) == 0).any()
    assert np.array_equal(cover & 254, control & 254)
    assert np.abs(cover.astype(int) - control.astype(int)).max() == 1
    identity = a.digest(
        a.encode(
            {"recipe": d.RECIPE, "original": ref.source.original_sha256, "mode": mode, "seed": 123}
        )
    )
    count = (cover.size * 2 // 5) // 8 * 8
    selected = np.random.Generator(np.random.PCG64(int(identity[16:24], 16))).permutation(
        cover.size
    )[:count]
    expected_payload = (
        np.random.Generator(np.random.PCG64(int(identity[:16], 16)))
        .integers(0, 256, count // 8, dtype="u1")
        .tobytes()
    )
    recovered = np.packbits(control.ravel()[selected] % 2, bitorder="big").tobytes()
    assert recovered == expected_payload
    assert receipt["payload_sha256"] == a.digest(recovered)
    mask = np.ones(cover.size, dtype=bool)
    mask[selected] = False
    assert np.array_equal(cover.ravel()[mask], control.ravel()[mask])
    x, labels = d.batch(cache, mode, (0,), 123)
    assert x.dtype == np.dtype("<f4") and x.shape == (2, cover.shape[2], 256, 256)
    assert labels.tolist() == [0, 1] and labels.dtype == np.dtype("<i8")
    assert np.array_equal(x[0].transpose(1, 2, 0), cover)
    assert np.array_equal(x[1].transpose(1, 2, 0), control)
    assert np.array_equal(x.astype("u1").astype("<f4"), x)
    assert ref.cover.read_bytes() == before
    assert d.pair(cache, 0, 123)[2] == receipt
    assert d.pair(cache, 0, 124)[2]["control_sha256"] != receipt["control_sha256"]
    with pytest.raises(FileExistsError):
        d.prepare((ref,), cache.root)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.source.role = "train"


def test_transitive_author_group_page_and_cross_mode_proposal():
    rows = (
        source(0, author="  Ａlice  SMITH "),
        source(1, "RGB", author="alice smith"),
        source(2, "RGB", group="group-1"),
        source(3, page="https://commons.wikimedia.org/wiki/File:Fixture%202.png"),
        source(4),
    )
    assert d.components(rows) == (("source-0", "source-1", "source-2", "source-3"), ("source-4",))
    # Independent graph closure: shared author 0--1, group 1--2, page 2--3.
    edges = [(0, 1), (1, 2), (2, 3)]
    reached = {0}
    for _ in rows:
        reached |= {b for a_, b in edges if a_ in reached} | {a_ for a_, b in edges if b in reached}
    assert {rows[i].id for i in reached} == set(d.components(rows)[0])
    report = d.feasibility(rows)
    assert report["shared_mode_components"] == 1
    assert report["domains"]["L"]["group_sizes_descending"] == [2, 1]
    assert not report["joint_partition_feasibility_proven"]
    roles = {row.id: "train" for row in rows}
    roles["source-4"] = "future_reserved"
    d.proposal(rows, roles)
    roles["source-3"] = "development"
    with pytest.raises(ValueError, match="leakage"):
        d.proposal(rows, roles)
    for invalid in (None, {}, {row.id: "unassigned" for row in rows}):
        with pytest.raises(ValueError, match="proposal"):
            d.proposal(rows, invalid)
    assert all(row.role == "unassigned" for row in rows)


def test_capacity_bounds_have_no_implied_assignment():
    rows = tuple(source(i, "RGB", group=f"group-{i // 8}") for i in range(128))
    report = d.feasibility(rows)
    rgb = report["domains"]["RGB"]
    assert rgb["ten_group_train_capacity_upper_bound"] == 80
    assert rgb["minimum_known_groups_covering_80"] == 10
    assert rgb["target_128_inventory_count_met"] and rgb["ten_group_80_train_capacity_met"]
    assert report["domains"]["L"]["minimum_known_groups_covering_80"] is None
    assert not report["unknown_common_ancestry_ruled_out"]


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "../secret"},
        {"group": ""},
        {"original_sha256": "no"},
        {"crop_sha256": "A" * 64},
        {"evidence_sha256": "0" * 63},
        {"mode": "RGBA"},
        {"mode": None},
        {"role": "train"},
        {"author": " "},
        {"author": None},
        {"author": "a" * 1025},
        {"license": "unreviewed"},
        {"license": None},
        {"license_url": "https://invalid.test"},
        {"page": None},
        {"page": "x" * 4097},
        {"page": "file:///etc/passwd"},
        {"page": "https://commons.wikimedia.org/wiki/File:a.png?download=1"},
        {"page": "https://commons.wikimedia.org/wiki/File:a.png#anchor"},
    ],
)
def test_source_provenance_bounds(changes):
    with pytest.raises(ValueError):
        source(**changes)


@pytest.mark.parametrize("rows", [None, (), [], ({},), tuple(source(i) for i in range(257))])
def test_catalog_shape_limits(rows):
    with pytest.raises(ValueError):
        d.sources(rows)


def test_duplicate_original_crop_and_per_mode_limit():
    first = source()
    for other in (
        first,
        source(1, original_sha256=first.original_sha256),
        source(1, crop_sha256=first.crop_sha256),
    ):
        with pytest.raises(ValueError, match="duplicate"):
            d.sources((first, other))
    with pytest.raises(ValueError, match="per-mode"):
        d.sources(tuple(source(i) for i in range(129)))


def snapshot(rows):
    originals = [{**dataclasses.asdict(row), "native_mode": row.mode} for row in rows]
    return {
        "schema": "spatial-native-expansion-snapshot-v1",
        "status": "ready_prepared_subset_only",
        "roles": "unassigned",
        "originals": originals,
    }


def test_snapshot_metadata_only_checksum_role_and_fields(tmp_path):
    path = tmp_path / "snapshot.json"
    value = snapshot((source(), source(1, "RGB")))

    def read(value):
        raw = a.encode(value)
        path.write_bytes(raw)
        return d.reviewed_snapshot(path, a.digest(raw))

    assert read(value) == (source(), source(1, "RGB"))
    with pytest.raises(ValueError, match="hash"):
        d.reviewed_snapshot(path, "0" * 64)
    for invalid in (
        [],
        {**value, "status": "pending"},
        {**value, "roles": "train"},
        {**value, "originals": []},
    ):
        with pytest.raises(ValueError, match="snapshot"):
            read(invalid)
    value["originals"][0]["role"] = "development"
    with pytest.raises(ValueError, match="role"):
        read(value)
    value["originals"][0]["role"] = "unassigned"
    del value["originals"][0]["author"]
    with pytest.raises(ValueError, match="fields"):
        read(value)
    value["originals"] = [None]
    with pytest.raises(ValueError, match="role"):
        read(value)


def test_prepare_input_failures_and_resource_precheck(tmp_path, monkeypatch):
    ref, _ = reference(tmp_path)
    for bad in (None, [], ({},), ()):
        with pytest.raises(ValueError):
            d.prepare(bad, tmp_path / "bad")
    for args in (
        ({}, ref.cover, ref.sha256),
        (ref.source, "host-path", ref.sha256),
        (ref.source, tmp_path / "source-0-lsb-control.png", ref.sha256),
    ):
        with pytest.raises(ValueError):
            d.Input(*args)
    with pytest.raises(ValueError, match="cover hash"):
        d.prepare((dataclasses.replace(ref, sha256="0" * 64),), tmp_path / "wrong-cover")
    with pytest.raises(ValueError, match="crop hash"):
        d.prepare((dataclasses.replace(ref, source=source()),), tmp_path / "wrong-crop")
    monkeypatch.setattr(d, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError, match="output limit"):
        d.prepare((ref,), tmp_path / "too-large")
    assert not (tmp_path / "too-large").exists()


def test_mode_crop_header_and_corrupt_media(tmp_path):
    ref, _ = reference(tmp_path, mode="RGB")
    with pytest.raises(ValueError, match="mode"):
        d.prepare(
            (dataclasses.replace(ref, source=dataclasses.replace(ref.source, mode="L")),),
            tmp_path / "mode",
        )
    large, _ = reference(tmp_path, 1, size=(256, 257))
    with pytest.raises(ValueError, match="fixed"):
        d.prepare((large,), tmp_path / "size")
    bmp, _ = reference(tmp_path, 2, extension="bmp")
    bmp.cover.write_bytes(ref.cover.read_bytes())
    with pytest.raises(ValueError, match="BMP"):
        d.prepare((dataclasses.replace(bmp, sha256=ref.sha256),), tmp_path / "signature")
    with pytest.raises(ValueError, match="mode"):
        good_bmp, _ = reference(tmp_path, 3, mode="RGB", extension="bmp")
        d.prepare(
            (dataclasses.replace(good_bmp, source=dataclasses.replace(good_bmp.source, mode="L")),),
            tmp_path / "bmp-mode",
        )
    ref.cover.write_bytes(b"broken")
    with pytest.raises((ValueError, OSError)):
        d.prepare((dataclasses.replace(ref, sha256=a.digest(b"broken")),), tmp_path / "broken")


def test_source_cache_symlinks_nonregular_and_overwrite(tmp_path):
    ref, _ = reference(tmp_path)
    real = tmp_path / "retained.png"
    ref.cover.rename(real)
    ref.cover.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        d.prepare((ref,), tmp_path / "out")
    assert real.exists()
    ref.cover.unlink()
    os.mkfifo(ref.cover)
    with pytest.raises(ValueError, match="nonregular"):
        d.prepare((ref,), tmp_path / "fifo-out")
    ref.cover.unlink()
    ref, _ = reference(tmp_path)
    cache, _ = cache_from(tmp_path, [ref])
    original = (cache.root / "source-0.u8").read_bytes()
    (cache.root / "source-0.u8").unlink()
    (cache.root / "source-0.u8").symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        d.pair(cache, 0, 1)
    (cache.root / "source-0.u8").unlink()
    (cache.root / "source-0.u8").write_bytes(original[:-1])
    with pytest.raises(ValueError, match="extent"):
        d.pair(cache, 0, 1)
    (cache.root / "source-0.u8").write_bytes(original + b"x")
    with pytest.raises(ValueError, match="byte limit"):
        d.pair(cache, 0, 1)
    link = tmp_path / "parent-link"
    link.symlink_to(cache.root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        d.load(link, cache.manifest_sha256)
    assert real.exists()


def test_cache_schema_and_data_corruption(tmp_path, monkeypatch):
    ref, _ = reference(tmp_path)
    cache, manifest = cache_from(tmp_path, [ref])
    with pytest.raises(ValueError, match="manifest hash"):
        d.load(cache.root, "0" * 64)

    def check(value):
        raw = a.encode(value)
        (cache.root / "cache.json").write_bytes(raw)
        return d.load(cache.root, a.digest(raw))

    for value in (
        [],
        {**manifest, "extra": True},
        {**manifest, "roles": "train"},
        {**manifest, "sources": []},
        {**manifest, "optimizer_updates": False},
        {**manifest, "sources": [None]},
        {**manifest, "sources": [{"source": {}, "input_sha256": ref.sha256}]},
        {**manifest, "pixel_bytes": True},
        {**manifest, "pixel_bytes": 0},
    ):
        with pytest.raises(ValueError):
            check(value)
    assert check(manifest) == cache
    with pytest.raises(ValueError, match="directory"):
        d.Cache("host-path", cache.rows, cache.manifest_sha256)
    pixels_path = cache.root / "source-0.u8"
    pixels_path.write_bytes(b"\0" * pixels_path.stat().st_size)
    with pytest.raises(ValueError, match="hash"):
        d.pair(cache, 0, 1)
    monkeypatch.setattr(d, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError, match="accounting"):
        check(manifest)


def test_batch_pair_boundaries_and_recovery_failure(tmp_path, monkeypatch):
    one, _ = reference(tmp_path)
    two, _ = reference(tmp_path, 1, "RGB")
    cache, _ = cache_from(tmp_path, [one, two])
    for invalid in ([], (), (0, 0), (True,), (-1,), (2,), (0, 1), (0, 1, 2, 3, 4)):
        with pytest.raises(ValueError):
            d.batch(cache, "L", invalid, 1)
    with pytest.raises(ValueError):
        d.batch(cache, "RGBA", (0,), 1)
    with pytest.raises(ValueError):
        d.batch(None, "L", (0,), 1)
    for index in (-1, 2, True):
        with pytest.raises(ValueError):
            d.pair(cache, index, 1)
    with pytest.raises(ValueError):
        d.pair(None, 0, 1)
    monkeypatch.setattr(d.scattered, "recover", lambda *_: b"broken")
    with pytest.raises(ValueError, match="recovery"):
        d.pair(cache, 0, 1)
