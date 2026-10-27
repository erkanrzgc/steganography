"""Generated lossless transport, independent payloads and hostile-file limits."""

import copy
import io

import numpy as np
import pytest
from PIL import Image

from core import colab_artifacts as a
from core import spatial_control_io as t
from core import spatial_training_data as d


@pytest.fixture(params=["L", "RGB"])
def setup(request, tmp_path):
    spec = d.GeneratedSpec(request.param, 2026102802, "train", 2)
    corpus = {"profile": [d.manifest(spec)]}
    root = tmp_path / "images"
    manifest = t.prepare(root, corpus)
    return root, spec, corpus, manifest


def test_both_formats_preserve_exact_samples_and_payloads(setup):
    root, spec, corpus, manifest = setup
    proof = t.verify(root, manifest, corpus)
    assert proof["image_checks"] == 8 and proof["independent_payload_checks"] == 4
    assert manifest["bytes"] <= t.MAX_SCRATCH_BYTES
    for fmt in ("PNG", "BMP"):
        actual = t.batch(root, manifest, spec, (1, 0), fmt)
        expected, _ = d.batch(spec, (1, 0))
        assert np.array_equal(actual, expected)
    with pytest.raises(FileExistsError):
        t.prepare(root, corpus)


def test_paths_hashes_channels_and_pixels_fail_closed(setup, tmp_path):
    root, _, _, manifest = setup
    row = manifest["images"][0]
    for name in ("../escape.png", "/absolute.png", "x.png"):
        with pytest.raises(ValueError, match="name"):
            t.read_image(root, {**row, "name": name})
    with pytest.raises(ValueError, match="file checksum"):
        t.read_image(root, {**row, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="pixel checksum"):
        t.read_image(root, {**row, "pixel_sha256": "0" * 64})
    with pytest.raises(ValueError, match="format/channel/extent"):
        t.read_image(root, {**row, "format": "GIF"})
    buffer = io.BytesIO()
    Image.new("RGB", (128, 256)).save(buffer, format="PNG")
    name = "g-" + "a" * 64 + "-cover.png"
    a.put(root / name, buffer.getvalue())
    with pytest.raises(ValueError, match="format/channel/extent"):
        t.read_image(root, {**row, "name": name, "sha256": a.digest(buffer.getvalue())})
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        t.read_image(link, row)


def test_invalid_batches_and_forged_transport(setup, monkeypatch):
    root, spec, corpus, manifest = setup
    for indices, fmt in [((0,), "GIF"), ((), "PNG"), ((0, 0), "PNG"), ((0, 1, 0, 1, 0), "PNG")]:
        with pytest.raises(ValueError):
            t.batch(root, manifest, spec, indices, fmt)
    changed = copy.deepcopy(manifest)
    changed["images"].append(changed["images"][0])
    with pytest.raises(ValueError, match="missing/duplicate"):
        t.batch(root, changed, spec, (0,), "PNG")
    with pytest.raises(ValueError, match="missing/duplicate"):
        t.batch(root, {**manifest, "images": []}, spec, (0,), "PNG")
    monkeypatch.setattr(
        t, "read_image", lambda *_: np.zeros((256, 256, 1 if spec.mode == "L" else 3), dtype="u1")
    )
    with pytest.raises(ValueError, match="in-memory"):
        t.batch(root, manifest, spec, (0,), "PNG")


def test_scratch_and_source_limits_before_file_output(tmp_path, monkeypatch):
    corpus = {"profile": [d.manifest(d.GeneratedSpec("L", 1, "train", 1))]}
    corrupt = copy.deepcopy(corpus)
    corrupt["profile"][0]["sources"][0]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source mismatch"):
        t.prepare(tmp_path / "bad", corrupt)
    monkeypatch.setattr(t, "MAX_SCRATCH_BYTES", 1)
    with pytest.raises(ValueError, match="scratch"):
        t.prepare(tmp_path / "limit", corpus)
    assert not list((tmp_path / "limit").iterdir())


def test_oriented_image_rejected_after_integrity_verification(setup):
    root, _, _, manifest = setup
    row = manifest["images"][0]
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 3
    Image.new(row["spec"]["mode"], (256, 256)).save(buffer, format="PNG", exif=exif)
    name = "g-" + "b" * 64 + "-cover.png"
    a.put(root / name, buffer.getvalue())
    with pytest.raises(ValueError, match="orientation"):
        t.read_image(root, {**row, "name": name, "sha256": a.digest(buffer.getvalue())})


def test_payload_and_extra_artifact_cannot_be_claimed_verified(setup, monkeypatch):
    root, spec, corpus, manifest = setup
    # Isolate the transport oracle from the generator's own scalar check.
    known = t.batch(root, manifest, spec, (0,), "PNG")
    monkeypatch.setattr(t, "batch", lambda *_: known)
    monkeypatch.setattr(d, "recover", lambda *_: b"wrong")
    with pytest.raises(ValueError, match="payload/LSB"):
        t.verify(root, manifest, corpus)
    monkeypatch.undo()
    extra = copy.deepcopy(manifest)
    extra["images"].append({**extra["images"][0], "index": 99})
    with pytest.raises(ValueError, match="extra image"):
        t.verify(root, extra, corpus)
