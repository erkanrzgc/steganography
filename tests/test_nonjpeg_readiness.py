import hashlib
import json

import pytest

from core import nonjpeg_readiness as readiness
from core import nonjpeg_unicode as unicode_controls
from core.research_acquisition import UNICODE_FILES
from steganography.research_features import feature_inputs
from tests.test_nonjpeg_features import image_bytes, wav_bytes


@pytest.fixture
def corpus(tmp_path):
    # This fixture tests binding/feature plumbing. Scalar payload oracles live
    # in test_nonjpeg_preparation and are rerun against the real corpus separately.
    root = tmp_path / "media"
    root.mkdir()
    rows = []
    for origin in ("BOWS2", "DIV2K", "ESC-50", "FSDD-v1.0.10"):
        for split in ("train", "validation"):
            for label in ("cover", "stego"):
                audio = origin in {"ESC-50", "FSDD-v1.0.10"}
                index = len(rows)
                data = (
                    wav_bytes(rate=8000 + index)
                    if audio
                    else image_bytes(mode="L" if origin == "BOWS2" else "RGB", color=index)
                )
                name = f"{index}.wav" if audio else f"{index}.png"
                (root / name).write_bytes(data)
                rows.append(
                    {
                        "path": name,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "lineage": f"{origin}-{split}",
                        "source_group": origin,
                        "split": split,
                        "label": label,
                        "format": "WAV" if audio else "PNG",
                    }
                )
    document = {"samples": rows, "source_sha256": {"fixture": "0" * 64}}
    (root / "manifest.json").write_text(json.dumps(document))
    return root


def bind_audit(prepared, tmp_path):
    root = prepared
    document = json.loads((root / "manifest.json").read_text())
    result = {
        "schema_version": "nonjpeg-independent-audit-v1",
        "status": "completed",
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "test_media_opened": False,
        "source_sha256": document["source_sha256"],
        "artifact_count": len(document["samples"]),
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(result))
    return root, path


def test_preparation_exports_ten_bound_source_split_artifacts(corpus, tmp_path):  # noqa: F811
    root, path = bind_audit(corpus, tmp_path)
    out = tmp_path / "features"
    result = readiness.prepare_features(root, path, out)
    assert result["status"] == "ready" and len(result["artifacts"]) == 10
    assert result["test_media_opened"] is False and result["optimizer_updates"] == 0
    assert str(tmp_path) not in (out / "report.json").read_text()
    for row in result["artifacts"]:
        x, _, provenance = feature_inputs(
            {
                "manifest": str(out / row["manifest"]),
                "features": str(out / row["artifact"]),
                "features_sha256": row["sha256"],
            },
            split=row["split"],
        )
        assert x.shape == (row["samples"], row["features"])
        assert provenance["manifest_sha256"] == row["manifest_sha256"]
        assert row["originals"] == 1
    with pytest.raises(FileExistsError):
        readiness.prepare_features(root, path, out)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        readiness.prepare_features(root, path, link / "bad")


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("manifest_sha256", "0" * 64),
        ("source_sha256", {}),
        ("artifact_count", 1),
        ("test_media_opened", True),
        ("schema_version", "wrong"),
    ],
)
def test_wrong_audit_bindings_fail(corpus, tmp_path, field, value):  # noqa: F811
    root, path = bind_audit(corpus, tmp_path)
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="audit binding"):
        readiness.prepare_features(root, path, tmp_path / "out")


@pytest.mark.parametrize(
    "field,value,message",
    [("split", "test", "development-only"), ("source_group", "../other", "source domain")],
)
def test_wrong_split_domain_fail_before_output(corpus, tmp_path, field, value, message):  # noqa: F811
    root, path = bind_audit(corpus, tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["samples"][0][field] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    record = json.loads(path.read_text())
    record["manifest_sha256"] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(record))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        readiness.prepare_features(root, path, out)
    assert not out.exists()


@pytest.mark.parametrize("limit", ["time", "bytes"])
def test_preparation_limits_are_failures(corpus, tmp_path, monkeypatch, limit):  # noqa: F811
    root, path = bind_audit(corpus, tmp_path)
    monkeypatch.setattr(readiness, "MAX_SECONDS" if limit == "time" else "MAX_OUTPUT_BYTES", 0)
    with pytest.raises((ValueError, TimeoutError), match="deadline|budget"):
        readiness.prepare_features(root, path, tmp_path / "out")


def unicode_sources(tmp_path):
    acquisition = {"unicode": {}}
    for name in UNICODE_FILES:
        text = f"# {name} 17.0.0\n# comment\n@Part0\n\n"
        if name == "NormalizationTest.txt":
            text += "00C0;0041 0300;00C0;0041 0300;00C0; # normative columns\n"
        elif name in {"BidiCharacterTest.txt", "emoji-test.txt"}:
            text += "0627 200D 200C 0020; 0;\n"
        else:
            text += "÷ 0020 × 200D × 200C ÷ # normal boundary control\n"
            text += "÷ D800 ÷\n÷ 110000 ÷\n" + " ".join(["0041"] * 129) + "\n"
        raw = text.encode()
        (tmp_path / name).write_bytes(raw)
        acquisition["unicode"][name] = {
            "status": "ready",
            "version": "17.0.0",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return acquisition


def test_unicode_pinned_scalar_sequences_and_incomplete_status(tmp_path):
    acquisition = unicode_sources(tmp_path)
    out = tmp_path / "controls.json"
    result = unicode_controls.controls(tmp_path, acquisition, out)
    assert result["status"] == "ready" and len(result["source_sha256"]) == 6
    assert result["natural_document_coverage"] == "unavailable"
    for row in result["rows"]:
        assert list(map(ord, row["text"])) == row["codepoints"]
        assert row["source_sha256"] == acquisition["unicode"][row["source"]]["sha256"]
        if row["source"] == "NormalizationTest.txt":
            assert row["normalization_columns"] == [[192], [65, 768], [192], [65, 768], [192]]
    with pytest.raises(FileExistsError):
        unicode_controls.controls(tmp_path, acquisition, out)
    result = unicode_controls.controls(tmp_path, {}, tmp_path / "incomplete.json")
    assert result["status"] == "incomplete" and len(result["missing"]) == 6
    (tmp_path / "GraphemeBreakTest.txt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        unicode_controls.controls(tmp_path, acquisition, tmp_path / "bad.json")
