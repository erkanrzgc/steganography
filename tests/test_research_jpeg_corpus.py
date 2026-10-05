import hashlib
import json

import pytest

from steganography import research_jpeg_corpus as corpus
from steganography.research import ResearchManifestError
from tests.test_research_spatial import development  # noqa: F401


def args(root, reserved):
    return {
        "source_sha256": hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        "reserved_manifests": [reserved],
        "count": 2,
    }


def test_real_local_generation_and_no_overwrite(development, tmp_path):  # noqa: F811
    pytest.importorskip("conseal")
    pytest.importorskip("jpeglib")
    root, reserved, _ = development
    out = tmp_path / "jpeg"
    before = {p: p.read_bytes() for p in root.iterdir()}
    result = corpus.generate_boss(root, out, **args(root, reserved))
    assert result["original_lineages"] == 2 and result["jpeg_files"] == 12
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(s["coefficient_changes"] > 0 for s in manifest["samples"] if s["label"] == "stego")
    assert manifest["generation"]["methods"] == ["JUNIWARD", "UERD"]
    assert {s["quality_factor"] for s in manifest["samples"]} == {75, 95}
    assert str(tmp_path) not in json.dumps(manifest)
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        corpus.generate_boss(root, out, **args(root, reserved))


def test_selection_and_reserved_binding(development, tmp_path):  # noqa: F811
    root, reserved, source = development
    original = (root / "selection.json").read_bytes()
    (root / "selection.json").write_text('{"members": []}')
    with pytest.raises(ResearchManifestError, match="selection"):
        corpus.generate_boss(root, tmp_path / "selection", **args(root, reserved))
    (root / "selection.json").write_bytes(original)
    source["reserved_manifest_sha256"] = ["b" * 64]
    (root / "source.json").write_text(json.dumps(source))
    with pytest.raises(ResearchManifestError, match="reserved provenance"):
        corpus.generate_boss(root, tmp_path / "provenance", **args(root, reserved))


@pytest.mark.parametrize("value", [0, 1, 129, True, "2"])
def test_count_bounds(development, tmp_path, value):  # noqa: F811
    root, reserved, _ = development
    with pytest.raises(ResearchManifestError, match="count"):
        corpus.generate_boss(root, tmp_path / "out", **{**args(root, reserved), "count": value})


def test_preflight_provenance_and_worker_failures(development, tmp_path, monkeypatch):  # noqa: F811
    root, reserved, source = development
    with pytest.raises(ResearchManifestError, match="checksum"):
        corpus.generate_boss(
            root, tmp_path / "out", **{**args(root, reserved), "source_sha256": "0" * 64}
        )
    with pytest.raises(ResearchManifestError, match="required"):
        corpus.generate_boss(
            root, tmp_path / "out", **{**args(root, reserved), "reserved_manifests": []}
        )
    with monkeypatch.context() as patch:
        patch.setattr(corpus.importlib.metadata, "version", lambda n: "invalid")
        with pytest.raises(ResearchManifestError, match="conseal"):
            corpus.generate_boss(root, tmp_path / "out", **args(root, reserved))

    def timeout(*a, **k):
        raise corpus.subprocess.TimeoutExpired("fixed worker", 90)

    with monkeypatch.context() as patch:
        patch.setattr(corpus.subprocess, "run", timeout)
        with pytest.raises(ResearchManifestError, match="timed out"):
            corpus.generate_boss(root, tmp_path / "timeout", **args(root, reserved))
    assert not (tmp_path / "timeout/manifest.json").exists()
    reserved.write_text(json.dumps({"samples": [source["samples"][0]]}))
    with pytest.raises(ResearchManifestError, match="overlap"):
        corpus.generate_boss(root, tmp_path / "reserved", **args(root, reserved))


@pytest.mark.parametrize("fault", ["exit", "limit", "count", "recipe", "identity", "features"])
def test_worker_output_fail_closed(development, tmp_path, monkeypatch, fault):  # noqa: F811
    root, reserved, _ = development

    def fake_run(command, *, stdout, **kwargs):
        lineage, split = command[-2:]
        rows = [
            {
                "lineage": lineage,
                "split": split,
                "method": m,
                "quality_factor": q,
                "path": f"{lineage}/q{q}-{m or 'cover'}.jpg",
                "label": "cover" if m is None else "stego",
                "source_group": "BOSSbase-1.01",
                "values": [0] * 1098,
            }
            for q in corpus.QUALITIES
            for m in (None, *corpus.METHODS)
        ]
        if fault == "count":
            rows.pop()
        elif fault == "recipe":
            rows[0]["method"] = "invalid"
        elif fault == "identity":
            rows[0]["path"] = "../escape.jpg"
        elif fault == "features":
            rows[0]["values"][0] = True
        stdout.write(
            b"x" * (corpus.MAX_WORKER_OUTPUT + 1) if fault == "limit" else json.dumps(rows).encode()
        )
        return corpus.subprocess.CompletedProcess(command, 1 if fault == "exit" else 0)

    monkeypatch.setattr(corpus.subprocess, "run", fake_run)
    with pytest.raises(ResearchManifestError):
        corpus.generate_boss(root, tmp_path / "out", **args(root, reserved))
    assert not (tmp_path / "out/manifest.json").exists()


def test_worker_direct_round_trip_and_guards(development, tmp_path, monkeypatch):  # noqa: F811
    cl = pytest.importorskip("conseal")
    root, _, source = development
    sample = source["samples"][0]
    data = (root / sample["path"]).read_bytes()
    kwargs = (sample["sha256"], sample["split"])
    rows = corpus.generate_lineage(data, tmp_path / "direct", *kwargs)
    assert len(rows) == 6 and all(len(r["values"]) == 1098 for r in rows)
    with pytest.raises(FileExistsError):
        corpus.generate_lineage(data, tmp_path / "direct", *kwargs)
    for invalid, match in ((b"", "bounded"), (b"wrong", "identity")):
        with pytest.raises(ResearchManifestError, match=match):
            corpus.generate_lineage(invalid, tmp_path / "bad", *kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(cl.juniward, "simulate_single_channel", lambda x, y, *a, **k: y.copy())
        with pytest.raises(ResearchManifestError, match="coefficient changes"):
            corpus.generate_lineage(data, tmp_path / "unchanged", *kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(corpus.np, "array_equal", lambda *a, **k: False)
        with pytest.raises(ResearchManifestError, match="round-trip"):
            corpus.generate_lineage(data, tmp_path / "roundtrip", *kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(corpus, "MAX_IMAGE_BYTES", 1)
        with pytest.raises(ResearchManifestError, match="exceeds"):
            corpus.generate_lineage(data, tmp_path / "limit", *kwargs)
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.new("L", (32, 32)).save(stream, format="PPM")
    small = stream.getvalue()
    with pytest.raises(ResearchManifestError, match="512x512"):
        corpus.generate_lineage(
            small, tmp_path / "dimensions", hashlib.sha256(small).hexdigest(), "train"
        )


@pytest.mark.parametrize(
    "fault,match",
    [
        ("purpose", "development originals"),
        ("schema", "development originals"),
        ("empty", "sample count"),
        ("size", "roles"),
        ("split", "both development"),
    ],
)
def test_original_preflight(development, tmp_path, fault, match):  # noqa: F811
    root, reserved, source = development
    if fault == "purpose":
        source["purpose"] = "evaluation"
    elif fault == "schema":
        source["schema_version"] = "invalid"
    elif fault == "empty":
        source["samples"] = []
    elif fault == "size":
        source["samples"][0]["size"] = 2 * 1024 * 1024
    else:
        for s in source["samples"]:
            s["split"] = "train"
    (root / "source.json").write_text(json.dumps(source))
    with pytest.raises(ResearchManifestError, match=match):
        corpus.generate_boss(root, tmp_path / "out", **args(root, reserved))


def test_worker_main_without_changing_test_process_limits(monkeypatch, capsys):
    import io
    import resource
    from types import SimpleNamespace

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr("sys.argv", ["worker"])
    assert corpus.main() == 1
    monkeypatch.setattr("sys.argv", ["module", "worker", "owned", "a" * 64, "train"])
    monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=io.BytesIO(b"test")))
    monkeypatch.setattr(corpus, "generate_lineage", lambda *a: [])
    assert corpus.main() == 0 and json.loads(capsys.readouterr().out) == []

    def fail(*a):
        raise ValueError("private path")

    monkeypatch.setattr(corpus, "generate_lineage", fail)
    assert corpus.main() == 1 and not capsys.readouterr().out
