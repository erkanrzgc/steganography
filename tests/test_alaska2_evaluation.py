import copy
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from core.result import AnalysisResult, FileAnalysis, FileInfo, Signal
from steganography.benchmarking import alaska2, metrics


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    samples = []
    for index in range(2):
        image = io.BytesIO()
        Image.new("RGB", (512, 512), (index * 50, 30, 90)).save(image, format="JPEG")
        cover = image.getvalue()
        lineage = alaska2.digest(cover)
        for folder in ("Cover", *alaska2.METHODS):
            # Source-label ambiguity is deliberate; this is not an embedding test.
            data = cover if folder in ("Cover", "UERD") else cover + folder.encode()
            path = root / folder / f"{index:05d}.jpg"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            samples.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": alaska2.digest(data),
                    "lineage": lineage,
                    "size": len(data),
                    "split": "test",
                    "label": "cover" if folder == "Cover" else "stego",
                    "method": None if folder == "Cover" else folder,
                    "format": "JPEG",
                    "width": 512,
                    "height": 512,
                    "source_group": "ALASKA2",
                }
            )
    selection = {"members": [{"path": r["path"]} for r in samples]}
    (root / "selection.json").write_text(json.dumps(selection))
    source = {
        "schema_version": "alaska2-acquisition-v1",
        "source_group": "ALASKA2",
        "selection_sha256": alaska2.digest((root / "selection.json").read_bytes()),
        "samples": samples,
    }
    (root / "source.json").write_text(json.dumps(source))
    return root, source


def rewrite(root, source):
    raw = json.dumps(source).encode()
    (root / "source.json").write_bytes(raw)
    return alaska2.digest(raw)


def result_for(sample, *, score=0, status="completed"):
    return {
        **sample,
        "score": score,
        "status": status,
        "seconds": 0.1,
        "signals": [],
        "coverage": dict.fromkeys(alaska2.REQUIRED, "ok"),
    }


def test_metrics_can_disable_threshold_search(monkeypatch):
    def forbidden(*args):
        pytest.fail("held-out data must not search thresholds")

    monkeypatch.setattr(metrics, "_recommended_threshold", forbidden)
    assert (
        metrics.classification_metrics([(True, 70)], threshold=70, recommend_threshold=False)[
            "recommended_threshold"
        ]
        is None
    )


def test_source_validation_and_bounds(corpus, tmp_path):
    root, source = corpus
    valid = rewrite(root, source)
    assert alaska2.load_source(root, valid)[0] == source
    with pytest.raises(ValueError, match="checksum"):
        alaska2.load_source(root, "0" * 64)
    cases = [
        ({"schema_version": "other"}, "schema"),
        ({"source_group": "other"}, "source"),
        ({"samples": []}, "count"),
        ({"selection_sha256": "0" * 64}, "selection checksum"),
        ({"samples": source["samples"][:-1]}, "incomplete"),
        ({"samples": [*source["samples"], source["samples"][0]]}, "duplicate"),
    ]
    for mutation, message in cases:
        altered = {**source, **mutation}
        with pytest.raises(ValueError, match=message):
            alaska2.load_source(root, rewrite(root, altered))
    for field, value, message in (
        ("path", "../escape.jpg", "unsafe"),
        ("path", 123, "unsafe"),
        ("split", "train", "provenance"),
        ("label", "stego", "provenance"),
        ("method", "UERD", "provenance"),
        ("format", "PNG", "provenance"),
        ("width", 9999999, "provenance"),
        ("size", True, "provenance"),
        ("size", alaska2.MAX_IMAGE + 1, "provenance"),
        ("sha256", "bad", "provenance"),
        ("lineage", "0" * 64, "lineage"),
        ("source_group", "other", "provenance"),
    ):
        altered = copy.deepcopy(source)
        altered["samples"][0][field] = value
        with pytest.raises(ValueError, match=message):
            alaska2.load_source(root, rewrite(root, altered))
    altered = copy.deepcopy(source)
    for row in altered["samples"][4:]:
        row["lineage"] = source["samples"][0]["lineage"]
    altered["samples"][4]["sha256"] = source["samples"][0]["lineage"]
    with pytest.raises(ValueError, match="lineage"):
        alaska2.load_source(root, rewrite(root, altered))
    (root / "selection.json").write_text('{"members": []}')
    source["selection_sha256"] = alaska2.digest((root / "selection.json").read_bytes())
    with pytest.raises(ValueError, match="membership"):
        alaska2.load_source(root, rewrite(root, source))
    empty = tmp_path / "empty"
    empty.touch()
    assert alaska2.bounded_read(empty, 10) == b""
    empty.write_bytes(b"123")
    with pytest.raises(ValueError, match="bounded"):
        alaska2.bounded_read(empty, 2)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        alaska2.bounded_read(link / "source.json", alaska2.MAX_MANIFEST)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="bounded"):
        alaska2.bounded_read(fifo, 10)


def test_sample_scoring_safe_failure_and_ai_isolation(corpus, monkeypatch):
    root, source = corpus
    sample = source["samples"][0]
    info = FileInfo(
        "/private/location",
        "00000.jpg",
        sample["size"],
        sample["sha256"],
        "jpeg",
        "image/jpeg",
        ".jpg",
        False,
    )
    results = (
        AnalysisResult("image_jpeg", 40, (Signal("test", 40, "private detail"),), None),
        AnalysisResult("ai_triage", 100, (Signal("claimed", 100, "secret"),), None),
    )
    report = FileAnalysis(info, 40, "medium", "balanced", results)

    def service(**options):
        assert options == {
            "profile": "balanced",
            "ai_provider": None,
            "max_file_size": alaska2.MAX_IMAGE,
        }
        return SimpleNamespace(analyze=lambda path: report)

    monkeypatch.setattr(alaska2, "AnalysisService", service)
    row = alaska2.score_sample((str(root), sample))
    assert row["score"] == 40 and row["status"] == "completed"
    assert row["coverage"] == {"image_jpeg": "ok"}
    assert len(row["signals"]) == 1
    assert "private" not in json.dumps(row) and "secret" not in json.dumps(row)
    report = replace(report, file=replace(info, sha256="0" * 64))
    assert alaska2.score_sample((str(root), sample))["status"] == "failed"
    for change in ({"path": "../escape"}, {"sha256": "0" * 64}, {"size": 1}):
        failed = alaska2.score_sample((str(root), {**sample, **change}))
        assert failed["score"] is None and failed["failure"] == "ValueError"
    Image.new("RGB", (1, 1)).save(root / sample["path"])
    data = (root / sample["path"]).read_bytes()
    failed = alaska2.score_sample(
        (str(root), {**sample, "size": len(data), "sha256": alaska2.digest(data)})
    )
    assert failed["status"] == "failed"


def test_bootstrap_ties_pairing_and_incomplete_coverage(corpus, monkeypatch):
    _, source = corpus
    rows = [result_for(r) for r in source["samples"] if r["method"] in (None, "UERD")]
    original = alaska2.fixed_metrics

    def paired(observations):
        assert sum(r["label"] == "cover" for r in observations) == len(observations) / 2
        assert all(
            sum(r["label"] == "cover" and r["lineage"] == lineage for r in observations)
            == sum(r["label"] == "stego" and r["lineage"] == lineage for r in observations)
            for lineage in {r["lineage"] for r in observations}
        )
        return original(observations)

    monkeypatch.setattr(alaska2, "fixed_metrics", paired)
    result = alaska2.summarize(rows)
    assert result["status"] == "completed"
    assert result["metrics"]["confusion"] == {"tp": 0, "tn": 2, "fp": 0, "fn": 2}
    assert result["metrics"]["roc_auc"] == 0.5
    assert result["lineage_bootstrap_95_ci"]["roc_auc"] == [0.5, 0.5]
    assert "recommended_threshold" not in result["metrics"]
    assert "average_precision" not in result["metrics"]
    rows[0]["coverage"]["stegseek"] = "unavailable"
    assert alaska2.summarize(rows)["status"] == "completed"
    for mutation in (
        {"status": "failed", "score": None},
        {"coverage": {}},
        {"coverage": {**rows[0]["coverage"], "image_jpeg_dct": "unavailable"}},
        {"coverage": {**rows[0]["coverage"], "exiftool": "error"}},
    ):
        incomplete = [{**rows[0], **mutation}, *rows[1:]]
        assert alaska2.summarize(incomplete)["metrics"] is None
    assert alaska2.summarize([])["status"] == "unavailable"


def test_independent_jpeg_uses_real_shared_service(corpus, monkeypatch):
    root, source = corpus
    monkeypatch.setattr("shutil.which", lambda executable: None)
    sample = source["samples"][0]
    row = alaska2.score_sample((str(root), sample))
    assert row["status"] == "completed"
    assert row["sha256"] == sample["sha256"]
    assert row["coverage"]["image_jpeg"] == "ok"
    assert row["coverage"]["stegseek"] == "unavailable"
    assert row["coverage"]["image_bitplane"] == "unsupported"
    assert "ai_triage" not in row["coverage"]


def test_evaluation_exclusive_outputs_shared_covers_and_provenance(corpus, tmp_path, monkeypatch):
    root, source = corpus
    monkeypatch.setattr(alaska2, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(alaska2, "score_sample", lambda item: result_for(item[1]))
    out = tmp_path / "report"
    options = {"source_sha256": rewrite(root, source), "protocol": root / "selection.json"}
    report = alaska2.evaluate(root, out, **options)
    assert report["status"] == "completed" and report["samples"] == 8
    assert report["lineages"] == 2
    assert all(
        r["metrics"]["positives"] == r["metrics"]["negatives"] == 2
        for r in report["by_method"].values()
    )
    assert report["byte_identical_stego_paths"] == ["UERD/00000.jpg", "UERD/00001.jpg"]
    assert report["uerd_sensitivity_excluding_identical_pairs"]["samples"] == 0
    assert report["cross_source_gate"] == report["ece"]["status"] == "unavailable"
    assert (
        report["scores_sha256"] == hashlib.sha256((out / "scores.jsonl").read_bytes()).hexdigest()
    )
    assert len((out / "scores.jsonl").read_text().splitlines()) == 8
    assert "/home/" not in (out / "report.json").read_text()
    assert not report["threshold_search_performed"]
    with pytest.raises(FileExistsError):
        alaska2.evaluate(root, out, **options)
    with pytest.raises(ValueError, match="workers"):
        alaska2.evaluate(root, tmp_path / "invalid", workers=0, **options)

    def absent_package(name):
        if name == "jpeglib":
            raise alaska2.importlib.metadata.PackageNotFoundError(name)
        return "fixture"

    monkeypatch.setattr(alaska2.importlib.metadata, "version", absent_package)
    monkeypatch.setattr(alaska2, "score_sample", lambda item: result_for(item[1], status="failed"))
    failed = alaska2.evaluate(root, tmp_path / "failed", **options)
    assert failed["status"] == "unavailable"
    assert failed["environment"]["packages"]["jpeglib"] == "unavailable"


def test_cli_redacts_failures_and_unavailable_exit(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["alaska2", "--source", "s", "--out", "o", "--source-sha256", "x", "--protocol", "p"],
    )
    monkeypatch.setattr(alaska2, "evaluate", lambda *a, **kw: {"status": "completed", "samples": 4})
    alaska2.main()
    assert "completed: 4" in capsys.readouterr().out
    monkeypatch.setattr(
        alaska2, "evaluate", lambda *a, **kw: {"status": "unavailable", "samples": 4}
    )
    with pytest.raises(SystemExit) as error:
        alaska2.main()
    assert error.value.code == 1

    def fail(*args, **kwargs):
        raise ValueError("password: /home/private")

    monkeypatch.setattr(alaska2, "evaluate", fail)
    with pytest.raises(SystemExit) as error:
        alaska2.main()
    assert error.value.code == 2
    assert "private" not in capsys.readouterr().err
