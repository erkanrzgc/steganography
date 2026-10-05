import hashlib
import json

import pytest

from cli import main
from core.features import FEATURE_NAMES
from steganography import research_generalization as rg
from steganography.research import ResearchManifestError
from tests.test_research_features import setup_features


@pytest.fixture
def diagnostic_inputs(tmp_path):
    _, manifest, path = setup_features(tmp_path)
    for sample in manifest["samples"]:
        sample.update(
            method=None if sample["label"] == "cover" else "sequential",
            rate_percent=0 if sample["label"] == "cover" else 5,
            format="png",
        )
    path.write_text(json.dumps(manifest))
    names = list(FEATURE_NAMES)
    provenance = {
        "feature_version": "spatial-summary-v1",
        "feature_names": names,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split": "validation",
        "samples": 2,
    }
    predictions = {
        "provenance": provenance,
        "predictions": [
            {
                **{
                    k: s[k]
                    for k in ("sha256", "lineage", "label", "method", "rate_percent", "format")
                },
                "score": 0.1 if s["label"] == "cover" else 0.9,
            }
            for s in manifest["samples"]
            if s["split"] == "validation"
        ],
    }
    card = {
        "domain": "spatial-summary-linear-v1",
        "training_provenance": {**provenance, "split": "train"},
        "preprocessing": {"feature_version": provenance["feature_version"], "feature_names": names},
    }
    prediction_path, card_path = tmp_path / "predictions.json", tmp_path / "card.json"
    prediction_path.write_text(json.dumps(predictions))
    card_path.write_text(json.dumps(card))
    return path, prediction_path, card_path


def diagnose(inputs, out, **kwargs):
    manifest, predictions, card = inputs
    return rg.diagnose_validation(
        manifest,
        predictions,
        card,
        out,
        predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        model_card_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),
        **kwargs,
    )


def rebind_manifest(inputs, mutate):
    manifest, predictions, card = inputs
    document = json.loads(manifest.read_bytes())
    mutate(document)
    manifest.write_text(json.dumps(document))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for path, field in ((predictions, "provenance"), (card, "training_provenance")):
        doc = json.loads(path.read_bytes())
        doc[field]["manifest_sha256"] = digest
        path.write_text(json.dumps(doc))


def test_cached_audit_cli_and_no_image_access(diagnostic_inputs, tmp_path, capsys):
    # Remove every image: the diagnostic only reads bound cached documents.
    for image in (tmp_path / "images").iterdir():
        image.unlink()
    out = tmp_path / "report.json"
    result = diagnose(diagnostic_inputs, out)
    assert not result["deployed"] and result["support_status"] == "experimental"
    assert result["source_separation"]["shared_source_groups"] == 1
    assert result["source_separation"]["independent_source_evidence"] == "unavailable"
    assert result["metadata_coverage"]["quality_factor"]["unknown_rows"] == 2
    for cell in result["cells"]:
        assert cell["metrics"]["roc_auc"] == 1
        assert "recommended_threshold" not in cell["metrics"]
        assert cell["method_family"] == "sequential"
    assert str(tmp_path) not in json.dumps(result)
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        diagnose(diagnostic_inputs, out)
    assert out.read_bytes() == before
    manifest, predictions, card = diagnostic_inputs
    assert (
        main(
            [
                "--quiet",
                "research",
                "diagnose",
                "--manifest",
                str(manifest),
                "--predictions",
                str(predictions),
                "--model-card",
                str(card),
                "--predictions-sha256",
                hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "--model-card-sha256",
                hashlib.sha256(card.read_bytes()).hexdigest(),
                "--out",
                str(tmp_path / "cli.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == result["schema_version"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_bad_threshold(diagnostic_inputs, tmp_path, value):
    with pytest.raises(ResearchManifestError, match="threshold"):
        diagnose(diagnostic_inputs, tmp_path / "out.json", threshold=value)


@pytest.mark.parametrize(
    "index,mutation,message",
    [
        (1, lambda d: d.update(provenance=None), "provenance"),
        (2, lambda d: d.update(training_provenance=[]), "provenance"),
        (2, lambda d: d.update(preprocessing="bad"), "provenance"),
        (1, lambda d: d["provenance"].pop("feature_version"), "missing"),
        (1, lambda d: d["provenance"].update(split="test"), "contract"),
        (2, lambda d: d["training_provenance"].update(split="validation"), "contract"),
        (2, lambda d: d["training_provenance"].update(manifest_sha256="0" * 64), "contract"),
        (2, lambda d: d.update(domain="jpeg-dct-summary-linear-v1"), "contract"),
        (1, lambda d: d["predictions"].pop(), "identity"),
        (1, lambda d: d.update(predictions=[None, None]), "identity"),
        (1, lambda d: d["predictions"][0].update(score=True), "identity"),
        (1, lambda d: d["predictions"][0].update(score=float("nan")), "identity"),
        (1, lambda d: d["predictions"][0].update(score=1.1), "identity"),
        (1, lambda d: d["predictions"].reverse(), "identity"),
    ],
)
def test_mutated_contracts(diagnostic_inputs, tmp_path, index, mutation, message):
    path = diagnostic_inputs[index]
    document = json.loads(path.read_bytes())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ResearchManifestError, match=message):
        diagnose(diagnostic_inputs, tmp_path / "out.json")


def test_checksum_limits_and_symlinks(diagnostic_inputs, tmp_path, monkeypatch):
    manifest, predictions, card = diagnostic_inputs
    with pytest.raises(ResearchManifestError, match="checksum"):
        rg.diagnose_validation(
            manifest,
            predictions,
            card,
            tmp_path / "out.json",
            predictions_sha256="0" * 64,
            model_card_sha256="0" * 64,
        )
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        diagnose(diagnostic_inputs, link / "out.json")
    with monkeypatch.context() as patch:
        patch.setattr(rg, "MAX_ROWS", 1)
        with pytest.raises(ResearchManifestError, match="row limit"):
            diagnose(diagnostic_inputs, tmp_path / "out.json")
    monkeypatch.setattr(rg, "MAX_CELLS", 1)
    with pytest.raises(ResearchManifestError, match="cell limit"):
        diagnose(diagnostic_inputs, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("quality", [0, 101, True, "90", float("nan")])
def test_invalid_quality(diagnostic_inputs, tmp_path, quality):
    rebind_manifest(diagnostic_inputs, lambda m: m["samples"][2].update(quality_factor=quality))
    with pytest.raises(ResearchManifestError, match="quality factor"):
        diagnose(diagnostic_inputs, tmp_path / "out.json")


def test_unseen_source_and_single_label_context(diagnostic_inputs, tmp_path):
    def mutate(manifest):
        for sample in manifest["samples"]:
            if sample["split"] == "validation":
                sample.update(
                    source_group="/home/private/TOKEN_SECRET",
                    quality_factor=90,
                    format="jpg" if sample["label"] == "cover" else "bmp",
                )

    rebind_manifest(diagnostic_inputs, mutate)
    path = diagnostic_inputs[1]
    document = json.loads(path.read_bytes())
    document["predictions"][0]["format"] = "jpg"
    document["predictions"][1]["format"] = "bmp"
    path.write_text(json.dumps(document))
    result = diagnose(diagnostic_inputs, tmp_path / "out.json")
    assert result["source_separation"]["independent_source_evidence"] == "candidate_not_qualified"
    assert result["metadata_coverage"]["quality_factor"]["known_rows"] == 2
    assert any(c["context"] == "qf-90" for c in result["cells"])
    formats = [c for c in result["cells"] if c["dimension"] == "format"]
    assert {c["context"] for c in formats} == {"jpeg", "bmp"}
    assert all(c["status"] == "unavailable" and c["metrics"] is None for c in formats)
    assert "TOKEN_SECRET" not in json.dumps(result) and "/home/private" not in json.dumps(result)


def test_unknown_source_cannot_claim_independence(diagnostic_inputs, tmp_path):
    rebind_manifest(diagnostic_inputs, lambda m: m["samples"][2].update(source_group=None))
    result = diagnose(diagnostic_inputs, tmp_path / "out.json")
    assert not result["source_separation"]["metadata_complete"]
    assert result["source_separation"]["independent_source_evidence"] == "unavailable"


@pytest.mark.parametrize("rate", [True, [], -1, 101, float("inf")])
def test_invalid_payload_rate(diagnostic_inputs, tmp_path, rate):
    rebind_manifest(diagnostic_inputs, lambda m: m["samples"][2].update(rate_percent=rate))
    path = diagnostic_inputs[1]
    document = json.loads(path.read_bytes())
    document["predictions"][0]["rate_percent"] = rate
    path.write_text(json.dumps(document))
    with pytest.raises(ResearchManifestError, match="payload rate"):
        diagnose(diagnostic_inputs, tmp_path / "out.json")


def test_metadata_changes_do_not_change_pooled_scores(diagnostic_inputs, tmp_path):
    original = diagnose(diagnostic_inputs, tmp_path / "original.json")
    rebind_manifest(
        diagnostic_inputs,
        lambda m: [
            s.update(source_group="/private/TOKEN_SECRET")
            for s in m["samples"]
            if s["split"] != "test"
        ],
    )
    changed = diagnose(diagnostic_inputs, tmp_path / "changed.json")
    assert "TOKEN_SECRET" not in json.dumps(changed)
    old_pooled = [c["metrics"] for c in original["cells"] if c["dimension"] == "pooled"]
    new_pooled = [c["metrics"] for c in changed["cells"] if c["dimension"] == "pooled"]
    assert old_pooled == new_pooled


def test_legacy_jpeg_adapter_never_changes_score_or_source_documents(diagnostic_inputs, tmp_path):
    from core.jpeg_features import FEATURE_NAMES as JPEG_NAMES

    names = list(JPEG_NAMES)
    manifest, predictions, card = diagnostic_inputs
    document = json.loads(predictions.read_bytes())
    document["schema_version"] = "jpeg-development-predictions-v1"
    document["provenance"].update(feature_version="jpeg-dct-summary-v1", feature_names=names)
    for row in document["predictions"]:
        row.pop("format")
        row.pop("rate_percent")
    predictions.write_text(json.dumps(document))
    model = json.loads(card.read_bytes())
    model["domain"] = "jpeg-dct-summary-linear-v1"
    for field in ("training_provenance", "preprocessing"):
        model[field].update(feature_version="jpeg-dct-summary-v1", feature_names=names)
    card.write_text(json.dumps(model))
    rebind_manifest(diagnostic_inputs, lambda m: [s.update(format="JPEG") for s in m["samples"]])
    before = {p: p.read_bytes() for p in diagnostic_inputs}
    result = diagnose(diagnostic_inputs, tmp_path / "legacy.json")
    assert result["legacy_fields_from_bound_manifest"] == 4
    assert any(c["context"] == "jpeg" for c in result["cells"])
    assert all(c["metrics"]["roc_auc"] == 1 for c in result["cells"])
    assert all(p.read_bytes() == data for p, data in before.items())
    document = json.loads(predictions.read_bytes())
    document["schema_version"] = "unknown"
    predictions.write_text(json.dumps(document))
    with pytest.raises(ResearchManifestError, match="identity"):
        diagnose(diagnostic_inputs, tmp_path / "unsupported-legacy.json")
