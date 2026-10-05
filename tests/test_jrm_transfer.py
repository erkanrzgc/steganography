"""Train-source exclusion and provenance guards; tiny fixtures are not accuracy evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from steganography import research_jrm as rj
from steganography.research_jrm_transfer import run_transfer
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha


@pytest.fixture
def prepared(corpus, tmp_path):
    pytest.importorskip("sealwatch")
    manifest, source = corpus
    for split in ("train", "validation"):
        rj.extract_cache(manifest, tmp_path / split, source=source, split=split)
    model_dir = tmp_path / "reference"
    rj.train_reference(
        manifest,
        tmp_path / "train/cache.json",
        model_dir,
        cache_sha256=sha(tmp_path / "train/cache.json"),
    )
    predictions = tmp_path / "reference-predictions.json"
    rj.predict_reference(
        manifest,
        tmp_path / "validation/cache.json",
        model_dir,
        predictions,
        cache_sha256=sha(tmp_path / "validation/cache.json"),
        card_sha256=sha(model_dir / "model-card.json"),
    )
    return {
        "stage": "source-transfer",
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "train_cache": str(tmp_path / "train/cache.json"),
        "train_cache_sha256": sha(tmp_path / "train/cache.json"),
        "validation_cache": str(tmp_path / "validation/cache.json"),
        "validation_cache_sha256": sha(tmp_path / "validation/cache.json"),
        "reference_predictions": str(predictions),
        "reference_predictions_sha256": sha(predictions),
    }, model_dir


def test_scope_and_single_source_actual_training(prepared, tmp_path, monkeypatch):
    import sealwatch as sw

    trainer = sw.ensemble_classifier.FldEnsembleTrainer
    seen = []

    def spy(covers, stegos, **kwargs):
        seen.append((covers.copy(), stegos.copy()))
        return trainer(covers, stegos, **kwargs)

    monkeypatch.setattr(sw.ensemble_classifier, "FldEnsembleTrainer", spy)
    config, _ = prepared
    path = tmp_path / "train/cache.json"
    features, samples, _ = rj.load_cache(
        Path(config["manifest"]), path, checksum=sha(path), split="train"
    )
    indices, all_scope = rj.training_scope(samples)
    assert indices.tolist() == list(range(6)) and all_scope["rows"] == 6
    for source in all_scope["source_ids"]:
        selected, scope = rj.training_scope(samples, source)
        assert len(selected) == 3 and len(scope["excluded_source_ids"]) == 1
        assert len({samples[i]["source_group"] for i in selected}) == 1
        card = rj.train_reference(
            Path(config["manifest"]),
            path,
            tmp_path / source,
            cache_sha256=sha(path),
            training_source_id=source,
        )
        assert card["training_scope"] == scope and card["paired_training_rows"] == 2
        assert card["upstream_training_vote_parity"]
        covers, stegos = rj.paired_indices([samples[i] for i in selected])
        np.testing.assert_array_equal(seen[-1][0], features[selected][covers])
        np.testing.assert_array_equal(seen[-1][1], features[selected][stegos])
        assert features.shape == (6, 11255)
    for bad in ("unknown", [], True):
        with pytest.raises(ValueError, match="source ID"):
            rj.training_scope(samples, bad)
    for bad_samples in ([], [{"source_group": None}], [{"source_group": " "}]):
        with pytest.raises(ValueError, match="named source"):
            rj.training_scope(bad_samples)


@pytest.mark.parametrize(
    "scope",
    [
        None,
        {},
        {"recipe": "bad"},
        {"recipe": "single-declared-source-v1", "source_ids": None},
        {"recipe": "single-declared-source-v1", "source_ids": ["unknown"]},
        "changed-count",
    ],
)
def test_scope_tampering_rejected_before_model(prepared, tmp_path, monkeypatch, scope):
    config, model_dir = prepared
    card_path = model_dir / "model-card.json"
    card = json.loads(card_path.read_bytes())
    if scope == "changed-count":
        card["training_scope"]["rows"] += 1
    else:
        card["training_scope"] = scope
    card_path.write_text(json.dumps(card))
    monkeypatch.setattr(rj, "load_model", lambda *a, **kw: pytest.fail("unsafe scope accepted"))
    with pytest.raises(ValueError, match="scope|source ID"):
        rj.predict_reference(
            Path(config["manifest"]),
            Path(config["validation_cache"]),
            model_dir,
            tmp_path / "bad.json",
            cache_sha256=config["validation_cache_sha256"],
            card_sha256=sha(card_path),
        )


def test_legacy_card_and_cli_transfer(prepared, tmp_path):
    config, model_dir = prepared
    card_path = model_dir / "model-card.json"
    card = json.loads(card_path.read_bytes())
    card.pop("training_scope")
    card_path.write_text(json.dumps(card))
    rj.predict_reference(
        Path(config["manifest"]),
        Path(config["validation_cache"]),
        model_dir,
        tmp_path / "legacy.json",
        cache_sha256=config["validation_cache_sha256"],
        card_sha256=sha(card_path),
    )
    config_path = tmp_path / "transfer.json"
    config_path.write_text(json.dumps(config))
    result = rj.run_reference(config_path, tmp_path / "transfer")
    assert result["qualification"] == "unavailable" and not result["deployed"]
    cells = result["comparison_cells"]
    assert len(cells) == 8 and sum(c["training_excluded_target"] for c in cells) == 4
    for cell in cells:
        assert cell["new"]["positives"] == cell["new"]["negatives"] == 1
        confusion = cell["new"]["confusion"]
        assert confusion["tp"] + confusion["fn"] == 1
        assert confusion["fp"] + confusion["tn"] == 1
        for key, delta in cell["delta_new_minus_reference"].items():
            assert delta == pytest.approx(cell["new"][key] - cell["reference"][key])
    assert "/home/" not in (tmp_path / "transfer/report.json").read_text()
    with pytest.raises(FileExistsError):
        run_transfer(config, tmp_path / "transfer")


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "reference-hash",
        "cache-hash",
        "score",
        "rows",
        "identity",
        "schema",
        "sources",
        "missing-family",
        "failed-training",
    ],
)
def test_transfer_preflight_and_partial_failure(prepared, tmp_path, monkeypatch, fault):
    config, _ = prepared
    if fault == "manifest":
        config["manifest_sha256"] = "0" * 64
    elif fault == "reference-hash":
        config["reference_predictions_sha256"] = "0" * 64
    elif fault == "cache-hash":
        config["train_cache_sha256"] = "0" * 64
    elif fault in {"score", "rows", "identity", "schema"}:
        path = Path(config["reference_predictions"])
        doc = json.loads(path.read_bytes())
        if fault == "score":
            doc["predictions"][0]["score"] = float("nan")
        if fault == "rows":
            doc["predictions"].pop()
        if fault == "identity":
            doc["predictions"][0]["lineage"] = "changed"
        if fault == "schema":
            doc["schema_version"] = "unknown"
        path.write_text(json.dumps(doc))
        config["reference_predictions_sha256"] = sha(path)
    elif fault == "sources":
        monkeypatch.setattr(
            rj, "training_scope", lambda *a: (np.array([0]), {"source_ids": ["only"]})
        )
    elif fault == "missing-family":
        monkeypatch.setattr(
            rj, "paired_indices", lambda *a: (_ for _ in ()).throw(ValueError("pairs"))
        )
    else:
        monkeypatch.setattr(
            rj, "train_reference", lambda *a, **k: (_ for _ in ()).throw(ValueError("failed"))
        )
    with pytest.raises(ValueError):
        run_transfer(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid/report.json").exists()
