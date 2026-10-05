import hashlib
import json
import math

import numpy as np
import pytest

from core.spatial_cooccurrence import FEATURE_NAMES
from steganography import research_operating_point as op
from steganography.research import ResearchManifestError

torch = pytest.importorskip("torch")


def fixture(tmp_path, *, complete=True, roles=True):
    groups = {"fit": [], "assessment": []}
    index = 0
    while any(len(g) < 2 for g in groups.values()):
        lineage = hashlib.sha256(f"fixture-{index}".encode()).hexdigest()
        index += 1
        r = op.role(lineage)
        if len(groups[r]) < 2:
            groups[r].append(lineage)
    lineages = [*groups["fit"], *groups["assessment"]] if roles else groups["fit"]
    samples = []
    for lineage in lineages:
        for method, rate in [
            (None, 0),
            *[(m, r) for m in ("sequential", "scattered") for r in (5, 20, 40)],
        ]:
            digest = hashlib.sha256(f"{lineage}:{method}:{rate}".encode()).hexdigest()
            samples.append(
                {
                    "path": digest + ".png",
                    "sha256": digest,
                    "size": 1,
                    "lineage": lineage,
                    "label": "cover" if method is None else "stego",
                    "split": "validation",
                    "method": method,
                    "rate_percent": rate,
                    "source_group": "fixture",
                }
            )
    if not complete:
        samples.pop()
    manifest = {
        "schema_version": "1.0",
        "samples": samples,
        "source": ".",
        "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [
        {k: s[k] for k in ("sha256", "lineage", "label")} | {"values": [0.0] * 468} for s in samples
    ]
    artifact = tmp_path / "features.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "research-features-1",
                "feature_version": "spatial-cooccurrence-v1",
                "feature_names": list(FEATURE_NAMES),
                "manifest_sha256": digest,
                "split": "validation",
                "rows": rows,
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "manifest": str(path),
                "features": str(artifact),
                "features_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
    )
    linear = torch.nn.Linear(468, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "features": 468,
            "state_dict": linear.state_dict(),
            "domain": "spatial-cooccurrence-linear-v1",
            "preprocessing": {
                "feature_version": "spatial-cooccurrence-v1",
                "feature_names": list(FEATURE_NAMES),
                "normalization": None,
                "inference_arithmetic": "float64",
            },
            "training_provenance": {"manifest_sha256": digest},
        },
        checkpoint,
    )
    return config, checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_quantile_ties_saturation_and_invalid_inputs():
    for scores in ([0.5] * 100, [1.0] * 10, [0.0] * 10, list(np.linspace(0, 1, 200))):
        threshold = op.choose_threshold(scores)
        assert 0 <= threshold <= 1
        clipped = np.clip(scores, 1e-12, 1 - 1e-12)
        assert sum(s >= threshold for s in clipped) <= math.floor(0.03 * len(scores))
    for scores in ([], [0.1], [float("nan"), 0.2], [float("inf"), 0.3], [-0.1, 0.5]):
        with pytest.raises(ResearchManifestError, match="finite"):
            op.choose_threshold(scores)


def test_disjoint_roles_and_exact_tied_threshold(tmp_path, monkeypatch):
    config, checkpoint, digest = fixture(tmp_path)
    result = op.run_assessment(config, checkpoint, tmp_path / "out", checkpoint_sha256=digest)
    assert result["fit_cover_count"] == 2 and result["assessment_lineages"] == 2
    assert result["fit_false_positives"] == 0
    assert not result["deployed"] and not result["probability_calibrated"]
    assert str(tmp_path) not in json.dumps(result)
    for cell in result["by_method_rate"].values():
        assert cell["before"]["false_positive_rate"] == 1
        assert cell["after"]["false_positive_rate"] == 0
        assert cell["before"]["recall"] == 1 and cell["after"]["recall"] == 0
        assert cell["before"]["roc_auc"] == cell["after"]["roc_auc"] == 0.5
        assert (
            cell["before"]["expected_calibration_error"]
            == cell["after"]["expected_calibration_error"]
        )
    rows = json.loads((tmp_path / "out/scores.json").read_bytes())["rows"]
    assert all(r["role"] == op.role(r["lineage"]) for r in rows)
    with pytest.raises(FileExistsError):
        op.run_assessment(config, checkpoint, tmp_path / "out", checkpoint_sha256=digest)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        op.run_assessment(config, checkpoint, link / "out", checkpoint_sha256=digest)
    monkeypatch.setattr(op, "run_assessment", lambda *a, **k: result)
    monkeypatch.setattr(
        "sys.argv",
        [
            "point",
            str(config),
            str(checkpoint),
            str(tmp_path / "cli"),
            "--checkpoint-sha256",
            digest,
        ],
    )
    op.main()


@pytest.mark.parametrize(
    "complete,roles,match", [(False, True, "complete"), (True, False, "two lineages")]
)
def test_incomplete_or_empty_role_refused(tmp_path, complete, roles, match):
    config, checkpoint, digest = fixture(tmp_path, complete=complete, roles=roles)
    with pytest.raises(ResearchManifestError, match=match):
        op.run_assessment(config, checkpoint, tmp_path / "out", checkpoint_sha256=digest)
    assert not (tmp_path / "out").exists()


def test_wrong_precision_and_nonfinite_refused(tmp_path):
    config, checkpoint, _ = fixture(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    saved["preprocessing"]["inference_arithmetic"] = "float32"
    torch.save(saved, checkpoint)
    with pytest.raises(ResearchManifestError, match="contract"):
        op.run_assessment(
            config,
            checkpoint,
            tmp_path / "bad",
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
    saved["preprocessing"]["inference_arithmetic"] = "float64"
    saved["state_dict"]["bias"].fill_(float("nan"))
    torch.save(saved, checkpoint)
    with pytest.raises(ResearchManifestError, match="nonfinite"):
        op.run_assessment(
            config,
            checkpoint,
            tmp_path / "nan",
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
