"""Read-only post-fit replay is not in-sample accuracy qualification."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from steganography import research_srnet_sanity as service
from tests.test_srnet_sanity import inputs as inputs  # noqa: F401


@pytest.fixture
def audit_inputs(inputs, tmp_path, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    config, out = inputs
    service.run(config, out)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    path = Path(__file__).resolve().parents[1] / "scripts/audit-srnet-tiny-sanity.py"
    spec = importlib.util.spec_from_file_location("tiny_sanity_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    monkeypatch.setattr(audit, "load_pixels", service.load_pixels)

    class Model:
        def state_dict(self):
            return {"bn.num_batches_tracked": torch.tensor(400)}

    monkeypatch.setattr(audit.srnet_model, "load_model", lambda *a, **kw: Model())
    monkeypatch.setattr(
        audit.srnet_reference,
        "reference_logits",
        lambda a, v: np.array([[-5.0, 5.0]]) if v[0, 0, 0, 0] else np.array([[5.0, -5.0]]),
    )
    return audit, config_path, out


def test_complete_replay_input_differences_and_no_validation(audit_inputs):
    audit, config, out = audit_inputs
    result = audit.audit(config, out)
    assert result["passed"] and result["native_singleton_rows"] == 24
    assert result["bn_updates_verified"] == 400
    assert len(result["independent_forward_audit"]) == 9
    assert len(result["input_pair_differences"]) == 16
    assert all(d["input_difference_rms"] == 1 for d in result["input_pair_differences"])
    assert not result["validation_pixels_loaded"]
    assert result["accuracy_qualification"] == "unavailable"


@pytest.mark.parametrize("edit", ["status", "identities", "schedule", "objectives", "scores"])
def test_modified_report_rejected(audit_inputs, edit):
    audit, config, out = audit_inputs
    path = out / "sanity.json"
    report = json.loads(path.read_bytes())
    if edit == "status":
        report["status"] = "failed"
    elif edit == "identities":
        report["selected_rows"][0]["sha256"] = "f" * 64
    elif edit == "schedule":
        report["epoch_pair_schedule"][0]["pairs"] = 15
    elif edit == "objectives":
        report["sanity_objectives"]["final_batch_loss_at_most_0_35"] = False
    else:
        report["stored_bn_singleton_train_metrics"]["scores"][0] = 0.5
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit.audit(config, out)


def test_failed_reference_is_retained_and_cli_exit_nonzero(audit_inputs, monkeypatch, tmp_path):
    audit, config, out = audit_inputs
    monkeypatch.setattr(audit.srnet_reference, "reference_logits", lambda a, v: np.zeros((1, 2)))
    result = audit.audit(config, out)
    assert not result["passed"] and len(result["independent_forward_audit"]) == 9
    monkeypatch.setattr(
        audit.sys,
        "argv",
        [
            "audit",
            "--config",
            str(config),
            "--result",
            str(out),
            "--out",
            str(tmp_path / "audit.json"),
        ],
    )
    assert audit.main() == 2
