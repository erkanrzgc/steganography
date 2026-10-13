"""Portable dated execution evidence must never masquerade as learning results."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def receipt():
    return json.loads((ROOT / "benchmarks/jpeg-segmented-gpu-start-20261013.json").read_bytes())


def test_complete_frozen_source_mapping_is_self_contained(receipt):
    mapping = receipt["source_sha256"]
    assert len(mapping) == receipt["execution_source_files"] == 52
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in mapping)
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in mapping.values())
    assert hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest() == \
        receipt["execution_source_closure_sha256"]


def test_observation_never_claims_completed_fit_or_improvement(receipt):
    assert receipt["status"] == "running_observed"
    assert not receipt["training_complete_at_observation"]
    assert not receipt["learning_gain_established"] and not receipt["deployed"]
    assert receipt["accuracy_qualification"] == "unavailable"
    assert not receipt["historical_failed_trials_reclassified"]
    assert not receipt["historical_snapshots_rebound_or_imported"]
    assert "scores" not in receipt and "metrics" not in receipt


def test_explicit_budget_does_not_reduce_exposure_or_raise_worker_caps(receipt):
    control = receipt["controls"]
    assert control["required_trials"] == 8
    assert control["presented_rows_per_logical_trial"] == \
        control["rows_per_segment"] * control["segments_per_trial"] == 32768
    assert control["per_worker_cpu_soft_seconds"] == control["per_worker_wall_seconds"] == 1800
    assert control["per_worker_cpu_hard_seconds"] == 1801
    assert control["per_worker_parent_timeout_seconds"] == 1830
    assert control["controls_budget_seconds"] == 21600
    assert control["service_watchdog_seconds"] == 21720
    assert not control["later_stage_extra_budget_authorized"]
    assert not control["later_stages_started"] and not receipt["cloud_spending"]


def test_native_proofs_are_generated_and_all_four_half_estimates_eligible(receipt):
    gates = receipt["generated_gates"]
    assert gates["status"] == "completed" and len(gates["receipts"]) == 12
    assert gates["archived_optimizer_bitwise_parity_all_arms"]
    assert gates["disk_resume_bitwise_parity_all_arms"]
    assert gates["fixture_counters_generated_not_real_fit"] and not gates["real_data_used"]
    assert len({p["name"] for p in gates["receipts"]}) == 12
    for value in receipt["generated_half_timing"]:
        expected = value["whole_seconds_per_update"] * (16384 // value["rows"]) * 1.25 + 60
        assert value["conservative_half_seconds"] == expected < 1800
        assert value["half_budget_eligible"] and value["measured_updates"] == 64


def test_historical_private_controller_error_not_silently_reclassified(receipt):
    fault = receipt["preserved_failed_preflight_controller"]
    assert fault["status"] == "failed" and fault["reason"] == "receipt_schema_key_error"
    assert not fault["native_generated_probes_failed"]
    assert receipt["verification"]["ci_jobs_passed"] == 8
    assert receipt["verification"]["full_suite_passed"] == 2743
    assert receipt["verification"]["total_coverage_percent"] >= 90
    assert receipt["verification"]["new_segment_core_coverage_percent"] >= 95


def test_portable_observation_retains_capacity_oom_without_secrets(receipt):
    capacity = receipt["capacity"]
    assert [p["rows"] for p in capacity] == [8, 16, 32]
    assert [p["status"] for p in capacity] == ["completed", "completed", "unavailable"]
    assert capacity[-1]["reason"] == "gpu_allocator_out_of_memory"
    encoded = json.dumps(receipt)
    assert all(secret not in encoded for secret in (
        "/home/", "/tmp/",  # noqa: S108 - forbid leaked host paths, not create a temporary file
        "password=", "api_key=", "SSH_AUTH_SOCK", "unredacted_command"))
