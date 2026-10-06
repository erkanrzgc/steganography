"""Independent schedule counts/identity oracles, not detector accuracy."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import cli
from core import jpeg_float256 as fp
from core import srnet_sampling as sampling
from steganography import research_pixels as rp
from steganography import research_srnet_plan as planner
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha


def rows():
    result = []
    # Unequal origins and Q-contexts: source A has 5 pairs/family, source B
    # has 1 at Q75 and 2 at Q95. Source balance is NOT flat cell balance.
    for source, quality, count in (("A", None, 5), ("B", 75, 1), ("B", 95, 2)):
        for lineage in range(count):
            for method in (None, "JUNIWARD", "UERD"):
                result.append(
                    {
                        "sha256": hashlib.sha256(str(len(result)).encode()).hexdigest(),
                        "lineage": f"{source}-{quality}-{lineage}",
                        "source_group": source,
                        "quality_factor": quality,
                        "label": "cover" if method is None else "stego",
                        "method": method,
                        "split": "train",
                        "format": "JPEG",
                    }
                )
    return result


def test_independent_pair_counts_complete_coverage_and_determinism():
    samples = rows()
    original = copy.deepcopy(samples)
    state = np.random.get_state()
    pairs, record = sampling.epoch_pairs(samples, seed=91, epoch=0)
    assert (
        record["ordered_pair_sha256"]
        == "3e806205e7677f2bf5881278c575c4bacde4f8c1d6b84eeab92019d6ebfa38ab"
    )
    assert samples == original and not pairs.flags.writeable
    assert np.array_equal(np.random.get_state()[1], state[1])
    np.testing.assert_array_equal(pairs, sampling.epoch_pairs(samples, seed=91, epoch=0)[0])
    assert not np.array_equal(pairs, sampling.epoch_pairs(samples, seed=91, epoch=1)[0])
    # Independent counting against input metadata, not implementation buckets.
    counts = Counter()
    seen = set()
    for cover, stego in pairs:
        c, s = samples[cover], samples[stego]
        assert c["label"] == "cover" and s["label"] == "stego"
        assert all(c[k] == s[k] for k in ("lineage", "source_group", "quality_factor"))
        counts[(s["source_group"], s["quality_factor"], s["method"])] += 1
        seen.add(int(stego))
    assert seen == {i for i, s in enumerate(samples) if s["label"] == "stego"}
    assert record["pairs"] == 24 and record["pairs_per_source"] == 12
    assert counts == {
        ("A", None, "JUNIWARD"): 6,
        ("A", None, "UERD"): 6,
        ("B", 75, "JUNIWARD"): 3,
        ("B", 75, "UERD"): 3,
        ("B", 95, "JUNIWARD"): 3,
        ("B", 95, "UERD"): 3,
    }
    identity = [[samples[c]["sha256"], samples[s]["sha256"]] for c, s in pairs]
    assert (
        record["ordered_pair_sha256"]
        == hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    )
    assert all(
        record["cells"][i]["sampled_pairs"] >= cell["original_pairs"]
        for i, cell in enumerate(record["cells"])
    )


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "not-list",
        "not-row",
        "validation",
        "format",
        "label",
        "source",
        "lineage",
        "quality",
        "quality-bool",
        "hash",
        "duplicate",
        "missing-cover",
        "missing-method",
        "cover-method",
        "method-type",
        "duplicate-method",
        "row-limit",
    ],
)
def test_sampling_rejects_unbound_incomplete_or_nontrain_rows(fault):
    samples = rows()
    if fault == "empty":
        samples = []
    if fault == "not-list":
        samples = ()
    if fault == "not-row":
        samples[0] = None
    edits = {
        "validation": ("split", "validation"),
        "format": ("format", "PNG"),
        "label": ("label", []),
        "source": ("source_group", ""),
        "lineage": ("lineage", ""),
        "quality": ("quality_factor", 101),
        "quality-bool": ("quality_factor", True),
        "hash": ("sha256", "bad"),
        "cover-method": ("method", "UERD"),
    }
    if fault in edits:
        key, value = edits[fault]
        samples[0][key] = value
    if fault == "duplicate":
        samples[1]["sha256"] = samples[0]["sha256"]
    if fault == "missing-cover":
        samples.pop(0)
    if fault == "missing-method":
        samples.pop(1)
    if fault == "method-type":
        samples[1]["method"] = []
    if fault == "duplicate-method":
        samples[1]["method"] = "UERD"
    if fault == "row-limit":
        samples *= 1000
    with pytest.raises(ValueError):
        sampling.epoch_pairs(samples, seed=1, epoch=0)


@pytest.mark.parametrize(
    "seed,epoch", [(True, 0), (-1, 0), (2**32, 0), (1, True), (1, -1), (1, 50)]
)
def test_bounded_schedule_parameters(seed, epoch):
    with pytest.raises(ValueError):
        sampling.epoch_pairs(rows(), seed=seed, epoch=epoch)


def test_source_and_expansion_limits(monkeypatch):
    samples = rows()
    monkeypatch.setattr(sampling, "MAX_PAIRS", 23)
    with pytest.raises(ValueError, match="pair limit"):
        sampling.epoch_pairs(samples, seed=1, epoch=0)
    many = []
    for i in range(9):
        for sample in rows()[:3]:
            many.append(
                {
                    **sample,
                    "source_group": str(i),
                    "sha256": hashlib.sha256(str(len(many)).encode()).hexdigest(),
                }
            )
    with pytest.raises(ValueError, match="source limit"):
        sampling.epoch_pairs(many, seed=1, epoch=0)


@pytest.fixture
def config(corpus, tmp_path, monkeypatch):
    manifest, source = corpus
    monkeypatch.setattr(fp, "decoder_contract", lambda: {"fixture": "generated"})
    monkeypatch.setattr(
        fp, "region", lambda _: {"image_size": [256, 256], "region": [0, 0, 256, 256]}
    )
    monkeypatch.setattr(
        fp, "pixel_batch", lambda files: np.full((len(files), 1, 256, 256), 128.125, dtype="<f4")
    )
    rp.extract_pixels(manifest, tmp_path / "cache", source=source, split="train", _float=True)
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "cache": str(tmp_path / "cache/cache.json"),
        "cache_sha256": sha(tmp_path / "cache/cache.json"),
        "epochs": 2,
        "seed": 91,
    }


def test_full_service_cli_scope_and_exclusive_outputs(config, tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "plan.json"
    assert cli.main(["research", "srnet-plan", "--config", str(path), "--out", str(out)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["epochs"][0]["pairs"] == 4 and len(record["epochs"]) == 2
    assert record["real_training"] == "unavailable" and not record["validation_used"]
    assert not record["deployed"] and str(tmp_path) not in out.read_text()
    for source in record["training_scope"]["source_ids"]:
        scoped = planner.plan_training({**config, "training_source_id": source}, tmp_path / source)
        assert scoped["training_scope"]["rows"] == 3
        assert scoped["epochs"][0]["pairs"] == 2
        assert {c["source_id"] for c in scoped["epochs"][0]["cells"]} == {source}
    with pytest.raises(FileExistsError):
        planner.plan_training(config, out)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(FileExistsError):
        planner.plan_training(config, link)


@pytest.mark.parametrize(
    "edit",
    [
        {"validation_cache": "forbidden"},
        {"epochs": True},
        {"epochs": 51},
        {"seed": -1},
        {"seed": 2**32},
        {"cache_sha256": "bad"},
        {"manifest": []},
        {"manifest_sha256": "0" * 64},
        {"cache_sha256": "0" * 64},
        {"training_source_id": "missing"},
    ],
)
def test_bad_config_never_publishes_complete_plan(config, edit, tmp_path):
    out = tmp_path / "bad-plan.json"
    with pytest.raises(ValueError):
        planner.plan_training({**config, **edit}, out)
    assert not out.exists()


def test_config_object_defaults_and_missing_keys(config):
    with pytest.raises(ValueError):
        planner.settings([])
    with pytest.raises(ValueError):
        planner.settings({})
    config = {k: v for k, v in config.items() if k not in {"epochs", "seed"}}
    assert planner.settings(config) == {"epochs": 10, "seed": 20261012}


def test_audit_entry_uses_checkout_not_stale_installed_package(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit-srnet-sampling.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(  # noqa: S603 — fixed checkout script, never corpus output
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0 and b"--manifest" in result.stdout
