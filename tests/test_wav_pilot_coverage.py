"""Coverage for the bound AudioMNIST pilot modules using generated PCM only."""

import builtins
import contextlib
import copy
import hashlib
import io
import json
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from core import nonjpeg_features, nonjpeg_linear, wav_pilot, wav_pilot_training

SEEDS = wav_pilot_training.SEEDS
REAL_RUN_LINEAR_PAIR = wav_pilot_training.run_linear_pair


def wav_bytes(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def chunk(name, payload):
    return name + len(payload).to_bytes(4, "little") + payload + b"\0" * (len(payload) & 1)


def riff(*chunks):
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Write 400 train /200 development generated originals in 30 speaker groups."""
    root = tmp_path_factory.mktemp("wav-corpus")
    records = []
    for role, count in (("train", 400), ("development", 200)):
        for index in range(count):
            speaker = f"generated-{role}-{index // 20:02d}"
            seed = hashlib.sha256(f"{role}:{index}".encode()).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed, "little"))
            raw = wav_bytes(rng.integers(-32768, 32768, size=480, dtype=np.int16))
            relative = f"data/{speaker}/{index % 10}_{speaker}_{(index // 10) % 2}.wav"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            records.append(
                {
                    "relative_source_path": relative,
                    "role": role,
                    "speaker_group": speaker,
                    "parent_recording_id": None,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return root, records


@pytest.fixture
def cached_rows(monkeypatch):
    """Cache materialized rows and features so repeated epochs stay fast."""
    real_materialize = wav_pilot_training.materialize
    real_features = nonjpeg_features.audio_features
    materialized, features = {}, {}

    def materialize(root, row, recipe):
        key = (row["relative_source_path"], recipe)
        if key not in materialized:
            materialized[key] = real_materialize(root, row, recipe)
        return materialized[key]

    def audio_features(data):
        if data not in features:
            features[data] = real_features(data)
        return features[data]

    monkeypatch.setattr(wav_pilot_training, "materialize", materialize)
    monkeypatch.setattr(nonjpeg_features, "audio_features", audio_features)


def manifest_doc(records):
    return {
        "schema_version": "wav-audiomnist-acquisition-v1",
        "source_commit": "630d7dab4c040882834de6fa21baf9a60372accd",
        "status": "acquired_native_headers_verified_not_training_ready",
        "reserved_speakers_accessed": False,
        "optimizer_updates": 0,
        "files": copy.deepcopy(records),
    }


def development_rows():
    rows = []
    for index in range(200):
        lineage, group = f"source-{index}", f"speaker-{index % 10}"
        source = {"role": "development"}
        rows.append(
            {"source": source, "lineage": lineage, "group": group, "label": "cover", "recipe": None}
        )
        for recipe in wav_pilot.RECIPES:
            rows.append(
                {
                    "source": source,
                    "lineage": lineage,
                    "group": group,
                    "label": "stego",
                    "recipe": recipe,
                }
            )
    return rows


def tiny_network(torch, scale=1.0):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, values):
            return self.linear(self.pool(values).flatten(1)) * scale

    return lambda _domain: Head()


# --- wav_pilot ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"RIFF\0\0\0\0WAVX", "invalid RIFF/WAVE"),
        (b"RIFF\0\0\0\0WAVE" + b"data" + (99).to_bytes(4, "little"), "chunk extent"),
        (riff(*[chunk(b"junk", b"")] * 65), "chunk extent"),
        (riff(chunk(b"data", b"ab"), chunk(b"data", b"cd")), "ambiguous"),
        (riff(chunk(b"fmt ", b"x")), "chunk layout"),
        (riff(chunk(b"data", b"ab")) + b"tail", "chunk layout"),
    ],
)
def test_wav_data_offset_rejects_ambiguous_or_truncated_layouts(raw, message):
    with pytest.raises(ValueError, match=message):
        wav_pilot._wav_data_offset(raw)


def test_load_manifest_accepts_bound_acquisition(tmp_path, corpus):
    _root, records = corpus
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_doc(records)))
    manifest, rows = wav_pilot.load_manifest(tmp_path)
    assert manifest["optimizer_updates"] == 0 and len(rows) == 600


def _mutate_header(doc):
    doc["reserved_speakers_accessed"] = True


def _mutate_count(doc):
    doc["files"] = doc["files"][:599]


def _mutate_path(doc):
    doc["files"][0]["relative_source_path"] = "data/other/0.wav"


def _mutate_duplicate_hash(doc):
    doc["files"][1]["sha256"] = doc["files"][0]["sha256"]


def _mutate_parent(doc):
    doc["files"][0]["parent_recording_id"] = "recording-0"


def _mutate_cross_role(doc):
    row = doc["files"][400]
    row["speaker_group"] = "generated-train-00"
    row["relative_source_path"] = "data/generated-train-00/crossed.wav"


def _mutate_group_count(doc):
    row = doc["files"][0]
    row["speaker_group"] = "generated-train-99"
    row["relative_source_path"] = "data/generated-train-99/extra.wav"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_mutate_header, "manifest binding"),
        (_mutate_count, "manifest binding"),
        (_mutate_path, "unsafe acquisition row"),
        (_mutate_duplicate_hash, "unsafe acquisition row"),
        (_mutate_parent, "unsafe acquisition row"),
        (_mutate_cross_role, "crosses roles"),
        (_mutate_group_count, "role/group count"),
    ],
)
def test_load_manifest_rejects_changed_bindings(tmp_path, corpus, mutate, message):
    doc = manifest_doc(corpus[1])
    mutate(doc)
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=message):
        wav_pilot.load_manifest(tmp_path)


def test_load_manifest_rejects_symlinked_manifest(tmp_path, corpus):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(manifest_doc(corpus[1])))
    root = tmp_path / "root"
    root.mkdir()
    (root / "manifest.json").symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        wav_pilot.load_manifest(root)


@pytest.mark.parametrize("rel", ["/etc/passwd", "data/../x.wav", "data//x.wav", "./x.wav"])
def test_materialize_rejects_unsafe_relative_paths(tmp_path, rel):
    with pytest.raises(ValueError, match="unsafe source path"):
        wav_pilot.materialize(tmp_path, {"relative_source_path": rel, "sha256": "0" * 64}, None)


def _source(tmp_path):
    raw = wav_bytes(np.arange(160, dtype=np.int16))
    (tmp_path / "source.wav").write_bytes(raw)
    return {"relative_source_path": "source.wav", "sha256": hashlib.sha256(raw).hexdigest()}


def test_materialize_returns_cover_bytes_and_rejects_unknown_recipe(tmp_path):
    row = _source(tmp_path)
    assert wav_pilot.materialize(tmp_path, row, None) == (tmp_path / "source.wav").read_bytes()
    with pytest.raises(ValueError, match="unknown controlled WAV recipe"):
        wav_pilot.materialize(tmp_path, row, ("sequential", 5, 2))


def test_materialize_rejects_pcm_extent_beyond_file(tmp_path, monkeypatch):
    row = _source(tmp_path)
    monkeypatch.setattr(wav_pilot, "pcm16_samples", lambda _raw: np.zeros(10_000, dtype=np.int16))
    with pytest.raises(ValueError, match="PCM data extent"):
        wav_pilot.materialize(tmp_path, row, None)


def test_materialize_independent_oracle_rejects_wrong_embedding(tmp_path, monkeypatch):
    row = _source(tmp_path)
    real = wav_pilot.replace
    monkeypatch.setattr(
        wav_pilot, "replace", lambda values, *args: (values.copy(), real(values, *args)[1])
    )
    with pytest.raises(ValueError, match="payload oracle"):
        wav_pilot.materialize(tmp_path, row, ("sequential", 40, 0))


def test_materialize_rejects_structural_change(tmp_path, monkeypatch):
    row = _source(tmp_path)
    real = wav_pilot.pcm16_samples
    calls = []

    def changing(raw):
        calls.append(raw)
        samples = real(raw)
        return samples if len(calls) == 1 else samples[:-1]

    monkeypatch.setattr(wav_pilot, "pcm16_samples", changing)
    with pytest.raises(ValueError, match="structural integrity"):
        wav_pilot.materialize(tmp_path, row, ("scattered", 5, 1))


def test_rows_for_role_expands_complete_families_and_keeps_roles_closed(corpus):
    _root, records = corpus
    train = wav_pilot.rows_for_role(records, "train")
    development = wav_pilot.rows_for_role(records, "development")
    assert len(train) == 3600 and len(development) == 1800
    assert sum(row["label"] == "cover" for row in train) == 400
    assert {row["recipe"] for row in train} == {None, *wav_pilot.RECIPES}
    with pytest.raises(ValueError, match="closed WAV role"):
        wav_pilot.rows_for_role(records, "reserved")
    with pytest.raises(ValueError, match="role/group count"):
        wav_pilot.rows_for_role(records[1:], "train")


def test_generated_row_smoke_detects_oracle_failure(monkeypatch):
    real = wav_pilot.replace
    monkeypatch.setattr(
        wav_pilot, "replace", lambda values, *args: (values.copy(), real(values, *args)[1])
    )
    with pytest.raises(ValueError, match="generated WAV scalar oracle"):
        wav_pilot.generated_row_smoke()


# --- wav_pilot_training: contracts and linear arm -------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": True},
        {"label": 2},
        {"label": 0, "row_count": 3601},
        {"label": 0, "row_count": 0, "originals": 0},
        {"label": 0, "row_count": 9.0, "originals": 1},
    ],
)
def test_balanced_row_weight_rejects_changed_contract(kwargs):
    label = kwargs.pop("label")
    with pytest.raises(ValueError, match="class-balance"):
        wav_pilot_training.balanced_row_weight(label, **kwargs)


def test_generated_schedule_smoke_rejects_changed_schedule(monkeypatch):
    monkeypatch.setattr(wav_pilot_training, "epoch_schedule", lambda _rows: [64] * 56)
    with pytest.raises(ValueError, match="virtual batch schedule"):
        wav_pilot_training.generated_schedule_smoke()


def test_feature_matrix_maps_rows_to_bounded_linear_metadata(corpus, cached_rows):
    root, records = corpus
    rows = wav_pilot.rows_for_role(records, "development")
    x, y, metadata = wav_pilot_training.feature_matrix(root, rows, "development")
    assert x.shape == (1800, 760) and np.isfinite(x).all()
    assert int(y.sum()) == 1600
    assert metadata[0]["method"] is None and metadata[0]["channel"] is None
    assert metadata[1]["channel"] == "mono" and metadata[1]["split"] == "development"


def test_feature_matrix_rejects_role_count_split_and_feature_contract(corpus, monkeypatch):
    root, records = corpus
    rows = wav_pilot.rows_for_role(records, "train")
    with pytest.raises(ValueError, match="role/count"):
        wav_pilot_training.feature_matrix(root, rows, "reserved")
    with pytest.raises(ValueError, match="role/count"):
        wav_pilot_training.feature_matrix(root, rows[:-1], "train")
    with pytest.raises(ValueError, match="split mismatch"):
        wav_pilot_training.feature_matrix(
            root, wav_pilot.rows_for_role(records, "development") * 2, "train"
        )
    for bad in (np.full(760, np.nan), np.full(760, 2.0), np.zeros(761)):
        monkeypatch.setattr(nonjpeg_features, "audio_features", lambda _data, bad=bad: bad)
        with pytest.raises(ValueError, match="audio feature contract"):
            wav_pilot_training.feature_matrix(root, rows, "train")


def test_linear_arm_rejects_changed_seed_exposure_and_invalid_scores(corpus, monkeypatch):
    root, records = corpus
    rows = wav_pilot.rows_for_role(records, "train")
    with pytest.raises(ValueError, match="seed/exposure"):
        wav_pilot_training.fit_linear_seed(root, rows, 1)
    with pytest.raises(ValueError, match="seed/exposure"):
        wav_pilot_training.fit_linear_seed(root, rows[:9], SEEDS[0])
    monkeypatch.setattr(
        wav_pilot_training,
        "feature_matrix",
        lambda *_args, **_kwargs: (np.zeros((3600, 760)), np.zeros(3600), []),
    )
    monkeypatch.setattr(
        nonjpeg_linear,
        "fit",
        lambda *_args, **_kwargs: {"optimizer_updates": 1139, "row_presentations": 72000},
    )
    with pytest.raises(ValueError, match="linear exposure mismatch"):
        wav_pilot_training.fit_linear_seed(root, rows, SEEDS[0])
    monkeypatch.setattr(nonjpeg_linear, "predict", lambda _model, x: np.full(len(x), np.nan))
    with pytest.raises(ValueError, match="linear development scores"):
        wav_pilot_training.predict_linear(root, rows, {})


def test_linear_pair_runs_both_seeds_with_matched_exposure(corpus, cached_rows):
    root, records = corpus
    result = wav_pilot_training.run_linear_pair(
        root,
        wav_pilot.rows_for_role(records, "train"),
        wav_pilot.rows_for_role(records, "development"),
    )
    assert result["status"] == "completed_linear_pair"
    assert result["optimizer_updates"] == 2280 and result["checkpoint_written"] is False
    assert [row["seed"] for row in result["seeds"]] == list(SEEDS)
    assert all(len(row["cells"]) == 8 for row in result["seeds"])
    assert all(row["fit"]["row_presentations"] == 72000 for row in result["seeds"])


def test_linear_pair_enforces_cpu_ceiling(corpus):
    root, records = corpus
    rows = wav_pilot.rows_for_role(records, "train")
    with pytest.raises(ValueError, match="positive CPU ceiling"):
        wav_pilot_training.run_linear_pair(root, rows, rows, ceiling_seconds=0)
    with pytest.raises(TimeoutError, match="CPU ceiling"):
        wav_pilot_training.run_linear_pair(root, rows, rows, ceiling_seconds=1e-12)


# --- wav_pilot_training: development metrics ------------------------------------


def _scores():
    return np.asarray([0.1 if row["label"] == "cover" else 0.9 for row in development_rows()])


def _drop_identity(rows):
    rows[0]["lineage"] = None


def _duplicate_cover(rows):
    rows[9] = dict(rows[0])


def _unknown_recipe(rows):
    rows[1]["recipe"] = ("sequential", 5, 2)


def _missing_cover(rows):
    rows[0] = dict(rows[1])


def _unmatched_cell(rows):
    rows[1]["recipe"] = rows[2]["recipe"]


def _group_mismatch(rows):
    rows[1]["group"] = "speaker-other"


def _one_cluster(rows):
    for row in rows:
        row["group"] = "speaker-0"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_identity, "bounded development identity"),
        (_duplicate_cover, "duplicate/invalid development cover"),
        (_unknown_recipe, "unknown development row recipe"),
        (_missing_cover, "incomplete development source families"),
        (_unmatched_cell, "incomplete matched development cell"),
        (_group_mismatch, "speaker group mismatch"),
        (_one_cluster, "speaker-cluster count"),
    ],
)
def test_development_metrics_rejects_changed_families(mutate, message):
    rows = [dict(row) for row in development_rows()]
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        wav_pilot_training.development_metrics(rows, _scores())


def test_development_metrics_rejects_invalid_score_contract():
    rows = development_rows()
    for scores in (_scores()[:-1], np.full(1800, 1.5), np.full(1800, np.nan)):
        with pytest.raises(ValueError, match="development-only score contract"):
            wav_pilot_training.development_metrics(rows, scores)
    train_rows = [dict(row, source={"role": "train"}) for row in rows]
    with pytest.raises(ValueError, match="development-only score contract"):
        wav_pilot_training.development_metrics(train_rows, _scores())


# --- wav_pilot_training: CNN arm (CPU-only test doubles for CUDA) ----------------


def test_fit_seed_rejects_contract_and_missing_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="seed/exposure"):
        wav_pilot_training.fit_seed(tmp_path, [{}] * 3600, 1)
    with pytest.raises(ValueError, match="seed/exposure"):
        wav_pilot_training.fit_seed(tmp_path, [{}] * 3600, SEEDS[0], deadline_seconds=0)
    with pytest.raises(ValueError, match="owned CUDA device"):
        wav_pilot_training.fit_seed(tmp_path, [{}] * 3600, SEEDS[0], device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device missing"):
        wav_pilot_training.fit_seed(tmp_path, [{}] * 3600, SEEDS[0])


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_fit_seed_restores_backend_policy_after_isolated_fit(tmp_path, monkeypatch, device):
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda devices: calls.append(devices) or contextlib.nullcontext(),
    )
    seen = {}

    def impl(_torch, _root, _rows, seed, **kwargs):
        seen.update(
            seed=seed,
            tf32=torch.backends.cuda.matmul.allow_tf32,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            **kwargs,
        )
        return "model", {"status": "stub"}

    monkeypatch.setattr(wav_pilot_training, "_fit_seed_impl", impl)
    prior = (torch.backends.cuda.matmul.allow_tf32, torch.are_deterministic_algorithms_enabled())
    model, fit = wav_pilot_training.fit_seed(tmp_path, [{}] * 3600, SEEDS[1], device=device)
    assert (model, fit) == ("model", {"status": "stub"}) and calls == [[0]]
    assert seen["seed"] == SEEDS[1] and seen["tf32"] is False and seen["deterministic"] is True
    assert seen["device"] == device and seen["deadline_seconds"] == 3600
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    ) == prior


def test_cnn_seed_fit_runs_frozen_schedule_on_generated_rows(corpus, cached_rows, monkeypatch):
    torch = pytest.importorskip("torch")
    root, records = corpus
    rows = wav_pilot.rows_for_role(records, "train")
    monkeypatch.setattr(wav_pilot_training, "EPOCHS", 1)
    monkeypatch.setattr(wav_pilot_training.nonjpeg_models, "network", tiny_network(torch))
    updates = []
    model, fit = wav_pilot_training._fit_seed_impl(
        torch, root, rows, SEEDS[0], device="cpu", on_update=lambda: updates.append(1)
    )
    assert fit["optimizer_updates"] == 57 == len(updates)
    assert fit["row_presentations"] == 3600 and fit["partial_batch_rows"] == 16
    assert fit["checkpoint_written"] is False and np.isfinite(fit["mean_batch_loss"])
    assert model.training is True

    development = wav_pilot.rows_for_role(records, "development")
    scores = wav_pilot_training.development_scores(root, development, model, device="cpu")
    assert scores.shape == (1800,) and np.all((scores >= 0) & (scores <= 1))
    assert model.training is True
    cells = wav_pilot_training.development_metrics(development, scores)
    assert len(cells) == 8


def _impl_rows(corpus):
    return [dict(row) for row in wav_pilot.rows_for_role(corpus[1], "train")]


def test_cnn_seed_fit_rejects_incomplete_or_crossed_families(corpus):
    torch = pytest.importorskip("torch")
    root = corpus[0]
    rows = _impl_rows(corpus)
    rows[0] = dict(rows[0], source={"role": "development"})
    with pytest.raises(ValueError, match="frozen train speakers"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu")
    rows = _impl_rows(corpus)
    rows[1]["group"] = "generated-train-19"
    with pytest.raises(ValueError, match="multiple speaker groups"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu")
    rows = _impl_rows(corpus)
    rows[1]["recipe"] = rows[2]["recipe"]
    with pytest.raises(ValueError, match="nine-row source families"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu")


def test_cnn_seed_fit_enforces_deadline_finite_gradients_and_accounting(
    corpus, cached_rows, monkeypatch
):
    torch = pytest.importorskip("torch")
    root = corpus[0]
    rows = _impl_rows(corpus)
    monkeypatch.setattr(wav_pilot_training.nonjpeg_models, "network", tiny_network(torch))
    with pytest.raises(TimeoutError, match="shared GPU ceiling"):
        wav_pilot_training._fit_seed_impl(
            torch, root, rows, SEEDS[0], device="cpu", deadline_seconds=1e-12
        )
    with pytest.raises(TimeoutError, match="shared GPU ceiling"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu", deadline_at=0)
    monkeypatch.setattr(
        wav_pilot_training.nonjpeg_models, "network", tiny_network(torch, float("nan"))
    )
    with pytest.raises(ValueError, match="non-finite AudioMNIST gradient"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu")
    monkeypatch.setattr(wav_pilot_training.nonjpeg_models, "network", tiny_network(torch))
    monkeypatch.setattr(wav_pilot_training, "EPOCHS", 1)
    monkeypatch.setattr(wav_pilot_training, "epoch_schedule", lambda _rows: [64] * 58)
    with pytest.raises(ValueError, match="exposure accounting mismatch"):
        wav_pilot_training._fit_seed_impl(torch, root, rows, SEEDS[0], device="cpu")


def test_development_scores_rejects_closed_rows_timeouts_and_invalid_scores(
    corpus, cached_rows, monkeypatch
):
    torch = pytest.importorskip("torch")
    root, records = corpus
    development = wav_pilot.rows_for_role(records, "development")
    model = tiny_network(torch)(None)
    with pytest.raises(ValueError, match="closed development role"):
        wav_pilot_training.development_scores(root, development[:-1], model, device="cpu")
    with pytest.raises(TimeoutError, match="shared GPU ceiling"):
        wav_pilot_training.development_scores(root, development, model, device="cpu", deadline_at=0)
    nan_model = tiny_network(torch, float("nan"))(None).eval()
    with pytest.raises(ValueError, match="invalid development score"):
        wav_pilot_training.development_scores(root, development, nan_model, device="cpu")
    assert nan_model.training is False
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("optional torch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="optional Torch missing"):
        wav_pilot_training.development_scores(root, development, model, device="cpu")


def test_cnn_pair_shares_one_ceiling_across_seeds(tmp_path, monkeypatch):
    calls = []

    def fit_seed(_root, rows, seed, **kwargs):
        calls.append((seed, kwargs["deadline_at"]))
        return f"model-{seed}", {"seed": seed, "optimizer_updates": 1140}

    monkeypatch.setattr(wav_pilot_training, "fit_seed", fit_seed)
    monkeypatch.setattr(
        wav_pilot_training, "development_scores", lambda *_args, **_kwargs: _scores()
    )
    result = wav_pilot_training.run_cnn_pair(tmp_path, [], development_rows(), device="cuda")
    assert result["status"] == "completed_cnn_pair" and result["optimizer_updates"] == 2280
    assert [seed for seed, _ in calls] == list(SEEDS) and calls[0][1] == calls[1][1]
    assert all(len(row["cells"]) == 8 for row in result["seeds"])
    with pytest.raises(ValueError, match="positive GPU ceiling"):
        wav_pilot_training.run_cnn_pair(tmp_path, [], [], ceiling_seconds=0)
    with pytest.raises(TimeoutError, match="shared GPU ceiling"):
        wav_pilot_training.run_cnn_pair(tmp_path, [], [], ceiling_seconds=1e-12)


# --- wav_pilot_training: generated CUDA preflight --------------------------------


@pytest.mark.parametrize(
    ("train", "development", "kwargs"),
    [
        ([160] * 399, [160] * 200, {}),
        ([160] * 400, [159] * 200, {}),
        ([160] * 400, [160.0] * 200, {}),
        ([160] * 400, [220501] * 200, {}),
        ([160] * 400, [160] * 200, {"device": "cpu"}),
        ([160] * 400, [160] * 200, {"ceiling_seconds": 0}),
    ],
)
def test_generated_preflight_rejects_changed_bounds(train, development, kwargs):
    with pytest.raises(ValueError, match="generated WAV preflight bounds"):
        wav_pilot_training.generated_cnn_preflight(train, development, **kwargs)


def _fake_cuda(monkeypatch, torch, peak):
    fractions = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: types.SimpleNamespace(total_memory=8 * 1024**3),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: fractions.append((fraction, index)),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak)
    return fractions


def _fake_pairs(monkeypatch, updates=1140):
    seen = {}

    def cnn(root, train_rows, development_rows, *, device, ceiling_seconds):
        seen["cnn"] = (Path(root), len(train_rows), len(development_rows), device)
        seen["files"] = len(list(Path(root).rglob("*.wav")))
        fit = {
            "epochs": 20,
            "row_presentations": 72000,
            "optimizer_updates": updates,
            "virtual_batch": 64,
            "partial_batch_rows": 16,
            "seconds": 1.0,
        }
        return {
            "seeds": [{"fit": dict(fit, seed=seed)} for seed in SEEDS],
            "optimizer_updates": 2 * updates,
            "seconds": 2.0,
        }

    def linear(root, train_rows, development_rows, *, ceiling_seconds):
        seen["linear"] = ceiling_seconds
        fit = {
            "epochs": 20,
            "row_presentations": 72000,
            "optimizer_updates": 1140,
            "exposure_sha256": "e" * 64,
        }
        # Mirror run_linear_pair: core.nonjpeg_linear.fit has no "seed" key.
        return {
            "seeds": [{"seed": seed, "fit": dict(fit)} for seed in SEEDS],
            "optimizer_updates": 2280,
            "seconds": 3.0,
        }

    monkeypatch.setattr(wav_pilot_training, "run_cnn_pair", cnn)
    monkeypatch.setattr(wav_pilot_training, "run_linear_pair", linear)
    return seen


def test_generated_preflight_reports_complete_generated_only_receipt(monkeypatch):
    torch = pytest.importorskip("torch")
    fractions = _fake_cuda(monkeypatch, torch, peak=606 * 1024**2)
    seen = _fake_pairs(monkeypatch)
    lengths = [160 + index for index in range(400)]
    result = wav_pilot_training.generated_cnn_preflight(lengths, [200] * 200)
    assert fractions == [(0.5, 0)]
    assert seen["cnn"][1:] == (3600, 1800, "cuda") and seen["files"] == 600
    assert seen["linear"] == 600 and not seen["cnn"][0].exists()
    assert result["status"] == "passed_generated_only"
    assert result["generated_optimizer_updates"] == 4560
    assert result["real_source_optimizer_updates"] == 0 and result["real_media_opened"] is False
    assert result["generated_file_count"] == 600
    assert (result["frame_count_min"], result["frame_count_max"]) == (160, 559)
    assert [row["seed"] for row in result["cnn_seed_receipts"]] == list(SEEDS)
    assert "exposure_sha256" in result["linear_seed_receipts"][0]


@pytest.mark.parametrize(("peak", "updates"), [(5 * 1024**3, 1140), (1, 1139)])
def test_generated_preflight_fails_closed_on_memory_or_exposure(monkeypatch, peak, updates):
    torch = pytest.importorskip("torch")
    _fake_cuda(monkeypatch, torch, peak=peak)
    _fake_pairs(monkeypatch, updates=updates)
    with pytest.raises(ValueError, match="preflight gate failed"):
        wav_pilot_training.generated_cnn_preflight([160] * 400, [160] * 200)


def test_generated_preflight_reports_missing_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA missing"):
        wav_pilot_training.generated_cnn_preflight([160] * 400, [160] * 200)


def test_generated_preflight_receipt_accepts_real_linear_pair(monkeypatch, cached_rows):
    """Regression for the 2026-10-19 attempt: real linear fits carry no seed key."""
    torch = pytest.importorskip("torch")
    _fake_cuda(monkeypatch, torch, peak=1)
    seen = _fake_pairs(monkeypatch)
    monkeypatch.setattr(wav_pilot_training, "run_linear_pair", REAL_RUN_LINEAR_PAIR)
    result = wav_pilot_training.generated_cnn_preflight([480] * 400, [480] * 200)
    assert seen["cnn"][1:] == (3600, 1800, "cuda")
    assert result["generated_linear_optimizer_updates"] == 2280
    receipts = result["linear_seed_receipts"]
    assert [row["seed"] for row in receipts] == list(SEEDS)
    assert all(row["row_presentations"] == 72000 for row in receipts)
    assert all(len(row["exposure_sha256"]) == 64 for row in receipts)
