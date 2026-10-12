"""Independent arithmetic, grouping and CPU controls, not GPU accuracy evidence."""

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import jpeg_study as c
from core import jpeg_study_audit as audit
from core import jpeg_study_data as d
from core import jpeg_study_entropy as entropy
from core import jpeg_study_optimizer as opt
from core import jpeg_study_state as state
from core import jpeg_study_views as views
from core import srnet_model
from steganography import research_jpeg_study as frontend


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def sample_rows(role="fit", count=2):
    rows = []
    for source in d.ORIGINS:
        for i in range(count):
            lineage = hashlib.sha256(f"{source}:{role}:{i}".encode()).hexdigest()
            for label in ("cover", "stego"):
                rows.append({"lineage": lineage, "source_group": source, "role": role,
                             "label": label})
    return rows


@pytest.fixture
def original_inputs(tmp_path):
    reserved_path = tmp_path / "reserved.json"
    reserved_sha = write(reserved_path, {"samples": [{"sha256": "f" * 64}]})
    inputs = []
    for source_index, source in enumerate(d.ORIGINS):
        root = tmp_path / source
        root.mkdir()
        samples = []
        for i in range(5):
            yy, xx = np.indices((512, 512))
            values = ((xx // (i + 2) + yy // (source_index + 2) + i * 37) % 256).astype("u1")
            stream = io.BytesIO()
            Image.fromarray(values).save(stream, format="PPM")
            raw = stream.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{i}.pgm"
            (root / name).write_bytes(raw)
            samples.append({"path": name, "sha256": digest, "lineage": digest,
                            "size": len(raw), "source_group": source,
                            "split": "train" if i < 4 else "validation"})
        manifest = {"schema_version": "boss-development-acquisition-v1" if source_index == 0
                    else "bows2-original-acquisition-v1", "samples": samples,
                    "purpose": "development covers only", "status": "completed",
                    "license": "test fixture only", "source_url": "https://example.invalid",
                    "reserved_manifest_sha256": [reserved_sha]}
        sha = write(root / "source.json", manifest)
        inputs.append({"root": str(root), "sha256": sha, "source_group": source})
    return inputs, [{"path": str(reserved_path), "sha256": reserved_sha}]


def fixture_generator(raw, lineage):
    # Independent deterministic coefficient change, NOT a claimed .4 embedder.
    jpeglib = pytest.importorskip("jpeglib")
    import tempfile

    assert hashlib.sha256(raw).hexdigest() == lineage
    with tempfile.TemporaryDirectory() as folder:
        cover, stego = Path(folder) / "cover.jpg", Path(folder) / "stego.jpg"
        with Image.open(io.BytesIO(raw)) as image:
            image.save(cover, quality=75, optimize=False)
        jpeglib.version.set("6b")
        jpeg = jpeglib.read_dct(str(cover))
        jpeg.load()
        values = jpeg.Y.copy()
        values[20, 20, 1, 1] += 1
        jpeg.Y = values
        jpeg.write_dct(str(stego))
        return [cover.read_bytes(), stego.read_bytes()]


@pytest.fixture
def prepared(tmp_path, original_inputs):
    inputs, reserved = original_inputs
    out = tmp_path / "prepared"
    manifest = d.prepare(inputs, reserved, out, generator=fixture_generator,
                         deadline=lambda: None, fit=2, development=2)
    sha = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    return out, manifest, sha


def test_reference_equations_and_first_step_ratio():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = opt.optimizer(model, reference=True, corrected_l2=True)
    v = np.zeros((1, 2))
    u = np.zeros((1, 2))
    weights = np.zeros((1, 2))
    for gradient in ([.01, -.02], [.005, -.00001], [-.04, .03], [0., 0.]):
        g = np.array([gradient], dtype=np.float64)
        model.weight.grad = torch.tensor(g, dtype=torch.float32)
        v = .9 * v + .1 * g
        u = np.maximum(.999 * u + 1e-8, np.abs(g))
        weights -= .001 * v / u
        optimizer.step()
        np.testing.assert_allclose(model.weight.detach().numpy(), weights, rtol=1e-6, atol=1e-9)
    assert optimizer.state[model.weight]["step"] == 4
    standard = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        standard.weight.zero_()
    standard.weight.grad = torch.tensor([[.01]])
    old = opt.optimizer(standard, corrected_l2=True)
    old.step()
    assert float(standard.weight.detach().abs()) / .0001 == pytest.approx(10, rel=1e-5)


def test_convolution_only_l2_independent_gradient():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
                                torch.nn.Flatten(), torch.nn.Linear(2, 2))
    penalty = opt.convolution_penalty(model)
    expected = float(np.square(model[0].weight.detach().numpy().astype(np.float64)).sum()) * 1e-4
    assert float(penalty.detach()) == pytest.approx(expected, rel=1e-6)
    penalty.backward()
    np.testing.assert_allclose(model[0].weight.grad.numpy(),
                               model[0].weight.detach().numpy() * 2e-4, rtol=1e-6)
    assert all(p.grad is None for n, p in model.named_parameters() if n != "0.weight")
    for reference in (False, True):
        optimizer = opt.optimizer(model, reference=reference, corrected_l2=True)
        assert optimizer.param_groups[0]["weight_decay"] == 0
    assert opt.optimizer(model, reference=True).param_groups[0]["weight_decay"] == .0001
    with pytest.raises(ValueError, match="convolution"):
        opt.convolution_penalty(torch.nn.Linear(1, 2))


@pytest.mark.parametrize("kwargs", [
    {"reference": 1}, {"corrected_l2": "yes"}, {"learning_rate": True},
    {"learning_rate": float("nan")}, {"learning_rate": 0}, {"learning_rate": 1},
])
def test_optimizer_configuration_rejections(kwargs):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        opt.optimizer(torch.nn.Linear(1, 2), **kwargs)


def test_optimizer_bad_gradient_state_and_closure_fail_closed():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 2, bias=False)
    optimizer = opt.optimizer(model, reference=True)
    before = model.weight.detach().clone()
    for gradient in (None, torch.full_like(model.weight, float("nan")),
                     torch.ones_like(model.weight).to_sparse()):
        model.weight.grad = gradient
        with pytest.raises(ValueError, match="gradients"):
            optimizer.step()
        assert torch.equal(model.weight, before)
    with pytest.raises(ValueError, match="closures"):
        optimizer.step(lambda: 0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.state[model.weight]["exp_inf"].fill_(-1)
    with pytest.raises(ValueError, match="state"):
        optimizer.step()
    model.double()
    model.weight.grad = torch.ones_like(model.weight)
    with pytest.raises(ValueError, match="gradients"):
        optimizer.step()


def test_preparation_reader_and_independent_audit(prepared):
    pytest.importorskip("scipy")
    root, manifest, sha = prepared
    d.validate_manifest(manifest)
    assert not manifest["deployed"] and not manifest["blind_test"]
    assert len(manifest["samples"]["fit"]) == 8
    assert not {r["lineage"] for r in manifest["samples"]["fit"]} & {
        r["lineage"] for r in manifest["samples"]["development"]
    }
    with d.Reader(root, sha, "fit", deadline=lambda: None) as reader:
        values = reader.batch(np.arange(8, dtype=np.int64))
        assert values.shape == (8, 1, 256, 256) and values.dtype == np.dtype("<f4")
        assert np.any(values[0] != values[1])
        for indices in (np.array([-1]), np.array([8]), np.array([0.0]),
                        np.empty(0, dtype=np.int64), np.zeros(33, dtype=np.int64)):
            with pytest.raises(ValueError):
                reader.batch(indices)
    proof = audit.audit(root, sha, deadline=lambda: None)
    assert proof["idct_rows"] == {"fit": 8, "development": 8}
    assert proof["maximum_absolute_pixel_difference"] <= 1e-4
    assert proof["optimizer_updates"] == 0


def test_fit_reader_never_opens_development_pixels(prepared, monkeypatch):
    root, _, sha = prepared
    real_open = d.regular_open
    opened = []

    def inspect(path):
        opened.append(path.name)
        return real_open(path)

    monkeypatch.setattr(d, "regular_open", inspect)
    with d.Reader(root, sha, "fit", deadline=lambda: None):
        pass
    assert "development.f32" not in opened


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(schema_version="wrong"),
    lambda m: m["decoder"].update(rounding="integers"),
    lambda m: m["samples"].update(development=[]),
    lambda m: m["samples"]["fit"][0].update(label="stego"),
    lambda m: m["samples"]["fit"][1].update(lineage="f" * 64),
    lambda m: m["samples"]["fit"][0].update(source_group="unknown"),
    lambda m: m["samples"]["fit"][0].update(role="development"),
    lambda m: m["samples"]["fit"][0].update(historical_role="test"),
    lambda m: m["samples"]["fit"][1].update(payload_rate=.2),
    lambda m: m["samples"]["fit"][0].update(size=True),
    lambda m: m["originals_per_source"].update(fit=99),
])
def test_manifest_adversarial_mutations(prepared, mutation):
    _, manifest, _ = prepared
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(ValueError):
        d.validate_manifest(changed)


def test_input_reservations_original_roles_and_overwrite(original_inputs, tmp_path):
    inputs, reserved = original_inputs
    forbidden = d.reservations(reserved)
    selected = d.originals(Path(inputs[0]["root"]), inputs[0]["sha256"], d.ORIGINS[0], forbidden)
    assert len(selected) == 4 and all(r["split"] == "train" for r in selected)
    assert selected == d.originals(Path(inputs[0]["root"]), inputs[0]["sha256"],
                                   d.ORIGINS[0], forbidden)
    with pytest.raises(ValueError, match="reserved"):
        d.originals(Path(inputs[0]["root"]), inputs[0]["sha256"], d.ORIGINS[0], {})
    with pytest.raises(ValueError):
        d.reservations([])
    bad = tmp_path / "empty.json"
    bad_sha = write(bad, {"samples": []})
    with pytest.raises(ValueError):
        d.reservations([{"path": str(bad), "sha256": bad_sha}])
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        d.prepare(inputs, reserved, out, generator=fixture_generator,
                  deadline=lambda: None, fit=2, development=2)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        d.fresh(link)
    with pytest.raises(ValueError):
        d.prepare(inputs, reserved, tmp_path / "bad", generator=fixture_generator,
                  deadline=lambda: None, fit=True, development=2)
    with pytest.raises(ValueError, match="insufficient"):
        d.prepare(inputs, reserved, tmp_path / "large", generator=fixture_generator,
                  deadline=lambda: None)


def test_duplicate_originals_fail_before_output(original_inputs, tmp_path):
    inputs, reserved = original_inputs
    root = Path(inputs[1]["root"])
    value = json.loads((root / "source.json").read_bytes())
    value["samples"].append(copy.deepcopy(value["samples"][0]))
    inputs[1]["sha256"] = write(root / "source.json", value)
    out = tmp_path / "duplicate"
    with pytest.raises(ValueError, match="identity"):
        d.prepare(inputs, reserved, out, generator=fixture_generator,
                  deadline=lambda: None, fit=2, development=2)
    assert not out.exists()


def test_simulation_bad_output_and_budget(original_inputs, tmp_path, monkeypatch):
    inputs, reserved = original_inputs
    for i, generator in enumerate((lambda *_: [], lambda raw, lineage: (
        fixture_generator(raw, lineage)[:1] * 2
    ))):
        with pytest.raises(ValueError, match="simulation|identical"):
            d.prepare(inputs, reserved, tmp_path / f"bad-{i}", generator=generator,
                      deadline=lambda: None, fit=2, development=2)
    monkeypatch.setattr(d, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        d.prepare(inputs, reserved, tmp_path / "budget", generator=fixture_generator,
                  deadline=lambda: None, fit=2, development=2)


def test_reader_mutation_nonfinite_and_wrong_role(prepared):
    root, manifest, sha = prepared
    with pytest.raises(ValueError, match="role"):
        d.Reader(root, sha, "test", deadline=lambda: None)
    with d.Reader(root, sha, "fit", deadline=lambda: None) as reader:
        with (root / "fit.f32").open("ab") as stream:
            stream.write(b"x")
        with pytest.raises(ValueError, match="changed"):
            reader.verify()
        with pytest.raises(ValueError):
            reader.batch(np.array([0]))
    with pytest.raises(ValueError, match="length"):
        d.Reader(root, sha, "fit", deadline=lambda: None)
    raw = np.zeros((8, 1, 256, 256), dtype="<f4")
    raw[0, 0, 0, 0] = np.nan
    encoded = raw.tobytes()
    (root / "fit.f32").write_bytes(encoded)
    manifest["tensors"]["fit"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    sha = write(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="nonfinite"):
        d.Reader(root, sha, "fit", deadline=lambda: None)


def test_schedule_matched_row_and_augmentation_exposure():
    samples = sample_rows(count=4)
    order, angles = c.schedule(samples, c.SEEDS[0], 4, 128)
    wider, wider_angles = c.schedule(samples, c.SEEDS[0], 16, 128)
    np.testing.assert_array_equal(order.ravel(), wider.ravel())
    np.testing.assert_array_equal(angles.ravel(), wider_angles.ravel())
    assert len(order) == 32 and len(wider) == 8
    assert np.all(order[:, 1::2] == order[:, ::2] + 1)
    assert np.all(angles[:, 1::2] == angles[:, ::2])
    assert set(np.bincount(order.ravel())) == {8}
    assert not order.flags.writeable
    pixels = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 1, 256, 256)
    result = c.augment(pixels, [0, 1, 4, 7])
    np.testing.assert_array_equal(result[0], pixels[0])
    np.testing.assert_array_equal(result[1], np.rot90(pixels[1], axes=(-2, -1)))
    np.testing.assert_array_equal(result[2], pixels[2][..., ::-1])
    with pytest.raises(ValueError):
        c.augment(pixels, [0, 0, 0, 8])
    with pytest.raises(ValueError):
        c.augment(pixels.astype(np.float64), [0] * 4)


@pytest.mark.parametrize("arm,seed,rows", [
    ("unknown", c.SEEDS[0], 4), ("legacy", True, 4), ("legacy", c.SEEDS[0], 8),
    ("genuine-batch", c.SEEDS[0], 4), ("legacy", c.SEEDS[0], True),
])
def test_recipe_invalid(arm, seed, rows):
    with pytest.raises(ValueError):
        c.recipe(arm, seed, rows)


def test_schedule_rejects_bad_roles_counts_settings():
    samples = sample_rows()
    for seed, rows, budget in ((True, 4, 16), (c.SEEDS[0], 2, 16), (c.SEEDS[0], 4, 3)):
        with pytest.raises(ValueError):
            c.schedule(samples, seed, rows, budget)
    with pytest.raises(ValueError):
        c.schedule(samples[:2], c.SEEDS[0], 4, 16)
    samples[0]["role"] = "development"
    with pytest.raises(ValueError):
        c.schedule(samples, c.SEEDS[0], 4, 16)
    samples = sample_rows(count=3)[:10]
    with pytest.raises(ValueError):
        c.schedule(samples, c.SEEDS[0], 4, 16)


def test_independent_auc_bootstrap_metrics_and_gate():
    samples = sample_rows(role="development", count=64)
    scores = np.tile([.1, .9], 128)
    baseline = np.full(len(scores), .5)
    assert c.auc(scores.reshape(-1, 2)) == 1
    assert c.auc(baseline.reshape(-1, 2)) == .5
    for cell in c.statistics(samples, scores, baseline=baseline, bootstrap=100):
        assert cell["metrics"]["roc_auc"] == 1
        assert cell["metrics"]["false_positive_rate"] == 0
        assert cell["auc_ci95"] == [1, 1]
        assert cell["paired_auc_delta_ci95"] == [.5, .5]
        assert cell["ece"] == pytest.approx(.1)
        assert cell["learning_gate"] and cell["support_status"] == "experimental"
    assert not any(v["learning_gate"] for v in c.statistics(samples, baseline, bootstrap=100))
    for bad in (scores[:-1], np.full(len(scores), np.nan), np.full(len(scores), 2)):
        with pytest.raises(ValueError):
            c.statistics(samples, bad)
    with pytest.raises(ValueError):
        c.statistics(samples, scores, baseline=baseline[:-1])
    with pytest.raises(ValueError):
        c.statistics(samples, scores, bootstrap=99)
    with pytest.raises(ValueError):
        c.statistics(samples[:128], scores[:128])


def tiny():
    torch = pytest.importorskip("torch")
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())


def test_cpu_kernel_controls_are_not_real_gpu_evidence(prepared, tmp_path):
    torch = pytest.importorskip("torch")
    root, _, sha = prepared
    previous = torch.get_num_threads()
    results = []
    for arm in c.ARMS[:3]:
        out = tmp_path / arm
        out.mkdir()
        results.append(c.kernel(root, sha, out, arm, c.SEEDS[0], 4, {}, device="cpu",
                                factory=tiny, row_budget=8))
    assert len({r["initial_learned_sha256"] for r in results}) == 1
    assert all(r["optimizer_updates"] == 2 for r in results)
    assert all(not r["real_gpu_fit"] and not r["deployed"] for r in results)
    assert torch.get_num_threads() == previous
    with pytest.raises(ValueError, match="budget"):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu", row_budget=3)


def test_missing_gpu_fails_before_opening_data(monkeypatch, tmp_path):
    from core.srnet_cuda import CUDAUnavailable

    monkeypatch.setattr(c.srnet_cuda, "inspect", lambda: (_ for _ in ()).throw(
        CUDAUnavailable("test_no_cuda")))
    with pytest.raises(CUDAUnavailable):
        c.fit(tmp_path / "does-not-exist", "0" * 64, tmp_path, "legacy", c.SEEDS[0], 4, {})


def test_cancellation_and_fail_closed_math(prepared, tmp_path):
    root, _, sha = prepared
    out = tmp_path / "cancel"
    out.mkdir()
    (out / "CANCEL").touch()
    with pytest.raises(InterruptedError):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu", factory=tiny,
                 row_budget=8)
    (out / "CANCEL").unlink()
    (out / "CANCEL").symlink_to(tmp_path / "none")
    with pytest.raises(ValueError, match="symlink"):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu", factory=tiny,
                 row_budget=8)


@pytest.mark.parametrize("fault", ["label", "role", "lineage", "source", "duplicate"])
def test_metrics_reject_mislabeled_or_leaked_pairs(fault):
    rows = sample_rows("development")
    if fault == "duplicate":
        rows[2]["lineage"] = rows[3]["lineage"] = rows[0]["lineage"]
    else:
        rows[1][fault if fault != "source" else "source_group"] = "wrong"
    with pytest.raises(ValueError, match="pair/role/ancestry"):
        c.statistics(rows, [.1, .9] * 4)


def test_small_perfect_cell_is_not_learning_evidence():
    rows = sample_rows("development")
    values = c.statistics(rows, [.1, .9] * 4, baseline=[.5] * 8, bootstrap=100)
    assert all(not cell["learning_gate"] for cell in values)


@pytest.fixture(scope="module")
def numeric_snapshot():
    arrays = {}
    for name, (shape, dtype) in state.specs().items():
        value = np.zeros(shape, dtype=dtype)
        if name.endswith("num_batches_tracked") or name.endswith("/step"):
            value[...] = 4096
        if name.endswith("running_var"):
            value[...] = 1
        arrays[name] = value
    result = {"metadata": {"schema_version": "jpeg-study-snapshot-v1",
                           "binding": "b" * 64, "presented_rows": 16384}, "arrays": arrays}
    state.validate(result)
    return result


@pytest.mark.parametrize("fault", ["container", "metadata", "rows", "binding", "keys",
                                   "shape", "dtype", "finite", "inf_norm", "step", "bn"])
def test_snapshot_rejects_hostile_numeric_state(numeric_snapshot, fault):
    value = {"metadata": copy.deepcopy(numeric_snapshot["metadata"]),
             "arrays": dict(numeric_snapshot["arrays"])}
    key = "optimizer/classifier.weight/exp_inf"
    if fault == "container":
        value["private_path"] = "/not-published"
    elif fault == "metadata":
        value["metadata"]["extra"] = "secret"
    elif fault == "rows":
        value["metadata"]["presented_rows"] = True
    elif fault == "binding":
        value["metadata"]["binding"] = "invalid"
    elif fault == "keys":
        del value["arrays"][key]
    elif fault == "shape":
        value["arrays"][key] = np.zeros(1, dtype="<f4")
    elif fault == "dtype":
        value["arrays"][key] = value["arrays"][key].astype("f8")
    elif fault == "finite":
        value["arrays"][key] = np.full_like(value["arrays"][key], np.nan)
    elif fault == "inf_norm":
        value["arrays"][key] = np.full_like(value["arrays"][key], -1)
    elif fault == "step":
        value["arrays"]["optimizer/classifier.weight/step"] = np.array(3, dtype="<f4")
    elif fault == "bn":
        value["arrays"]["model/front.0.1.num_batches_tracked"] = np.array(3, dtype="<i8")
    with pytest.raises(ValueError):
        state.validate(value)


def test_snapshot_roundtrip_and_recipe_binding(numeric_snapshot, tmp_path):
    path = tmp_path / "snapshot.npz"
    sha = state.save(numeric_snapshot, path)
    loaded = state.load(path, sha, "b" * 64)
    assert loaded["metadata"] == numeric_snapshot["metadata"]
    for name, array in loaded["arrays"].items():
        np.testing.assert_array_equal(array, numeric_snapshot["arrays"][name])
    with pytest.raises(FileExistsError):
        state.save(numeric_snapshot, path)
    with pytest.raises(ValueError, match="checksum"):
        state.load(path, "0" * 64, "b" * 64)
    with pytest.raises(ValueError, match="binding"):
        state.load(path, sha, "a" * 64)


def test_snapshot_restore_and_capture(numeric_snapshot):
    torch = pytest.importorskip("torch")
    model = c.srnet_samepool.network()
    optimizer = opt.optimizer(model, reference=True, corrected_l2=True)
    rows = state.restore(numeric_snapshot, model, optimizer, "b" * 64, 4)
    assert rows == 16384
    captured = state.capture(model, optimizer, "b" * 64, rows)
    for name, values in captured["arrays"].items():
        np.testing.assert_array_equal(values, numeric_snapshot["arrays"][name])
    before = model.classifier.weight.detach().clone()
    with pytest.raises(ValueError, match="batch"):
        state.restore(numeric_snapshot, model, optimizer, "b" * 64, 8)
    with pytest.raises(ValueError, match="binding"):
        state.restore(numeric_snapshot, model, optimizer, "a" * 64, 4)
    assert torch.equal(before, model.classifier.weight)
    assert set(model.state_dict()) == set(srnet_model.SHAPES)


@pytest.mark.parametrize("fault", ["members", "compression", "symlink", "shape", "meta",
                                   "length", "expansion"])
def test_snapshot_archive_preflight(numeric_snapshot, tmp_path, monkeypatch, fault):
    import zipfile

    # Small hostile archives are rejected before loading any real-sized arrays.
    buffer = io.BytesIO()
    if fault == "expansion":
        monkeypatch.setattr(state, "MAX_BYTES", 64)
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, (shape, dtype) in state.specs().items():
            if fault == "members" and name == "model/classifier.weight":
                continue
            raw = io.BytesIO()
            np.lib.format.write_array_header_1_0(raw, {
                "shape": (999,) if fault == "shape" else shape,
                "fortran_order": False, "descr": dtype.str,
            })
            entry = zipfile.ZipInfo(name + ".npy")
            if fault == "compression":
                entry.compress_type = zipfile.ZIP_DEFLATED
            if fault == "symlink":
                entry.external_attr = 0o120777 << 16
            archive.writestr(entry, raw.getvalue())
        raw = io.BytesIO()
        np.save(raw, np.zeros(0 if fault == "meta" else 1, dtype="u1"))
        archive.writestr("__metadata__.npy", raw.getvalue())
    path = tmp_path / "bad.npz"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        state.load(path, hashlib.sha256(path.read_bytes()).hexdigest(), "b" * 64)


def control_configuration():
    return {"schema_version": "jpeg-study-config-v1", "action": "controls", "root": "absent",
            "manifest_sha256": "a" * 64, "audit": "absent-audit", "audit_sha256": "b" * 64}


@pytest.mark.parametrize("fault", ["schema", "action", "extra", "root", "sha", "resume",
                                   "resume_duplicate", "resume_seed", "resume_path"])
def test_frontend_rejects_bad_configuration(fault):
    config = control_configuration()
    if fault == "schema":
        config["schema_version"] = "legacy"
    elif fault == "action":
        config["action"] = "automatic-deploy"
    elif fault == "extra":
        config["unexpected"] = "not-reportable"
    elif fault == "root":
        config["root"] = ""
    elif fault == "sha":
        config["audit_sha256"] = "invalid"
    else:
        row = {"arm": "legacy", "seed": c.SEEDS[0], "path": "resume.npz", "sha256": "a" * 64}
        config["resume"] = [row]
        if fault == "resume":
            config["resume"] = "wrong"
        elif fault == "resume_duplicate":
            config["resume"].append(row)
        elif fault == "resume_seed":
            row["seed"] = True
        elif fault == "resume_path":
            row["path"] = ""
    with pytest.raises(ValueError):
        frontend.configuration(config)


def test_frontend_cuda_absence_portable_report_before_data(tmp_path, monkeypatch):
    from core.srnet_cuda import CUDAUnavailable

    path = tmp_path / "config.json"
    write(path, control_configuration())
    monkeypatch.setattr(frontend.srnet_cuda, "inspect", lambda: (_ for _ in ()).throw(
        CUDAUnavailable("cuda_not_visible_or_cpu_torch")))
    report = frontend.run(path, tmp_path / "out")
    assert report["status"] == "unavailable" and report["optimizer_updates"] == 0
    assert not report["deployed"] and not report["cpu_fallback"]
    assert str(tmp_path) not in json.dumps(report)
    assert frontend.main(["--config", str(path), "--out", str(tmp_path / "second")]) == 2
    with pytest.raises(FileExistsError):
        frontend.run(path, tmp_path / "out")
    large = tmp_path / "large.json"
    large.write_bytes(b" " * 65537)
    with pytest.raises(ValueError, match="bounds"):
        frontend.run(large, tmp_path / "third")


def test_owned_worker_output_timeout_cancel_and_cleanup(tmp_path):
    import os
    import sys

    assert frontend.child([sys.executable, "-c", "print('ok')"], timeout=5) == b"ok\n"
    with pytest.raises(ValueError, match="failed"):
        frontend.child([sys.executable, "-c", "raise SystemExit(1)"], timeout=5)
    with pytest.raises(ValueError, match="output"):
        frontend.child([sys.executable, "-c", "print('x' * 300000)"], timeout=5)
    pid_path = tmp_path / "pid"
    command = [sys.executable, "-c",
               "import os,time;from pathlib import Path;"
               f"Path({str(pid_path)!r}).write_text(str(os.getpid()));time.sleep(30)"]
    with pytest.raises(TimeoutError):
        frontend.child(command, timeout=.5)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(InterruptedError):
        frontend.child(command, timeout=5, cancelled=lambda: True)
    with pytest.raises(ValueError, match="input"):
        frontend.child(command, timeout=5, payload=b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="limit"):
        frontend.child(command, timeout=5, maximum=2)


def test_real_simulation_and_independent_coefficient_audit(original_inputs):
    pytest.importorskip("conseal")
    jpeglib = pytest.importorskip("jpeglib")
    import base64
    import tempfile

    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(123).integers(
        0, 256, (512, 512), dtype="u1")).save(stream, format="PPM")
    raw = stream.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    encoded = frontend.simulate(raw, sha)
    cover, stego = [base64.b64decode(value, validate=True) for value in encoded]
    with tempfile.TemporaryDirectory() as folder:
        paths = [Path(folder) / name for name in ("cover.jpg", "stego.jpg")]
        before, after = [], []
        for path, value, target in zip(paths, [cover, stego], [before, after], strict=True):
            path.write_bytes(value)
            jpeg = jpeglib.read_dct(str(path))
            jpeg.load()
            target.extend([jpeg.Y.copy(), jpeg.get_component_qt(0).copy()])
        delta = after[0].astype(np.int64) - before[0].astype(np.int64)
        assert np.any(delta) and np.max(np.abs(delta)) == 1
        np.testing.assert_array_equal(before[1], after[1])
        import conseal

        seed = int.from_bytes(hashlib.sha256(f"jpeg-study:{sha}".encode()).digest()[:4], "big")
        spatial = jpeglib.read_spatial(str(paths[0]), jpeglib.JCS_GRAYSCALE).spatial[..., 0]
        reference = conseal.juniward.simulate_single_channel(
            spatial, before[0], before[1], .4, seed=seed)
        np.testing.assert_array_equal(after[0], reference)
    with pytest.raises(ValueError, match="identity"):
        frontend.simulate(raw, "0" * 64)


def test_simulation_response_validation(monkeypatch):
    for response in (["eA=="], {"status": "completed"}, ["***", "eA=="]):
        monkeypatch.setattr(frontend, "child", lambda *a, response=response, **k:
                            json.dumps(response).encode())
        with pytest.raises(ValueError):
            frontend.generate(b"raw", "a" * 64)
    monkeypatch.setattr(frontend, "child", lambda *a, **k: b'{"status":"unavailable"}')
    with pytest.raises(ImportError):
        frontend.generate(b"raw", "a" * 64)
    monkeypatch.setattr(frontend, "child", lambda *a, **k: b'["eA==","eQ=="]')
    assert frontend.generate(b"raw", "a" * 64) == [b"x", b"y"]


@pytest.mark.parametrize("error", [ImportError, ValueError, TimeoutError, InterruptedError])
def test_preparation_failure_retained_without_path_secrets(tmp_path, monkeypatch, error):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": []}
    path = tmp_path / "config.json"
    write(path, config)
    monkeypatch.setattr(frontend.jpeg_study_data, "prepare", lambda *a, **k:
                        (_ for _ in ()).throw(error("/private/secret")))
    result = frontend.run(path, tmp_path / "out")
    assert result["status"] == ("cancelled" if error is InterruptedError else "unavailable")
    assert result["optimizer_updates"] == 0 and "secret" not in json.dumps(result)


def control_mocks(monkeypatch, tmp_path, *, fault=None, representation=views.CENTER):
    import time

    before = {"source": "a" * 64}
    manifest = {"samples": {"fit": sample_rows(count=256),
                            "development": sample_rows("development", count=64)},
                "originals_per_source": {"fit": 256, "development": 64}}
    proof = {"schema_version": "jpeg-study-independent-audit-v1", "status": "completed",
             "manifest_sha256": "a" * 64, "source_sha256": before,
             "idct_rows": {"fit": 1024, "development": 256}}
    if representation == views.FIVE:
        manifest.update(schema_version=views.SCHEMA, representation=views.contract())
        proof.update(schema_version="jpeg-study-independent-audit-v2",
                     representation=views.contract(), idct_views={"fit": 5120, "development": 1280})
        if fault == "missing_view_audit":
            proof["idct_views"]["development"] = 256
        elif fault == "wrong_aggregation":
            proof["representation"]["inference"] = "max-positive-crop"
    monkeypatch.setattr(frontend, "document", lambda path, sha:
                        manifest if path.name == "manifest.json" else proof)
    monkeypatch.setattr(frontend.jpeg_study_data, "validate_manifest", lambda m: None)
    monkeypatch.setattr(frontend, "capacity_sources", lambda: before)

    def child(command, **kwargs):
        rows = int(command[command.index("--rows") + 1])
        if "steganography.research_batch_probe" in command:
            result = {"status": "completed", "rows": rows, "source_sha256": before}
            if fault == "no_batch":
                result["status"] = "unavailable"
            elif fault == "bad_probe":
                result["rows"] = 99
            return json.dumps(result).encode()
        arm = command[command.index("--arm") + 1]
        seed = int(command[command.index("--seed") + 1])
        result = {"status": "completed", "recipe": c.recipe(
            arm, seed, rows, representation=representation),
                  "presented_rows": c.PRESENTED_ROWS, "real_gpu_fit": True,
                  "source_sha256": before, "initial_learned_sha256": "a" * 64,
                  "schedule_sha256": "c" * 64,
                  "scores": [.5] * 256 if arm == "legacy" else [.1, .9] * 128}
        if fault == "timeout":
            raise TimeoutError
        if fault == "cancel":
            raise InterruptedError
        if fault == "missing_source":
            del result["source_sha256"]
        if fault == "partial":
            result["presented_rows"] = 4
        if fault == "different_source":
            result["source_sha256"] = {}
        if fault == "different_initialization" and arm != "legacy":
            result["initial_learned_sha256"] = "b" * 64
        if fault == "different_schedule" and arm != "legacy":
            result["schedule_sha256"] = "d" * 64
        if fault == "center_recipe":
            result["recipe"] = c.recipe(arm, seed, rows)
        if fault == "cpu":
            result["real_gpu_fit"] = False
        if fault == "invalid_status":
            result["status"] = "passed"
        return json.dumps(result).encode()

    monkeypatch.setattr(frontend, "child", child)
    out = tmp_path / "out"
    out.mkdir()
    def run(start=None):
        return frontend.controls(control_configuration(), out, before,
                                 time.monotonic() if start is None else start)

    return run


def test_control_matrix_gates_require_all_sources_seeds_and_baseline(monkeypatch, tmp_path):
    run = control_mocks(monkeypatch, tmp_path)
    result = run()
    assert result["status"] == "completed" and result["learning_gate"]
    assert result["passing_arms"] == list(c.ARMS[1:])
    assert len(result["trials"]) == 8
    assert all(t["recipe"]["rows"] == (32 if t["arm"] == "genuine-batch" else 4)
               for t in result["trials"])
    assert result["baseline_stage"] == result["adaptation_stage"] == "not_started"
    assert len(list((tmp_path / "out").glob("trial-*.json"))) == 8


@pytest.mark.parametrize("fault", ["missing_source", "partial", "different_source", "cpu",
                                   "invalid_status", "different_initialization", "bad_probe",
                                   "different_schedule"])
def test_control_matrix_fails_closed_on_unbound_results(monkeypatch, tmp_path, fault):
    run = control_mocks(monkeypatch, tmp_path, fault=fault)
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize("fault", ["no_batch", "timeout", "cancel"])
def test_control_matrix_retains_unavailable_or_cancelled_trials(monkeypatch, tmp_path, fault):
    run = control_mocks(monkeypatch, tmp_path, fault=fault)
    result = run()
    assert result["status"] == ("cancelled" if fault == "cancel" else "unavailable")
    assert not all(t["status"] == "completed" for t in result["trials"]) or fault == "cancel"


def test_control_budget_and_cancellation_markers(monkeypatch, tmp_path):
    import time

    run = control_mocks(monkeypatch, tmp_path)
    result = run(time.monotonic() - 14401)
    assert result["reason"] == "controls_budget_exhausted"
    marker = tmp_path / "out" / "CANCEL"
    marker.touch()
    # Cancel in owned child before capacity probes are allowed to finish.
    monkeypatch.setattr(frontend, "child", lambda *a, **k: (_ for _ in ()).throw(
        InterruptedError()))
    assert run()["status"] == "cancelled"
    marker.unlink()
    marker.symlink_to(tmp_path / "nonexistent")
    monkeypatch.setattr(frontend, "child", lambda *a, **k: k["cancelled"]())
    with pytest.raises(ValueError, match="symlink"):
        run()


@pytest.mark.parametrize("failure", [None, "gpu", "import", "timeout", "cancel", "oom"])
def test_worker_audit_status_and_no_report_secrets(monkeypatch, tmp_path, failure):
    import sys
    from types import SimpleNamespace

    from core.srnet_cuda import CUDAUnavailable

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(
        json.dumps(control_configuration()).encode())))
    errors = {"gpu": CUDAUnavailable("test_cuda_absent"), "import": ImportError(),
              "timeout": TimeoutError(), "cancel": InterruptedError(),
              "oom": torch.cuda.OutOfMemoryError()}

    def audit(*args, **kwargs):
        if failure:
            raise errors[failure]
        return {"status": "completed"}

    monkeypatch.setattr(frontend.jpeg_study_audit, "audit", audit)
    result = frontend.worker(SimpleNamespace(worker="audit"))
    assert result["status"] == ("completed" if failure is None else
                                "cancelled" if failure == "cancel" else "unavailable")
    assert str(tmp_path) not in json.dumps(result)


def test_worker_fit_resume_simulation_and_source_mutation(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: None)
    config = control_configuration()
    config["resume"] = [{"arm": "legacy", "seed": c.SEEDS[0], "path": "private.npz",
                         "sha256": "a" * 64}]
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(
        buffer=io.BytesIO(json.dumps(config).encode())))
    monkeypatch.setattr(frontend.srnet_cuda, "inspect", lambda: {})
    seen = []

    def fit(*args, **kwargs):
        seen.append(kwargs["resume"])
        return {"status": "completed"}

    monkeypatch.setattr(frontend.jpeg_study, "fit", fit)
    args = SimpleNamespace(worker="fit", out=tmp_path / "fit", arm="legacy",
                           seed=c.SEEDS[0], rows=4)
    assert frontend.worker(args)["status"] == "completed"
    assert seen == [{"path": Path("private.npz"), "sha256": "a" * 64}]
    args.out = None
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(
        buffer=io.BytesIO(json.dumps(config).encode())))
    with pytest.raises(ValueError, match="fresh"):
        frontend.worker(args)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"raw")))
    monkeypatch.setattr(frontend, "simulate", lambda raw, sha: ["eA==", "eQ=="])
    assert frontend.worker(SimpleNamespace(worker="simulate", lineage="a" * 64)) == [
        "eA==", "eQ=="]
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(
        buffer=io.BytesIO(json.dumps(config).encode())))
    monkeypatch.setattr(frontend.jpeg_study_audit, "audit", lambda *a, **k: {"status": "completed"})
    values = iter([{"x": "a" * 64}, {"x": "b" * 64}])
    monkeypatch.setattr(frontend, "sources", lambda: next(values))
    with pytest.raises(ValueError, match="source mutation"):
        frontend.worker(SimpleNamespace(worker="audit"))


@pytest.mark.parametrize("failure", [False, True])
def test_preparation_report_requires_independent_audit(tmp_path, monkeypatch, failure):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": [{"path": "reserved", "sha256": "b" * 64}]}
    path = tmp_path / "config.json"
    write(path, config)

    def prepare(inputs, reserved, out, *, generator, deadline):
        deadline()
        generator(b"raw", "a" * 64)
        out.mkdir()
        result = {"samples": {"fit": sample_rows(), "development": sample_rows("development")}}
        write(out / "manifest.json", result)
        return result

    monkeypatch.setattr(frontend.jpeg_study_data, "prepare", prepare)
    monkeypatch.setattr(frontend, "generate", lambda raw, sha, **k: k["deadline"]())
    monkeypatch.setattr(frontend, "child", lambda *a, **k: json.dumps({
        "status": "unavailable" if failure else "completed",
        "source_sha256": frontend.sources()}).encode())
    report = frontend.run(path, tmp_path / "out")
    assert report["status"] == ("unavailable" if failure else "completed")
    assert report["optimizer_updates"] == 0 and not report["real_model_trained"]
    assert str(tmp_path) not in json.dumps(report)


def test_source_bound_config_and_entrypoint_errors(monkeypatch, tmp_path):
    from types import SimpleNamespace

    original_worker = frontend.worker
    with pytest.raises(SystemExit):
        frontend.main([])
    monkeypatch.setattr(frontend, "worker", lambda args: {"status": "completed"})
    assert frontend.main(["--worker", "audit"]) == 0
    monkeypatch.setattr(frontend, "worker", original_worker)
    config = control_configuration()
    path = tmp_path / "config.json"
    write(path, config)
    monkeypatch.setattr(frontend.srnet_cuda, "inspect", lambda: {})
    monkeypatch.setattr(frontend, "controls", lambda *a: {"status": "completed"})
    assert frontend.main(["--config", str(path), "--out", str(tmp_path / "out")]) == 0
    values = iter([{"a": "a" * 64}, {"b": "b" * 64}])
    monkeypatch.setattr(frontend, "sources", lambda: next(values))
    with pytest.raises(ValueError, match="sources changed"):
        frontend.run(path, tmp_path / "mutated")
    monkeypatch.setattr(frontend, "sources", lambda: {})
    import sys

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b" " * 65537)))
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: None)
    with pytest.raises(ValueError, match="bounds"):
        frontend.worker(SimpleNamespace(worker="audit"))


@pytest.mark.parametrize("fault", ["inputs", "fields", "root", "reserved"])
def test_prepare_config_is_bounded_and_exact(fault):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": [{"path": "reserved", "sha256": "b" * 64}]}
    if fault == "inputs":
        config["inputs"] = []
    elif fault == "fields":
        config["inputs"][0]["extra"] = True
    elif fault == "root":
        config["inputs"][0]["root"] = False
    else:
        config["reserved_manifests"][0] = {"sha256": "b" * 64}
    with pytest.raises(ValueError):
        frontend.configuration(config)


def test_source_and_checkpoint_size_bounds(monkeypatch, tmp_path, numeric_snapshot):
    monkeypatch.setattr(frontend, "regular_open", lambda path: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError, match="source closure"):
        frontend.sources()
    monkeypatch.setattr(state, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        state.save(numeric_snapshot, tmp_path / "snapshot.npz")


def test_control_audit_binding_and_budget_cancel_between_trials(monkeypatch, tmp_path):
    run = control_mocks(monkeypatch, tmp_path)
    marker = tmp_path / "out" / "CANCEL"
    real_child = frontend.child
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if "steganography.research_jpeg_study" in command:
            marker.touch()
        return real_child(command, **kwargs)

    monkeypatch.setattr(frontend, "child", child)
    assert run()["status"] == "cancelled"
    marker.unlink()
    monkeypatch.setattr(frontend, "document", lambda path, sha: {})
    with pytest.raises(ValueError):
        run()


def test_training_logit_and_singleton_parity_fail_closed(prepared, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root, _, sha = prepared

    class Invalid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, values):
            return self.weight.expand(len(values), 1)

    out = tmp_path / "bad-logits"
    out.mkdir()
    with pytest.raises(ValueError, match="logits"):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu",
                 factory=Invalid, row_budget=4)
    with (d.Reader(root, sha, "development", deadline=lambda: None) as reader,
          pytest.raises(ValueError, match="parity")):
        c.evaluate(Invalid(), reader, "cpu")
    monkeypatch.setattr(c.time, "monotonic", iter([0, 1801]).__next__)
    with pytest.raises(TimeoutError):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu",
                 factory=tiny, row_budget=4)


def test_resume_orchestration_next_row_and_separate_snapshot(prepared, tmp_path, monkeypatch):
    root, _, sha = prepared
    out = tmp_path / "resume"
    out.mkdir()
    captured = []
    monkeypatch.setattr(c.jpeg_study_state, "load", lambda *a: {})
    monkeypatch.setattr(c.jpeg_study_state, "restore", lambda *a: c.PRESENTED_ROWS - 4)
    monkeypatch.setattr(c.jpeg_study_state, "capture", lambda *a: captured.append(a[-1]))
    monkeypatch.setattr(c.jpeg_study_state, "save", lambda *a: "a" * 64)
    result = c.kernel(root, sha, out, "reference-adamax", c.SEEDS[0], 4, {}, device="cpu",
                      factory=tiny, resume={"path": out / "previous.npz", "sha256": "b" * 64})
    assert captured == [c.PRESENTED_ROWS] and result["optimizer_updates"] == 1
    assert result["total_optimizer_updates"] == c.PRESENTED_ROWS // 4
    assert result["checkpoints"][0]["path"] == "rows-32768.npz"
    monkeypatch.setattr(c.jpeg_study_state, "restore", lambda *a: c.PRESENTED_ROWS)
    with pytest.raises(ValueError, match="remaining"):
        c.kernel(root, sha, out, "legacy", c.SEEDS[0], 4, {}, device="cpu", factory=tiny,
                 resume={"path": out / "previous.npz", "sha256": "b" * 64})


def test_independent_audit_rejects_wrong_cached_pixels(prepared):
    root, manifest, _ = prepared
    path = root / "fit.f32"
    values = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
    values[0] += 1
    path.write_bytes(values.tobytes())
    manifest["tensors"]["fit"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sha = write(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="IDCT"):
        audit.audit(root, sha, deadline=lambda: None)


def test_simulation_rejects_uncertain_payload_and_invalid_shape(original_inputs, monkeypatch):
    conseal = pytest.importorskip("conseal")
    inputs, _ = original_inputs
    root = Path(inputs[0]["root"])
    row = json.loads((root / "source.json").read_text())["samples"][0]
    raw = (root / row["path"]).read_bytes()

    def warning(*args, **kwargs):
        import warnings

        warnings.warn("independent unknown numerical warning", RuntimeWarning, stacklevel=1)
        return (np.zeros((512, 512)), np.zeros((512, 512))), 1

    monkeypatch.setattr(conseal.simulate._ternary, "probability", warning)
    with pytest.raises(ValueError, match="unaudited"):
        frontend.simulate(raw, row["sha256"])
    monkeypatch.setattr(conseal.simulate._ternary, "probability", lambda *a, **k:
                        ((np.zeros((512, 512)), np.zeros((512, 512))), 1))
    monkeypatch.setattr(frontend.jpeg_study_entropy, "refine", lambda *a:
                        (_ for _ in ()).throw(ValueError("entropy residual")))
    with pytest.raises(ValueError, match="entropy residual"):
        frontend.simulate(raw, row["sha256"])
    stream = io.BytesIO()
    Image.new("L", (32, 32)).save(stream, format="PPM")
    raw = stream.getvalue()
    with pytest.raises(ValueError, match="grayscale PGM"):
        frontend.simulate(raw, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(frontend.importlib.metadata, "version", lambda name: "unreviewed")
    with pytest.raises(ValueError, match="pinned"):
        frontend.simulate(raw, hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("status,expected", [("completed", 0), ("unavailable", 2)])
def test_shared_cli_entrypoint(status, expected, monkeypatch):
    import cli

    seen = []

    def run(config, out):
        seen.append((config, out))
        return {"status": status, "action": "controls"}

    monkeypatch.setattr(frontend, "run", run)
    args = cli.build_parser().parse_args(["research", "jpeg-study", "--config", "config.json",
                                         "--out", "out"])
    assert args.fn(args) == expected
    assert seen == [(Path("config.json"), Path("out"))]


def test_independent_gibbs_entropy_refinement_and_iteration_bounds(monkeypatch):
    import math

    costs = (np.ones((4, 4)), np.full((4, 4), 2.0))
    probabilities = entropy.refine(costs, 6)
    p, q = float(probabilities[0][0, 0]), float(probabilities[1][0, 0])
    value = -16 * sum(v * math.log2(v) for v in (p, q, 1 - p - q))
    assert abs(value - 6) <= 6e-6
    # Gibbs ratios for costs 1 and 2 must obey q/p0 == (p/p0)**2.
    assert q / (1 - p - q) == pytest.approx((p / (1 - p - q))**2, abs=1e-12)
    with pytest.raises(ValueError, match="capacity"):
        entropy.refine((np.ones((1, 1)), np.ones((1, 1))), 2)
    with pytest.raises(ValueError, match="bracketed"):
        entropy.refine((np.zeros((4, 4)), np.zeros((4, 4))), 6)
    monkeypatch.setattr(entropy.np, "exp", lambda values: (values >= -3).astype(float))
    with pytest.raises(ValueError, match="converge"):
        entropy.refine((np.ones((4, 4)), np.ones((4, 4))), 6)


@pytest.mark.parametrize("fault", ["container", "target", "count", "shape", "dtype", "nan",
                                   "negative", "bound", "empty"])
def test_entropy_refinement_rejects_malformed_costs(fault):
    costs, target = (np.ones((4, 4)), np.ones((4, 4))), 6
    if fault == "container":
        costs = list(costs)
    elif fault == "target":
        target = True
    elif fault == "count":
        costs = (costs[0],)
    elif fault == "shape":
        costs = (costs[0], np.ones((3, 4)))
    elif fault == "dtype":
        costs = (costs[0], np.ones((4, 4), dtype="f4"))
    elif fault == "nan":
        costs = (costs[0], np.full((4, 4), np.nan))
    elif fault == "negative":
        costs = (costs[0], np.full((4, 4), -1.0))
    elif fault == "bound":
        costs = (costs[0], np.full((4, 4), 1e15))
    else:
        costs = (costs[0], np.empty((0, 0)))
    with pytest.raises(ValueError):
        entropy.refine(costs, target)


def test_parent_interrupt_retains_cancelled_control_report(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    write(path, control_configuration())
    monkeypatch.setattr(frontend.srnet_cuda, "inspect", lambda: {})
    monkeypatch.setattr(frontend, "controls", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    result = frontend.run(path, tmp_path / "out")
    assert result["status"] == "cancelled" and result["trial_reports"] == []


def test_native_audit_reads_private_copies_not_mutable_corpus_paths(prepared, monkeypatch):
    jpeglib = pytest.importorskip("jpeglib")
    root, _, sha = prepared
    original = jpeglib.read_dct
    reads = []

    def read(path, *args, **kwargs):
        assert not Path(path).is_relative_to(root)
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(jpeglib, "read_dct", read)
    assert audit.audit(root, sha, deadline=lambda: None)["status"] == "completed"
    assert len(reads) == 16 and all(not Path(path).exists() for path in reads)


def test_invalid_worker_does_not_import_optional_torch(monkeypatch):
    import builtins
    import sys
    from types import SimpleNamespace

    original = builtins.__import__
    calls = []

    def importing(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            calls.append(name)
            raise ImportError("independent absent optional backend fixture")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b" " * 65537)))
    with pytest.raises(ValueError, match="bounds"):
        frontend.worker(SimpleNamespace(worker="audit"))
    assert calls == []


def corner_generator(raw, lineage):
    jpeglib = pytest.importorskip("jpeglib")
    import tempfile

    assert hashlib.sha256(raw).hexdigest() == lineage
    with tempfile.TemporaryDirectory() as folder:
        cover, stego = Path(folder) / "cover.jpg", Path(folder) / "stego.jpg"
        with Image.open(io.BytesIO(raw)) as image:
            image.save(cover, quality=75, optimize=False)
        jpeglib.version.set("6b")
        jpeg = jpeglib.read_dct(str(cover))
        jpeg.load()
        coefficients = jpeg.Y.copy()
        coefficients[2, 2, 1, 1] += 1
        jpeg.Y = coefficients
        jpeg.write_dct(str(stego))
        return [cover.read_bytes(), stego.read_bytes()]


@pytest.fixture
def five_prepared(tmp_path, original_inputs):
    inputs, reserved = original_inputs
    out = tmp_path / "five"
    manifest = d.prepare(inputs, reserved, out, generator=corner_generator,
                         deadline=lambda: None, fit=2, development=2, representation=views.FIVE)
    sha = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    return out, manifest, sha


def test_five_view_keeps_signal_free_centers_and_all_originals(five_prepared, original_inputs):
    pytest.importorskip("scipy")
    root, manifest, sha = five_prepared
    assert manifest["schema_version"] == views.SCHEMA
    assert manifest["representation"] == views.contract()
    assert manifest["tensors"]["fit"]["bytes"] == 8 * views.ROW_BYTES
    inputs, reserved = original_inputs
    forbidden = d.reservations(reserved)
    for entry in inputs:
        selected = d.originals(Path(entry["root"]), entry["sha256"],
                               entry["source_group"], forbidden)
        for role, expected in (("fit", selected[:2]), ("development", selected[2:4])):
            actual = [row["lineage"] for row in manifest["samples"][role][::2]
                      if row["source_group"] == entry["source_group"]]
            assert actual == [row["sha256"] for row in expected]
    with d.Reader(root, sha, "fit", deadline=lambda: None) as reader:
        indices = np.array([0, 1], dtype=np.int64)
        centers = reader.batch(indices, np.array([0, 0]))
        top_left = reader.batch(indices, np.array([1, 1]))
        assert np.array_equal(centers[0], centers[1])
        assert not np.array_equal(top_left[0], top_left[1])
        for invalid in (None, np.array([0]), np.array([0., 1.]), np.array([-1, 1]),
                        np.array([5, 0]), np.array([[0, 1]])):
            with pytest.raises(ValueError, match="five-view"):
                reader.batch(indices, invalid)
    proof = audit.audit(root, sha, deadline=lambda: None)
    assert proof["schema_version"] == "jpeg-study-independent-audit-v2"
    assert proof["idct_rows"] == {"fit": 8, "development": 8}
    assert proof["idct_views"] == {"fit": 40, "development": 40}
    assert proof["identical_coefficient_view_pairs"] == {
        "fit": [4, 0, 4, 4, 4], "development": [4, 0, 4, 4, 4]}
    assert proof["maximum_absolute_pixel_difference"] <= 1e-4
    assert proof["optimizer_updates"] == 0 and not proof["deployed"]


def test_five_view_contract_and_cache_mutations(five_prepared, prepared):
    root, manifest, sha = five_prepared
    for mutate in (
        lambda m: m.update(schema_version=d.SCHEMA),
        lambda m: m["representation"].update(inference="max-positive-crop"),
        lambda m: m["representation"]["regions"].reverse(),
        lambda m: m.pop("representation"),
    ):
        altered = copy.deepcopy(manifest)
        mutate(altered)
        with pytest.raises(ValueError, match="contract"):
            d.validate_manifest(altered)
    old_root, old_manifest, old_sha = prepared
    with (d.Reader(old_root, old_sha, "fit", deadline=lambda: None) as reader,
          pytest.raises(ValueError, match="center")):
        reader.batch(np.array([0]), np.array([0]))
    old_manifest["representation"] = views.contract()
    with pytest.raises(ValueError, match="contract"):
        d.validate_manifest(old_manifest)
    path = root / "fit.f32"
    with path.open("r+b") as stream:
        # Corrupt the LAST corner, not just the historically tested center.
        stream.seek(4 * d.ROW_BYTES)
        value = np.frombuffer(stream.read(4), dtype="<f4")[0] + 1
        stream.seek(4 * d.ROW_BYTES)
        stream.write(np.array([value], dtype="<f4").tobytes())
    manifest["tensors"]["fit"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sha = write(root / "manifest.json", manifest)
    pytest.importorskip("scipy")
    with pytest.raises(ValueError, match="IDCT"):
        audit.audit(root, sha, deadline=lambda: None)


def test_five_view_identical_full_pair_is_not_replaced(original_inputs, tmp_path):
    inputs, reserved = original_inputs
    calls = []

    def identical(raw, lineage):
        calls.append(lineage)
        pair = corner_generator(raw, lineage)
        return [pair[0], pair[0]]

    with pytest.raises(ValueError, match="identical"):
        d.prepare(inputs, reserved, tmp_path / "bad-five", generator=identical,
                  deadline=lambda: None, fit=2, development=2, representation=views.FIVE)
    assert len(calls) == 1
    assert not (tmp_path / "bad-five" / "manifest.json").exists()
    with pytest.raises(ValueError, match="limits"):
        d.prepare(inputs, reserved, tmp_path / "bad-representation", generator=identical,
                  deadline=lambda: None, representation="best-crop")


def test_five_view_independent_idct_and_legacy_center_parity():
    scipy = pytest.importorskip("scipy.fft")
    coefficients = np.random.default_rng(42).integers(-20, 21, (64, 64, 8, 8), dtype=np.int16)
    qt = np.arange(1, 65, dtype=np.uint16).reshape(8, 8)
    decoded = scipy.idctn(coefficients.astype(np.float64) * qt, axes=(-2, -1), norm="ortho") + 128
    full = decoded.transpose(0, 2, 1, 3).reshape(512, 512).astype("<f4")
    actual = views.reconstruct(coefficients, qt)
    for tensor, (left, top, width, height) in zip(actual, views.REGIONS, strict=True):
        np.testing.assert_allclose(tensor[0], full[top:top + height, left:left + width], atol=1e-4)
    np.testing.assert_array_equal(actual[0, 0], d.jpeg_float256.reconstruct(
        coefficients, qt, 512, 512))
    assert actual.shape == (5, 1, 256, 256) and actual.dtype == np.dtype("<f4")
    assert min(qt.ravel()) == 1 and np.any(actual < 0) and np.any(actual > 255)


def test_five_view_schedule_is_matched_cyclic_and_label_independent():
    samples = sample_rows()
    hashes = []
    for rows in (4, 8, 16, 32):
        indices, angles = c.schedule(samples, c.SEEDS[0], rows, 128)
        selected = c.view_schedule(samples, c.SEEDS[0], indices)
        assert not selected.flags.writeable
        assert np.array_equal(selected.ravel()[::2], selected.ravel()[1::2])
        hashes.append(hashlib.sha256(indices.tobytes() + angles.tobytes() +
                                     selected.tobytes()).hexdigest())
        for index in range(0, len(samples), 2):
            visits = selected.ravel()[indices.ravel() == index]
            assert set(visits[:5]) == set(range(5))
            assert np.all((visits[1:] - visits[:-1]) % 5 == 1)
        changed = copy.deepcopy(samples)
        for row in changed:
            row["label"] = "label-not-available"
            row["prediction"] = 1.0
        np.testing.assert_array_equal(selected, c.view_schedule(changed, c.SEEDS[0], indices))
    assert len(set(hashes)) == 1
    assert c.recipe("legacy", c.SEEDS[0], 4) != c.recipe(
        "legacy", c.SEEDS[0], 4, representation=views.FIVE)
    with pytest.raises(ValueError):
        c.recipe("legacy", c.SEEDS[0], 4, representation="adaptive")
    for invalid in (None, np.array([0, 1]), np.zeros((1, 3), dtype=int),
                    np.full((1, 4), 999), np.array([[0, 0, 2, 3]])):
        with pytest.raises(ValueError):
            c.view_schedule(samples, c.SEEDS[0], invalid)
    altered = copy.deepcopy(samples)
    altered[1]["lineage"] = "different"
    with pytest.raises(ValueError, match="ancestry"):
        c.view_schedule(altered, c.SEEDS[0], np.array([[0, 1, 2, 3]]))


def test_five_view_evaluation_is_mean_probability_not_max_or_logit():
    torch = pytest.importorskip("torch")

    class Reader:
        view_count = 5
        samples = sample_rows("development")
        calls: list = []

        def batch(self, indices, selected):
            self.calls.append((indices.copy(), selected.copy()))
            return np.broadcast_to(selected[:, None, None, None],
                                   (len(indices), 1, 256, 256)).astype("<f4")

        def verify(self):
            self.verified = True

    class Model(torch.nn.Module):
        def forward(self, values):
            score = values.mean(dim=(1, 2, 3))
            return torch.stack([torch.zeros_like(score), score], dim=1)

    reader = Reader()
    model = Model()
    result = c.evaluate(model, reader, "cpu")
    expected = np.mean(1 / (1 + np.exp(-np.arange(5))))
    np.testing.assert_allclose(result, [expected] * 8, atol=1e-7)
    assert not np.isclose(expected, 1 / (1 + np.exp(-2)))
    assert not np.isclose(expected, 1 / (1 + np.exp(-4)))
    assert reader.verified and not model.training
    assert [int(v[0]) for _, v in reader.calls] == list(range(5)) * 2


def test_five_view_kernel_matches_arms_without_more_updates(five_prepared, tmp_path):
    pytest.importorskip("torch")
    root, _, sha = five_prepared
    results = []
    for arm, rows in (("legacy", 4), ("reference-adamax", 4), ("genuine-batch", 8)):
        out = tmp_path / arm
        out.mkdir()
        results.append(c.kernel(root, sha, out, arm, c.SEEDS[0], rows, {}, device="cpu",
                                factory=tiny, row_budget=16))
    assert len({r["initial_learned_sha256"] for r in results}) == 1
    assert len({r["schedule_sha256"] for r in results}) == 1
    assert all(r["recipe"]["representation"] == views.contract() for r in results)
    assert all(r["presented_rows"] == 16 and not r["real_gpu_fit"] for r in results)
    assert [r["optimizer_updates"] for r in results] == [4, 4, 2]


@pytest.mark.parametrize("value", [None, "max-crop", [], {}, 5])
def test_five_view_prepare_config_rejects_implicit_or_adaptive_view(value):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": [], "representation": value}
    with pytest.raises(ValueError, match="representation"):
        frontend.configuration(config)


def test_five_view_frontend_forwards_representation_and_preserves_failure(tmp_path, monkeypatch):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": [], "representation": views.FIVE}
    assert frontend.configuration(config) == config
    path = tmp_path / "config.json"
    write(path, config)

    def fail(*args, **kwargs):
        assert kwargs["representation"] == views.FIVE
        raise ValueError("invalid pair")

    monkeypatch.setattr(d, "prepare", fail)
    report = frontend.run(path, tmp_path / "out")
    assert report["status"] == "unavailable" and report["optimizer_updates"] == 0
    assert report["representation"] == views.contract()
    assert str(tmp_path) not in json.dumps(report)


def test_five_view_decoder_rejects_wrong_format_bounds_and_nonfinite(monkeypatch):
    for raw in (b"", b"x" * (views.MAX_JPEG + 1), "not-bytes"):
        with pytest.raises(ValueError, match="bounds"):
            views.validate(raw)
    for mode, size in (("RGB", (512, 512)), ("L", (256, 256))):
        stream = io.BytesIO()
        Image.new(mode, size).save(stream, format="JPEG")
        with pytest.raises(ValueError, match="gray"):
            views.validate(stream.getvalue())
    with pytest.raises(ValueError):
        views.reconstruct(np.zeros((32, 32, 8, 8), dtype=np.int16), np.ones((8, 8), dtype=int))
    monkeypatch.setattr(views.jpeg_float256, "reconstruct", lambda *a: None)
    monkeypatch.setattr(views.np, "einsum", lambda *a, **k: np.full((64, 64, 8, 8), np.nan))
    with pytest.raises(ValueError):
        views.reconstruct(np.zeros((64, 64, 8, 8), dtype=np.int16), np.ones((8, 8), dtype=int))


@pytest.mark.parametrize("fault", ["timeout", "nonzero", "truncated", "oversize", "nan", "ok"])
def test_five_view_owned_worker_bounds_and_reaping(tmp_path, monkeypatch, fault):
    import subprocess

    killed, waited = [], []
    monkeypatch.setattr(views, "validate", lambda raw: None)
    monkeypatch.setattr(views.jpeg_float256, "decoder_contract", lambda: {})
    monkeypatch.setattr(views.os, "killpg", lambda *args: killed.append(args))

    class Process:
        pid = 123456789
        returncode = 1 if fault == "nonzero" else 0

        def __init__(self, command, **kwargs):
            assert command[-1] == "core.jpeg_study_views" and kwargs["start_new_session"]
            self.output = kwargs["stdout"]

        def communicate(self, payload, timeout):
            assert timeout == 20 and payload == b"\x01\x00\x00\x00\x01\x00\x00\x00x"
            if fault == "timeout":
                raise subprocess.TimeoutExpired("fixed worker", 20)
            raw = np.zeros((5, 1, 256, 256), dtype="<f4").tobytes()
            if fault == "truncated":
                raw = raw[:-1]
            elif fault == "oversize":
                raw += b"x"
            elif fault == "nan":
                raw = np.array([np.nan], dtype="<f4").tobytes() + raw[4:]
            self.output.write(raw)

        def wait(self):
            waited.append(True)

    monkeypatch.setattr(views.subprocess, "Popen", Process)
    if fault == "ok":
        assert views.pixel_batch([b"x"]).shape == (1, 5, 1, 256, 256)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            views.pixel_batch([b"x"])
    assert waited and killed[0][0] == Process.pid
    for files in (None, [], [b"x"] * 3):
        with pytest.raises(ValueError):
            views.pixel_batch(files)


@pytest.mark.parametrize("raw,code", [
    (b"", 2), ((3).to_bytes(4, "little"), 2),
    ((1).to_bytes(4, "little") + b"\x00", 2),
    ((1).to_bytes(4, "little") + (views.MAX_JPEG + 1).to_bytes(4, "little"), 2),
    ((1).to_bytes(4, "little") + (2).to_bytes(4, "little") + b"x", 2),
    ((1).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"xz", 2),
    ((1).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"x", 0),
])
def test_five_view_worker_frame_and_resource_limits(monkeypatch, raw, code):
    import sys
    from types import SimpleNamespace

    limits = []
    monkeypatch.setattr(views.resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(views, "decode", lambda raw: b"tensor")
    assert views.main() == code
    assert len(limits) == 4
    assert limits[2] == (views.resource.RLIMIT_FSIZE, (2 * views.ROW_BYTES,) * 2)


def test_five_view_direct_decoder_private_copy_and_structure(original_inputs, monkeypatch):
    jpeglib = pytest.importorskip("jpeglib")
    from types import SimpleNamespace

    inputs, _ = original_inputs
    root = Path(inputs[0]["root"])
    row = json.loads((root / "source.json").read_bytes())["samples"][0]
    pair = corner_generator((root / row["path"]).read_bytes(), row["sha256"])
    calls = []
    original = jpeglib.read_dct

    def read(path):
        assert Path(path).read_bytes() == pair[0]
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(jpeglib, "read_dct", read)
    tensor = np.frombuffer(views.decode(pair[0]), dtype="<f4").reshape(5, 1, 256, 256)
    np.testing.assert_array_equal(tensor, views.pixel_batch([pair[0]])[0])
    assert calls and all(not path.exists() for path in calls)
    monkeypatch.setattr(jpeglib, "read_dct", lambda path: SimpleNamespace(
        load=lambda: None, jpeg_color_space=SimpleNamespace(name="JCS_YCbCr")))
    with pytest.raises(ValueError, match="structure"):
        views.decode(pair[0])


@pytest.mark.parametrize("fault", [None, "missing_view_audit", "wrong_aggregation",
                                   "center_recipe", "different_schedule"])
def test_five_view_control_gate_requires_whole_representation(monkeypatch, tmp_path, fault):
    run = control_mocks(monkeypatch, tmp_path, fault=fault, representation=views.FIVE)
    if fault:
        with pytest.raises(ValueError):
            run()
    else:
        result = run()
        assert result["status"] == "completed" and result["learning_gate"]
        assert result["representation"] == views.contract()
        assert len(result["trials"]) == 8


@pytest.mark.parametrize("error", [TimeoutError, InterruptedError, KeyboardInterrupt, ValueError])
def test_preparation_retains_audit_failure_without_training(tmp_path, monkeypatch, error):
    config = {"schema_version": "jpeg-study-config-v1", "action": "prepare", "inputs": [
        {"root": "private", "sha256": "a" * 64, "source_group": source}
        for source in d.ORIGINS], "reserved_manifests": [], "representation": views.FIVE}
    path = tmp_path / "config.json"
    write(path, config)

    def prepare(*args, **kwargs):
        out = args[2]
        out.mkdir()
        result = {"samples": {"fit": sample_rows(), "development": sample_rows("development")}}
        write(out / "manifest.json", result)
        return result

    monkeypatch.setattr(d, "prepare", prepare)
    monkeypatch.setattr(frontend, "child", lambda *a, **k: (_ for _ in ()).throw(error("secret")))
    report = frontend.run(path, tmp_path / "out")
    assert report["status"] == ("cancelled" if error in (InterruptedError, KeyboardInterrupt)
                                else "unavailable")
    assert report["optimizer_updates"] == 0 and not report["real_model_trained"]
    assert "secret" not in json.dumps(report) and "manifest_sha256" in report
