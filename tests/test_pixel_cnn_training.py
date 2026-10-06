"""Bounded pixel learning/serialization contracts; generated fixtures are not accuracy."""

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

import cli
from core import jpeg_cnn as cnn
from core import jpeg_cnn_model as cm
from steganography import research_pixel_cnn as pc
from steganography import research_pixels as rp
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_pixel_research import pixel_corpus as pixel_corpus

torch = pytest.importorskip("torch")


@pytest.fixture
def configs(pixel_corpus, tmp_path):
    manifest, source = pixel_corpus
    for split in ("train", "validation"):
        rp.extract_pixels(manifest, tmp_path / split, source=source, split=split)
    base = {
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "epochs": 1,
        "batch_size": 2,
        "threads": 1,
    }
    return (
        {
            **base,
            "stage": "train",
            "cache": str(tmp_path / "train/cache.json"),
            "cache_sha256": sha(tmp_path / "train/cache.json"),
        },
        {
            **base,
            "stage": "predict",
            "cache": str(tmp_path / "validation/cache.json"),
            "cache_sha256": sha(tmp_path / "validation/cache.json"),
        },
    )


def test_numeric_roundtrip_and_cpu_dtype(tmp_path):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = cnn.network().eval()
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
        assert model.filters.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous)
    path = tmp_path / "model.npz"
    checksum = cm.save_model(model, path)
    loaded = cm.load_model(path, checksum=checksum)
    raw = np.zeros((2, 1, 128, 128), dtype="u1")
    np.testing.assert_array_equal(cnn.pixel_logits(model, raw), cnn.pixel_logits(loaded, raw))
    with pytest.raises(FileExistsError):
        cm.save_model(model, path)
    with pytest.raises(ValueError, match="checksum"):
        cm.load_model(path, checksum="0" * 64)
    symlink = tmp_path / "link.npz"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        cm.load_model(symlink, checksum=checksum)
    with pytest.raises(FileExistsError):
        cm.save_model(model, symlink)


@pytest.mark.parametrize("fault", ["nan", "filters", "shape", "missing", "dtype"])
def test_numeric_parameters_rejected(fault):
    arrays = {k: v.numpy() for k, v in cnn.network().state_dict().items()}
    if fault == "nan":
        arrays["layers.10.bias"][0] = np.nan
    if fault == "filters":
        arrays["filters"][0, 0, 0, 0] = 5
    if fault == "shape":
        arrays["layers.10.bias"] = np.zeros(2, dtype="f4")
    if fault == "missing":
        arrays.pop("filters")
    if fault == "dtype":
        arrays["filters"] = arrays["filters"].astype("f8")
    with pytest.raises(ValueError):
        cm.validate(arrays)


@pytest.mark.parametrize(
    "fault", ["giant", "object", "version", "truncated", "unknown", "compressed-size"]
)
def test_headers_rejected_before_numpy_load(tmp_path, monkeypatch, fault):
    model = cnn.network().eval()
    original = tmp_path / "original.npz"
    cm.save_model(model, original)
    target = tmp_path / "bad.npz"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(target, "w") as output:
        for member in source.infolist():
            raw = source.read(member)
            if member.filename == "filters.npy":
                if fault in {"giant", "object"}:
                    stream = io.BytesIO()
                    np.lib.format.write_array_header_1_0(
                        stream,
                        {
                            "descr": "|O" if fault == "object" else "<f4",
                            "fortran_order": False,
                            "shape": (10**12,) if fault == "giant" else (3, 1, 3, 3),
                        },
                    )
                    raw = stream.getvalue() + b"x"
                if fault == "version":
                    raw = b"\x93NUMPY\x03\x00" + raw[8:]
                if fault == "truncated":
                    raw = raw[:-1]
                if fault == "compressed-size":
                    raw = bytes(cm.MAX_BYTES + 1)
                if fault == "unknown":
                    member.filename = "unknown.npy"
            output.writestr(member, raw, compress_type=zipfile.ZIP_DEFLATED)
    monkeypatch.setattr(cm.np, "load", lambda *a, **k: pytest.fail("unsafe header reached numpy"))
    with pytest.raises(ValueError):
        cm.load_model(target, checksum=sha(target))


def test_model_byte_limits_and_header_v2(tmp_path, monkeypatch):
    model = cnn.network().eval()
    path = tmp_path / "model.npz"
    cm.save_model(model, path)
    alternative = tmp_path / "v2.npz"
    with zipfile.ZipFile(alternative, "w") as output:
        for key, shape in cm.SHAPES.items():
            values = model.state_dict()[key].numpy()
            stream = io.BytesIO()
            np.lib.format.write_array_header_2_0(
                stream, {"descr": "<f4", "fortran_order": False, "shape": shape}
            )
            output.writestr(key + ".npy", stream.getvalue() + values.tobytes())
    cm.load_model(alternative, checksum=sha(alternative))
    monkeypatch.setattr(cm, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="bounded"):
        cm.load_model(path, checksum=sha(path))
    with pytest.raises(ValueError, match="byte limit"):
        cm.save_model(model, tmp_path / "too-big.npz")


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 0),
        ("epochs", True),
        ("batch_size", 65),
        ("threads", 3),
        ("seed", -1),
        ("max_seconds", 1801),
        ("learning_rate", float("nan")),
        ("weight_decay", -1),
    ],
)
def test_settings_bounds(key, value):
    with pytest.raises(ValueError):
        pc.settings({key: value})
    with pytest.raises(ValueError):
        pc.settings(None)


def test_train_predict_cli_repeatability_and_scope(configs, tmp_path, capsys):
    train, predict = configs
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    progress = []
    card = pc.train_pixels(train, tmp_path / "first", progress=progress.append)
    assert torch.equal(torch.get_rng_state(), rng) and torch.get_num_threads() == threads
    assert len(progress) == 1 and card["training_scope"]["rows"] == 6
    repeated = pc.train_pixels(train, tmp_path / "second")
    assert card["model_sha256"] == repeated["model_sha256"]
    assert card["epoch_losses"] == repeated["epoch_losses"]
    for source in card["training_scope"]["source_ids"]:
        chosen = pc.train_pixels({**train, "training_source_id": source}, tmp_path / source)
        assert (
            chosen["training_scope"]["rows"] == 3
            and len(chosen["training_scope"]["excluded_source_ids"]) == 1
        )
        assert chosen["weighting"]["minimum"] == 0.75 and chosen["weighting"]["maximum"] == 1.5
        pc.predict_pixels(
            {
                **predict,
                "model_dir": str(tmp_path / source),
                "card_sha256": sha(tmp_path / source / "model-card.json"),
            },
            tmp_path / f"{source}-predictions.json",
        )
    config_path = tmp_path / "predict.json"
    config_path.write_text(
        json.dumps(
            {
                **predict,
                "model_dir": str(tmp_path / "first"),
                "card_sha256": sha(tmp_path / "first/model-card.json"),
            }
        )
    )
    assert (
        cli.main(
            [
                "research",
                "pixel-cnn",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "predictions.json"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["predictions"]) == 6 and not result["deployed"]
    assert all(0 <= p["score"] <= 1 for p in result["predictions"])
    with pytest.raises(FileExistsError):
        pc.run_pixel_cnn(config_path, tmp_path / "predictions.json")
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train))
    pc.run_pixel_cnn(train_path, tmp_path / "cli-training")
    with pytest.raises(FileExistsError):
        pc.train_pixels(train, tmp_path / "first")
    train_path.write_text(json.dumps({"stage": "unknown"}))
    with pytest.raises(ValueError, match="stage"):
        pc.run_pixel_cnn(train_path, tmp_path / "unknown")


@pytest.mark.parametrize(
    "fault", ["manifest", "source", "nan-loss", "nan-gradient", "before-deadline", "after-deadline"]
)
def test_failed_training_never_gets_card(configs, tmp_path, monkeypatch, fault):
    train, _ = configs
    if fault == "manifest":
        train["manifest_sha256"] = "0" * 64
    if fault == "source":
        train["training_source_id"] = "unknown"
    if fault in {"nan-loss", "nan-gradient"}:
        factory = cnn.network

        def bad_network():
            model = factory()
            if fault == "nan-loss":
                with torch.no_grad():
                    model.layers[-1].bias.fill_(float("nan"))
            else:
                model.layers[-1].bias.register_hook(lambda grad: grad * float("nan"))
            return model

        monkeypatch.setattr(cnn, "network", bad_network)
    if fault in {"before-deadline", "after-deadline"}:
        times = iter([0, 1801] if fault == "before-deadline" else [0, 0, 0, 0, 1801])
        monkeypatch.setattr(pc, "time", SimpleNamespace(monotonic=lambda: next(times)))
    threads = torch.get_num_threads()
    with pytest.raises(ValueError):
        pc.train_pixels(train, tmp_path / "failed")
    assert not (tmp_path / "failed/model-card.json").exists() and torch.get_num_threads() == threads


@pytest.mark.parametrize(
    "fault",
    [
        "checksum",
        "decoder",
        "scope-type",
        "scope-recipe",
        "scope-rows",
        "single-bad",
        "settings",
        "missing-settings",
        "settings-extra",
        "architecture",
    ],
)
def test_prediction_provenance_guards(configs, tmp_path, monkeypatch, fault):
    train, predict = configs
    model_dir = tmp_path / "model"
    card = pc.train_pixels(train, model_dir)
    path = model_dir / "model-card.json"
    if fault == "decoder":
        card["decoder"]["pillow"] = "changed"
    if fault == "scope-type":
        card["training_scope"] = None
    if fault == "scope-recipe":
        card["training_scope"]["recipe"] = "unknown"
    if fault == "scope-rows":
        card["training_scope"]["rows"] += 1
    if fault == "single-bad":
        card["training_scope"] = {"recipe": "single-declared-source-v1", "source_ids": ["unknown"]}
    if fault == "settings":
        card["settings"]["threads"] = 10000
    if fault == "missing-settings":
        card.pop("settings")
    if fault == "settings-extra":
        card["settings"]["unexpected"] = True
    if fault == "architecture":
        card["architecture"] = "unknown"
    path.write_text(json.dumps(card))
    monkeypatch.setattr(
        pc, "load_model", lambda *a, **k: pytest.fail("bad provenance reached model")
    )
    with pytest.raises(ValueError):
        pc.predict_pixels(
            {
                **predict,
                "model_dir": str(model_dir),
                "card_sha256": "0" * 64 if fault == "checksum" else sha(path),
            },
            tmp_path / "invalid.json",
        )
