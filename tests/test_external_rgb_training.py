"""External-cover RGB training: generated CPU QA only."""

import json

import numpy as np
import pytest
from PIL import Image

from research import external_rgb_training as ext
from research import spatial_matching_probe as matching
from research import spatial_parity_probe as probe
from tests.test_spatial_parity_probe import toy

torch = pytest.importorskip("torch")


def photos(folder, count=3, size=(300, 280)):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        y, x = np.mgrid[0 : size[1], 0 : size[0]]
        base = 128 + 60 * np.sin(x / (9.0 + i)) * np.cos(y / 13.0)
        pixels = np.clip(base[:, :, None] + rng.normal(0, 3, (size[1], size[0], 3)), 0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(folder / f"{i:04d}.png")
    return folder


def banks(tmp_path, crops=2):
    train = ext.build_bank(ext.sources(photos(tmp_path / "train")), crops, 1, tmp_path / "bank")
    valid = ext.build_bank(ext.sources(photos(tmp_path / "valid", 2)), 1, 2, tmp_path / "vbank")
    return tmp_path / "bank", tmp_path / "vbank", train, valid


def test_curriculum_adds_lower_rates():
    assert ext.rates(0) == (0.4,) and ext.rates(3) == (0.2, 0.4)
    assert ext.rates(7) == (0.1, 0.2, 0.4) and ext.rates(50) == (0.05, 0.1, 0.2, 0.4)
    for bad in (-1, 1.5):
        with pytest.raises(ValueError):
            ext.rates(bad)


def test_bank_crops_are_exact_grouped_and_exclusive(tmp_path):
    bank, _, manifest, _ = banks(tmp_path)
    crops = ext.load_bank(bank)
    groups = np.load(bank / "groups.npy")
    assert crops.shape == (6, 256, 256, 3) and list(groups) == [0, 0, 1, 1, 2, 2]
    source = np.asarray(Image.open(tmp_path / "train" / "0001.png"))
    windows = np.lib.stride_tricks.sliding_window_view(source, (256, 256, 3))[:, :, 0]
    assert any(np.array_equal(crops[2], w) for row in windows for w in row)
    assert manifest["sources"] == ["0000.png", "0001.png", "0002.png"]
    with pytest.raises(ValueError, match="exists"):
        ext.build_bank(ext.sources(tmp_path / "train"), 2, 1, bank)


def test_bank_skips_non_rgb_and_small_images_and_rejects_empty(tmp_path):
    folder = photos(tmp_path / "mixed", 1)
    Image.new("L", (300, 300)).save(folder / "gray.png")
    Image.new("RGB", (100, 300)).save(folder / "small.png")
    (folder / "notes.txt").write_text("x")
    manifest = ext.build_bank(ext.sources(folder), 1, 3, tmp_path / "b")
    assert manifest["sources"] == ["0000.png"] and sorted(manifest["skipped"]) == [
        "gray.png", "small.png"]
    with pytest.raises(ValueError, match="no usable"):
        ext.build_bank([folder / "gray.png"], 1, 3, tmp_path / "c")
    with pytest.raises(ValueError, match="crops per image"):
        ext.build_bank([folder / "0000.png"], 0, 3, tmp_path / "d")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no PNG"):
        ext.sources(tmp_path / "empty")


def test_load_bank_rejects_wrong_shapes(tmp_path):
    (tmp_path / "x").mkdir()
    np.save(tmp_path / "x" / "crops.npy", np.zeros((1, 8, 8, 3), np.uint8))
    with pytest.raises(ValueError):
        ext.load_bank(tmp_path / "x")


def test_pairs_are_deterministic_fresh_per_epoch_and_bounded(tmp_path):
    bank, *_ = banks(tmp_path)
    pairs = ext.Pairs(ext.load_bank(bank))
    first, again = pairs[1], pairs[1]
    assert first.shape == (2, 256, 256, 3) and np.array_equal(first, again)
    assert np.abs(first[1].astype(int) - first[0]).max() <= 1
    assert any(np.array_equal(first[0], matching.d4(np.asarray(pairs.crops[1]), k))
               for k in range(8))
    pairs.epoch = 9
    assert not np.array_equal(first, pairs[1]) and len(pairs) == 6


def test_device_parity_matches_reference_parity():
    images = np.random.default_rng(1).integers(0, 256, (2, 4, 4, 3), dtype=np.uint8)
    got = ext.to_input(torch.from_numpy(images), "cpu").numpy()
    expected = probe.with_parity(images.transpose(0, 3, 1, 2).astype("<f4"))
    assert np.array_equal(got, expected)


def test_validation_stegos_are_fixed_per_condition(tmp_path):
    _, vbank, *_ = banks(tmp_path)
    covers, stegos = ext.validation(ext.load_bank(vbank))
    again = ext.validation(ext.load_bank(vbank))[1]
    assert set(stegos) == {f"{k}-{r}" for k, r in ext.CONDITIONS} and len(stegos) == 8
    assert all(np.array_equal(stegos[k], again[k]) for k in stegos)
    assert stegos["matching-0.05"].shape == covers.shape


def test_toy_training_evaluates_saves_and_resumes(tmp_path):
    bank, vbank, *_ = banks(tmp_path)
    out, logs = tmp_path / "run", []
    first = ext.train(bank=bank, valid=vbank, out=out, device="cpu", epochs=1, workers=0,
                      pairs_per_batch=2, build=toy, log=logs.append)
    assert first["status"] == "completed" and first["epochs"] == 1 and len(logs) == 1
    metrics = first["history"][0]["validation"]
    assert set(metrics) == {f"{k}-{r}" for k, r in ext.CONDITIONS}
    assert all(0 <= m["auc"] <= 1 and 0 <= m["accuracy"] <= 1 for m in metrics.values())
    assert (out / "checkpoint.pt").exists() and (out / "model.pt").exists()
    second = ext.train(bank=bank, valid=vbank, out=out, device="cpu", epochs=3, decay_epoch=2,
                       workers=0, pairs_per_batch=2, build=toy, log=logs.append)
    assert second["epochs"] == 3 and [h["epoch"] for h in second["history"]] == [1, 2, 3]
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-4)


def test_warm_start_deadline_and_nonfinite(tmp_path):
    bank, vbank, *_ = banks(tmp_path)
    init = tmp_path / "init.pt"
    model = toy()
    torch.save(model.state_dict(), init)
    stopped = ext.train(bank=bank, valid=vbank, out=tmp_path / "a", device="cpu", epochs=2,
                        workers=0, build=toy, init_state=init, max_seconds=1e-9, log=print)
    assert stopped["status"] == "deadline" and stopped["epochs"] == 0

    def broken():
        net = toy()
        with torch.no_grad():
            net[2].weight.fill_(float("nan"))
        return net

    with pytest.raises(ValueError, match="nonfinite"):
        ext.train(bank=bank, valid=vbank, out=tmp_path / "b", device="cpu", epochs=1,
                  workers=0, pairs_per_batch=2, build=broken, log=print)
    with pytest.raises(ValueError, match="smaller than one batch"):
        ext.train(bank=bank, valid=vbank, out=tmp_path / "c", device="cpu", epochs=1,
                  workers=0, pairs_per_batch=7, build=toy, log=print)


@pytest.mark.parametrize(
    "extra",
    [{"epochs": 0}, {"epochs": 201}, {"decay_epoch": 1}, {"pairs_per_batch": 0},
     {"learning_rate": 0.5}, {"max_seconds": 0}],
)
def test_training_contract_bounds(tmp_path, extra):
    arguments = {"bank": tmp_path, "valid": tmp_path, "out": tmp_path / "o", "device": "cpu",
                 "epochs": 1, "build": toy}
    arguments.update(extra)
    with pytest.raises(ValueError):
        ext.train(**arguments)


def test_cli_banks_and_trains_real_srnet(tmp_path, capsys):
    photos(tmp_path / "t", 1)
    photos(tmp_path / "v", 1)
    assert ext.main(["bank", str(tmp_path / "t"), str(tmp_path / "tb"), "--crops", "2"]) == 0
    assert ext.main(["bank", str(tmp_path / "v"), str(tmp_path / "vb"), "--crops", "1"]) == 0
    code = ext.main(["train", str(tmp_path / "tb"), str(tmp_path / "vb"), str(tmp_path / "o"),
                     "--device", "cpu", "--epochs", "1", "--batch-pairs", "2", "--workers", "0"])
    assert code == 0
    result = json.loads((tmp_path / "o" / "result.json").read_text())
    assert result["status"] == "completed" and result["pairs"] == 2
    assert '"status": "completed"' in capsys.readouterr().out
