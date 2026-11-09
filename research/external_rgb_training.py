"""Large-scale external-cover training of the parity-input RGB SRNet (GPU oriented).

The native corpus has only 64 training originals. This module trains on 256x256
crops of lossless 8-bit RGB PNG photographs (for example DIV2K HR), banked once
and grouped by source image so that validation can stay image-disjoint. Every
presentation applies a random pixel-exact D4 transform and a fresh LSB
replacement or +/-1 matching payload at a curriculum rate. Validation scores
fixed stegos per condition. Checkpoints are written atomically after every
epoch so a disconnected Colab runtime can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from research import spatial_matching_probe as matching
from research import spatial_parity_probe as probe

CROP = 256
MAX_PIXELS = 50_000_000
KINDS = matching.KINDS
CURRICULUM: tuple[tuple[int, tuple[float, ...]], ...] = (
    (0, (0.4,)),
    (2, (0.2, 0.4)),
    (4, (0.1, 0.2, 0.4)),
    (8, (0.05, 0.1, 0.2, 0.4)),
)
CONDITIONS: tuple[tuple[str, float], ...] = tuple(
    (kind, rate) for kind in KINDS for rate in (0.4, 0.2, 0.1, 0.05)
)
TRAIN_SEED = 2026110905
VALID_SEED = 2026110906
MODEL_SEED = 2026110907
DECAY_FACTOR = 0.1


def rates(epoch: int) -> tuple[float, ...]:
    """Payload rates sampled uniformly during a zero-based epoch."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("non-negative epoch index required")
    return [choices for start, choices in CURRICULUM if epoch >= start][-1]


def sources(folder: Path) -> list[Path]:
    """Sorted regular (non-symlink) PNG files of one folder."""
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() == ".png" and p.is_file() and not p.is_symlink()
    )
    if not paths:
        raise ValueError("no PNG sources found")
    return paths


def _rgb(path: Path) -> np.ndarray | None:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.width * image.height > MAX_PIXELS:
            return None
        if image.width < CROP or image.height < CROP:
            return None
        return np.asarray(image)


def build_bank(paths: Sequence[Path], crops_per_image: int, seed: int, out: Path) -> dict[str, Any]:
    """Write ``crops.npy`` (N,256,256,3 uint8), ``groups.npy`` and a manifest, exclusively."""
    if type(crops_per_image) is not int or not 1 <= crops_per_image <= 64:
        raise ValueError("1..64 crops per image required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "crops.npy").exists():
        raise ValueError("crop bank already exists")
    images = [(path, _rgb(path)) for path in paths]
    usable = [(path, rgb) for path, rgb in images if rgb is not None]
    if not usable:
        raise ValueError("no usable 8-bit RGB source of at least 256x256")
    bank = np.lib.format.open_memmap(
        out / "crops.npy", mode="w+", dtype=np.uint8,
        shape=(len(usable) * crops_per_image, CROP, CROP, 3),
    )
    for group, (_, rgb) in enumerate(usable):
        rng = np.random.default_rng([seed, group])
        height, width = rgb.shape[:2]
        for j in range(crops_per_image):
            y = int(rng.integers(0, height - CROP + 1))
            x = int(rng.integers(0, width - CROP + 1))
            bank[group * crops_per_image + j] = rgb[y : y + CROP, x : x + CROP]
    bank.flush()
    del bank
    np.save(out / "groups.npy", np.repeat(np.arange(len(usable)), crops_per_image))
    manifest = {
        "schema": "external-rgb-crop-bank-v1",
        "seed": seed,
        "crops_per_image": crops_per_image,
        "sources": [path.name for path, _ in usable],
        "skipped": [path.name for path, rgb in images if rgb is None],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_bank(folder: Path) -> np.ndarray:
    crops = np.load(folder / "crops.npy", mmap_mode="r")
    if crops.dtype != np.uint8 or crops.ndim != 4 or crops.shape[1:] != (CROP, CROP, 3):
        raise ValueError("crop bank shape")
    return crops


class Pairs:
    """Map-style dataset of (cover, stego) uint8 pairs with fresh per-epoch payloads."""

    def __init__(self, crops: np.ndarray, seed: int = TRAIN_SEED):
        self.crops, self.seed, self.epoch = crops, seed, 0

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch, int(index)])
        cover = matching.d4(np.asarray(self.crops[index]), int(rng.integers(0, 8)))
        kind = KINDS[int(rng.integers(0, len(KINDS)))]
        choices = rates(self.epoch)
        rate = choices[int(rng.integers(0, len(choices)))]
        return np.stack([cover, matching.embed(cover, kind, rate, rng)])


def to_input(images, device: str):
    """(N,H,W,3) uint8 tensor -> (N,6,H,W) float RGB plus +/-PARITY_SCALE LSB sign."""
    import torch

    x = images.to(device).permute(0, 3, 1, 2).float()
    return torch.cat([x, (x.remainder(2) * 2 - 1) * probe.PARITY_SCALE], 1)


def validation(crops: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    covers = np.array(crops)
    stegos = {
        f"{kind}-{rate}": np.stack([
            matching.embed(cover, kind, rate, np.random.default_rng([VALID_SEED, k, i]))
            for i, cover in enumerate(covers)
        ])
        for k, (kind, rate) in enumerate(CONDITIONS)
    }
    return covers, stegos


def score(model, images: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    import torch

    model.eval()
    values = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=device.startswith("cuda")):
        for start in range(0, len(images), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(images[start : start + batch]))
            logits = model(to_input(chunk, device)).float()
            values.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(values).astype(np.float64)


def evaluate(model, covers, stegos, device: str) -> dict[str, dict[str, float]]:
    base = score(model, covers, device)
    result = {}
    for key, values in stegos.items():
        marked = score(model, values, device)
        result[key] = {
            "auc": probe.pairwise_auc(np.stack([base, marked], 1)),
            "accuracy": float((np.sum(base < 0.5) + np.sum(marked >= 0.5)) / (2 * len(base))),
        }
    return result


def _save(path: Path, value: dict[str, Any]) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class _Run:
    def __init__(self, *, out: Path, device: str, learning_rate: float, init_state: Path | None,
                 build: Callable[[], Any]):
        import torch

        torch.manual_seed(MODEL_SEED)
        self.out, self.device = out, device
        self.model = build().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                           weight_decay=0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
        self.epoch, self.history = 0, []
        checkpoint = out / "checkpoint.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            self.epoch, self.history = state["epoch"], state["history"]
        elif init_state is not None:
            self.model.load_state_dict(torch.load(init_state, map_location=device,
                                                  weights_only=True))

    def save(self) -> None:
        _save(self.out / "checkpoint.pt", {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(), "epoch": self.epoch, "history": self.history,
        })
        _save(self.out / "model.pt", {k: v.detach().cpu() for k, v in
                                       self.model.state_dict().items()})

    def epoch_loss(self, pairs: Pairs, pairs_per_batch: int, workers: int) -> float:
        import torch

        generator = torch.Generator().manual_seed(MODEL_SEED + self.epoch)
        if len(pairs) < pairs_per_batch:
            raise ValueError("crop bank smaller than one batch")
        loader: Any = torch.utils.data.DataLoader(
            cast(Any, pairs), batch_size=pairs_per_batch, shuffle=True, generator=generator,
            num_workers=workers, drop_last=True, pin_memory=self.device.startswith("cuda"),
        )
        losses = []
        for batch in loader:
            images = batch.reshape(-1, CROP, CROP, 3)
            labels = torch.tensor([0, 1] * len(batch), device=self.device)
            with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                loss = torch.nn.functional.cross_entropy(
                    self.model(to_input(images, self.device)), labels
                )
            if not bool(torch.isfinite(loss)):
                raise ValueError("external training nonfinite loss")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))


def train(
    *, bank: Path, valid: Path, out: Path, device: str, epochs: int,
    pairs_per_batch: int = 8, learning_rate: float = 1e-3, decay_epoch: int | None = None,
    init_state: Path | None = None, max_seconds: float = 6 * 3600, workers: int = 2,
    build: Callable[[], Any] = probe.network, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train or resume; evaluate after every epoch; stop before an epoch would miss the deadline."""
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError("1..200 epochs required")
    if decay_epoch is not None and not 1 <= decay_epoch < epochs:
        raise ValueError("decay epoch must precede the final epoch")
    if not 1 <= pairs_per_batch <= 64 or not 0 < learning_rate <= 1e-2 or max_seconds <= 0:
        raise ValueError("bounded batch, learning rate and deadline required")
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    run = _Run(out=out, device=device, learning_rate=learning_rate, init_state=init_state,
               build=build)
    pairs = Pairs(load_bank(bank))
    covers, stegos = validation(load_bank(valid))
    status, last = "completed", 0.0
    while run.epoch < epochs:
        if time.monotonic() - started + 1.2 * last > max_seconds:
            status = "deadline"
            break
        begun = time.monotonic()
        decayed = decay_epoch is not None and run.epoch >= decay_epoch
        for group in run.optimizer.param_groups:
            group["lr"] = learning_rate * (DECAY_FACTOR if decayed else 1.0)
        pairs.epoch = run.epoch
        loss = run.epoch_loss(pairs, pairs_per_batch, workers)
        metrics = evaluate(run.model, covers, stegos, device)
        run.epoch += 1
        run.history.append({"epoch": run.epoch, "loss": loss, "validation": metrics,
                            "seconds": time.monotonic() - begun})
        run.save()
        last = time.monotonic() - begun
        log(json.dumps({"epoch": run.epoch, "loss": round(loss, 4), "seconds": round(last, 1),
                        "auc": {k: round(v["auc"], 4) for k, v in metrics.items()}}))
    return {"schema": "external-rgb-training-v1", "status": status, "epochs": run.epoch,
            "history": run.history, "pairs": len(pairs), "validation_covers": len(covers)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    bank = commands.add_parser("bank", help="extract a crop bank from a PNG folder")
    bank.add_argument("source", type=Path)
    bank.add_argument("out", type=Path)
    bank.add_argument("--crops", type=int, default=8)
    bank.add_argument("--seed", type=int, default=TRAIN_SEED)
    fit = commands.add_parser("train", help="train or resume on a crop bank")
    fit.add_argument("bank", type=Path)
    fit.add_argument("valid", type=Path)
    fit.add_argument("out", type=Path)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--epochs", type=int, default=16)
    fit.add_argument("--decay-epoch", type=int, default=None)
    fit.add_argument("--batch-pairs", type=int, default=8)
    fit.add_argument("--lr", type=float, default=1e-3)
    fit.add_argument("--init", type=Path, default=None)
    fit.add_argument("--hours", type=float, default=6.0)
    fit.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "bank":
        manifest = build_bank(sources(args.source), args.crops, args.seed, args.out)
        counts = {key: len(manifest[key]) for key in ("sources", "skipped")}
        print(json.dumps(counts))
        return 0
    result = train(bank=args.bank, valid=args.valid, out=args.out, device=args.device,
                   epochs=args.epochs, pairs_per_batch=args.batch_pairs, learning_rate=args.lr,
                   decay_epoch=args.decay_epoch, init_state=args.init,
                   max_seconds=args.hours * 3600, workers=args.workers)
    (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"status": result["status"], "epochs": result["epochs"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
