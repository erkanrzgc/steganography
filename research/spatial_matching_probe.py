"""LSB-matching fine-tune of the parity-input RGB SRNet (post-freeze research).

The frozen 40%-replacement states transfer to 40% LSB matching (development AUC
.85) but weaken at 10-20%, where Weighted Stego is at chance. This experiment
warm-starts from those states and trains on +/-1 LSB matching only, with a
decreasing-rate curriculum and pixel-exact D4 (rotation/flip) augmentation.
Development scoring reuses the fixed hard-condition stegos (seed 2026110903),
so epoch 0 reproduces the frozen comparison. Reserved and L sources stay closed.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from core import colab_artifacts as a
from core import spatial_native_data as data
from research import spatial_parity_probe as probe

SCHEMA = "rgb-parity-matching-result-v1"
EPOCHS = 40
MAX_EPOCHS = 60
DECAY_EPOCH = 30
DECAY_FACTOR = 0.1
LEARNING_RATE = 1e-3
BATCH_PAIRS = 4
TRAIN_SEED = 2026110904
EVAL_SEED = 2026110903
EVAL_EVERY = 5
MAX_SECONDS = 4 * 3600
KINDS = ("replacement", "matching")
CURRICULUM: tuple[tuple[int, tuple[float, ...]], ...] = (
    (0, (0.4,)),
    (10, (0.2, 0.4)),
    (20, (0.1, 0.2, 0.4)),
)
CONDITIONS: tuple[tuple[str, float], ...] = (
    ("replacement", 0.40),
    ("replacement", 0.20),
    ("replacement", 0.10),
    ("replacement", 0.05),
    ("matching", 0.40),
    ("matching", 0.20),
    ("matching", 0.10),
)
GATE = {"matching-0.2": 0.80, "matching-0.1": 0.65}


def embed(cover: np.ndarray, kind: str, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Embed ``rate * samples`` random bits at random positions (replacement or +/-1 matching)."""
    if not isinstance(cover, np.ndarray) or cover.dtype != np.uint8 or cover.ndim != 3:
        raise ValueError("uint8 HxWxC cover required")
    if kind not in KINDS or not isinstance(rate, float) or not 0.0 < rate <= 1.0:
        raise ValueError("embedding kind and rate in (0, 1] required")
    flat = cover.reshape(-1).astype(np.int16)
    chosen = rng.permutation(flat.size)[: int(round(flat.size * rate))]
    bits = rng.integers(0, 2, chosen.size)
    values = flat[chosen]
    if kind == "replacement":
        values = (values & ~1) | bits
    else:
        step = np.where(rng.integers(0, 2, chosen.size) == 1, 1, -1)
        step = np.where(values == 0, 1, np.where(values == 255, -1, step))
        values = np.where((values & 1) != bits, values + step, values)
    flat[chosen] = values
    return flat.reshape(cover.shape).astype(np.uint8)


def d4(image: np.ndarray, k: int) -> np.ndarray:
    """One of the eight pixel-exact rotations/reflections of an HxWxC image."""
    if type(k) is not int or not 0 <= k < 8:
        raise ValueError("D4 index 0..7 required")
    out = np.rot90(image, k % 4, axes=(0, 1))
    return np.ascontiguousarray(out[:, ::-1] if k >= 4 else out)


def rates(epoch: int) -> tuple[float, ...]:
    """Matching rates sampled uniformly during a zero-based epoch."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("non-negative epoch index required")
    return [choices for start, choices in CURRICULUM if epoch >= start][-1]


def eval_stegos(covers: Sequence[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Fixed development stegos, identical to the frozen hard-condition comparison."""
    return {
        f"{kind}-{rate}": [
            embed(cover, kind, rate, np.random.default_rng(EVAL_SEED + 1000 * k + i))
            for i, cover in enumerate(covers)
        ]
        for k, (kind, rate) in enumerate(CONDITIONS)
    }


def training_pair(
    cover: np.ndarray, epoch: int, row: int, model_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A fresh D4-augmented cover and its matching stego for one presentation."""
    rng = np.random.default_rng([TRAIN_SEED, model_seed, epoch, row])
    image = d4(cover, int(rng.integers(0, 8)))
    choices = rates(epoch)
    rate = choices[int(rng.integers(0, len(choices)))]
    return image, embed(image, "matching", rate, rng)


def _tensor(images: Sequence[np.ndarray]):
    import torch

    x = np.stack([image.transpose(2, 0, 1) for image in images]).astype("<f4")
    return torch.from_numpy(probe.with_parity(x))


class _Trainer:
    def __init__(
        self, model, covers, *, device: str, learning_rate: float, model_seed: int,
        epochs: int, max_seconds: float,
    ):
        import torch

        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self.model = model.to(device).train()
        self.covers, self.device, self.model_seed = covers, device, model_seed
        self.schedule = probe.order(model_seed, len(covers), epochs)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, foreach=False, fused=False,
        )
        self.updates = 0

    def check(self) -> None:
        if time.monotonic() - self.started > self.max_seconds:
            raise ValueError("matching probe deadline exceeded")

    def epoch(self, index: int) -> float:
        import torch

        count = len(self.covers)
        rows = [int(v % count) for v in self.schedule[index * count : (index + 1) * count]]
        losses = []
        for start in range(0, count, BATCH_PAIRS):
            self.check()
            images: list[np.ndarray] = []
            for row in rows[start : start + BATCH_PAIRS]:
                images.extend(training_pair(self.covers[row], index, row, self.model_seed))
            labels = torch.tensor([0, 1] * (len(images) // 2), device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                self.model(_tensor(images).to(self.device)), labels
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("matching probe nonfinite loss")
            loss.backward()
            self.optimizer.step()
            self.updates += 1
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def scores(self, covers, stegos: dict[str, list[np.ndarray]]) -> dict[str, float]:
        import torch

        self.check()
        self.model.eval()
        try:
            with torch.inference_mode():

                def p(image: np.ndarray) -> float:
                    logits = self.model(_tensor([image]).to(self.device)).double()
                    return float(torch.softmax(logits, 1)[0, 1])

                base = [p(cover) for cover in covers]
                return {
                    key: probe.pairwise_auc([[c, p(x)] for c, x in zip(base, values, strict=True)])
                    for key, values in stegos.items()
                }
        finally:
            self.model.train()


def _validate(cache, split, *, device, epochs, decay_epoch, learning_rate, model_seed,
              max_seconds, eval_every, init_state, init_sha256, state_path) -> None:
    probe._validate(cache, split, "cpu", epochs, min(max_seconds, probe.MAX_SECONDS))
    checks = (
        (device in ("cpu", "cuda:0"), "explicit cpu or cuda:0 device required"),
        (isinstance(max_seconds, (int, float)) and 0 < max_seconds <= MAX_SECONDS,
         "bounded deadline required"),
        (decay_epoch is None or (type(decay_epoch) is int and 1 <= decay_epoch < epochs),
         "decay epoch must precede the final epoch"),
        (isinstance(learning_rate, float) and 0 < learning_rate <= 1e-2,
         "bounded learning rate required"),
        (model_seed in probe.MODEL_SEEDS, "original model seed required"),
        (type(eval_every) is int and eval_every >= 1, "positive evaluation interval required"),
        ((init_state is None) == (init_sha256 is None), "initial state and its hash go together"),
        (len(split.train) >= 1 and len(split.development) >= 1, "train and development required"),
        (state_path is None or not state_path.exists(), "state path must be new"),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


def _load(model, path: Path, expected: str) -> None:
    import torch

    raw = path.read_bytes()
    if a.digest(raw) != expected:
        raise ValueError("initial state hash mismatch")
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))


def run(
    cache,
    split,
    *,
    model_seed: int,
    init_state: Path | None,
    init_sha256: str | None,
    device: str = "cpu",
    epochs: int = EPOCHS,
    decay_epoch: int | None = DECAY_EPOCH,
    learning_rate: float = LEARNING_RATE,
    max_seconds: float = MAX_SECONDS,
    eval_every: int = EVAL_EVERY,
    build: Callable[[], Any] = probe.network,
    state_path: Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune one seed on LSB matching; score fixed development stegos along the way."""
    import torch

    _validate(cache, split, device=device, epochs=epochs, decay_epoch=decay_epoch,
              learning_rate=learning_rate, model_seed=model_seed, max_seconds=max_seconds,
              eval_every=eval_every, init_state=init_state, init_sha256=init_sha256,
              state_path=state_path)
    torch.manual_seed(model_seed)
    model = build()
    if init_state is not None and init_sha256 is not None:
        _load(model, init_state, init_sha256)
    lookup = {row.id: i for i, row in enumerate(cache.rows)}
    covers = {
        role: [data.pair(cache, lookup[s], probe.PAYLOAD_SEED)[0] for s in getattr(split, role)]
        for role in ("train", "development")
    }
    stegos = eval_stegos(covers["development"])
    with probe._policy(device):
        trainer = _Trainer(model, covers["train"], device=device, learning_rate=learning_rate,
                           model_seed=model_seed, epochs=epochs, max_seconds=max_seconds)
        history: list[dict[str, Any]] = []

        def record(epoch: int, loss: float | None) -> None:
            scores = trainer.scores(covers["development"], stegos)
            entry = {"epoch": epoch, "loss": loss, "auc": scores}
            history.append(entry)
            if progress is not None:
                progress(entry)

        record(0, None)
        losses, learning_rates = [], []
        for index in range(epochs):
            decayed = decay_epoch is not None and index >= decay_epoch
            rate = learning_rate * (DECAY_FACTOR if decayed else 1.0)
            for group in trainer.optimizer.param_groups:
                group["lr"] = rate
            learning_rates.append(rate)
            losses.append(trainer.epoch(index))
            if (index + 1) % eval_every == 0 or index + 1 == epochs:
                record(index + 1, losses[-1])
    final = history[-1]["auc"]
    gate = {key: {"auc": final[key], "threshold": t, "passed": final[key] >= t}
            for key, t in GATE.items()}
    return {
        "schema": SCHEMA,
        "architecture": probe.ARCHITECTURE,
        "recipe": {
            "embedding": "matching", "curriculum": [list(c) for c in CURRICULUM],
            "augmentation": "d4", "batch_pairs": BATCH_PAIRS, "train_seed": TRAIN_SEED,
            "eval_seed": EVAL_SEED, "learning_rate": learning_rate, "decay_epoch": decay_epoch,
            "decay_factor": DECAY_FACTOR, "epochs": epochs,
        },
        "model_seed": model_seed,
        "init_sha256": init_sha256,
        "losses": losses,
        "learning_rates": learning_rates,
        "history": history,
        "gate": gate,
        "passed": all(item["passed"] for item in gate.values()),
        "updates": trainer.updates,
        "elapsed_seconds": time.monotonic() - trainer.started,
        "state_sha256": probe._save_state(trainer.model, state_path) if state_path else None,
        "environment": probe._environment(device),
        "future_reserved_opened": False,
    }
