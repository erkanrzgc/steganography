"""One-variable native RGB diagnosis: the failed SRNet recipe plus LSB-parity inputs.

The 2026-11-07 native RGB fit failed with train AUC near .5 while a parameter-free
Weighted Stego estimator separates the same train pairs. The hypothesis is that
a ReLU CNN cannot derive sample parity from raw intensities. This probe keeps
the split, payload, model seed, initial weights, order, batch, optimizer and
gate of seed 2026102301 and changes only the input: three channels of
``±PARITY_SCALE`` LSB sign are appended and the first convolution is widened.
No checkpoint is written; reserved and L sources are never opened.
"""

from __future__ import annotations

import math
import platform
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from core import colab_artifacts as a
from core import nonjpeg_models as models
from core import spatial_native_data as data
from modules import lsb_replacement as ws

ARCHITECTURE = "srnet-rgb-parity-input-v1"
MODEL_SEED = 2026102301
MODEL_SEEDS = (2026102301, 2026102302)
PAYLOAD_SEED = 2026102805
EPOCHS = 20
MAX_EPOCHS = 60
MAX_SECONDS = 7200
DECAY_FACTOR = 0.1
BATCH_IMAGES = 8
PARITY_SCALE = 64.0
PARITY_INIT_SEED = 2026110801
LEARNING_RATE = 1e-4
TRAIN_AUC_GATE = 0.85
DEVELOPMENT_AUC_GATE = 0.70
LOSS_RATIO_GATE = 0.8


def binding(
    fresh_payload: bool = False,
    learning_rate: float = LEARNING_RATE,
    model_seed: int = MODEL_SEED,
    epochs: int = EPOCHS,
    decay_epoch: int | None = None,
) -> str:
    """Bind the recipe; the original fixed-payload 1e-4 binding is unchanged."""
    extra: dict[str, Any] = {"training_payload": "fresh-per-epoch-v1"} if fresh_payload else {}
    if learning_rate != LEARNING_RATE:
        extra["learning_rate"] = learning_rate
    if epochs != EPOCHS:
        extra["epochs"] = epochs
    if decay_epoch is not None:
        extra["decay_epoch"] = decay_epoch
        extra["decay_factor"] = DECAY_FACTOR
    return a.digest(
        a.encode(
            {
                **extra,
                "architecture": ARCHITECTURE,
                "model_seed": model_seed,
                "payload_seed": PAYLOAD_SEED,
                "parity_scale": PARITY_SCALE,
                "parity_init_seed": PARITY_INIT_SEED,
                "optimizer": "scalar-adamw-1e-4-v1",
            }
        )
    )


def with_parity(x: np.ndarray) -> np.ndarray:
    """Append the scaled LSB sign of every raw 0..255 RGB sample as three channels."""
    if (
        not isinstance(x, np.ndarray)
        or x.dtype != np.float32
        or x.ndim != 4
        or x.shape[1] != 3
        or not np.isfinite(x).all()
        or np.any((x < 0) | (x > 255) | (x != np.rint(x)))
    ):
        raise ValueError("raw integral 0..255 NCHW RGB samples required")
    sign = np.where(x.astype(np.uint8) & 1, PARITY_SCALE, -PARITY_SCALE).astype(np.float32)
    return np.concatenate((x, sign), axis=1)


def network():
    """Original RGB SRNet initialisation, with a parity-widened first convolution."""
    import torch

    model = models.network(models.RGB_DOMAIN)
    old = model.front[0][0]
    conv = torch.nn.Conv2d(6, 64, 3, padding=1, dtype=torch.float32, device="cpu")
    generator = torch.Generator().manual_seed(PARITY_INIT_SEED)
    with torch.no_grad():
        conv.weight[:, :3].copy_(old.weight)
        conv.weight[:, 3:].normal_(0.0, math.sqrt(2.0 / 27.0), generator=generator)
        if conv.bias is None or old.bias is None:
            raise ValueError("biased first convolution required")
        conv.bias.copy_(old.bias)
    model.front[0][0] = conv
    model.architecture = ARCHITECTURE
    return model


def pairwise_auc(values) -> float:
    pairs = np.asarray(values, dtype=np.float64)
    difference = pairs[:, 1, None] - pairs[None, :, 0]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def order(model_seed: int, count: int, epochs: int) -> np.ndarray:
    """The native engine's per-epoch permutation order, extended beyond twenty epochs."""
    rng = np.random.Generator(np.random.PCG64(model_seed))
    return np.concatenate([rng.permutation(count) + epoch * count for epoch in range(epochs)])


def training_seed(epoch_index: int, fresh_payload: bool) -> int:
    """Fixed payload seed, or a fresh payload/position seed for every epoch."""
    return (PAYLOAD_SEED + 1 + epoch_index) % 2**32 if fresh_payload else PAYLOAD_SEED


class _Probe:
    def __init__(
        self, cache, split, device: str, epochs: int, max_seconds: float, build,
        fresh_payload: bool = False, learning_rate: float = LEARNING_RATE,
        model_seed: int = MODEL_SEED,
    ):
        import torch

        self.started = time.monotonic()
        self.max_seconds = max_seconds
        self.cache, self.split, self.device = cache, split, device
        self.lookup = {row.id: i for i, row in enumerate(cache.rows)}
        self.fresh_payload = fresh_payload
        self.schedule = order(model_seed, len(split.train), epochs)
        self.model = build().to(device).train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )
        self.cursor = self.updates = self.presentations = 0

    def check(self) -> None:
        if time.monotonic() - self.started > self.max_seconds:
            raise ValueError("parity probe deadline exceeded")

    def inputs(self, indices: tuple[int, ...], seed: int = PAYLOAD_SEED):
        import torch

        x, labels = data.batch(self.cache, "RGB", indices, seed)
        return torch.from_numpy(with_parity(x)).to(self.device), labels

    def step(self) -> float:
        import torch

        self.check()
        count = len(self.split.train)
        stop = min(self.cursor + BATCH_IMAGES // 2, (self.cursor // count + 1) * count)
        rows = tuple(int(v % count) for v in self.schedule[self.cursor : stop])
        seed = training_seed(self.cursor // count, self.fresh_payload)
        values, labels = self.inputs(tuple(self.lookup[self.split.train[i]] for i in rows), seed)
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            self.model(values), torch.from_numpy(labels).to(self.device)
        )
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(
            p.grad is None or not bool(torch.isfinite(p.grad).all())
            for p in self.model.parameters()
        ):
            raise ValueError("parity probe nonfinite/absent gradients")
        self.optimizer.step()
        self.updates += 1
        self.presentations += 2 * len(rows)
        self.cursor = stop
        return float(loss.detach().cpu())

    def scores(self, role: str) -> dict[str, Any]:
        import torch

        self.check()
        probabilities, margins = [], []
        self.model.eval()
        try:
            with torch.inference_mode():
                for source in getattr(self.split, role):
                    values, _ = self.inputs((self.lookup[source],))
                    logits = self.model(values).double()
                    probabilities.append(torch.softmax(logits, 1)[:, 1].cpu().tolist())
                    margins.append((logits[:, 1] - logits[:, 0]).cpu().tolist())
        finally:
            self.model.train()
        return {
            "scores": probabilities,
            "margins": margins,
            "auc": pairwise_auc(probabilities),
            "margin_auc": pairwise_auc(margins),
        }


def ws_baseline(cache, split) -> dict[str, Any]:
    lookup = {row.id: i for i, row in enumerate(cache.rows)}
    result = {}
    for role in ("train", "development"):
        estimates = []
        for source in getattr(split, role):
            cover, stego, _ = data.pair(cache, lookup[source], PAYLOAD_SEED)
            estimates.append([ws.weighted_stego(cover).payload, ws.weighted_stego(stego).payload])
        result[role] = {"estimates": estimates, "auc": pairwise_auc(estimates)}
    return result


def _environment(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if device == "cuda:0":
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _policy(device: str):
    if device != "cuda:0":
        return nullcontext()
    from core import spatial_colab_profile as profile

    return profile.cuda_policy()


def _save_state(model, path: Path) -> str:
    """Write the final CPU state dict exclusively and return its SHA-256."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    raw = buffer.getvalue()
    with path.open("xb") as stream:
        stream.write(raw)
    return a.digest(raw)


def _validate(cache, split, device: str, epochs: int, max_seconds: float) -> None:
    if device not in ("cpu", "cuda:0"):
        raise ValueError("explicit cpu or cuda:0 device required")
    if type(epochs) is not int or not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError("bounded epoch count required")
    if device == "cuda:0" and epochs < EPOCHS:
        raise ValueError("physical runs must keep at least twenty epochs")
    if not isinstance(max_seconds, (int, float)) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("bounded deadline required")
    if split.mode != "RGB":
        raise ValueError("RGB split required")
    split.validate(cache)


def run(
    cache,
    split,
    *,
    device: str = "cuda:0",
    epochs: int = EPOCHS,
    max_seconds: float = 1500,
    build: Callable[[], Any] = network,
    fresh_payload: bool = False,
    learning_rate: float = LEARNING_RATE,
    model_seed: int = MODEL_SEED,
    decay_epoch: int | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Train one seed under the original recipe with parity inputs; return all scores."""
    import torch

    _validate(cache, split, device, epochs, max_seconds)
    if not isinstance(learning_rate, float) or not 0 < learning_rate <= 1e-2:
        raise ValueError("bounded learning rate required")
    if model_seed not in MODEL_SEEDS:
        raise ValueError("original model seed required")
    if decay_epoch is not None and (type(decay_epoch) is not int or not 1 <= decay_epoch < epochs):
        raise ValueError("decay epoch must precede the final epoch")
    if state_path is not None and state_path.exists():
        raise FileExistsError("model state path already exists")
    with _policy(device), torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
        torch.random.default_generator.manual_seed(model_seed)
        if device == "cuda:0":
            torch.cuda.manual_seed(model_seed)
        random.seed(model_seed)
        np.random.seed(model_seed)
        state = _Probe(
            cache, split, device, epochs, max_seconds, build,
            fresh_payload, learning_rate, model_seed,
        )
        initial = {role: state.scores(role) for role in ("train", "development")}
        steps = math.ceil(len(split.train) / (BATCH_IMAGES // 2))
        losses, rows, rates = [], [], []
        for epoch in range(1, epochs + 1):
            rate = learning_rate * (DECAY_FACTOR if decay_epoch and epoch > decay_epoch else 1.0)
            for group in state.optimizer.param_groups:
                group["lr"] = rate
            rates.append(rate)
            epoch_losses = [state.step() for _ in range(steps)]
            losses.extend(epoch_losses)
            development = state.scores("development")
            rows.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "development_auc": development["auc"],
                    "development_margin_auc": development["margin_auc"],
                }
            )
            print(a.encode(rows[-1]).decode().strip(), flush=True)
        final = {"train": state.scores("train"), "development": development}
        elapsed = time.monotonic() - state.started
        environment = _environment(device)
        state_sha256 = _save_state(state.model, state_path) if state_path is not None else None
    first, last = float(np.mean(losses[:steps])), float(np.mean(losses[-steps:]))
    gate = {
        "train_auc": final["train"]["auc"],
        "development_auc": final["development"]["auc"],
        "loss_ratio": last / first,
    }
    gate["passed"] = bool(
        gate["train_auc"] >= TRAIN_AUC_GATE
        and gate["development_auc"] >= DEVELOPMENT_AUC_GATE
        and gate["loss_ratio"] <= LOSS_RATIO_GATE
    )
    return {
        "schema": "rgb-parity-probe-result-v1",
        "binding_sha256": binding(fresh_payload, learning_rate, model_seed, epochs, decay_epoch),
        "epoch_count": epochs,
        "decay_epoch": decay_epoch,
        "learning_rates": rates,
        "state_sha256": state_sha256,
        "model_seed": model_seed,
        "learning_rate": learning_rate,
        "fresh_payload": fresh_payload,
        "training_payload_seeds": [training_seed(i, fresh_payload) for i in range(epochs)],
        "architecture": ARCHITECTURE,
        "device": device,
        "environment": environment,
        "split_manifest_sha256": split.manifest_sha256,
        "epochs": rows,
        "losses": losses,
        "schedule": [int(v % len(split.train)) for v in state.schedule],
        "initial": initial,
        "final": final,
        "gate": gate,
        "ws_baseline": ws_baseline(cache, split),
        "optimizer_updates": state.updates,
        "presentations": state.presentations,
        "elapsed_seconds": elapsed,
        "checkpoints": 0 if state_sha256 is None else 1,
        "future_reserved_opened": False,
    }
