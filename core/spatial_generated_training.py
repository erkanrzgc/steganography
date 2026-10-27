"""Generated-only SRNet step/evaluation/numeric-resume core; no native fit CLI."""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core import colab_artifacts as a
from core import nonjpeg_models as models
from core import spatial_colab as capacity
from core import spatial_colab_state as numeric
from core import spatial_training_data as data
from core import srnet_cuda

MAX_OUTPUT_BYTES = 512 * a.MIB


@dataclass(frozen=True)
class Plan:
    spec: data.GeneratedSpec
    model_seed: int
    epochs: int
    batch_images: int
    source_sha256: str

    def __post_init__(self) -> None:
        data.seed(self.model_seed)
        a.identity(self.source_sha256)
        if (
            type(self.spec) is not data.GeneratedSpec
            or self.spec.split != "train"
            or type(self.epochs) is not int
            or not 1 <= self.epochs <= 20
            or type(self.batch_images) is not int
            or self.batch_images not in (2, 4, 8)
        ):
            raise ValueError("bounded generated training plan required")


def order(plan: Plan) -> np.ndarray:
    if type(plan) is not Plan:
        raise ValueError("generated plan required")
    rng = np.random.Generator(np.random.PCG64(plan.model_seed))
    count = plan.spec.originals
    return np.concatenate(
        [rng.permutation(count) + epoch * count for epoch in range(plan.epochs)]
    ).astype("<i8")


class _Session:
    def __init__(self, plan: Plan, device: str, max_seconds: float):
        import torch

        self.plan, self.device, self.max_seconds = plan, device, max_seconds
        self.started = time.monotonic()
        self.prior_seconds = 0.0
        self.failed = False
        self.cursor = self.updates = self.presentations = self.output_bytes = 0
        self.schedule = order(plan)
        self.data_sha256 = a.digest(a.encode(data.manifest(plan.spec)))
        self.binding = a.digest(
            a.encode(
                {
                    "plan": asdict(plan),
                    "device": device,
                    "recipe": data.RECIPE,
                    "optimizer": "scalar-adamw-1e-4-v1",
                }
            )
        )
        self.model = models.network(plan.spec.domain).to(device).train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )

    def check(self) -> None:
        if self.failed or self.prior_seconds + time.monotonic() - self.started >= self.max_seconds:
            self.failed = True
            raise ValueError("generated session unavailable: failed/deadline")

    def step(self) -> dict:
        import torch

        self.check()
        if self.cursor == len(self.schedule):
            raise StopIteration
        count = self.plan.spec.originals
        stop = min(self.cursor + self.plan.batch_images // 2, (self.cursor // count + 1) * count)
        indices = tuple(int(v % count) for v in self.schedule[self.cursor : stop])
        try:
            x, labels = data.batch(self.plan.spec, indices)
            values = torch.from_numpy(x).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                self.model(values), torch.from_numpy(labels).to(self.device)
            )
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(
                p.grad is None or not bool(torch.isfinite(p.grad).all())
                for p in self.model.parameters()
            ):
                raise ValueError("generated nonfinite/absent gradients")
            self.check()
            self.optimizer.step()
            # Charge immediately, including an update that subsequently fails a check.
            self.updates += 1
            self.presentations += 2 * len(indices)
            self.cursor = stop
            if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
                raise ValueError("generated nonfinite model state")
            self.check()
        except Exception:
            self.failed = True
            raise
        return {
            "optimizer_updates": self.updates,
            "presentations": self.presentations,
            "cursor": self.cursor,
            "loss": float(loss.detach().cpu()),
            "generated_only": True,
        }

    def scores(self, spec: data.GeneratedSpec) -> dict:
        import torch

        from core.jpeg_study import auc

        self.check()
        if type(spec) is not data.GeneratedSpec or spec.mode != self.plan.spec.mode:
            raise ValueError("generated evaluation domain mismatch")
        if spec.split == "train" and spec != self.plan.spec:
            raise ValueError("generated train evaluation must match bound training data")
        previous = self.model.training
        scores = []
        try:
            self.model.eval()
            with torch.inference_mode():
                for index in range(spec.originals):
                    self.check()
                    values, _ = data.batch(spec, (index,))
                    pair = (
                        torch.softmax(self.model(torch.from_numpy(values).to(self.device)), 1)[:, 1]
                        .cpu()
                        .numpy()
                    )
                    if not np.isfinite(pair).all() or np.any((pair < 0) | (pair > 1)):
                        raise ValueError("generated nonfinite evaluation scores")
                    scores.append(pair.astype(np.float64))
        finally:
            self.model.train(previous)
        self.check()
        values = np.stack(scores)
        return {
            "scores": values,
            "auc": auc(values),
            "generated_only": True,
            "split": spec.split,
            "optimizer_updates": self.updates,
        }

    def checkpoint(self, path: Path) -> str:
        self.check()
        meta, arrays = numeric.capture(
            self.model,
            self.optimizer,
            source_sha256=self.binding,
            data_sha256=self.data_sha256,
            order=self.schedule,
            consumption={
                "epoch": self.cursor // self.plan.spec.originals,
                "cursor": self.cursor,
                "optimizer_updates": self.updates,
                "presentations": self.presentations,
                "elapsed_seconds": self.prior_seconds + time.monotonic() - self.started,
                "output_bytes": self.output_bytes,
            },
        )
        # Include this file in its own cumulative byte debit. Header length converges
        # once the number of decimal digits stabilizes; numeric arrays are fixed-size.
        body_bytes = sum(value.nbytes for value in arrays.values())
        while True:
            total = self.output_bytes + 12 + len(a.encode(meta)) + body_bytes
            if total == meta["consumption"]["output_bytes"]:
                break
            meta["consumption"]["output_bytes"] = total
        if total > MAX_OUTPUT_BYTES:
            raise ValueError("generated cumulative checkpoint output limit")
        checksum = numeric.save(path, meta, arrays)
        self.output_bytes = total
        self.check()
        return checksum

    def resume(self, path: Path, checksum: str) -> None:
        self.check()
        if self.cursor or self.updates or self.output_bytes:
            raise ValueError("resume requires a fresh generated session")
        meta, arrays = numeric.load(
            path,
            checksum=checksum,
            environment=capacity.environment(),
            source_sha256=self.binding,
            data_sha256=self.data_sha256,
        )
        count = meta["consumption"]
        cursor, originals, pairs = (
            count["cursor"],
            self.plan.spec.originals,
            self.plan.batch_images // 2,
        )
        updates = (cursor // originals) * math.ceil(originals / pairs) + math.ceil(
            (cursor % originals) / pairs
        )
        if (
            not np.array_equal(arrays["data/order"], self.schedule)
            or count["epoch"] != cursor // originals
            or cursor % originals % pairs != 0
            or count["optimizer_updates"] != updates
            or count["presentations"] != 2 * cursor
            or count["elapsed_seconds"] >= self.max_seconds
            or count["output_bytes"] > MAX_OUTPUT_BYTES
            or count["output_bytes"] < path.stat().st_size
        ):
            raise ValueError("generated checkpoint schedule/consumption mismatch")
        numeric.restore(self.model, self.optimizer, meta, arrays)
        self.cursor, self.updates, self.presentations = cursor, updates, count["presentations"]
        self.output_bytes, self.prior_seconds = count["output_bytes"], count["elapsed_seconds"]
        self.check()


@contextmanager
def session(plan: Plan, *, device: str = "cpu", max_seconds: float = 120):
    """Software API for generated QA; actual experiments need a bound supervisor."""
    import torch

    if (
        type(plan) is not Plan
        or type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 1800
    ):
        raise ValueError("generated session plan/deadline")
    srnet_cuda.device(device)
    python, numpy, threads = random.getstate(), np.random.get_state(), torch.get_num_threads()
    policy = srnet_cuda.policy() if device == "cuda:0" else capacity.fp32_policy()
    try:
        with policy, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.set_num_threads(2)
            torch.random.default_generator.manual_seed(plan.model_seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(plan.model_seed)
            random.seed(plan.model_seed)
            np.random.seed(plan.model_seed)
            yield _Session(plan, device, max_seconds)
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_num_threads(threads)
