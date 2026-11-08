"""Role-bound native SRNet adapter; experiment authority belongs to its runner."""

from __future__ import annotations

import math
import random
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core import colab_artifacts as a
from core import nonjpeg_models as models
from core import spatial_colab as capacity
from core import spatial_colab_state as numeric
from core import spatial_native_data as data
from core import spatial_training_data as seeds
from core import srnet_cuda

MAX_OUTPUT_BYTES = 512 * a.MIB


@dataclass(frozen=True)
class Split:
    manifest_sha256: str
    mode: str
    seed: int
    train: tuple[str, ...]
    development: tuple[str, ...]
    future_reserved: tuple[str, ...]
    unassigned: tuple[str, ...]

    def __post_init__(self) -> None:
        a.identity(self.manifest_sha256)
        seeds.seed(self.seed)
        fields = (self.train, self.development, self.future_reserved, self.unassigned)
        if (
            self.mode not in ("L", "RGB")
            or any(not isinstance(ids, tuple) for ids in fields)
            or any(not ids for ids in fields[:3])
        ):
            raise ValueError("immutable nonempty native partitions required")
        ids = tuple(i for values in fields for i in values)
        if (
            not 1 <= len(ids) <= data.MAX_SOURCES
            or any(not isinstance(i, str) or not re.fullmatch(r"[a-z0-9-]{1,80}", i) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("native split identity/overlap")

    def validate(self, cache: data.Cache) -> None:
        if type(cache) is not data.Cache or cache.manifest_sha256 != self.manifest_sha256:
            raise ValueError("native split cache binding")
        roles = {
            i: role
            for role in ("train", "development", "future_reserved", "unassigned")
            for i in getattr(self, role)
        }
        if set(roles) != {row.id for row in cache.rows}:
            raise ValueError("complete global native split required")
        if any(row.mode != self.mode and roles[row.id] != "unassigned" for row in cache.rows):
            raise ValueError("native split crosses modes")
        if any(len({roles[i] for i in group}) != 1 for group in data.components(cache.rows)):
            raise ValueError("native split common-ancestry leakage")


def split_document(value: dict) -> Split:
    if not isinstance(value, dict) or set(value) != set(Split.__dataclass_fields__):
        raise ValueError("native split document fields")
    values = dict(value)
    for key in ("train", "development", "future_reserved", "unassigned"):
        if not isinstance(values[key], list) or len(values[key]) > data.MAX_SOURCES:
            raise ValueError("native split document rows")
        values[key] = tuple(values[key])
    return Split(**values)


def partition(cache: data.Cache, mode: str, counts: tuple[int, int, int], seed: int) -> Split:
    """Exact metadata-only group assignment, without opening any pixel file."""
    seeds.seed(seed)
    if (
        type(cache) is not data.Cache
        or mode not in ("L", "RGB")
        or not isinstance(counts, tuple)
        or len(counts) != 3
        or any(type(n) is not int or not 1 <= n <= 128 for n in counts)
        or sum(counts) > 128
    ):
        raise ValueError("bounded native partition quotas")
    lookup = {row.id: row for row in cache.rows}
    groups = [
        group for group in data.components(cache.rows) if all(lookup[i].mode == mode for i in group)
    ]
    groups.sort(key=lambda group: a.digest(a.encode({"seed": seed, "sources": list(group)})))
    if sum(map(len, groups)) != sum(counts):
        raise ValueError("native partition inventory does not match quotas")
    layers: list[dict[tuple[int, int], tuple[int, int, int]]] = []
    states = {(0, 0)}
    consumed = 0
    for group in groups:
        size = len(group)
        following: dict[tuple[int, int], tuple[int, int, int]] = {}
        for train, dev in sorted(states):
            for role, (new_train, new_dev) in enumerate(
                ((train + size, dev), (train, dev + size), (train, dev))
            ):
                reserved = consumed + size - new_train - new_dev
                if new_train <= counts[0] and new_dev <= counts[1] and 0 <= reserved <= counts[2]:
                    following.setdefault((new_train, new_dev), (train, dev, role))
        layers.append(following)
        states = set(following)
        consumed += size
    state = counts[:2]
    if state not in states:
        raise ValueError("native grouped partition is infeasible")
    chosen: list[list[str]] = [[], [], []]
    for group, layer in zip(reversed(groups), reversed(layers), strict=True):
        train, dev, role = layer[state]
        chosen[role].extend(group)
        state = train, dev
    assigned = {i for values in chosen for i in values}
    result = Split(
        cache.manifest_sha256,
        mode,
        seed,
        tuple(sorted(chosen[0])),
        tuple(sorted(chosen[1])),
        tuple(sorted(chosen[2])),
        tuple(sorted(set(lookup) - assigned)),
    )
    result.validate(cache)
    return result


@dataclass(frozen=True)
class Plan:
    split: Split
    model_seed: int
    payload_seed: int
    epochs: int
    batch_images: int
    source_sha256: str

    def __post_init__(self) -> None:
        seeds.seed(self.model_seed)
        seeds.seed(self.payload_seed)
        a.identity(self.source_sha256)
        if (
            type(self.split) is not Split
            or type(self.epochs) is not int
            or not 1 <= self.epochs <= 20
            or type(self.batch_images) is not int
            or self.batch_images not in (2, 4, 8)
        ):
            raise ValueError("bounded native training plan required")

    @property
    def domain(self) -> str:
        return models.GRAY_DOMAIN if self.split.mode == "L" else models.RGB_DOMAIN


def order(plan: Plan) -> np.ndarray:
    if type(plan) is not Plan:
        raise ValueError("native plan required")
    rng = np.random.Generator(np.random.PCG64(plan.model_seed))
    n = len(plan.split.train)
    return np.concatenate([rng.permutation(n) + epoch * n for epoch in range(plan.epochs)]).astype(
        "<i8"
    )


class _Session:
    def __init__(self, cache: data.Cache, plan: Plan, device: str, max_seconds: float):
        import torch

        self.cache = data.load(cache.root, plan.split.manifest_sha256)
        if self.cache.rows != cache.rows:
            raise ValueError("native cache source mismatch")
        plan.split.validate(self.cache)
        self.lookup = {row.id: i for i, row in enumerate(self.cache.rows)}
        self.plan, self.device, self.max_seconds = plan, device, max_seconds
        self.started = time.monotonic()
        self.prior_seconds = 0.0
        self.failed = False
        self.cursor = self.updates = self.presentations = self.output_bytes = 0
        self.schedule = order(plan)
        self.data_sha256 = a.digest(
            a.encode(
                {
                    "manifest": cache.manifest_sha256,
                    "split": asdict(plan.split),
                    "recipe": data.RECIPE,
                    "payload_seed": plan.payload_seed,
                }
            )
        )
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
        self.model = models.network(plan.domain).to(device).train()
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
            raise ValueError("native session unavailable: failed/deadline")

    def step(self) -> dict:
        import torch

        self.check()
        if self.cursor == len(self.schedule):
            raise StopIteration
        count = len(self.plan.split.train)
        stop = min(self.cursor + self.plan.batch_images // 2, (self.cursor // count + 1) * count)
        indices = tuple(int(v % count) for v in self.schedule[self.cursor : stop])
        try:
            x, labels = data.batch(
                self.cache,
                self.plan.split.mode,
                tuple(self.lookup[self.plan.split.train[i]] for i in indices),
                self.plan.payload_seed,
            )
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
                raise ValueError("native nonfinite/absent gradients")
            self.check()
            self.optimizer.step()
            # Charge immediately, including an update that subsequently fails a check.
            self.updates += 1
            self.presentations += 2 * len(indices)
            self.cursor = stop
            if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
                raise ValueError("native nonfinite model state")
            self.check()
        except Exception:
            self.failed = True
            raise
        return {
            "optimizer_updates": self.updates,
            "presentations": self.presentations,
            "cursor": self.cursor,
            "loss": float(loss.detach().cpu()),
            "scope": "native_adapter",
        }

    def scores(self, role: str, *, record_scores: bool = True) -> dict:
        import torch

        from core.jpeg_study import auc

        self.check()
        if role not in ("train", "development") or type(record_scores) is not bool:
            raise ValueError("native train/development scoring scope required")
        previous = self.model.training
        scores, durations = [], []
        try:
            self.model.eval()
            with torch.inference_mode():
                for source_id in getattr(self.plan.split, role):
                    self.check()
                    started = time.monotonic()
                    x, _ = data.batch(
                        self.cache,
                        self.plan.split.mode,
                        (self.lookup[source_id],),
                        self.plan.payload_seed,
                    )
                    pair = (
                        torch.softmax(self.model(torch.from_numpy(x).to(self.device)), 1)[:, 1]
                        .cpu()
                        .numpy()
                    )
                    if not np.isfinite(pair).all() or np.any((pair < 0) | (pair > 1)):
                        raise ValueError("native nonfinite evaluation scores")
                    durations.append(time.monotonic() - started)
                    if record_scores:
                        scores.append(pair.astype(np.float64))
        finally:
            self.model.train(previous)
        self.check()
        result: dict = {
            "originals": len(durations),
            "max_pair_seconds": max(durations),
            "seconds": sum(durations),
            "role": role,
            "optimizer_updates": self.updates,
        }
        if record_scores:
            values = np.stack(scores)
            result.update({"scores": values, "auc": auc(values)})
        return result

    def checkpoint(self, path: Path) -> str:
        self.check()
        meta, arrays = numeric.capture(
            self.model,
            self.optimizer,
            source_sha256=self.binding,
            data_sha256=self.data_sha256,
            order=self.schedule,
            consumption={
                "epoch": self.cursor // len(self.plan.split.train),
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
            raise ValueError("native cumulative checkpoint output limit")
        checksum = numeric.save(path, meta, arrays)
        self.output_bytes = total
        self.check()
        return checksum

    def resume(self, path: Path, checksum: str) -> None:
        self.check()
        if self.cursor or self.updates or self.output_bytes:
            raise ValueError("resume requires a fresh native session")
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
            len(self.plan.split.train),
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
            raise ValueError("native checkpoint schedule/consumption mismatch")
        numeric.restore(self.model, self.optimizer, meta, arrays)
        self.cursor, self.updates, self.presentations = cursor, updates, count["presentations"]
        self.output_bytes, self.prior_seconds = count["output_bytes"], count["elapsed_seconds"]
        self.check()


@contextmanager
def session(cache: data.Cache, plan: Plan, *, device: str = "cpu", max_seconds: float = 120):
    """Software API for native QA; actual experiments need a bound supervisor."""
    import torch

    if (
        type(cache) is not data.Cache
        or type(plan) is not Plan
        or type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 1800
    ):
        raise ValueError("native session plan/deadline")
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
            yield _Session(cache, plan, device, max_seconds)
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_num_threads(threads)
