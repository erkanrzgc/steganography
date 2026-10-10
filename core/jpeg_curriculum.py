"""Frozen two-source Q75 curriculum; development evidence, not deployment."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from core import (
    jpeg_float256,
    jpeg_population_control,
    srnet_cuda,
    srnet_groupnorm,
    srnet_model,
    srnet_norm_training,
    srnet_population_bn,
    srnet_samepool,
    srnet_training,
)
from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA, TimingReader
from core.srnet_stream import ROW_BYTES, deadline_after, document, regular_open, signature

SEED = 20261011
EPOCHS = 40
UPDATES = 96


def selection(samples):
    # Existing independently audited kit partitions by original lineage, not file hash.
    fit, probe = srnet_population_bn.partition(samples)
    allowed = set(fit.flat)
    pairs = []
    for source in ("BOSSbase-1.01", "BOWS2"):
        groups = sorted(
            {samples[i]["lineage"] for i in allowed if samples[i]["source_group"] == source}
        )
        if len(groups) != 24:
            raise ValueError("curriculum requires 24 fit originals per source")
        for lineage in groups:
            pair = []
            for method in (None, "JUNIWARD"):
                rows = [
                    i
                    for i in allowed
                    if samples[i]["source_group"] == source
                    and samples[i]["lineage"] == lineage
                    and samples[i]["quality_factor"] == 75
                    and samples[i]["method"] == method
                    and samples[i]["label"] == ("cover" if method is None else "stego")
                ]
                if len(rows) != 1:
                    raise ValueError("curriculum matched Q75 family missing")
                if method is not None and (
                    samples[rows[0]].get("payload_rate") != 0.2
                    or samples[rows[0]].get("payload_unit") != "bpnzAC"
                ):
                    raise ValueError("curriculum adaptation payload mismatch")
                pair.append(rows[0])
            pairs.append(pair)
    return np.array(pairs, dtype=np.int64), probe


def schedule(pairs, epoch):
    if (
        pairs.shape != (48, 2)
        or pairs.dtype != np.int64
        or type(epoch) is not int
        or not 0 <= epoch < EPOCHS
    ):
        raise ValueError("curriculum schedule bounds")
    rng = np.random.default_rng(SEED + epoch)
    # Four complete exposures per original per epoch; never repeat by outcome.
    return np.concatenate([pairs[rng.permutation(48)].reshape(24, 4) for _ in range(4)])


def augment(values, orientation):
    if (
        values.shape != (4, 1, 256, 256)
        or values.dtype != np.dtype("<f4")
        or type(orientation) is not int
        or not 0 <= orientation < 8
    ):
        raise ValueError("curriculum D4 bounds")
    transformed = np.flip(values, axis=3) if orientation >= 4 else values
    return np.rot90(transformed, orientation % 4, axes=(2, 3)).copy()


def probe_metrics(samples, logits):
    result = jpeg_population_control.cells(samples, logits)
    for cell in result:
        scores: dict[str, list[float]] = {"cover": [], "stego": []}
        for i, values in logits.items():
            row = samples[i]
            if (row["source_group"], row["quality_factor"]) == (
                cell["source_group"],
                cell["quality_factor"],
            ) and row["method"] in (None, cell["method"]):
                scores[row["label"]].append(float(values[1]) - float(values[0]))
        cell["roc_auc"] = (
            sum(
                (positive > negative) + 0.5 * (positive == negative)
                for positive in scores["stego"]
                for negative in scores["cover"]
            )
            / 64
        )
    return result


def checked_pixels(cover, stego):
    """Same native parser, independent SciPy IDCT math; no parser-oracle claim."""
    import tempfile
    from pathlib import Path

    import jpeglib
    from scipy.fft import idctn  # type: ignore[import-untyped]

    values = jpeg_float256.pixel_batch([cover, stego])
    decoded = []
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix="jpeg-curriculum-audit-") as folder:
        for position, data in enumerate((cover, stego)):
            path = Path(folder) / f"{position}.jpg"
            with path.open("xb") as stream:
                stream.write(data)
            jpeglib.version.set("6b")
            item = jpeglib.read_dct(str(path))
            item.load()
            info = jpeg_float256.region(data)
            if info["image_size"] != [512, 512] or item.jpeg_color_space.name != "JCS_GRAYSCALE":
                raise ValueError("curriculum requires fixed grayscale 512-square JPEG")
            coefficients, table = item.Y.copy(), item.get_component_qt(0).copy()
            decoded.append((coefficients, table))
            raster = (
                idctn(coefficients.astype(np.float64) * table, axes=(-2, -1), norm="ortho") + 128
            )
            image = raster.transpose(0, 2, 1, 3).reshape(512, 512)
            independent = image[128:384, 128:384].astype("<f4")
            maximum = max(maximum, float(np.max(np.abs(independent - values[position, 0]))))
    difference = decoded[1][0].astype(np.int64) - decoded[0][0].astype(np.int64)
    changes = int(np.count_nonzero(difference))
    if (
        maximum > 1e-4
        or not changes
        or np.abs(difference).max() > 1
        or not np.array_equal(decoded[0][1], decoded[1][1])
    ):
        raise ValueError("curriculum coefficient/IDCT audit failed")
    if np.array_equal(values[0], values[1]):
        raise ValueError("curriculum crop has no signal; retain failure, do not filter")
    return values, changes, maximum


def prepare(root, audit, out, generate):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("curriculum output exists or uses symlink")
    deadline = deadline_after(600)
    rows, digest = [], hashlib.sha256()
    with TimingReader(root, audit, deadline=deadline) as reader:
        pairs, _ = selection(reader.samples)
        out.mkdir(parents=False)
        with (out / "pixels.f32").open("xb") as stream:
            for cover_index, stego_index in pairs:
                deadline()
                row = reader.samples[int(cover_index)]
                with regular_open(root / "jpeg" / (row["sha256"] + ".jpg")) as source:
                    cover = source.read(256 * 1024 + 1)
                if len(cover) != row["size"] or hashlib.sha256(cover).hexdigest() != row["sha256"]:
                    raise ValueError("curriculum cover identity mismatch")
                stego = generate(cover, row["lineage"])
                if not 0 < len(stego) <= 256 * 1024:
                    raise ValueError("curriculum worker output bound")
                values, changes, maximum = checked_pixels(cover, stego)
                existing = reader.batch(
                    np.array([cover_index, stego_index, cover_index, stego_index])
                )
                if not np.array_equal(existing[0], values[0]):
                    raise ValueError("curriculum base preprocessing mismatch")
                with (out / f"{row['lineage']}.jpg").open("xb") as target:
                    target.write(stego)
                raw = values.tobytes()
                stream.write(raw)
                digest.update(raw)
                rows.append(
                    {
                        "indices": [int(cover_index), int(stego_index)],
                        "lineage": row["lineage"],
                        "source_group": row["source_group"],
                        "sha256": hashlib.sha256(stego).hexdigest(),
                        "bytes": len(stego),
                        "coefficient_changes": changes,
                        "idct_max_difference": maximum,
                    }
                )
        reader.verify()
    result = {
        "schema_version": "jpeg-q75-curriculum-cache-v1",
        "status": "completed",
        "manifest_sha256": MANIFEST_SHA,
        "audit_sha256": AUDIT_SHA,
        "rows": rows,
        "tensor_sha256": digest.hexdigest(),
        "tensor_bytes": 96 * ROW_BYTES,
        "payload_rate": 0.4,
        "payload_unit": "bpnzAC",
        "quality_factor": 75,
        "method": "JUNIWARD",
        "probe_pixels_read": 0,
        "independent_idct_rows": 96,
        "independent_native_parser": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
    return result


class HighReader:
    def __init__(self, root, checksum, pairs, deadline):
        self.stream = None
        self.root, self.deadline = root, deadline
        self.report = document(root / "manifest.json", checksum)
        r = self.report
        if (
            r.get("schema_version") != "jpeg-q75-curriculum-cache-v1"
            or r.get("status") != "completed"
            or r.get("manifest_sha256") != MANIFEST_SHA
            or r.get("audit_sha256") != AUDIT_SHA
            or r.get("payload_rate") != 0.4
            or r.get("payload_unit") != "bpnzAC"
            or r.get("quality_factor") != 75
            or r.get("method") != "JUNIWARD"
            or r.get("tensor_bytes") != 96 * ROW_BYTES
            or r.get("probe_pixels_read") != 0
            or r.get("independent_idct_rows") != 96
            or [x["indices"] for x in r["rows"]] != pairs.tolist()
        ):
            raise ValueError("curriculum high cache binding mismatch")
        self.positions = {int(index): i for i, index in enumerate(pairs.flat)}
        try:
            self.stream = regular_open(root / "pixels.f32")
            self.initial = signature(self.stream)
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        self.deadline()
        if (
            self.stream is None
            or signature(self.stream) != self.initial
            or self.initial[2] != 96 * ROW_BYTES
        ):
            raise ValueError("curriculum high tensor closed/mutated")
        self.stream.seek(0)
        digest = hashlib.sha256()
        while raw := self.stream.read(1024**2):
            self.deadline()
            values = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(values).all() or np.any(np.abs(values) > 2**36):
                raise ValueError("curriculum high values invalid")
            digest.update(raw)
        if (
            digest.hexdigest() != self.report["tensor_sha256"]
            or signature(self.stream) != self.initial
        ):
            raise ValueError("curriculum high tensor identity mismatch")

    def batch(self, indices):
        self.deadline()
        if (
            self.stream is None
            or signature(self.stream) != self.initial
            or indices.shape != (4,)
            or indices.dtype != np.int64
            or any(int(i) not in self.positions for i in indices)
        ):
            raise ValueError("curriculum high batch bounds")
        result = np.empty((4, 1, 256, 256), dtype="<f4")
        for target, index in zip(result, indices, strict=True):
            self.stream.seek(self.positions[int(index)] * ROW_BYTES)
            raw = self.stream.read(ROW_BYTES)
            if len(raw) != ROW_BYTES or signature(self.stream) != self.initial:
                raise ValueError("curriculum high row mutated")
            target[:] = np.frombuffer(raw, dtype="<f4").reshape(1, 256, 256)
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def fit(root, audit, high_root, high_sha):
    import torch

    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 2400:
            raise TimeoutError("curriculum exceeded frozen training deadline")

    execution = srnet_cuda.inspect()
    with srnet_cuda.policy(), TimingReader(root, audit, deadline=deadline) as reader:
        pairs, probe = selection(reader.samples)
        with HighReader(high_root, high_sha, pairs, deadline) as high:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEED)
                initial = srnet_samepool.network()
                initial_sha = srnet_groupnorm.parameter_sha(initial)
                del initial
            stages, model = [], None
            for stage, fetch, lr in (
                ("warmup-0.4", high.batch, 0.001),
                ("adapt-0.2", reader.batch, 0.0001),
            ):
                current = {"epoch": 0, "batch": 0}

                def batches(epoch, current=current):
                    current.update(epoch=epoch, batch=0)
                    return schedule(pairs, epoch)

                def augmented(indices, current=current, fetch=fetch):
                    orientation = (SEED + current["epoch"] * UPDATES + current["batch"]) % 8
                    current["batch"] += 1
                    return augment(fetch(indices), orientation)

                previous_model = model
                starting_sha = (
                    initial_sha if model is None else srnet_groupnorm.parameter_sha(model)
                )

                def resumed(saved: Any = previous_model) -> Any:
                    return saved

                factory = srnet_samepool.network if previous_model is None else resumed
                model, losses = srnet_norm_training.kernel(
                    fetch=augmented,
                    batches=batches,
                    epochs=EPOCHS,
                    seed=SEED,
                    params=srnet_training.settings({"learning_rate": lr}),
                    deadline=deadline,
                    device="cuda:0",
                    factory=factory,
                    validator=srnet_model.validate_tensors,
                )
                if len(losses) != EPOCHS or any(x["updates"] != UPDATES for x in losses):
                    raise ValueError("curriculum update accounting mismatch")
                stages.append(
                    {
                        "name": stage,
                        "learning_rate": lr,
                        "starting_parameter_sha256": starting_sha,
                        "epoch_losses": losses,
                        "learned_parameter_sha256": srnet_groupnorm.parameter_sha(model),
                        "optimizer_reset": True,
                    }
                )
            if model is None:
                raise ValueError("curriculum stages incomplete")
            model.to("cuda:0").eval()
            logits, parity = jpeg_population_control.score(model, reader, probe, deadline)
            reader.verify()
            high.verify()
            deadline()
            return {
                "schema_version": "jpeg-q75-curriculum-learning-v1",
                "status": "completed",
                "architecture": model.architecture,
                "execution": execution,
                "manifest_sha256": MANIFEST_SHA,
                "audit_sha256": AUDIT_SHA,
                "high_manifest_sha256": high_sha,
                "fit_rows": pairs.reshape(-1).tolist(),
                "fit_originals": 48,
                "probe_rows": 120,
                "probe_originals": 24,
                "original_groups_disjoint": True,
                "probe_previously_inspected": True,
                "probe_is_train_role_development": True,
                "validation_test_used": False,
                "initial_parameter_sha256": initial_sha,
                "seed": SEED,
                "stages": stages,
                "optimizer_updates": 2 * EPOCHS * UPDATES,
                "optimizer": {
                    "name": "Adamax",
                    "weight_decay": 0.0001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "foreach": False,
                },
                "paired_d4_augmentation": True,
                "schedule_sha256": hashlib.sha256(
                    b"".join(schedule(pairs, e).tobytes() for e in range(EPOCHS))
                ).hexdigest(),
                "probe_cells": probe_metrics(reader.samples, logits),
                "probe_logits": [
                    {"row": int(i), "logits": logits[int(i)].tolist()} for i in probe.flat
                ],
                "singleton_batch_parity": {"passed": parity, "atol": 1e-4, "rtol": 1e-4},
                "seconds": time.monotonic() - started,
                "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "real_model_trained": True,
                "model_discarded": True,
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
