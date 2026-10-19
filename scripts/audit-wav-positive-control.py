"""Generated-only positive control for the bound AudioMNIST PCM CNN recipe.

Diagnostic, not a protocol run: no real/reserved media, no checkpoint, reduced
exposure. Asks (1) does the unchanged network/optimizer/microbatch recipe learn
LSB replacement on quiet speech-like PCM, and (2) does eval-mode BatchNorm
(running stats, what the protocol scores with) match per-sample train-mode BN.
"""

import copy
import hashlib
import io
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
VARIANT = sys.argv[4] if len(sys.argv) > 4 else "v1"
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from core import nonjpeg_models, wav_pilot, wav_pilot_v3  # noqa: E402
from core.wav_pilot_training import (  # noqa: E402
    LEARNING_RATE,
    VIRTUAL_BATCH,
    WEIGHT_DECAY,
)

TRAIN_FILES, DEV_FILES, PER_GROUP = 120, 60, 10
SEED = 20261016
prep = json.loads((ROOT / "benchmarks/wav-audiomnist-preparation-20261017.json").read_text())
LENGTHS = [row["frames"] for row in prep["files"]]  # metadata only


def speech_like(rng, n):
    """Quiet voiced utterance with dithered near-silence, peak ~ AudioMNIST (<=2800)."""
    t = np.arange(n) / 48000
    f0 = rng.uniform(90, 250) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(2, 6) * t))
    phase = 2 * np.pi * np.cumsum(f0) / 48000
    voice = sum(
        rng.uniform(0.2, 1) / k * np.sin(k * phase + rng.uniform(0, 6.3)) for k in range(1, 12)
    )
    start, stop = int(n * rng.uniform(0.15, 0.3)), int(n * rng.uniform(0.7, 0.85))
    env = np.zeros(n)
    env[start:stop] = np.hanning(stop - start) ** 0.5
    signal = rng.uniform(500, 2800) * env * voice / np.abs(voice).max()
    signal += rng.normal(0, rng.uniform(0.6, 3), n)  # mic/dither floor
    return np.clip(np.round(signal), -32768, 32767).astype("<i2")


def write_corpus(root):
    rng = np.random.default_rng(SEED)
    records = []
    for role, count in (("train", TRAIN_FILES), ("development", DEV_FILES)):
        for index in range(count):
            speaker = f"gen-{role}-{index // PER_GROUP:02d}"
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(48000)
                audio.writeframes(speech_like(rng, int(rng.choice(LENGTHS))).tobytes())
            raw = buffer.getvalue()
            rel = f"data/{speaker}/{index}.wav"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(raw)
            records.append(
                {
                    "relative_source_path": rel,
                    "role": role,
                    "speaker_group": speaker,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    rows = {}
    for role in ("train", "development"):
        rows[role] = [
            {"source": src, "label": "cover" if recipe is None else "stego", "recipe": recipe}
            for src in records
            if src["role"] == role
            for recipe in (None, *wav_pilot.RECIPES)
        ]
    return rows


def model_inputs(data):
    if VARIANT == "v1":
        return nonjpeg_models.inputs(data, nonjpeg_models.AUDIO_DOMAIN)
    value = wav_pilot_v3.inputs(data)
    if VARIANT == "v3-noplanes":
        value = value.copy()
        value[:, 2:] = 0  # diagnostic ablation: residual channels only
    return value


def new_model():
    if VARIANT == "v1":
        return nonjpeg_models.network(nonjpeg_models.AUDIO_DOMAIN)
    return wav_pilot_v3.network()


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def score(model, root, rows, train_mode_bn):
    model = copy.deepcopy(model)
    model.train(train_mode_bn)
    out = []
    with torch.no_grad():
        for row in rows:
            x = torch.from_numpy(
                model_inputs(wav_pilot.materialize(root, row["source"], row["recipe"]))
            )
            out.append(float(torch.softmax(model(x), 1)[0, 1]))
    return np.asarray(out)


def cells(rows, scores):
    covers = scores[[i for i, r in enumerate(rows) if r["recipe"] is None]]
    result = {"all": auc(covers, scores[[i for i, r in enumerate(rows) if r["recipe"]]])}
    for recipe in wav_pilot.RECIPES:
        stegos = scores[[i for i, r in enumerate(rows) if r["recipe"] == recipe]]
        result["%s-%d%%-p%d" % recipe] = round(auc(covers, stegos), 4)  # noqa: UP031
    result["mean_cover_p"] = round(float(covers.mean()), 4)
    result["mean_stego_p"] = round(
        float(scores[[i for i, r in enumerate(rows) if r["recipe"]]].mean()), 4
    )
    result["ba_at_0.5"] = round(
        0.5
        * (
            float((covers < 0.5).mean())
            + float((scores[[i for i, r in enumerate(rows) if r["recipe"]]] >= 0.5).mean())
        ),
        4,
    )
    return result


def pov_baseline(root, rows):
    """Classical pairs-of-values statistic on plane 0 (lower = more equalized = stego)."""
    stat = []
    for row in rows:
        s = nonjpeg_models.pcm16_samples(wav_pilot.materialize(root, row["source"], row["recipe"]))
        h = np.bincount(s.astype(np.int64) + 32768, minlength=65536).astype(float)
        even, odd = h[0::2], h[1::2]
        mask = (even + odd) > 0
        stat.append(
            -float(np.sum((even[mask] - odd[mask]) ** 2 / (even[mask] + odd[mask])) / len(s))
        )
    return np.asarray(stat)


def main():
    started = time.monotonic()
    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory(prefix="wav-positive-control-") as directory:
        root = Path(directory)
        rows = write_corpus(root)
        train, dev = rows["train"], rows["development"]
        model = new_model()
        if VARIANT == "v1":
            decay = [
                p for n, p in model.named_parameters() if p.ndim > 1 and not n.endswith("bias")
            ]
            rest = [
                p
                for n, p in model.named_parameters()
                if not (p.ndim > 1 and not n.endswith("bias"))
            ]
            opt = torch.optim.SGD(
                [
                    {"params": decay, "weight_decay": WEIGHT_DECAY},
                    {"params": rest, "weight_decay": 0.0},
                ],
                lr=LEARNING_RATE,
            )
        else:
            opt = wav_pilot_v3.optimizer(torch, model)
        weight = {0: len(train) / (2 * TRAIN_FILES), 1: len(train) / (2 * 8 * TRAIN_FILES)}
        rng = np.random.default_rng(SEED ^ 0x4E4A5047)
        epoch_loss, updates = [], 0
        model.train()
        for _ in range(EPOCHS):
            order, losses = rng.permutation(len(train)), []
            for start in range(0, len(order), VIRTUAL_BATCH):
                batch = [train[int(i)] for i in order[start : start + VIRTUAL_BATCH]]
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for row in batch:
                    label = int(row["label"] == "stego")
                    x = torch.from_numpy(
                        model_inputs(wav_pilot.materialize(root, row["source"], row["recipe"]))
                    )
                    loss = torch.nn.functional.cross_entropy(
                        model(x), torch.tensor([label]), reduction="sum"
                    )
                    (loss * weight[label] / len(batch)).backward()
                    total += float(loss.detach()) / len(batch)
                opt.step()
                updates += 1
                losses.append(total)
            epoch_loss.append(round(float(np.mean(losses)), 4))
            print(
                "epoch",
                len(epoch_loss),
                epoch_loss[-1],
                round(time.monotonic() - started),
                flush=True,
            )
        result = {
            "diagnostic": "generated-only positive control; not a protocol run",
            "variant": VARIANT,
            "threads": torch.get_num_threads(),
            "real_media_opened": False,
            "checkpoint_written": False,
            "train_files": TRAIN_FILES,
            "dev_files": DEV_FILES,
            "epochs": EPOCHS,
            "optimizer_updates": updates,
            "presentations": EPOCHS * len(train),
            "epoch_mean_unweighted_ce": epoch_loss,
            "dev_eval_mode_bn_running_stats": cells(dev, score(model, root, dev, False)),
            "dev_train_mode_bn_per_sample": cells(dev, score(model, root, dev, True)),
            "train_eval_mode_bn": cells(train, score(model, root, train, False)),
            "dev_classical_pov_plane0_baseline": cells(dev, pov_baseline(root, dev)),
            "seconds": round(time.monotonic() - started, 1),
        }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


main()
