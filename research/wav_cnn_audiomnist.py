"""Raw-waveform CNN for AudioMNIST +/-1 LSB matching at 5% and 10% (Colab GPU).

The CPU feature studies left these two cells at 63.88/75.54% calibration BA
with 7,500 fit pairs, flat in the training-set size. This module tries a
different signal model on the same speaker-grouped roles: fixed difference
filters, truncation, a masked convolutional stack and statistics pooling.

- Data: only the 30 opened AudioMNIST speakers of the pinned commit are
  extracted (every file hash-checked); reserved speakers' bytes are never read.
- Roles: 15 fit, 5 calibration and 10 test speakers (500 clips each).
- Training: random 16,384-sample crops; every presentation gets a fresh
  matching payload. A payload curriculum 40 -> 20 -> 10 -> 5% initialises each
  stage from the previous stage's best calibration checkpoint.
- Calibration and test use the exact project embedding
  (``wav_lsb_features.embed``) with the earlier studies' seeds; digests of
  every fixed stego set are checked against the layout file.
- Selection: per target rate, the epoch with the best calibration BA (then
  AUC, then the earlier epoch) and its calibration threshold. Gate .80. The
  40% stage is a positive control: if it misses .80, training stops.
- Test: once, only for frozen choices that pass the gate.

Checkpoints are written atomically after every epoch so a disconnected Colab
runtime resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import wave
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research import power_fisher as power
from research import rich_color_detector as rich
from research import wav_lsb_features as wav

COMMIT = "630d7dab4c040882834de6fa21baf9a60372accd"
URL = f"https://github.com/soerenab/AudioMNIST/archive/{COMMIT}.zip"
MEMBER = re.compile(r"^AudioMNIST-" + COMMIT + r"/data/(\d\d)/(\d)_(\d\d)_(\d{1,2})\.wav$")
ROLES = ("fit", "calibration", "test")
# (payload rate, epochs, peak learning rate) of the curriculum stages.
STAGES: tuple[tuple[float, int, float], ...] = (
    (0.4, 12, 1e-3), (0.2, 12, 5e-4), (0.1, 24, 5e-4), (0.05, 24, 5e-4))
RATES = tuple(stage[0] for stage in STAGES)
TARGETS = (0.1, 0.05)
CONTROL_RATE = 0.4
OLD_CONDITIONS = tuple(f"{k}-{r}" for k in wav.KINDS for r in (0.05, 0.1, 0.2, 0.4))
CELLS = ("matching-0.05", "matching-0.1")
CROP = 16384
BATCH = 32
EVAL_BATCH = 16
WEIGHT_DECAY = 1e-4
TRUNCATION = 8.0
GATE = 0.80
MODEL_SEED = 2026111903
TRAIN_SEED = 2026111904
BOOTSTRAP_SEED = 2026111905
BOOTSTRAP_ROUNDS = 2000
MAX_MEMBER = 1 << 20
DIFFERENCES = ((0, 0, -1, 1, 0), (0, 1, -2, 1, 0), (0, -1, 3, -3, 1), (1, -4, 6, -4, 1))


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_new(path: Path, value: Any) -> None:
    """Write JSON once; an existing file is never replaced."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, allow_nan=False)


def rows(layout: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """Sorted clip rows of one role from the layout's file list."""
    if role not in ROLES:
        raise ValueError("unknown role")
    out = []
    for path in sorted(layout["files"]):
        digit, speaker, repetition = Path(path).stem.split("_")
        if layout["roles"][speaker] == role:
            out.append({"path": path, "speaker": speaker, "digit": int(digit),
                        "repetition": int(repetition)})
    return out


def seed(row: dict[str, Any], rate: float, old_index: dict[str, int]) -> list[int]:
    """Embedding seed: the earlier AudioMNIST studies' rule, new seeds otherwise."""
    cell = f"matching-{rate}"
    if rate not in RATES:
        raise ValueError("unknown payload rate")
    old = old_index.get(row["path"])
    if old is not None:
        return [2026111201, 1, OLD_CONDITIONS.index(cell), old]
    tail = [int(row["speaker"]), row["digit"], row["repetition"]]
    if cell in CELLS:
        return [2026111801, CELLS.index(cell), *tail]
    return [2026111901, RATES.index(rate), *tail]


def extract(archive: Path, out: Path, layout: dict[str, Any]) -> int:
    """Extract and hash-check only the opened speakers' files."""
    expected = layout["files"]
    seen = set()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            match = None if info.is_dir() else MEMBER.match(info.filename)
            if match is None:
                continue
            speaker, digit, owner, index = match.groups()
            relative = f"data/{speaker}/{digit}_{owner}_{index}.wav"
            if relative not in expected:
                continue  # Reserved speaker: its bytes are never read.
            if owner != speaker or info.file_size > MAX_MEMBER:
                raise ValueError(f"unexpected member: {relative}")
            raw = bundle.read(info)
            if sha256(raw) != expected[relative]:
                raise ValueError(f"opened file hash mismatch: {relative}")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            seen.add(relative)
    if seen != set(expected):
        raise ValueError(f"{len(set(expected) - seen)} opened files missing from the archive")
    return len(seen)


def read_pcm(path: Path) -> np.ndarray:
    """Mono 48 kHz PCM16 samples of one WAV file."""
    with wave.open(str(path), "rb") as stream:
        info = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                stream.getcomptype())
        raw = stream.readframes(stream.getnframes())
    if info != (1, 2, 48000, "NONE"):
        raise ValueError(f"unexpected WAV format: {path.name}")
    return np.frombuffer(raw, "<i2").astype(np.int16)


def digest(signals: Sequence[np.ndarray]) -> str:
    h = hashlib.sha256()
    for signal in signals:
        h.update(np.ascontiguousarray(signal, "<i2").tobytes())
    return h.hexdigest()


def stegos(signals: Sequence[np.ndarray], clips: Sequence[dict[str, Any]], rate: float,
           old_index: dict[str, int]) -> list[np.ndarray]:
    """Fixed +/-1 matching stegos with the project embedding and seeds."""
    return [wav.embed(x, "matching", rate, np.random.default_rng(seed(row, rate, old_index)))
            for x, row in zip(signals, clips, strict=True)]


def pack(signals: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    lengths = np.array([len(s) for s in signals], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return {"samples": np.concatenate(signals).astype(np.int16), "offsets": offsets,
            "lengths": lengths}


def unpack(bank: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [bank["samples"][o:o + n] for o, n in zip(bank["offsets"], bank["lengths"],
                                                     strict=True)]


def save_bank(path: Path, signals: Sequence[np.ndarray]) -> None:
    packed = pack(signals)
    with path.open("xb") as stream:
        np.savez(stream, samples=packed["samples"], offsets=packed["offsets"],
                 lengths=packed["lengths"])


def load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {k: stored[k] for k in ("samples", "offsets", "lengths")}


def checked(signals: Sequence[np.ndarray], expected: str, label: str) -> None:
    if digest(signals) != expected:
        raise ValueError(f"{label} digest differs from the layout; check the numpy version")


def prepare(archive: Path, data: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Extract opened speakers and bank fit/calibration covers and calibration stegos."""
    done = data / "prepared.json"
    if done.exists():
        return json.loads(done.read_text())
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != layout["archive_sha256"]:
        print("archive hash differs from the recorded one; per-file hashes decide", flush=True)
    data.mkdir(parents=True, exist_ok=True)
    count = extract(archive, data, layout)
    old = layout["old_index"]
    for role in ("fit", "calibration"):
        clips = rows(layout, role)
        covers = [read_pcm(data / r["path"]) for r in clips]
        checked(covers, layout["digests"]["covers"][role], f"{role} covers")
        save_bank(data / f"covers-{role}.npz", covers)
        if role == "calibration":
            for rate in RATES:
                fixed = stegos(covers, clips, rate, old)
                checked(fixed, layout["digests"]["calibration"][str(rate)], f"stegos {rate}")
                save_bank(data / f"stegos-calibration-{rate}.npz", fixed)
    summary = {"opened_files": count, "numpy": np.__version__, "created_unix": time.time()}
    write_new(done, summary)
    return summary


def metrics(covers: np.ndarray, stegos_: np.ndarray, threshold: float) -> dict[str, float]:
    tnr, tpr = float((covers < threshold).mean()), float((stegos_ >= threshold).mean())
    return {"balanced_accuracy": 0.5 * (tnr + tpr), "auc": rich.auc(covers, stegos_),
            "tpr": tpr, "fpr": 1.0 - tnr, "threshold": float(threshold)}


def calibrate(covers: np.ndarray, stegos_: np.ndarray) -> dict[str, float]:
    return metrics(covers, stegos_, power.calibration_threshold(covers, stegos_))


def better(new: dict[str, Any], old: dict[str, Any] | None) -> bool:
    """Higher calibration BA, then AUC; ties keep the earlier epoch."""
    if old is None:
        return True
    a, b = new["calibration"], old["calibration"]
    return (a["balanced_accuracy"], a["auc"]) > (b["balanced_accuracy"], b["auc"])


def network() -> Any:
    """Difference filters, truncation, masked conv stack and statistics pooling."""
    import torch
    from torch import nn
    from torch.nn import functional

    def layer(cin: int, cout: int, kernel: int, stride: int = 1) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU())

    def mask(h: Any, valid: Any) -> Any:
        keep = torch.arange(h.shape[-1], device=h.device)[None, :] < valid[:, None]
        return h * keep[:, None, :].to(h.dtype)

    class ResidualStatsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hpf = nn.Parameter(
                torch.tensor(DIFFERENCES, dtype=torch.float32)[:, None, :], requires_grad=False)
            self.front = nn.ModuleList([layer(4, 32, 5), layer(32, 32, 5)])
            widths = (32, 64, 128, 256)
            pairs = list(zip((32, *widths[:-1]), widths, strict=True))
            self.same = nn.ModuleList([layer(cin, cout, 3) for cin, cout in pairs])
            self.down = nn.ModuleList([layer(cout, cout, 3, 2) for _, cout in pairs])
            self.head = nn.Sequential(nn.Linear(2 * widths[-1], 128), nn.ReLU(),
                                      nn.Linear(128, 1))

        def forward(self, x: Any, lengths: Any) -> Any:
            valid = lengths
            h = functional.conv1d(mask(x[:, None], valid), self.hpf, padding=2)
            h = mask(torch.clamp(h, -TRUNCATION, TRUNCATION) / TRUNCATION, valid)
            for module in self.front:
                h = mask(module(h), valid)
            for same, down in zip(self.same, self.down, strict=True):
                h = mask(same(h), valid)
                valid = (valid + 1) // 2  # stride-2, kernel-3 output length
                h = mask(down(h), valid)
            keep = mask(torch.ones_like(h[:, :1]), valid)
            count = keep.sum(-1)
            mean = (h * keep).sum(-1) / count
            var = (((h - mean[..., None]) * keep) ** 2).sum(-1) / count
            pooled = torch.cat([mean, torch.sqrt(var + 1e-6)], 1)
            return self.head(pooled)[:, 0]

    return ResidualStatsNet()


def embed_batch(x: Any, rate: float, generator: Any) -> Any:
    """Fresh +/-1 matching on a float batch of int16 values (Bernoulli positions)."""
    import torch

    def draw(high: int) -> Any:
        return torch.randint(0, high, x.shape, generator=generator, device=x.device)

    chosen = torch.rand(x.shape, generator=generator, device=x.device) < rate
    bits, step = draw(2), draw(2) * 2 - 1
    step = torch.where(x >= 32767, -1, torch.where(x <= -32768, 1, step))
    change = chosen & (torch.remainder(x, 2) != bits)
    return x + torch.where(change, step, 0).to(x.dtype)


def crops(bank: dict[str, Any], clips: Any, generator: Any) -> Any:
    """Random CROP-sample windows of the given clips from a device-resident bank."""
    import torch

    room = (bank["lengths"][clips] - CROP + 1).to(torch.float32)
    start = (torch.rand(len(clips), generator=generator, device=room.device) * room).long()
    index = bank["offsets"][clips, None] + start[:, None]
    index = index + torch.arange(CROP, device=room.device)[None, :]
    return bank["samples"][index].to(torch.float32)


def scores(model: Any, signals: Sequence[np.ndarray], device: str) -> np.ndarray:
    """Full-length logits; clips are batched by length and zero-padded under a mask."""
    import torch

    order = np.argsort([len(s) for s in signals], kind="stable")
    out = np.empty(len(signals))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), EVAL_BATCH):
            chosen = order[start:start + EVAL_BATCH]
            width = max(len(signals[i]) for i in chosen)
            batch = np.zeros((len(chosen), width), np.float32)
            for row, i in enumerate(chosen):
                batch[row, :len(signals[i])] = signals[i]
            lengths = torch.tensor([len(signals[i]) for i in chosen], device=device)
            out[chosen] = model(torch.from_numpy(batch).to(device), lengths).double().cpu().numpy()
    return out


def save_atomic(torch: Any, value: Any, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _optimizer(torch: Any, model: Any, state: dict[str, Any], out: Path,
               steps: int) -> tuple[Any, Any]:
    """Stage optimizer; a fresh stage starts from the previous stage's best weights."""
    _, epochs, peak = STAGES[state["stage"]]
    if state["epoch"] == 0 and state["stage"] > 0:
        previous = out / f"best-{RATES[state['stage'] - 1]}.pt"
        model.load_state_dict(torch.load(previous, map_location="cpu", weights_only=False))
    optimizer = torch.optim.AdamW(model.parameters(), peak, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * steps, eta_min=peak / 20)
    if state["epoch"] > 0:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    return optimizer, scheduler


def _epoch(torch: Any, model: Any, fit: dict[str, Any], rate: float, generator: Any,
           optimizer: Any, scheduler: Any) -> float:
    """One pass over random crops of the fit clips with fresh payloads."""
    model.train()
    device = fit["samples"].device
    order = torch.randperm(len(fit["lengths"]), generator=generator, device=device)
    total, steps = 0.0, len(order) // BATCH
    for step in range(steps):
        x = crops(fit, order[step * BATCH:(step + 1) * BATCH], generator)
        batch = torch.cat([x, embed_batch(x, rate, generator)])
        target = torch.cat([torch.zeros(len(x)), torch.ones(len(x))]).to(device)
        lengths = torch.full((len(batch),), CROP, device=device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(batch, lengths), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total = total + loss.detach()
    return float(total) / steps


def _control_failed(out: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    """Stop after the 40% stage if it misses the gate (the recipe does not learn)."""
    control = state["best"].get(str(CONTROL_RATE))
    if state["stage"] == 0 or control["calibration"]["balanced_accuracy"] >= GATE:
        return None
    path = out / "control-failed.json"
    if not path.exists():
        write_new(path, {"status": "positive_control_failed", "control": control})
    return json.loads(path.read_text())


def train(data: Path, out: Path, device: str = "cuda") -> dict[str, Any]:
    """Run or resume the curriculum; freeze the per-target choices at the end."""
    import torch

    out.mkdir(parents=True, exist_ok=True)
    fit = {k: torch.from_numpy(v).to(device) for k, v in load_bank(data / "covers-fit.npz").items()}
    covers = unpack(load_bank(data / "covers-calibration.npz"))
    fixed = {r: unpack(load_bank(data / f"stegos-calibration-{r}.npz")) for r in RATES}
    path = out / "state.pt"
    torch.manual_seed(MODEL_SEED)
    model = network().to(device)
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED)
    state: dict[str, Any] = {"stage": 0, "epoch": 0, "history": [], "best": {}}
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        generator.set_state(state["generator"])
    optimizer = None
    while state["stage"] < len(STAGES):
        failed = _control_failed(out, state)
        if failed is not None:
            return failed
        rate, epochs, _ = STAGES[state["stage"]]
        if optimizer is None:
            optimizer, scheduler = _optimizer(torch, model, state, out,
                                              len(fit["lengths"]) // BATCH)
        started = time.monotonic()
        loss = _epoch(torch, model, fit, rate, generator, optimizer, scheduler)
        record = {"rate": rate, "epoch": state["epoch"], "loss": loss,
                  "calibration": calibrate(scores(model, covers, device),
                                           scores(model, fixed[rate], device)),
                  "seconds": time.monotonic() - started}
        state["history"].append(record)
        if better(record, state["best"].get(str(rate))):
            state["best"][str(rate)] = record
            save_atomic(torch, model.state_dict(), out / f"best-{rate}.pt")
        print(f"rate {rate} epoch {state['epoch']} loss {loss:.4f} "
              f"cal BA {record['calibration']['balanced_accuracy']:.4f} "
              f"AUC {record['calibration']['auc']:.4f} {record['seconds']:.0f}s", flush=True)
        state["epoch"] += 1
        state.update(model=model.state_dict(), generator=generator.get_state(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        if state["epoch"] == epochs:
            state["stage"], state["epoch"], optimizer = state["stage"] + 1, 0, None
        save_atomic(torch, state, path)
        (out / "history.json").write_text(json.dumps(state["history"], indent=1))
    failed = _control_failed(out, state)
    return failed if failed is not None else freeze(out, state["best"], torch)


def freeze(out: Path, best: dict[str, Any], torch: Any) -> dict[str, Any]:
    path = out / "choice-freeze.json"
    if path.exists():
        return json.loads(path.read_text())
    choices = {}
    for rate in TARGETS:
        record = best[str(rate)]
        weights = (out / f"best-{rate}.pt").read_bytes()
        choices[str(rate)] = {**record, "model_sha256": sha256(weights),
                              "passes_gate": record["calibration"]["balanced_accuracy"] >= GATE}
    frozen = {"choices": choices, "gate": GATE, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "created_unix": time.time()}
    write_new(path, frozen)
    return frozen


def intervals(covers: np.ndarray, stegos_: np.ndarray, threshold: float,
              speakers: Sequence[str]) -> dict[str, list[float]]:
    """Pair and speaker-cluster bootstrap 95% intervals of the test BA."""
    correct = ((covers < threshold).astype(float) + (stegos_ >= threshold)) / 2
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairs = correct[rng.integers(len(correct), size=(BOOTSTRAP_ROUNDS, len(correct)))].mean(1)
    groups = sorted(set(speakers))
    members = [np.flatnonzero(np.asarray(speakers) == g) for g in groups]
    clustered = []
    for draw in rng.integers(len(groups), size=(BOOTSTRAP_ROUNDS, len(groups))):
        clustered.append(correct[np.concatenate([members[g] for g in draw])].mean())
    return {"pairs": np.quantile(pairs, [0.025, 0.975]).tolist(),
            "speakers": np.quantile(clustered, [0.025, 0.975]).tolist()}


def evaluate(data: Path, out: Path, layout: dict[str, Any],
             device: str = "cuda") -> dict[str, Any]:
    """Score the test speakers once, only for frozen choices that pass the gate."""
    import torch

    frozen = json.loads((out / "choice-freeze.json").read_text())
    result_path = out / "test-result.json"
    if result_path.exists():
        raise FileExistsError("the test was already evaluated once")
    passing = [r for r in TARGETS if frozen["choices"][str(r)]["passes_gate"]]
    result: dict[str, Any] = {str(r): {"status": "not_evaluated_calibration_gate"}
                              for r in TARGETS if r not in passing}
    if passing:
        calibration = rows(layout, "calibration")
        reference = unpack(load_bank(data / "covers-calibration.npz"))
        clips = rows(layout, "test")  # Files were hash-checked at extraction.
        covers = [read_pcm(data / r["path"]) for r in clips]
        for rate in passing:
            choice = frozen["choices"][str(rate)]
            weights = out / f"best-{rate}.pt"
            if sha256(weights.read_bytes()) != choice["model_sha256"]:
                raise ValueError("frozen model changed")
            # The embedding in this runtime must still reproduce the recorded stegos.
            checked(stegos(reference, calibration, rate, layout["old_index"]),
                    layout["digests"]["calibration"][str(rate)], f"stegos {rate}")
            fixed = stegos(covers, clips, rate, layout["old_index"])
            model = network().to(device)
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=False))
            c, s = scores(model, covers, device), scores(model, fixed, device)
            threshold = choice["calibration"]["threshold"]
            result[str(rate)] = {
                "status": "evaluated_once", "test": metrics(c, s, threshold),
                "intervals_95": intervals(c, s, threshold, [r["speaker"] for r in clips]),
                "test_pairs": len(clips)}
    write_new(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "train", "test"))
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    layout = json.loads(args.layout.read_text())
    if args.command == "prepare":
        print(prepare(args.archive, args.data, layout))
    elif args.command == "train":
        print(json.dumps(train(args.data, args.out, args.device), indent=1))
    else:
        print(json.dumps(evaluate(args.data, args.out, layout, args.device), indent=1))


if __name__ == "__main__":
    main()
