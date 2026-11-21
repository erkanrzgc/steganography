"""Seed/crop ensemble with test-time augmentation for the AudioMNIST CNN (v4).

The v3 CNN (all 60 speakers) missed matching 10% by 0.49 points on its fresh
test speakers. v4 keeps the v3 data, roles and gate and changes the scorer:

- Members: two extra 10% models fine-tuned from the v3 20% checkpoint with
  new seeds; the second uses 32,768-sample crops. Each member's epoch is the
  best calibration BA, as in v3.
- Test-time augmentation (TTA): the mean logit over the identity, sign-flipped,
  time-reversed and both-flipped signal. +/-1 matching is symmetric under
  both maps, so the stego/cover contrast is unchanged.
- Candidates per target rate (single v3 model or ensemble, with or without
  TTA) are compared by leave-one-calibration-speaker-out BA only; ties prefer
  the higher worst speaker, then the simpler candidate. Gate .80.
- Test: once per passing rate. The test speakers were already scored once by
  v3, so this second look is adaptive evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research import wav_cnn_audiomnist as cnn

RATE = 0.1
EPOCHS = 24
PEAK = 5e-4
MEMBERS: tuple[dict[str, Any], ...] = (
    {"name": "seed-b", "seed": 2026112111, "crop": 16384, "batch": 32},
    {"name": "crop-32k", "seed": 2026112112, "crop": 32768, "batch": 16},
)
TRANSFORMS = ("identity", "negate", "reverse", "negate-reverse")
# (name, member models, transforms); order breaks ties toward simpler scorers.
CANDIDATES: dict[float, tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]] = {
    0.1: (("v3", ("v3",), ("identity",)),
          ("v3-tta", ("v3",), TRANSFORMS),
          ("ensemble", ("v3", "seed-b", "crop-32k"), ("identity",)),
          ("ensemble-tta", ("v3", "seed-b", "crop-32k"), TRANSFORMS)),
    0.05: (("v3", ("v3",), ("identity",)),
           ("v3-tta", ("v3",), TRANSFORMS)),
}


def transform(signal: np.ndarray, name: str) -> np.ndarray:
    """Float copy of a clip under one TTA map (no int16 overflow on negation)."""
    x = signal.astype(np.float32)
    if name not in TRANSFORMS:
        raise ValueError("unknown transform")
    if "negate" in name:
        x = -x
    if "reverse" in name:
        x = x[::-1]
    return np.ascontiguousarray(x)


def crops(bank: dict[str, Any], clips: Any, generator: Any, crop: int) -> Any:
    """Random windows of ``crop`` samples (whole clip padded when shorter)."""
    import torch

    lengths = bank["lengths"][clips]
    room = torch.clamp(lengths - crop + 1, min=1).to(torch.float32)
    start = (torch.rand(len(clips), generator=generator, device=room.device) * room).long()
    offset = torch.arange(crop, device=room.device)[None, :]
    inside = offset < lengths[:, None]
    index = bank["offsets"][clips, None] + start[:, None] + torch.where(inside, offset, 0)
    return torch.where(inside, bank["samples"][index].to(torch.float32), 0.0), \
        torch.clamp(lengths, max=crop)


def _epoch(torch: Any, model: Any, fit: dict[str, Any], spec: dict[str, Any], generator: Any,
           optimizer: Any, scheduler: Any) -> float:
    """One pass over random crops with fresh 10% matching payloads."""
    model.train()
    device = fit["samples"].device
    order = torch.randperm(len(fit["lengths"]), generator=generator, device=device)
    batch, total = spec["batch"], 0.0
    steps = len(order) // batch
    for step in range(steps):
        x, valid = crops(fit, order[step * batch:(step + 1) * batch], generator, spec["crop"])
        stego = cnn.embed_batch(x, RATE, generator)
        inside = torch.arange(x.shape[1], device=device)[None, :] < valid[:, None]
        pair = torch.cat([x, torch.where(inside, stego, 0.0)])
        target = torch.cat([torch.zeros(len(x)), torch.ones(len(x))]).to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(pair, torch.cat([valid, valid])), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total = total + loss.detach()
    return float(total) / steps


def train_member(data: Path, base: Path, out: Path, spec: dict[str, Any],
                 device: str = "cuda") -> dict[str, Any]:
    """Fine-tune one 10% member from the v3 20% checkpoint; resumable per epoch."""
    import torch

    folder = out / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "best.json"
    if done.exists():
        return json.loads(done.read_text())
    fit = {k: torch.from_numpy(v).to(device)
           for k, v in cnn.load_bank(data / "covers-fit.npz").items()}
    covers = cnn.unpack(cnn.load_bank(data / "covers-calibration.npz"))
    fixed = cnn.unpack(cnn.load_bank(data / f"stegos-calibration-{RATE}.npz"))
    model = cnn.network().to(device)
    init = base / "best-0.2.pt"
    model.load_state_dict(torch.load(init, map_location=device, weights_only=False))
    generator = torch.Generator(device=device).manual_seed(spec["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), PEAK, weight_decay=cnn.WEIGHT_DECAY)
    steps = len(fit["lengths"]) // spec["batch"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * steps,
                                                           eta_min=PEAK / 20)
    path = folder / "state.pt"
    state: dict[str, Any] = {"epoch": 0, "history": [], "best": None,
                             "init_sha256": cnn.sha256(init.read_bytes())}
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        generator.set_state(state["generator"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    while state["epoch"] < EPOCHS:
        started = time.monotonic()
        loss = _epoch(torch, model, fit, spec, generator, optimizer, scheduler)
        c, s = cnn.scores(model, covers, device), cnn.scores(model, fixed, device)
        record = {"member": spec["name"], "epoch": state["epoch"], "loss": loss,
                  "calibration": cnn.calibrate(c, s), "seconds": time.monotonic() - started}
        state["history"].append(record)
        if cnn.better(record, state["best"]):
            state["best"] = record
            cnn.save_atomic(torch, model.state_dict(), folder / "best.pt")
        print(f"{spec['name']} epoch {state['epoch']} loss {loss:.4f} "
              f"cal BA {record['calibration']['balanced_accuracy']:.4f} "
              f"AUC {record['calibration']['auc']:.4f} {record['seconds']:.0f}s", flush=True)
        state["epoch"] += 1
        state.update(model=model.state_dict(), generator=generator.get_state(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        cnn.save_atomic(torch, state, path)
        (folder / "history.json").write_text(json.dumps(state["history"], indent=1))
    best = {**state["best"], "init_sha256": state["init_sha256"],
            "model_sha256": cnn.sha256((folder / "best.pt").read_bytes())}
    cnn.write_new(done, best)
    return best


def weights(base: Path, out: Path, name: str, rate: float) -> Path:
    return base / f"best-{rate}.pt" if name == "v3" else out / name / "best.pt"


def score_set(signals: Sequence[np.ndarray], paths: dict[str, Path], members: Sequence[str],
              transforms: Sequence[str], device: str) -> np.ndarray:
    """Mean logit over members and TTA maps."""
    import torch

    total = np.zeros(len(signals))
    for name in members:
        model = cnn.network().to(device)
        model.load_state_dict(torch.load(paths[name], map_location=device, weights_only=False))
        for t in transforms:
            total += cnn.scores(model, [transform(x, t) for x in signals], device)
    return total / (len(members) * len(transforms))


def calibration_parts(out: Path, rate: float, signals: dict[str, list[np.ndarray]],
                      paths: dict[str, Path], device: str
                      ) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Calibration logits per (member, transform), cached as .npy for resume and audit."""
    parts: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for m in sorted(paths):
        for t in TRANSFORMS:
            parts[m, t] = {}
            for kind, clips in signals.items():
                path = out / "scores" / f"{rate}-{m}-{t}-{kind}.npy"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, score_set(clips, paths, (m,), (t,), device))
                parts[m, t][kind] = np.asarray(np.load(path))
    return parts


def compare(parts: dict[tuple[str, str], dict[str, np.ndarray]], rate: float,
            speakers: Sequence[str]) -> list[dict[str, Any]]:
    """LOSO record of every candidate scorer for one rate."""
    records = []
    for order, (name, members, transforms) in enumerate(CANDIDATES[rate]):
        c = np.mean([parts[m, t]["covers"] for m in members for t in transforms], axis=0)
        s = np.mean([parts[m, t]["stegos"] for m in members for t in transforms], axis=0)
        rule = cnn.threshold_rule(c, s, speakers)
        worst = min((r["loso_worst"] for r in rule["records"] if r["rule"] == rule["rule"]),
                    default=rule["loso_ba"])
        records.append({"candidate": name, "order": order, "members": list(members),
                        "transforms": list(transforms), "loso_ba": rule["loso_ba"],
                        "loso_worst": worst, "threshold_rule": rule,
                        "pooled": cnn.calibrate(c, s)})
    return records


def freeze(data: Path, base: Path, out: Path, speakers: Sequence[str],
           device: str = "cuda") -> dict[str, Any]:
    """Choose one scorer per rate by LOSO calibration BA and freeze it."""
    path = out / "choice-freeze.json"
    if path.exists():
        return json.loads(path.read_text())
    covers = cnn.unpack(cnn.load_bank(data / "covers-calibration.npz"))
    choices: dict[str, Any] = {}
    for rate, candidates in CANDIDATES.items():
        fixed = cnn.unpack(cnn.load_bank(data / f"stegos-calibration-{rate}.npz"))
        names = sorted({m for _, members, _ in candidates for m in members})
        paths = {m: weights(base, out, m, rate) for m in names}
        parts = calibration_parts(out, rate, {"covers": covers, "stegos": fixed}, paths, device)
        records = compare(parts, rate, speakers)
        best = max(records, key=lambda r: (r["loso_ba"], r["loso_worst"], -r["order"]))
        choices[str(rate)] = {
            "candidates": records, "selected": best["candidate"],
            "members": best["members"], "transforms": best["transforms"],
            "threshold": best["threshold_rule"]["threshold"],
            "model_sha256": {m: cnn.sha256(paths[m].read_bytes()) for m in best["members"]},
            "passes_gate": best["loso_ba"] >= cnn.GATE}
    frozen = {"choices": choices, "gate": cnn.GATE, "created_unix": time.time()}
    cnn.write_new(path, frozen)
    return frozen


def evaluate(data: Path, base: Path, out: Path, layout: dict[str, Any],
             device: str = "cuda") -> dict[str, Any]:
    """Score the test speakers once per passing rate (adaptive second look)."""
    frozen = json.loads((out / "choice-freeze.json").read_text())
    result_path = out / "test-result.json"
    if result_path.exists():
        raise FileExistsError("the test was already evaluated once")
    result: dict[str, Any] = {}
    clips = cnn.rows(layout, "test")
    covers: list[np.ndarray] | None = None
    for key, choice in frozen["choices"].items():
        if not choice["passes_gate"]:
            result[key] = {"status": "not_evaluated_calibration_gate"}
            continue
        rate = float(key)
        paths = {m: weights(base, out, m, rate) for m in choice["members"]}
        if {m: cnn.sha256(p.read_bytes()) for m, p in paths.items()} != choice["model_sha256"]:
            raise ValueError("frozen model changed")
        reference = cnn.unpack(cnn.load_bank(data / "covers-calibration.npz"))
        cnn.checked(cnn.stegos(reference, cnn.rows(layout, "calibration"), rate,
                               layout["old_index"]),
                    layout["digests"]["calibration"][key], f"stegos {key}")
        if covers is None:
            covers = [cnn.read_pcm(data / r["path"]) for r in clips]
        fixed = cnn.stegos(covers, clips, rate, layout["old_index"])
        c = score_set(covers, paths, choice["members"], choice["transforms"], device)
        s = score_set(fixed, paths, choice["members"], choice["transforms"], device)
        result[key] = {
            "status": "evaluated_once", "evidence": "second look at the v3 test speakers",
            "scorer": choice["selected"], "test": cnn.metrics(c, s, choice["threshold"]),
            "intervals_95": cnn.intervals(c, s, choice["threshold"],
                                          [r["speaker"] for r in clips]),
            "test_pairs": len(clips)}
    cnn.write_new(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "freeze", "test"))
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    layout = json.loads(args.layout.read_text())
    if args.command == "train":
        for spec in MEMBERS:
            print(json.dumps(train_member(args.data, args.base, args.out, spec, args.device)))
    elif args.command == "freeze":
        speakers = [r["speaker"] for r in cnn.rows(layout, "calibration")]
        print(json.dumps(freeze(args.data, args.base, args.out, speakers, args.device),
                         indent=1))
    else:
        print(json.dumps(evaluate(args.data, args.base, args.out, layout, args.device),
                         indent=1))


if __name__ == "__main__":
    main()
