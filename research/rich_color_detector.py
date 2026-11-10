"""Rich-model style RGB LSB detector: residual co-occurrences + Fisher discriminant (CPU).

Per-channel SRM-lite co-occurrences (first/second-order residuals, four
neighbours; third-order, three) are pooled over channels and horizontal/
vertical scans. Colour features count the joint (R, G, B) residual triplet at
each pixel: independent per-channel +/-1 changes break the strong inter-channel
residual correlation of natural photographs. A ridge-regularised Fisher linear
discriminant over standardised features scores images; zero is the decision
threshold. One detector covers LSB replacement and +/-1 matching at mixed rates.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research import external_rgb_training as ext
from research import spatial_matching_probe as matching

SCHEMA = "rich-color-detector-v1"
TRAIN_SEED = 2026111101
MIN_SIDE = 8
MAX_PIXELS = 16_777_216
RATES = (0.05, 0.1, 0.2, 0.4)
LAMBDAS = (0.03, 0.1, 0.3, 1.0)
WITHIN = (("r1", 2, 4), ("r2", 2, 4), ("r3", 2, 3))
CROSS = (("r1", 1), ("r2", 1), ("r2", 2), ("r3", 2))
CROSS_T = 3
DIMS = sum((2 * t + 1) ** k for _, t, k in WITHIN) + len(CROSS) * (2 * CROSS_T + 1) ** 3


def _residuals(x: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Horizontal and vertical first/second/third-order residuals of one channel."""
    return {
        "r1": (x[:, 1:] - x[:, :-1], (x[1:] - x[:-1]).T),
        "r2": (x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:], (x[:-2] - 2 * x[1:-1] + x[2:]).T),
        "r3": (-x[:, :-3] + 3 * x[:, 1:-2] - 3 * x[:, 2:-1] + x[:, 3:],
               (-x[:-3] + 3 * x[1:-2] - 3 * x[2:-1] + x[3:]).T),
    }


def _cooccurrence(residual: np.ndarray, t: int, k: int) -> np.ndarray:
    """Counts of k consecutive clipped residuals along each row."""
    base = 2 * t + 1
    r = np.clip(residual, -t, t) + t
    n = r.shape[1] - k + 1
    code = sum(r[:, j : j + n] * base ** (k - 1 - j) for j in range(k))
    return np.bincount(np.ravel(code), minlength=base**k)


def features(image: np.ndarray) -> np.ndarray:
    """Normalised co-occurrence features of a uint8 HxWx3 RGB image."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("uint8 HxWx3 image required")
    h, w, c = image.shape
    if c != 3 or min(h, w) < MIN_SIDE or h * w > MAX_PIXELS:
        raise ValueError("RGB image with bounded size required")
    x = image.astype(np.int32)
    channels = [_residuals(x[:, :, i]) for i in range(3)]
    parts = []
    for name, t, k in WITHIN:
        counts = np.sum([_cooccurrence(r, t, k) for ch in channels for r in ch[name]], axis=0)
        parts.append(counts / counts.sum())
    for name, q in CROSS:
        counts = np.zeros((2 * CROSS_T + 1) ** 3, np.int64)
        for d in range(2):
            rs = [np.clip(np.round(ch[name][d] / q), -CROSS_T, CROSS_T).astype(np.int32) + CROSS_T
                  for ch in channels]
            side = 2 * CROSS_T + 1
            counts += np.bincount(
                np.ravel((rs[0] * side + rs[1]) * side + rs[2]), minlength=side**3
            )
        parts.append(counts / counts.sum())
    return np.concatenate(parts).astype(np.float32)


def _pair(task: tuple[np.ndarray, int, int, bool]) -> tuple[np.ndarray, np.ndarray]:
    crop, index, salt, augment = task
    rng = np.random.default_rng([TRAIN_SEED, salt, index])
    if augment:
        crop = matching.d4(crop, int(rng.integers(0, 8)))
    kind = matching.KINDS[int(rng.integers(0, len(matching.KINDS)))]
    rate = RATES[int(rng.integers(0, len(RATES)))]
    return features(crop), features(matching.embed(crop, kind, rate, rng))


def _map(function, items: list, workers: int) -> list:
    if workers == 0:
        return [function(item) for item in items]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(function, items, chunksize=16))


def pair_features(
    covers: Sequence[np.ndarray], *, salt: int, augment: bool = False, workers: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Features of each cover and one fresh mixed-kind/mixed-rate stego of it."""
    if not covers or type(salt) is not int or type(workers) is not int or workers < 0:
        raise ValueError("covers, integer salt and non-negative workers required")
    out = _map(_pair, [(np.asarray(c), i, salt, augment) for i, c in enumerate(covers)], workers)
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


@dataclass(frozen=True)
class Detector:
    """Standardised ridge Fisher discriminant; positive scores mean stego."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float
    ridge: float

    def score(self, feats: np.ndarray) -> np.ndarray:
        if feats.ndim != 2 or feats.shape[1] != self.weights.size:
            raise ValueError("feature matrix width mismatch")
        return ((feats - self.mean) / self.scale) @ self.weights - self.threshold

    def save(self, path: Path) -> None:
        with path.open("xb") as handle:
            np.savez(handle, schema=np.array(SCHEMA), mean=self.mean, scale=self.scale,
                     weights=self.weights, threshold=self.threshold, ridge=self.ridge)

    @classmethod
    def load(cls, path: Path) -> Detector:
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["schema"]) != SCHEMA:
                raise ValueError("detector schema mismatch")
            return cls(stored["mean"], stored["scale"], stored["weights"],
                       float(stored["threshold"]), float(stored["ridge"]))


def fit(covers: np.ndarray, stegos: np.ndarray, ridge: float) -> Detector:
    """Fisher discriminant with ridge ``ridge`` on standardised features."""
    if covers.ndim != 2 or covers.shape != stegos.shape or len(covers) < 2 or not ridge > 0:
        raise ValueError("paired feature matrices and positive ridge required")
    both = np.concatenate([covers, stegos]).astype(np.float64)
    mean, scale = both.mean(0), both.std(0) + 1e-12
    cz, sz = (covers - mean) / scale, (stegos - mean) / scale
    within = np.cov(np.concatenate([cz - cz.mean(0), sz - sz.mean(0)]).T)
    weights = np.linalg.solve(within + ridge * np.eye(len(within)), sz.mean(0) - cz.mean(0))
    threshold = 0.5 * float((cz @ weights).mean() + (sz @ weights).mean())
    return Detector(mean, scale, weights, threshold, float(ridge))


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    n, p = len(negative), len(positive)
    if n == 0 or p == 0:
        raise ValueError("both classes required")
    _, inverse, counts = np.unique(
        np.concatenate([negative, positive]), return_inverse=True, return_counts=True
    )
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[n:].sum() - p * (p + 1) / 2) / (n * p))


def select_ridge(covers, stegos, groups, ridges=LAMBDAS) -> tuple[float, dict[str, float]]:
    """Pick the ridge by AUC on a held-out fifth of source-image groups."""
    hold = np.asarray(groups) % 5 == 0
    if hold.all() or not hold.any():
        raise ValueError("held-out and fitted groups required")
    scores = {}
    for ridge in ridges:
        model = fit(covers[~hold], stegos[~hold], ridge)
        scores[str(ridge)] = auc(model.score(covers[hold]), model.score(stegos[hold]))
    best = max(ridges, key=lambda r: scores[str(r)])
    return float(best), scores


def evaluate(detector: Detector, covers, stegos: Mapping[str, Any],
             workers: int = 0) -> dict[str, dict[str, float]]:
    """AUC, accuracy at zero and paired ordering per stego condition."""
    base = detector.score(np.stack(_map(features, list(covers), workers)))
    out = {}
    for key, values in stegos.items():
        s = detector.score(np.stack(_map(features, list(values), workers)))
        out[key] = {"auc": auc(base, s),
                    "accuracy": 0.5 * float((base < 0).mean() + (s >= 0).mean()),
                    "ordered": float((s > base).mean())}
    return out


def train(bank: Path, out: Path, *, valid: Path | None = None, workers: int = 0,
          limit: int | None = None) -> dict[str, Any]:
    """Fit a detector on a crop bank; optionally score the fixed validation stegos."""
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    crops = ext.load_bank(bank)
    groups = np.load(bank / "groups.npy")
    count = len(crops) if limit is None else limit
    covers, stegos = pair_features([crops[i] for i in range(count)], salt=0, workers=workers)
    ridge, holdout = select_ridge(covers, stegos, groups[:count])
    detector = fit(covers, stegos, ridge)
    detector.save(out / "detector.npz")
    result: dict[str, Any] = {"schema": SCHEMA, "pairs": count, "dims": DIMS, "ridge": ridge,
                              "holdout_auc": holdout, "train_seed": TRAIN_SEED}
    if valid is not None:
        vc, vs = ext.validation(ext.load_bank(valid))
        result["validation"] = evaluate(detector, list(vc), vs, workers)
    result["elapsed_seconds"] = time.monotonic() - started
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bank", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = train(args.bank, args.out, valid=args.valid, workers=args.workers, limit=args.limit)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
