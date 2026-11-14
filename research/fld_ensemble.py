"""Random-subspace Fisher ensemble (Kodovsky, Fridrich & Holub 2012) on paired features.

Each base learner is a ridge Fisher discriminant fitted on a random feature
subspace and a bootstrap sample of cover/stego pairs. The ensemble score is the
mean of the learners' tanh-squashed, spread-normalised projections; zero is the
decision threshold. Stored as plain ``npz`` without pickle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "fld-ensemble-v1"


@dataclass(frozen=True)
class Ensemble:
    """Fitted ensemble; positive scores mean stego."""

    mean: np.ndarray
    scale: np.ndarray
    columns: np.ndarray  # (learners, d_sub) feature indices
    weights: np.ndarray  # (learners, d_sub)
    thresholds: np.ndarray  # (learners,)
    spreads: np.ndarray  # (learners,)

    def score(self, feats: np.ndarray, *, batch_rows: int = 256) -> np.ndarray:
        """Score bounded row batches without an N x learners x subspace tensor."""
        if not isinstance(feats, np.ndarray) or feats.ndim != 2:
            raise ValueError("feature matrix width mismatch")
        if feats.shape[1] != self.mean.size:
            raise ValueError("feature matrix width mismatch")
        if type(batch_rows) is not int or not 1 <= batch_rows <= 4096:
            raise ValueError("batch_rows must be an integer from 1 to 4096")
        scores = np.empty(len(feats), dtype=np.float64)
        for start in range(0, len(feats), batch_rows):
            z = (feats[start:start + batch_rows] - self.mean) / self.scale
            votes = np.zeros(len(z))
            for columns, weights, threshold, spread in zip(
                self.columns, self.weights, self.thresholds, self.spreads, strict=True
            ):
                votes += np.tanh((z[:, columns] @ weights - threshold) / spread)
            scores[start:start + len(z)] = votes / len(self.columns)
        return scores

    def save(self, path: Path) -> None:
        with path.open("xb") as handle:
            np.savez(handle, schema=np.array(SCHEMA), mean=self.mean, scale=self.scale,
                     columns=self.columns, weights=self.weights, thresholds=self.thresholds,
                     spreads=self.spreads)

    @classmethod
    def load(cls, path: Path) -> Ensemble:
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["schema"]) != SCHEMA:
                raise ValueError("ensemble schema mismatch")
            return cls(*(stored[k] for k in ("mean", "scale", "columns", "weights",
                                             "thresholds", "spreads")))


def fit(covers: np.ndarray, stegos: np.ndarray, *, d_sub: int = 1000, learners: int = 51,
        ridge: float = 1.0, seed: int = 0) -> Ensemble:
    """Fit ``learners`` subspace Fisher discriminants on paired cover/stego features."""
    if covers.ndim != 2 or covers.shape != stegos.shape or len(covers) < 2:
        raise ValueError("paired feature matrices required")
    if d_sub < 1 or learners < 1 or not ridge > 0:
        raise ValueError("positive subspace size, learner count and ridge required")
    rng = np.random.default_rng(seed)
    both = np.concatenate([covers, stegos]).astype(np.float64)
    mean, scale = both.mean(0), both.std(0) + 1e-9
    cz, sz = (covers - mean) / scale, (stegos - mean) / scale
    n, d = cz.shape
    k = min(d_sub, d)
    cols, weights, thresholds, spreads = [], [], [], []
    for _ in range(learners):
        c_idx = np.sort(rng.choice(d, k, replace=False))
        rows = rng.integers(0, n, n)
        c, s = cz[rows][:, c_idx], sz[rows][:, c_idx]
        within = np.atleast_2d(np.cov(np.concatenate([c - c.mean(0), s - s.mean(0)]).T))
        w = np.linalg.solve(within + ridge * np.eye(k), s.mean(0) - c.mean(0))
        pc, ps = c @ w, s @ w
        cols.append(c_idx)
        weights.append(w)
        thresholds.append(0.5 * (pc.mean() + ps.mean()))
        spreads.append(0.5 * (pc.std() + ps.std()) + 1e-12)
    return Ensemble(mean, scale, np.array(cols), np.array(weights), np.array(thresholds),
                    np.array(spreads))
