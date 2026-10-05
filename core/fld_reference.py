"""Bounded numeric FLD inference; never deserialize upstream Python objects."""

from dataclasses import dataclass

import numpy as np

from core.jpeg_jrm import DIMENSIONS

LEARNERS = 31
SUBSPACE = 256


@dataclass(frozen=True)
class FLDReference:
    subspaces: np.ndarray
    weights: np.ndarray
    biases: np.ndarray

    def __post_init__(self):
        if (
            self.subspaces.shape != (LEARNERS, SUBSPACE)
            or self.subspaces.dtype.kind not in "iu"
            or np.any(self.subspaces < 0)
            or np.any(self.subspaces >= DIMENSIONS)
            or any(len(np.unique(row)) != SUBSPACE for row in self.subspaces)
            or self.weights.shape != (LEARNERS, SUBSPACE)
            or self.biases.shape != (LEARNERS,)
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.biases).all()
        ):
            raise ValueError("invalid bounded FLD parameters")
        # Own read-only copies: callers cannot mutate inference through aliases.
        for name in ("subspaces", "weights", "biases"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            features.ndim != 2
            or features.shape[1] != DIMENSIONS
            or not 1 <= len(features) <= 4000
            or not np.isfinite(features).all()
            or np.any(features < 0)
            or np.any(features > 1)
        ):
            raise ValueError("invalid bounded FLD input")
        votes = np.zeros(len(features), dtype=np.float64)
        for space, weight, bias in zip(self.subspaces, self.weights, self.biases, strict=True):
            with np.errstate(over="ignore", invalid="ignore"):
                margins = features[:, space].astype(np.float64) @ weight - bias
            if not np.isfinite(margins).all():
                raise ValueError("nonfinite FLD margins")
            votes += np.sign(margins)
        return (votes / LEARNERS + 1) / 2
