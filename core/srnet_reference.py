"""Independent NumPy float64 evaluation of frozen SRNet numeric state."""

from __future__ import annotations

import numpy as np

from core.srnet_model import validate

ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-5
SCORE_TOLERANCE = 1e-6


def convolution(values, weights, bias, *, stride=1):
    kernel = weights.shape[-1]
    padding = kernel // 2
    padded = np.pad(values, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    return (
        np.einsum("nchwkl,ockl->nohw", windows, weights, optimize=True) + bias[None, :, None, None]
    )


def average_reduce(values):
    height, width = values.shape[-2:]
    padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)))
    valid = np.pad(np.ones((height, width)), 1)
    patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))[:, :, ::2, ::2]
    counts = np.lib.stride_tricks.sliding_window_view(valid, (3, 3))[::2, ::2].sum(axis=(-2, -1))
    return patches.sum(axis=(-2, -1)) / counts


def reference_logits(arrays, pixels):
    validate(arrays)
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (1, 1, 256, 256)
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("reference requires one bounded float256 sample")
    state = {
        k: v.astype(np.float64) for k, v in arrays.items() if not k.endswith("num_batches_tracked")
    }

    def block(values, name, relu=False, stride=1):
        values = convolution(
            values, state[name + ".0.weight"], state[name + ".0.bias"], stride=stride
        )
        mean = state[name + ".1.running_mean"][None, :, None, None]
        variance = state[name + ".1.running_var"][None, :, None, None]
        gamma = state[name + ".1.weight"][None, :, None, None]
        beta = state[name + ".1.bias"][None, :, None, None]
        result = (values - mean) / np.sqrt(variance + 1e-5) * gamma + beta
        return np.maximum(result, 0) if relu else result

    values = pixels.astype(np.float64)
    for i in range(2):
        values = block(values, f"front.{i}", relu=True)
    for i in range(2, 7):
        branch = block(block(values, f"front.{i}.branch.0", relu=True), f"front.{i}.branch.1")
        values = values + branch
    for i in range(4):
        branch = block(block(values, f"middle.{i}.branch.0", relu=True), f"middle.{i}.branch.1")
        values = average_reduce(branch) + block(values, f"middle.{i}.skip", stride=2)
    values = block(block(values, "tail.0", relu=True), "tail.1").mean(axis=(-2, -1))
    result = values @ state["classifier.weight"].T
    if not np.isfinite(result).all():
        raise ValueError("reference produced nonfinite logits")
    return result


def compare(native, reference):
    if (
        not isinstance(native, np.ndarray)
        or not isinstance(reference, np.ndarray)
        or native.shape != (1, 2)
        or reference.shape != (1, 2)
        or not np.isfinite(native).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("invalid parity logits")
    error = np.abs(native - reference)
    native_score = np.exp(-np.logaddexp(0.0, native[:, 0].astype(np.float64) - native[:, 1]))
    reference_score = np.exp(
        -np.logaddexp(0.0, reference[:, 0].astype(np.float64) - reference[:, 1])
    )
    score_difference = float(np.max(np.abs(native_score - reference_score)))
    decisions_equal = bool(
        np.array_equal(native[:, 1] >= native[:, 0], reference[:, 1] >= reference[:, 0])
    )
    return {
        "maximum_logit_difference": float(error.max()),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "maximum_score_difference": score_difference,
        "score_tolerance": SCORE_TOLERANCE,
        "decisions_equal": decisions_equal,
        "passed": bool(
            np.all(error <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(reference))
            and decisions_equal
            and score_difference <= SCORE_TOLERANCE
        ),
    }
