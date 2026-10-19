"""AudioMNIST v3 PCM residual CNN recipe (never invoked by imports).

The v1 network scaled PCM by 1/32768, so a one-LSB change reached the model as
~3e-5, and BatchNorm saw single-waveform microbatches. A generated positive
control showed no learning even on training rows. v3 keeps the data, rows,
seeds, exposure and metrics; it changes only the CNN inputs, normalization and
optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.nonjpeg_features import pcm16_samples
from core.wav_pilot_training import CNNRecipe

AUDIO_V3_DOMAIN = "audio-pcm16-tlu-parity-gn1d-v3"
# Residual truncation in integer sample units (truncated linear unit).
TLU = 8
CHANNELS = 4
GROUPS = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def inputs(data: bytes) -> np.ndarray:
    """Truncated first/second differences plus centered bit planes 0 and 1."""
    raw = pcm16_samples(data).astype(np.int64)
    first = np.diff(raw, prepend=raw[0])
    second = np.diff(first, prepend=first[0])
    residual = np.clip(np.stack((first, second)), -TLU, TLU) / TLU
    # Two's-complement planes match the embedder's unsigned 16-bit view.
    planes = np.stack((raw & 1, (raw >> 1) & 1)) * 2 - 1
    return np.concatenate((residual, planes)).astype(np.float32)[None]


def network():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AudioMNIST v3 CNN unavailable: optional Torch missing") from exc

    def norm(channels):
        # Batch-independent: identical statistics in training and scoring.
        return torch.nn.GroupNorm(GROUPS, channels, dtype=torch.float32)

    class Conv(torch.nn.Sequential):
        def __init__(self, before, after, stride=1):
            super().__init__(
                torch.nn.Conv1d(before, after, 5, stride=stride, padding=2, dtype=torch.float32),
                norm(after),
                torch.nn.ReLU(),
            )

    class Residual(torch.nn.Module):
        def __init__(self, before, after, stride=1):
            super().__init__()
            self.branch = torch.nn.Sequential(
                Conv(before, after, stride),
                torch.nn.Conv1d(after, after, 5, padding=2, dtype=torch.float32),
                norm(after),
            )
            self.skip = (
                torch.nn.Identity()
                if before == after and stride == 1
                else torch.nn.Conv1d(before, after, 1, stride=stride, dtype=torch.float32)
            )

        def forward(self, values):
            return torch.relu(self.branch(values) + self.skip(values))

    class PCMNetV3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = AUDIO_V3_DOMAIN
            self.body = torch.nn.Sequential(
                Conv(CHANNELS, 32),
                Residual(32, 32),
                Residual(32, 32),
                Residual(32, 64, 2),
                Residual(64, 128, 2),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
            )
            self.classifier = torch.nn.Linear(128, 2, dtype=torch.float32)

        def forward(self, values):
            return self.classifier(self.body(values))

    return PCMNetV3()


def optimizer(torch: Any, model):
    """AdamW with decoupled L2 on weights only, as in v1's decay grouping."""
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 and not name.endswith("bias") else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )


V3_RECIPE = CNNRecipe(AUDIO_V3_DOMAIN, inputs, network, optimizer)
