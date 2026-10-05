"""Small opt-in pixel-residual research CNN; not SRNet or a deployed model."""

from __future__ import annotations

import numpy as np

from core.jpeg_pixels import SIDE

ARCHITECTURE = "jpeg-center128-residual-cnn8-v1"
MAX_INFERENCE_BATCH = 64
FILTERS = (
    ((0, 0, 0), (-1, 0, 1), (0, 0, 0)),
    ((0, -1, 0), (0, 0, 0), (0, 1, 0)),
    ((0, 1, 0), (1, -4, 1), (0, 1, 0)),
)


def network():
    """Torch stays optional; fixed filters, no labels/metadata as image inputs."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("pixel CNN requires the optional research extra") from exc

    class ResidualCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.filters: torch.Tensor
            self.register_buffer("filters", torch.tensor(FILTERS, dtype=torch.float32)[:, None])
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2),
                torch.nn.Conv2d(8, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(2),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 1),
            )

        def residuals(self, values):
            # Valid convolution excludes crop padding artifacts from the high-pass stage.
            return torch.clamp(torch.nn.functional.conv2d(values, self.filters), -1, 1)

        def forward(self, values):
            return self.layers(self.residuals(values))

    return ResidualCNN()


def pixel_logits(model, pixels: np.ndarray) -> np.ndarray:
    """Bounded inference on exact raw-cache tensors; no model download/loading."""
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 4
        or pixels.shape[1:] != (1, SIDE, SIDE)
        or not 1 <= pixels.shape[0] <= MAX_INFERENCE_BATCH
    ):
        raise ValueError("invalid bounded uint8 pixel tensor")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("pixel CNN requires the optional research extra") from exc

    if model.training:
        raise ValueError("pixel inference requires eval mode")
    with torch.no_grad():
        values = torch.from_numpy(pixels.astype(np.float32) / 255)
        scores = model(values).detach().cpu().numpy()
    if scores.shape != (len(pixels), 1) or not np.isfinite(scores).all():
        raise ValueError("invalid pixel model output")
    return scores
