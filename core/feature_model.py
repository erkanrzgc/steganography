"""Reconstruct an explicit research feature model, with optional train-only scaling."""

from typing import Any


def feature_model(checkpoint: dict[str, Any]) -> Any:
    import torch

    linear = torch.nn.Linear(int(checkpoint["features"]), 1)
    linear.load_state_dict(checkpoint["state_dict"])
    normalization = checkpoint["preprocessing"].get("normalization")
    if normalization is None:
        return linear
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    scale = torch.tensor(normalization["scale"], dtype=torch.float32)
    if (
        normalization.get("method") != "train-only-standardization"
        or mean.shape != (linear.in_features,)
        or scale.shape != mean.shape
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(scale).all())
        or not bool((scale > 0).all())
    ):
        raise ValueError("invalid feature normalization contract")

    class StandardizedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.register_buffer("mean", mean)
            self.register_buffer("scale", scale)

        def forward(self, values):
            return self.linear((values - self.mean) / self.scale)

    return StandardizedLinear()
