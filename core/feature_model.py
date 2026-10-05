"""Reconstruct an explicit research feature model, with optional train-only scaling."""

from typing import Any


def feature_model(checkpoint: dict[str, Any]) -> Any:
    import torch

    linear = torch.nn.Linear(int(checkpoint["features"]), 1)
    linear.load_state_dict(checkpoint["state_dict"])
    arithmetic = checkpoint["preprocessing"].get("inference_arithmetic", "float32")
    if arithmetic not in {"float32", "float64"}:
        raise ValueError("invalid inference arithmetic contract")
    dtype = torch.float64 if arithmetic == "float64" else torch.float32
    linear = linear.to(dtype=dtype)
    normalization = checkpoint["preprocessing"].get("normalization")
    if normalization is None and arithmetic == "float32":
        return linear
    mean = torch.tensor(normalization["mean"], dtype=dtype) if normalization else None
    scale = torch.tensor(normalization["scale"], dtype=dtype) if normalization else None
    if normalization is not None and (
        mean is None
        or scale is None
        or normalization.get("method") != "train-only-standardization"
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
            values = values.to(dtype=dtype)
            if self.mean is not None:
                values = (values - self.mean) / self.scale
            # Accumulate opt-in double arithmetic; keep the existing float I/O.
            return self.linear(values).to(dtype=torch.float32)

    return StandardizedLinear()
