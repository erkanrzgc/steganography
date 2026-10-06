"""Optional SRNet-style architecture, not a trained or qualified detector."""

from __future__ import annotations

import numpy as np

ARCHITECTURE = "srnet-gray12-cpu-v1"
SIDES = (128, 256)
MAX_BATCH = 4


def network():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SRNet preparation requires the optional research extra") from exc

    class ConvBN(torch.nn.Sequential):
        def __init__(self, before, after, kernel=3, stride=1, relu=False):
            modules = [
                torch.nn.Conv2d(
                    before,
                    after,
                    kernel,
                    stride=stride,
                    padding=kernel // 2,
                    dtype=torch.float32,
                    device="cpu",
                ),
                torch.nn.BatchNorm2d(
                    after, eps=1e-5, momentum=0.1, dtype=torch.float32, device="cpu"
                ),
            ]
            if relu:
                modules.append(torch.nn.ReLU())
            super().__init__(*modules)

    class Residual(torch.nn.Module):
        def __init__(self, before, after, down=False):
            super().__init__()
            self.branch = torch.nn.Sequential(
                ConvBN(before, after, relu=True),
                ConvBN(after, after),
            )
            self.pool = (
                torch.nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
                if down
                else torch.nn.Identity()
            )
            self.skip = ConvBN(before, after, kernel=1, stride=2) if down else torch.nn.Identity()

        def forward(self, values):
            # No activation after the residual addition.
            return self.pool(self.branch(values)) + self.skip(values)

    class SRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = ARCHITECTURE
            self.front = torch.nn.Sequential(
                ConvBN(1, 64, relu=True),
                ConvBN(64, 16, relu=True),
                *(Residual(16, 16) for _ in range(5)),
            )
            self.middle = torch.nn.Sequential(
                *(
                    Residual(before, after, down=True)
                    for before, after in ((16, 16), (16, 64), (64, 128), (128, 256))
                ),
            )
            self.tail = torch.nn.Sequential(
                ConvBN(256, 512, relu=True),
                ConvBN(512, 512),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )
            self.classifier = torch.nn.Linear(512, 2, bias=False, dtype=torch.float32, device="cpu")
            for module in self.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0.2)
            torch.nn.init.normal_(self.classifier.weight, std=0.01)

        def forward(self, values):
            return self.classifier(self.tail(self.middle(self.front(values))))

    return SRNet()


def float_logits(model, pixels: np.ndarray) -> np.ndarray:
    return logits(model, pixels, _float=True)


def logits(model, pixels: np.ndarray, *, _float: bool = False) -> np.ndarray:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != (np.dtype("<f4") if _float else np.dtype("u1"))
        or pixels.ndim != 4
        or pixels.shape[1] != 1
        or pixels.shape[2] not in ((256,) if _float else SIDES)
        or pixels.shape[3] != pixels.shape[2]
        or not 1 <= len(pixels) <= MAX_BATCH
    ):
        raise ValueError("invalid bounded SRNet uint8 tensor")
    if _float and (not np.isfinite(pixels).all() or np.any(np.abs(pixels) > 2**36)):
        raise ValueError("invalid bounded SRNet float tensor")
    if any(module.training for module in model.modules()):
        raise ValueError("SRNet inference requires every module in eval mode")
    import torch

    with torch.no_grad():
        result = model(torch.from_numpy(pixels.astype(np.float32))).detach().cpu().numpy()
    if result.shape != (len(pixels), 2) or not np.isfinite(result).all():
        raise ValueError("invalid SRNet output")
    return result
