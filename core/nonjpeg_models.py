"""Optional FP32 research skeletons; no optimizer, downloads or checkpoints."""

from __future__ import annotations

import numpy as np

from core.nonjpeg_features import pcm16_samples, spatial_pixels

GRAY_DOMAIN = "spatial-native-gray-srnet-v1"
RGB_DOMAIN = "spatial-native-rgb-srnet-v1"
AUDIO_DOMAIN = "audio-native-pcm16-resnet1d-v1"
DOMAINS = (GRAY_DOMAIN, RGB_DOMAIN, AUDIO_DOMAIN)


def inputs(data: bytes, domain: str) -> np.ndarray:
    if domain == AUDIO_DOMAIN:
        raw = pcm16_samples(data)
        # Signed arithmetic before FP32 scaling; zero left boundary for deltas.
        first = np.diff(raw, prepend=raw[0])
        second = np.diff(first, prepend=first[0])
        return np.stack((raw, first, second)).astype(np.float32)[None] / 32768
    if domain not in (GRAY_DOMAIN, RGB_DOMAIN):
        raise ValueError("unknown native model domain")
    pixels = spatial_pixels(data, rgb=domain == RGB_DOMAIN)
    if domain == GRAY_DOMAIN and pixels.shape[2] != 1:
        raise ValueError("gray model requires native grayscale")
    return pixels.transpose(2, 0, 1).astype(np.float32)[None]


def network(domain: str):
    if domain not in DOMAINS:
        raise ValueError("unknown native model domain")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("native model skeleton unavailable: optional Torch missing") from exc

    if domain in (GRAY_DOMAIN, RGB_DOMAIN):
        from core.srnet import network as srnet_network

        model = srnet_network()
        if domain == RGB_DOMAIN:
            conv = torch.nn.Conv2d(3, 64, 3, padding=1, dtype=torch.float32, device="cpu")
            torch.nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            if conv.bias is not None:
                torch.nn.init.constant_(conv.bias, 0.2)
            model.front[0][0] = conv
        model.architecture = domain
        return model

    class Conv(torch.nn.Sequential):
        def __init__(self, before, after, stride=1):
            super().__init__(
                torch.nn.Conv1d(before, after, 5, stride=stride, padding=2, dtype=torch.float32),
                torch.nn.BatchNorm1d(after, dtype=torch.float32),
                torch.nn.ReLU(),
            )

    class Residual(torch.nn.Module):
        def __init__(self, before, after, stride=1):
            super().__init__()
            self.branch = torch.nn.Sequential(
                Conv(before, after, stride),
                torch.nn.Conv1d(after, after, 5, padding=2, dtype=torch.float32),
                torch.nn.BatchNorm1d(after, dtype=torch.float32),
            )
            self.skip = (
                torch.nn.Identity()
                if before == after and stride == 1
                else torch.nn.Conv1d(before, after, 1, stride=stride, dtype=torch.float32)
            )

        def forward(self, values):
            return torch.relu(self.branch(values) + self.skip(values))

    class PCMNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = AUDIO_DOMAIN
            self.body = torch.nn.Sequential(
                Conv(3, 32),
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

    return PCMNet()


def generated_smoke() -> dict:
    """Generated-only forward/backward; preserve RNG, parameters and thread policy."""
    try:
        import torch
    except ImportError:
        return {"status": "unavailable", "reason": "optional_torch_missing", "optimizer_updates": 0}
    previous_threads = torch.get_num_threads()
    results = {}
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261015)
            for domain in DOMAINS:
                model = network(domain).train()
                shape = (
                    (2, 3, 640)
                    if domain == AUDIO_DOMAIN
                    else (1, 3 if domain == RGB_DOMAIN else 1, 256, 256)
                )
                values = torch.randn(shape, dtype=torch.float32, requires_grad=True)
                before = [p.detach().clone() for p in model.parameters()]
                output = model(values)
                loss = torch.nn.functional.cross_entropy(
                    output, torch.arange(shape[0], dtype=torch.long) % 2
                )
                loss.backward()
                if (
                    output.shape != (shape[0], 2)
                    or not bool(torch.isfinite(output).all())
                    or values.grad is None
                    or not bool(torch.isfinite(values.grad).all())
                    or not bool((values.grad != 0).any())
                    or any(
                        p.grad is None or not bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    )
                    or any(
                        not torch.equal(p, old)
                        for p, old in zip(model.parameters(), before, strict=True)
                    )
                ):
                    raise ValueError("native generated forward/backward contract failed")
                results[domain] = {
                    "status": "ready",
                    "input_shape": list(shape),
                    "dtype": "float32",
                    "finite_gradients": True,
                    "parameters_unchanged": True,
                    "parameters": sum(p.numel() for p in model.parameters()),
                }
                del model, values, output, loss, before
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "status": "ready",
        "models": results,
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "weights_downloaded": False,
        "gpu_used": False,
    }
