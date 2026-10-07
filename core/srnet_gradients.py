"""State-preserving training-gradient probes, not inference or accuracy gates."""

import math

import numpy as np

from core import srnet, srnet_model, srnet_multibatch


def select_batches(samples):
    batches, record = srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=0)
    wanted = {
        (s["source_group"], s["quality_factor"], s["method"])
        for s in samples
        if s["label"] == "stego"
    }
    if len(wanted) != 6:
        raise ValueError("gradient diagnosis requires six complete declared training cells")
    seen: set[tuple] = set()
    chosen = []
    for batch in batches:
        cells = {
            (samples[i]["source_group"], samples[i]["quality_factor"], samples[i]["method"])
            for i in (batch[1], batch[3])
        }
        if cells - seen:
            chosen.append(batch.tolist())
            seen.update(cells)
    if seen != wanted or not 1 <= len(chosen) <= 6:
        raise ValueError("gradient diagnosis incomplete cell coverage")
    return chosen, record


def probe(model, pixels):
    if (
        getattr(model, "architecture", None) != srnet.ARCHITECTURE
        or any(m.training for m in model.modules())
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (4, 1, 256, 256)
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("gradient probe requires one bounded four-row batch and eval model")
    import torch

    state = model.state_dict()
    if any(v.device.type != "cpu" for v in state.values()):
        raise ValueError("gradient probe requires CPU state")
    srnet_model.validate({k: v.detach().numpy() for k, v in state.items()})
    saved = {k: v.detach().clone() for k, v in state.items()}
    flags = [(m, m.training) for m in model.modules()]
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if len(layers) != 26:
        raise ValueError("gradient probe requires the exact architecture")
    tracking = [(m, m.track_running_stats) for m in layers]
    params = list(model.named_parameters())
    if any(not p.requires_grad for _, p in params):
        raise ValueError("gradient probe requires all learned parameters enabled")
    previous = torch.get_num_threads()
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    def loss_for(values):
        logits = model(values)
        if logits.shape != (4, 2) or not bool(torch.isfinite(logits).all()):
            raise ValueError("gradient probe produced invalid logits")
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("gradient probe produced nonfinite loss")
        return loss

    def gradients(values):
        inputs = torch.from_numpy(values.copy()).requires_grad_(True)
        loss = loss_for(inputs)
        grads = torch.autograd.grad(loss, [p for _, p in params] + [inputs])
        if not all(bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError("gradient probe produced nonfinite gradients")
        return float(loss.detach()), [g.detach().numpy().copy() for g in grads]

    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            model.train()
            for layer in layers:
                layer.track_running_stats = False
            true_loss, true = gradients(pixels)
            null_pixels = pixels[[0, 0, 2, 2]].copy()
            null_loss, null = gradients(null_pixels)
            summaries, t2, n2, d2, dot = [], 0.0, 0.0, 0.0, 0.0
            for (name, _), t, n in zip(params, true[:-1], null[:-1], strict=True):
                a, b = t.astype(np.float64), n.astype(np.float64)
                ta, nb = float(np.sum(a * a)), float(np.sum(b * b))
                t2 += ta
                n2 += nb
                d2 += float(np.sum((a - b) ** 2))
                dot += float(np.sum(a * b))
                summaries.append(
                    {"parameter": name, "true_l2": math.sqrt(ta), "null_l2": math.sqrt(nb)}
                )
            classifier_index = [name for name, _ in params].index("classifier.weight")
            weight = params[classifier_index][1]
            grad = true[classifier_index].astype(np.float64)
            norm = float(np.linalg.norm(grad))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("classifier directional gradient unavailable")
            direction = torch.from_numpy((grad / norm).astype("<f4"))
            analytic = float(np.sum(grad * direction.numpy()))
            with torch.no_grad():
                weight.copy_(saved["classifier.weight"] + 0.001 * direction)
                plus = float(loss_for(torch.from_numpy(pixels.copy())))
                weight.copy_(saved["classifier.weight"] - 0.001 * direction)
                minus = float(loss_for(torch.from_numpy(pixels.copy())))
                weight.copy_(saved["classifier.weight"])
            numeric = (plus - minus) / 0.002
            error = abs(numeric - analytic)
            delta = pixels.astype(np.float64) - null_pixels
            result = {
                "true_cross_entropy": true_loss,
                "null_cross_entropy": null_loss,
                "parameters": summaries,
                "true_gradient_l2": math.sqrt(t2),
                "null_gradient_l2": math.sqrt(n2),
                "gradient_difference_l2": math.sqrt(d2),
                "relative_gradient_difference": math.sqrt(d2 / t2) if t2 else None,
                "gradient_cosine": dot / math.sqrt(t2 * n2) if t2 and n2 else None,
                "input_gradient_l2": float(np.linalg.norm(true[-1].astype(np.float64))),
                "input_difference_rms": float(np.sqrt(np.mean(delta**2))),
                "input_directional_derivative": float(np.sum(true[-1].astype(np.float64) * delta)),
                "classifier_directional_check": {
                    "epsilon": 0.001,
                    "analytic": analytic,
                    "numeric": numeric,
                    "absolute_error": error,
                    "absolute_tolerance": 0.002,
                    "relative_tolerance": 0.02,
                    "passed": error <= 0.002 + 0.02 * abs(analytic),
                },
                "batch_statistics_training_diagnostic_only": True,
                "state_unchanged": True,
            }
            if not all(
                math.isfinite(v)
                for v in (
                    t2,
                    n2,
                    d2,
                    dot,
                    analytic,
                    numeric,
                    result["input_gradient_l2"],
                    result["input_directional_derivative"],
                )
            ):
                raise ValueError("gradient probe produced nonfinite summaries")
    finally:
        changed = any(not torch.equal(v, saved[k]) for k, v in model.state_dict().items())
        with torch.no_grad():
            model.load_state_dict(saved, strict=True)
        for module, flag in flags:
            module.training = flag
        for module, flag in tracking:
            module.track_running_stats = flag
        torch.set_num_threads(previous)
    if changed:
        raise ValueError("gradient probe attempted model mutation")
    return result
