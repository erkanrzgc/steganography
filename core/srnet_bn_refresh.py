"""Training-only BN context/refresh controls; source weights remain immutable."""

import numpy as np

from core import srnet, srnet_model, srnet_multibatch, srnet_reference, srnet_sanity


def checked(model, pixels, samples):
    if (
        getattr(model, "architecture", None) != srnet.ARCHITECTURE
        or any(m.training for m in model.modules())
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (24, 1, 256, 256)
        or len(samples) != 24
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("BN refresh requires 24 bounded train tensors and an eval model")
    import torch

    state = model.state_dict()
    if any(v.device.type != "cpu" for v in state.values()):
        raise ValueError("BN refresh requires CPU numeric state")
    srnet_model.validate({k: v.detach().numpy() for k, v in state.items()})
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if len(layers) != 26 or any(not m.track_running_stats for m in layers):
        raise ValueError("BN refresh requires 26 tracked BN layers")
    batches, schedule = srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=19)
    if batches.shape != (8, 4):
        raise ValueError("BN refresh requires eight complete final-epoch batches")
    return layers, batches, schedule


def context(model, pixels, samples):
    layers, batches, schedule = checked(model, pixels, samples)
    import torch

    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    flags = [(m, m.training) for m in model.modules()]
    tracking = [(m, m.track_running_stats) for m in layers]
    previous, results = torch.get_num_threads(), {}
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for mode in ("stored_bn", "batch_statistics_diagnostic_only"):
                if mode != "stored_bn":
                    for m in layers:
                        m.training, m.track_running_stats = True, False
                outputs, metrics = [], []
                for batch in batches:
                    logits = model(torch.from_numpy(pixels[batch].copy())).detach().numpy()
                    final = srnet_sanity.metrics(logits, [samples[i] for i in batch])
                    outputs.append({"indices": batch.tolist(), "logits": logits.tolist()})
                    metrics.append(final)
                results[mode] = {
                    "batches": outputs,
                    "presented_rows": 32,
                    "metrics": {
                        k: float(np.mean([m[k] for m in metrics]))
                        for k in (
                            "cross_entropy",
                            "balanced_accuracy",
                            "recall",
                            "false_positive_rate",
                        )
                    },
                }
    finally:
        changed = any(not torch.equal(v, saved[k]) for k, v in model.state_dict().items())
        model.load_state_dict(saved, strict=True)
        for m, flag in flags:
            m.training = flag
        for m, flag in tracking:
            m.track_running_stats = flag
        torch.set_num_threads(previous)
    if changed:
        raise ValueError("BN context attempted source-state mutation")
    return {"state_unchanged": True, "epoch_batch_schedule": schedule, **results}


def refresh(model, pixels, samples):
    _, batches, schedule = checked(model, pixels, samples)
    import torch

    source = {k: v.detach().clone() for k, v in model.state_dict().items()}
    source_flags = [(m, m.training) for m in model.modules()]
    source_bn = [
        (m, m.track_running_stats, m.momentum)
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]
    previous = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        clone = srnet.network().eval()
    clone.load_state_dict(source, strict=True)
    layers = [m for m in clone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momentum = [(m, m.momentum) for m in layers]
    try:
        torch.set_num_threads(2)
        for m in layers:
            m.reset_running_stats()
            m.momentum, m.training = None, True
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for batch in batches:
                logits = clone(torch.from_numpy(pixels[batch].copy())).detach().numpy()
                srnet_sanity.metrics(logits, [samples[i] for i in batch])
    finally:
        source_changed = any(not torch.equal(v, source[k]) for k, v in model.state_dict().items())
        model.load_state_dict(source, strict=True)
        for m, flag in source_flags:
            m.training = flag
        for m, tracking, value in source_bn:
            m.track_running_stats, m.momentum = tracking, value
        for m, value in momentum:
            m.momentum = value
        clone.eval()
        torch.set_num_threads(previous)
    arrays = {k: v.detach().numpy() for k, v in clone.state_dict().items()}
    srnet_model.validate(arrays)
    if any(int(v) != 8 for k, v in arrays.items() if k.endswith("num_batches_tracked")):
        raise ValueError("BN refresh clone counter mismatch")
    allowed = ("running_mean", "running_var", "num_batches_tracked")
    changed = [k for k, v in clone.state_dict().items() if not torch.equal(v, source[k])]
    if any(not k.endswith(allowed) for k in changed) or source_changed:
        raise ValueError("BN refresh changed learned parameters or source state")
    return clone, {
        "changed_buffers": changed,
        "source_state_unchanged": True,
        "learned_parameters_bit_identical": True,
        "bn_refresh_batches": 8,
        "optimizer_steps": 0,
        "epoch_batch_schedule": schedule,
        "exact_population_variance": False,
    }


def objectives(own, baseline):
    if any(
        type(d.get(k)) not in (int, float) or not np.isfinite(d[k]) or d[k] < 0
        for d in (own, baseline)
        for k in ("balanced_accuracy", "cross_entropy")
    ) or any(d["balanced_accuracy"] > 1 for d in (own, baseline)):
        raise ValueError("BN refresh requires finite training-only metrics")
    return {
        "own_train_balanced_accuracy_at_least_0_90": own["balanced_accuracy"] >= 0.90,
        "own_train_cross_entropy_below_baseline": own["cross_entropy"] < baseline["cross_entropy"],
    }


def oracles(model, pixels, samples, logits):
    checked(model, pixels, samples)
    selected, seen = [], set()
    for i, s in enumerate(samples):
        key = (s["source_group"], s["quality_factor"], s["label"], s["method"])
        if key not in seen:
            selected.append(i)
            seen.add(key)
    if len(selected) != 9:
        raise ValueError("BN refresh oracle coverage incomplete")
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    result = [
        {
            "row": i,
            "source_sha256": samples[i]["sha256"],
            **srnet_reference.compare(
                logits[i : i + 1], srnet_reference.reference_logits(arrays, pixels[i : i + 1])
            ),
        }
        for i in selected
    ]
    return {
        "independent_oracles": result,
        "numerical_gates_passed": all(r["passed"] for r in result),
    }
