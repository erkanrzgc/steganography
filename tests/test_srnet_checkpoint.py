"""Generated epoch resume and hostile numeric archives, not real learning."""

import copy
import hashlib
import io
import json
import zipfile
from contextlib import nullcontext

import numpy as np
import pytest

from core import srnet_checkpoint as checkpoint
from core import srnet_training as training

torch = pytest.importorskip("torch")
BINDING = hashlib.sha256(b"generated-frozen-two-epoch-control-v1").hexdigest()


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def learn(*, segment=None, random_input=False, **changes):
    values = np.arange(4 * 256 * 256, dtype="<f4").reshape(4, 1, 256, 256) % 255

    def fetch(_):
        noise = torch.rand(1).item() if random_input else 0.0
        return (values + noise).astype("<f4")

    args = {
        "fetch": fetch,
        "batches": lambda _: np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype="<i8"),
        "epochs": 2,
        "seed": 91,
        "params": training.settings({"threads": 1, "max_seconds": 180}),
        "target_pairs": 2,
        "segment": segment,
    }
    return training._learn(**{**args, **changes})


def segment(*, stop=1, resume=None, sink=None):
    return {
        "binding": BINDING,
        "stop_epoch": stop,
        "resume": resume,
        "sink": sink or (lambda _: None),
    }


@pytest.fixture(scope="module")
def snapshot():
    result = []
    learn(segment=segment(sink=result.append))
    return result[0]


@pytest.mark.parametrize("random_input", [False, True])
def test_disk_resume_exact_optimizer_bn_loss_rng_and_caller_state(tmp_path, random_input):
    rng = torch.get_rng_state().clone()
    baseline, records = learn(random_input=random_input)
    outputs = []
    learn(segment=segment(sink=outputs.append), random_input=random_input)
    path = tmp_path / "checkpoint.npz"
    digest = checkpoint.save(outputs[0], path)
    loaded = checkpoint.load(path, checksum=digest)
    other = []
    candidate, candidate_records = learn(
        segment=segment(stop=2, resume=loaded, sink=other.append),
        random_input=random_input,
    )
    assert records == candidate_records and len(records) == 2
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    # Compare final optimizer slots and generator bytes as well, not just weights.
    complete = []
    learn(segment=segment(stop=2, sink=complete.append), random_input=random_input)
    for name, value in complete[0]["arrays"].items():
        np.testing.assert_array_equal(value, other[0]["arrays"][name])
    assert complete[0]["metadata"] == other[0]["metadata"]
    assert other[0]["metadata"]["next_epoch"] == 2
    assert all(int(v) == 4 for k, v in other[0]["arrays"].items() if k.endswith("/step"))


@pytest.mark.parametrize(
    "fault",
    [
        "container",
        "metadata",
        "seed",
        "next_epoch",
        "loss",
        "counter",
        "step",
        "inf_norm",
        "nan",
        "shape",
        "dtype",
        "keys",
        "settings",
        "rng",
        "version",
        "scheduled_updates",
    ],
)
def test_rejects_malformed_snapshot_before_resume(snapshot, fault):
    value = {"metadata": copy.deepcopy(snapshot["metadata"]), "arrays": dict(snapshot["arrays"])}
    meta = value["metadata"]
    name = "optimizer/classifier.weight/exp_inf"
    if fault == "container":
        value["extra"] = None
    elif fault == "metadata":
        meta["host_path"] = "/private/secret"
    elif fault == "seed":
        meta["seed"] = True
    elif fault == "next_epoch":
        meta["next_epoch"] = 0
    elif fault == "loss":
        meta["epoch_training"][0]["mean_pair_loss"] = float("nan")
    elif fault == "counter":
        value["arrays"]["model/front.0.1.num_batches_tracked"] = np.array(1, dtype="<i8")
    elif fault == "step":
        value["arrays"]["optimizer/classifier.weight/step"] = np.array(1, dtype="<f4")
    elif fault == "inf_norm":
        value["arrays"][name] = np.full_like(value["arrays"][name], -1)
    elif fault == "nan":
        value["arrays"][name] = np.full_like(value["arrays"][name], np.inf)
    elif fault == "shape":
        value["arrays"][name] = value["arrays"][name][:1]
    elif fault == "dtype":
        value["arrays"][name] = value["arrays"][name].astype("f8")
    elif fault == "keys":
        del value["arrays"][name]
    elif fault == "settings":
        meta["settings"]["max_seconds"] = 1801
    elif fault == "rng":
        meta["cpu_rng_bytes"] = 16385
    elif fault == "version":
        meta["torch_version"] = "/private/secret"
    elif fault == "scheduled_updates":
        meta["scheduled_updates"][0] = 3
    with pytest.raises(ValueError):
        checkpoint.validate(value)


@pytest.mark.parametrize(
    "fault",
    [
        "binding",
        "seed",
        "settings",
        "schedule",
        "version",
        "rng",
        "repeat",
        "sink",
        "pairs",
        "accumulation",
        "stop",
        "segment_keys",
    ],
)
def test_resume_contract_failure_preserves_rng(snapshot, fault):
    value = {"metadata": copy.deepcopy(snapshot["metadata"]), "arrays": dict(snapshot["arrays"])}
    args, piece = {}, segment(stop=2, resume=value)
    if fault == "binding":
        piece["binding"] = "0" * 64
    elif fault == "seed":
        args["seed"] = 92
    elif fault == "settings":
        args["params"] = training.settings({"threads": 1, "max_seconds": 179})
    elif fault == "schedule":
        args["batches"] = lambda _: np.array([[2, 3, 0, 1], [0, 1, 2, 3]])
    elif fault == "version":
        value["metadata"]["torch_version"] = "0.0.0"
    elif fault == "rng":
        value["metadata"]["cpu_rng_bytes"] -= 1
        value["arrays"]["rng/cpu"] = value["arrays"]["rng/cpu"][:-1]
    elif fault == "repeat":
        piece["stop_epoch"] = 1
    elif fault == "sink":
        piece["sink"] = None
    elif fault == "pairs":
        args["target_pairs"] = 1
    elif fault == "accumulation":
        args["accumulate_context"] = True
    elif fault == "stop":
        piece["stop_epoch"] = True
    elif fault == "segment_keys":
        piece["extra"] = None
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        learn(segment=piece, **args)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "batches",
    [
        None,
        np.zeros((1, 2), dtype="i8"),
        np.zeros((1, 4), dtype="f4"),
        np.full((1, 4), -1),
        np.full((1, 4), 12000),
        np.zeros((0, 4), dtype="i8"),
    ],
)
def test_invalid_schedule_rejected(batches):
    with pytest.raises(ValueError):
        checkpoint.schedule(lambda _: batches, 1)


def test_schedule_epoch_bound_and_propagated_deadline():
    with pytest.raises(ValueError, match="epoch count"):
        checkpoint.schedule(lambda _: None, 51)
    with pytest.raises(ValueError, match="deadline"):
        checkpoint.schedule(
            lambda _: pytest.fail("read after expiration"),
            1,
            deadline=lambda: (_ for _ in ()).throw(ValueError("deadline")),
        )


def test_cuda_checkpoint_control_flow_only(monkeypatch):
    """All tensors remain on CPU. Not physical CUDA parity or timing."""
    from core import srnet_cuda

    tensor_to, module_to, tensor, fork = (
        torch.Tensor.to,
        torch.nn.Module.to,
        torch.tensor,
        torch.random.fork_rng,
    )
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **k: (
            tensor_to(self, "cpu") if a == ("cuda:0",) else tensor_to(self, *a, **k)
        ),
    )
    monkeypatch.setattr(
        torch.nn.Module,
        "to",
        lambda self, *a, **k: (
            module_to(self, "cpu") if a == ("cuda:0",) else module_to(self, *a, **k)
        ),
    )
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **k: tensor(
            *a, **{**k, **({"device": "cpu"} if k.get("device") == "cuda:0" else {})}
        ),
    )
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: fork(devices=[]))
    monkeypatch.setattr(srnet_cuda, "policy", lambda: nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _: torch.get_rng_state().clone())
    restored = []
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda value, _: restored.append(value.clone())
    )
    outputs = []
    learn(segment=segment(sink=outputs.append), device="cuda:0")
    meta = outputs[0]["metadata"]
    assert meta["device"] == "cuda:0" and meta["cuda_rng_bytes"] == len(torch.get_rng_state())
    other = []
    learn(segment=segment(stop=2, resume=outputs[0], sink=other.append), device="cuda:0")
    assert len(restored) == 1 and other[0]["metadata"]["next_epoch"] == 2


def test_metadata_output_bound_and_v2_header(snapshot, tmp_path, monkeypatch):
    stream = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        stream, {"descr": "<f4", "fortran_order": False, "shape": (2, 3)}
    )
    stream.seek(0)
    assert checkpoint.header(stream) == ((2, 3), False, np.dtype("<f4"))
    monkeypatch.setattr(checkpoint, "MAX_META", 1)
    with pytest.raises(ValueError, match="metadata size"):
        checkpoint.save(snapshot, tmp_path / "too-much-meta.npz")


def test_invalid_runtime_rng_does_not_leak_or_publish(snapshot):
    value = {"metadata": copy.deepcopy(snapshot["metadata"]), "arrays": dict(snapshot["arrays"])}
    value["arrays"]["rng/cpu"] = np.zeros_like(value["arrays"]["rng/cpu"])
    checkpoint.validate(value)  # Opaque runtime bytes are validated by Torch.
    rng = torch.get_rng_state().clone()
    outputs = []
    with pytest.raises(RuntimeError):
        learn(segment=segment(stop=2, resume=value, sink=outputs.append))
    assert outputs == [] and torch.equal(rng, torch.get_rng_state())


def test_no_overwrite_symlink_size_checksum_or_failed_job_snapshot(snapshot, tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    checksum = checkpoint.save(snapshot, path)
    with pytest.raises(FileExistsError):
        checkpoint.save(snapshot, path)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        checkpoint.save(snapshot, link)
    with pytest.raises(ValueError):
        checkpoint.load(link, checksum=checksum)
    with pytest.raises(ValueError):
        checkpoint.load(path, checksum="0" * 64)
    monkeypatch.setattr(checkpoint, "MAX_BYTES", 128)
    with pytest.raises(ValueError, match="size"):
        checkpoint.save(snapshot, tmp_path / "oversized.npz")
    with pytest.raises(ValueError, match="size"):
        checkpoint.load(path, checksum=checksum)
    assert not (tmp_path / "oversized.npz").exists()
    outputs = []
    with pytest.raises(ValueError, match="deadline"):
        learn(
            segment=segment(sink=outputs.append),
            outer_deadline=lambda: (_ for _ in ()).throw(ValueError("deadline")),
        )
    assert outputs == []


@pytest.mark.parametrize(
    "fault",
    [
        "compressed",
        "duplicate",
        "traversal",
        "object",
        "huge",
        "truncated",
        "header_version",
        "meta_dtype",
        "meta_shape",
        "meta_length",
        "model_shape",
        "model_length",
        "symlink",
    ],
)
def test_hostile_archive_preflight_before_numpy_allocation(snapshot, tmp_path, monkeypatch, fault):
    raw_meta = json.dumps(snapshot["metadata"]).encode()
    buffers = {}
    for name, array in {
        **snapshot["arrays"],
        "__metadata__": np.frombuffer(raw_meta, dtype="u1"),
    }.items():
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        buffers[name + ".npy"] = stream.getvalue()
    if fault in {"meta_dtype", "meta_shape"}:
        stream = io.BytesIO()
        np.save(
            stream,
            np.zeros(3, dtype="f4") if fault == "meta_dtype" else np.zeros((1, 3), dtype="u1"),
        )
        buffers["__metadata__.npy"] = stream.getvalue()
    if fault == "meta_length":
        buffers["__metadata__.npy"] += b"x"
    if fault in {"object", "huge", "header_version", "model_shape"}:
        stream = io.BytesIO()
        if fault == "header_version":
            stream.write(b"\x93NUMPY\x03\x00")
        else:
            shape = (10**12,) if fault == "huge" else (1,)
            np.lib.format.write_array_header_1_0(
                stream,
                {
                    "descr": "|O" if fault == "object" else "<f4",
                    "fortran_order": False,
                    "shape": shape,
                },
            )
        buffers["model/classifier.weight.npy"] = stream.getvalue()
    if fault == "model_length":
        buffers["model/classifier.weight.npy"] += b"x"
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED if fault == "compressed" else zipfile.ZIP_STORED,
    ) as archive:
        for name, raw in buffers.items():
            archive.writestr(name, raw)
        if fault == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("__metadata__.npy", buffers["__metadata__.npy"])
        if fault == "traversal":
            archive.writestr("../../escape.npy", b"x")
        if fault == "symlink":
            info = zipfile.ZipInfo("link.npy")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, b"target")
    raw = stream.getvalue()[:-10] if fault == "truncated" else stream.getvalue()
    path = tmp_path / "hostile.npz"
    path.write_bytes(raw)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("untrusted allocation reached"))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        checkpoint.load(path, checksum=hashlib.sha256(raw).hexdigest())
    assert not (tmp_path.parent / "escape.npy").exists()
