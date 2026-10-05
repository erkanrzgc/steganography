import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from core.feature_model import feature_model
from steganography import research_precision as rp
from steganography import research_spatial as rs
from steganography.research import ResearchManifestError, train_model
from steganography.research_features import extract_features, read_feature_checkpoint
from tests.test_research_spatial import development  # noqa: F401 - pytest fixture

torch = pytest.importorskip("torch")


@pytest.fixture
def trained(development, tmp_path):  # noqa: F811 - pytest fixture injection
    root, reserved, _ = development
    corpus = tmp_path / "corpus"
    rs.generate_corpus(
        root,
        corpus,
        source_sha256=hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        reserved_manifests=[reserved],
    )
    configs = {}
    for split in ("train", "validation"):
        artifact = tmp_path / f"{split}.json"
        result = extract_features(
            corpus / "manifest.json",
            artifact,
            source=corpus,
            split=split,
            feature_version="spatial-cooccurrence-v1",
        )
        config = tmp_path / f"{split}-config.json"
        config.write_text(
            json.dumps(
                {
                    "manifest": str(corpus / "manifest.json"),
                    "features": str(artifact),
                    "features_sha256": result["sha256"],
                    "epochs": 3,
                    "seed": 42,
                }
            )
        )
        configs[split] = config
    checkpoint = tmp_path / "source.pt"
    train_model(configs["train"], checkpoint)
    return checkpoint, configs["validation"], hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_precise_export_contract_and_cli(trained, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    source, config, digest = trained
    before = source.read_bytes()
    report = rp.audit_export(config, source, tmp_path / "audit", source_sha256=digest)
    assert report["passed"] and not report["deployed"] and not report["retrained"]
    assert report["samples"] == 7 and set(report["batches"]) == {"1", "7"}
    assert str(tmp_path) not in json.dumps(report)
    assert source.read_bytes() == before
    precise = read_feature_checkpoint(tmp_path / "audit/precise.pt")
    original = read_feature_checkpoint(source)
    assert precise["preprocessing"]["inference_arithmetic"] == "float64"
    assert precise["training_provenance"] == original["training_provenance"]
    for key, value in original["state_dict"].items():
        assert torch.equal(value, precise["state_dict"][key])
    with pytest.raises(FileExistsError):
        rp.audit_export(config, source, tmp_path / "audit", source_sha256=digest)
    with pytest.raises(FileExistsError):
        rp.derive_checkpoint(source, tmp_path / "audit/precise.pt", source_sha256=digest)
    monkeypatch.setattr(rp, "audit_export", lambda *a, **k: report)
    monkeypatch.setattr(
        "sys.argv",
        ["precision", str(config), str(source), str(tmp_path / "cli"), "--source-sha256", digest],
    )
    rp.main()


def test_invalid_contracts_and_symlink_outputs(trained, tmp_path):
    source, config, digest = trained
    with pytest.raises(ResearchManifestError, match="checksum"):
        rp.derive_checkpoint(source, tmp_path / "wrong.pt", source_sha256="0" * 64)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        rp.derive_checkpoint(source, link / "derived.pt", source_sha256=digest)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rp.audit_export(config, source, link / "audit", source_sha256=digest)
    saved = read_feature_checkpoint(source)
    saved["domain"] = "wrong"
    torch.save(saved, source)
    with pytest.raises(ResearchManifestError, match="contract"):
        rp.audit_export(
            config,
            source,
            tmp_path / "wrong-domain",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )


def test_double_arithmetic_independent_oracle_and_legacy_default():
    linear = torch.nn.Linear(3, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.31, -2.75, 0.021]]))
        linear.bias.fill_(0.15)
    c = {
        "features": 3,
        "state_dict": linear.state_dict(),
        "preprocessing": {
            "normalization": {
                "method": "train-only-standardization",
                "mean": [0.91, 0.75, 0.333],
                "scale": [0.0001, 0.0002, 0.017],
            }
        },
    }
    x = np.array([[0.9, 0.75, 0.335], [0.95, 0.7, 0.2]], dtype=np.float32)
    legacy = feature_model(c)
    assert legacy.linear.weight.dtype == torch.float32
    precise = copy.deepcopy(c)
    precise["preprocessing"]["inference_arithmetic"] = "float64"
    actual = feature_model(precise)(torch.from_numpy(x)).detach().numpy()
    norm = c["preprocessing"]["normalization"]
    expected = (
        ((x.astype("float64") - norm["mean"]) / norm["scale"])
        @ linear.weight.detach().numpy().astype("float64").T
        + linear.bias.detach().numpy()
    ).astype("float32")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert actual.dtype == np.float32
    precise["preprocessing"]["normalization"] = None
    assert feature_model(precise)(torch.from_numpy(x)).dtype == torch.float32
    precise["preprocessing"]["inference_arithmetic"] = "float16"
    with pytest.raises(ValueError, match="arithmetic"):
        feature_model(precise)


def test_checkpoint_bounds_before_weights_loading(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    with pytest.raises(ResearchManifestError, match="bounded"):
        read_feature_checkpoint(path)
    path.write_bytes(b"invalid archive")
    with pytest.raises(ResearchManifestError, match="ZIP"):
        read_feature_checkpoint(path)
    for entries, size in ((257, 0), (1, 16 * 1024 * 1024 + 1)):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for i in range(entries):
                z.writestr(str(i), b"x" * size)
        path.write_bytes(buffer.getvalue())
        with pytest.raises(ResearchManifestError, match="archive exceeds"):
            read_feature_checkpoint(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("fixture", b"small")
    path.write_bytes(buffer.getvalue())
    for value in ([], {}, {"features": True}, {"features": 4097}):
        monkeypatch.setattr(torch, "load", lambda *a, fixture_value=value, **k: fixture_value)
        with pytest.raises(ResearchManifestError, match="dimensions"):
            read_feature_checkpoint(path)
    original = tmp_path / "original.pt"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ResearchManifestError, match="bounded"):
        read_feature_checkpoint(path)
