"""Tests for ToolRunner, external tool adapters, and ModelRegistry guards."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.database import Database
from core.models import (
    ModelRegistry,
    ModelVerificationError,
    _key_bytes,
    _safe_component,
)
from core.tools import ToolRunner
from modules.external_tools import (
    ExifToolAdapter,
    StegseekAdapter,
    ZstegAdapter,
)


def test_tool_runner_execution_and_redaction(tmp_path: Path) -> None:
    runner = ToolRunner(timeout=5.0, max_output=1024)
    py = sys.executable

    # 1. Normal execution
    res = runner.run(py, ["-c", "print('hello_world')"], cwd=tmp_path)
    assert res.status == "completed"
    assert res.exit_code == 0
    assert "hello_world" in res.output

    # 2. Secret redaction
    res_redacted = runner.run(
        py,
        ["-c", "print('password is SENSITIVE_SECRET')"],
        cwd=tmp_path,
        secret_values=("SENSITIVE_SECRET",),
    )
    assert "SENSITIVE_SECRET" not in res_redacted.output
    assert "[REDACTED]" in res_redacted.output

    # 3. Output truncation
    runner_short = ToolRunner(timeout=5.0, max_output=10)
    res_truncated = runner_short.run(py, ["-c", "print('a' * 50)"], cwd=tmp_path)
    assert len(res_truncated.output) >= 10
    assert "[output truncated]" in res_truncated.output

    # 4. Failed command
    res_failed = runner.run(py, ["-c", "import sys; sys.exit(42)"], cwd=tmp_path)
    assert res_failed.status == "failed"
    assert res_failed.exit_code == 42
    assert "status 42" in (res_failed.error or "")

    # 5. Cancellation
    res_cancelled = runner.run(
        py, ["-c", "import time; time.sleep(1)"], cwd=tmp_path, should_cancel=lambda: True
    )
    assert res_cancelled.status == "cancelled"

    # 6. Timeout
    res_timeout = runner.run(
        py, ["-c", "import time; time.sleep(2)"], cwd=tmp_path, timeout=0.1
    )
    assert res_timeout.status == "timed_out"

    # 7. Version lookup for non-existent tool
    assert runner.version("non_existent_tool_xyz_123") is None


def test_external_tool_adapters(tmp_path: Path) -> None:
    # 1. ZstegAdapter
    zsteg = ZstegAdapter()
    assert zsteg.command(Path("test.png")) == ["zsteg", "--all", "test.png"]
    sig_zsteg = zsteg.interpret("b1,rgb,lsb,xy .. text: 'flag{zsteg}'\nb2,r,lsb .. file: zip", 0)
    assert len(sig_zsteg) == 1
    assert sig_zsteg[0].name == "zsteg_candidate"
    assert zsteg.interpret("nothing here", 0) == ()
    assert zsteg.analyze(tmp_path / "test.jpg").status == "unsupported"

    # 2. StegseekAdapter
    stegseek = StegseekAdapter()
    assert stegseek.command(Path("test.jpg")) == ["stegseek", "--seed", "test.jpg"]
    sig_stegseek = stegseek.interpret("Found passphrase: 'test'\nRecovered seed: 12345", 0)
    assert len(sig_stegseek) == 1
    assert sig_stegseek[0].name == "steghide_seed_recovered"
    assert stegseek.interpret("no seed found", 1) == ()
    assert stegseek.analyze(tmp_path / "test.png").status == "unsupported"

    # 3. ExifToolAdapter
    exif = ExifToolAdapter()
    assert exif.command(Path("test.png")) == [
        "exiftool",
        "-validate",
        "-warning",
        "-error",
        "test.png",
    ]
    sig_exif = exif.interpret("Warning: Corrupt metadata\nError: Invalid header", 0)
    assert len(sig_exif) == 1
    assert sig_exif[0].name == "metadata_validation"
    assert exif.interpret("All checks passed completely", 0) == ()
    assert exif.analyze(tmp_path / "test.txt").status == "unsupported"


def test_model_registry_guards_and_components(tmp_path: Path) -> None:
    # 1. _safe_component
    assert _safe_component("valid-id.1_0") == "valid-id.1_0"
    with pytest.raises(ModelVerificationError):
        _safe_component("invalid/path")
    with pytest.raises(ModelVerificationError):
        _safe_component("../escape")
    with pytest.raises(ModelVerificationError):
        _safe_component("")

    # 2. _key_bytes
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes_raw()
    assert _key_bytes(public_bytes) == public_bytes
    b64_key = base64.b64encode(public_bytes).decode("ascii")
    assert _key_bytes(b64_key) == public_bytes
    key_file = tmp_path / "key.pub"
    key_file.write_bytes(public_bytes)
    assert _key_bytes(key_file) == public_bytes

    with pytest.raises(ModelVerificationError):
        _key_bytes(b"too_short")
    with pytest.raises(ModelVerificationError):
        _key_bytes("not_base64_invalid_chars!!")

    # 3. verify_manifest
    manifest_no_sig = {"id": "test", "version": "1.0"}
    with pytest.raises(ModelVerificationError, match="no Ed25519 signature"):
        ModelRegistry.verify_manifest(manifest_no_sig, public_key=public_bytes)

    bad_sig = base64.b64encode(b"0" * 64).decode()
    manifest_bad_sig = {"id": "test", "version": "1.0", "signature": bad_sig}
    with pytest.raises(ModelVerificationError, match="invalid Ed25519 manifest signature"):
        ModelRegistry.verify_manifest(manifest_bad_sig, public_key=public_bytes)

    # 4. runtime_status
    status = ModelRegistry.runtime_status()
    assert "status" in status
    assert "available_providers" in status

    # 5. Database lookup error
    db = Database(tmp_path / "models.db")
    registry = ModelRegistry(db, tmp_path / "models_root")
    with pytest.raises(LookupError):
        registry.get("non_existent", "1.0")

    # 6. score with missing model
    score_res = registry.score("missing", "1.0", None, domain="spatial")
    assert score_res.status == "unavailable"
