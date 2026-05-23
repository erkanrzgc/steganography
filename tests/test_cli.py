import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 — args are fully controlled in this test
        [sys.executable, str(ROOT / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        check=False,
    )


def test_list_modules_exits_zero():
    r = _run(["list-modules"])
    assert r.returncode == 0
    assert "image_lsb" in r.stdout


def test_embed_extract_roundtrip(png_64x64: Path, tmp_outdir: Path, tmp_path: Path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"cli round trip")
    out = tmp_outdir / "out.png"
    extracted = tmp_path / "got.bin"
    r1 = _run(["embed", "--in", str(secret), "--carrier", str(png_64x64), "--out", str(out)])
    assert r1.returncode == 0, r1.stderr
    r2 = _run(["extract", "--in", str(out), "--out", str(extracted)])
    assert r2.returncode == 0, r2.stderr
    assert extracted.read_bytes() == b"cli round trip"


def test_analyze_returns_json_when_flag(png_64x64: Path):
    r = _run(["analyze", "--in", str(png_64x64), "--json"])
    assert r.returncode == 0
    assert r.stdout.strip().startswith("[")
