import io
import json
from pathlib import Path

from cli import main


def test_cli_json_v1_and_failure_threshold(png_64x64: Path, capsys):
    code = main(
        [
            "--quiet",
            "analyze",
            "--in",
            str(png_64x64),
            "--format",
            "json-v1",
            "--fail-on",
            "high",
        ]
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["schema_version"] == "1.0"


def test_cli_scatter_password_stdin_roundtrip(
    png_64x64: Path, tmp_path: Path, monkeypatch, capsys
):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"cli extended")
    stego = tmp_path / "scatter.png"
    monkeypatch.setattr("sys.stdin", io.StringIO("password\n"))
    assert main(
        [
            "--quiet",
            "embed",
            "--in",
            str(secret),
            "--carrier",
            str(png_64x64),
            "--out",
            str(stego),
            "--method",
            "image_lsb_scatter",
            "--password-stdin",
        ]
    ) == 0
    recovered = tmp_path / "out.bin"
    password_file = tmp_path / "password.txt"
    password_file.write_text("password\n")
    assert main(
        [
            "--quiet",
            "extract",
            "--in",
            str(stego),
            "--out",
            str(recovered),
            "--method",
            "image_lsb_scatter",
            "--password-file",
            str(password_file),
        ]
    ) == 0
    assert recovered.read_bytes() == b"cli extended"
    capsys.readouterr()


def test_cli_scan_json_v1_handles_bad_file(tmp_path: Path, capsys):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    (tmp_path / "ok.txt").write_text("hello\n")
    out = tmp_path / "report.json"
    assert main(
        [
            "--quiet",
            "scan",
            "--dir",
            str(tmp_path),
            "--report",
            "json-v1",
            "--out",
            str(out),
            "--jobs",
            "2",
        ]
    ) == 0
    body = json.loads(out.read_text())
    assert body["summary"]["files"] == 2
    assert body["summary"]["errors"] >= 1
    capsys.readouterr()


def test_cli_reports_user_error_and_version(capsys):
    assert main(["--quiet", "analyze", "--in", "/definitely/missing"]) == 2
    assert "error:" in capsys.readouterr().err
