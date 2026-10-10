"""Run inside the full image with read-only root and a writable /tmp tmpfs."""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from core.ctf import CTFLimits, CTFService
from core.service import StegoService
from core.tool_inventory import read_lock, verify_inventory
from core.tools import ToolRunner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(os.getuid() != 0, "smoke must run as non-root")
    root_mount = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    require(any(row[1] == "/" and "ro" in row[3].split(",") for row in root_mount),
            "smoke requires read-only root")
    inventory = verify_inventory(
        read_lock(Path("/usr/share/steganography/full-image.lock.json")), cwd=Path.cwd()
    )
    baseline = json.loads(Path("/usr/share/steganography/full-image-inventory.json").read_text())
    require(inventory["status"] == "verified" and inventory == baseline,
            "installed inventory differs from verified build")
    for tool in (
        "binwalk",
        "zsteg",
        "stegseek",
        "steghide",
        "outguess",
        "exiftool",
        "pngcheck",
        "ffmpeg",
        "sox",
        "gifsicle",
        "zbarimg",
        "openstego",
    ):
        require(shutil.which(tool) is not None, f"missing {tool}")
    with tempfile.TemporaryDirectory(prefix="ctf-smoke-") as directory:
        root = Path(directory)
        runner = ToolRunner(timeout=0.1, max_output=128)
        timed_out = runner.run("sleep", ["2"], cwd=root, _include_version=False)
        require(timed_out.status == "timed_out" and timed_out.exit_code is not None,
                "tool deadline ignored")
        started = time.monotonic()
        interrupted = runner.run(
            "sleep", ["2"], cwd=root, timeout=5,
            should_cancel=lambda: time.monotonic() - started > 0.05,
            _include_version=False,
        )
        require(interrupted.status == "cancelled" and interrupted.exit_code is not None,
                "live tool cancellation ignored")
        flood = runner.run(
            "python", ["-c", "print('x' * 10000)"], cwd=root, timeout=5, _include_version=False
        )
        require("[output truncated]" in flood.output and len(flood.output) < 200,
                "tool output budget ignored")
        payload = root / "payload.bin"
        payload.write_bytes(b"flag{independent_container_recovery}")
        cover = root / "cover.png"
        Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        ).save(cover)
        native = root / "native.png"
        StegoService().embed(payload, cover, native)
        report = CTFService().solve(native, root / "native-job", mode="quick")
        require(report.verdict == "confirmed", "native extraction not confirmed")
        require(
            any(a.path and a.path.read_bytes() == payload.read_bytes() for a in report.artifacts),
            "native payload mismatch",
        )
        try:
            CTFService().solve(native, root / "native-job", mode="quick")
        except FileExistsError:
            pass
        else:
            raise RuntimeError("existing output overwritten")
        link = root / "input-link.png"
        link.symlink_to(native)
        try:
            CTFService().solve(link, root / "link-job", mode="quick")
        except ValueError:
            pass
        else:
            raise RuntimeError("artifact input symlink accepted")
        cancelled = CTFService().solve(native, root / "cancelled-job", should_cancel=lambda: True)
        require(cancelled.status == "cancelled", "job cancellation ignored")
        bmp = root / "cover.bmp"
        with Image.open(cover) as image:
            image.save(bmp)
        stego = root / "external.bmp"
        executable = shutil.which("steghide")
        if executable is None:
            raise RuntimeError("missing steghide")
        subprocess.run(  # noqa: S603 - fixed executable and generated fixture
            [
                executable,
                "embed",
                "-cf",
                str(bmp),
                "-ef",
                str(payload),
                "-sf",
                str(stego),
                "-p",
                "fixture-password",
                "-f",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        external = CTFService().solve(
            stego,
            root / "external-job",
            password="fixture-password",  # noqa: S106 - public test fixture
            limits=CTFLimits(tool_timeout=5, job_timeout=60),
        )
        require(external.status == "completed", f"external job: {external.error}")
        require(
            any(a.path and a.path.read_bytes() == payload.read_bytes() for a in external.artifacts),
            "independent steghide payload mismatch",
        )
        require("fixture-password" not in str(external.to_dict()), "password leaked")
        require(str(root) not in str(external.to_dict()), "absolute sandbox path leaked")
        require(all(a.sha256 and a.provenance for a in external.artifacts),
                "artifact provenance missing")
        wordlist = root / "fixture-wordlist.txt"
        wordlist.write_text("wrong\nfixture-password\n", encoding="utf-8")
        seek_dir = root / "stegseek-budget"
        seek_dir.mkdir()
        seek = ToolRunner(timeout=10).run(
            "stegseek", ["../external.bmp", "../fixture-wordlist.txt", "payload.bin", "-f"],
            cwd=seek_dir,
        )
        require(seek.status == "completed", "Stegseek fixture recovery failed")
        require((seek_dir / "payload.bin").read_bytes() == payload.read_bytes(),
                "Stegseek exact payload mismatch")
        seek_report = CTFService().solve(stego, root / "wordlist-job", wordlist=wordlist)
        require(seek_report.status == "completed", "wordlist playbook failed")
        require(
            any(a.path and a.path.read_bytes() == payload.read_bytes()
                and "stegseek" in a.provenance for a in seek_report.artifacts),
            "wordlist playbook did not recover payload through Stegseek",
        )
        require(any(t.tool == "stegseek" and t.status == "completed" and t.exit_code == 0
                    for t in seek_report.tools), "Stegseek success provenance missing")
        require("fixture-password" not in str(seek_report.to_dict()),
                "recovered wordlist passphrase leaked")
        openstego = shutil.which("openstego")
        if openstego is None:
            raise RuntimeError("missing openstego")
        openstego_file = root / "openstego.png"
        subprocess.run(  # noqa: S603 - independent known fixture generator
            [
                openstego,
                "embed",
                "-a",
                "randomlsb",
                "-cf",
                str(cover),
                "-mf",
                str(payload),
                "-sf",
                str(openstego_file),
                "-E",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        require(openstego_file.is_file(), "OpenStego exited without producing a file")
        # Repeated cold JVM starts catch native-memory failures hidden by a single
        # successful extraction. Do not retry, raise budgets or accept crash output.
        runner = ToolRunner(timeout=5)
        for attempt in range(10):
            workdir = root / f"openstego-budget-{attempt}"
            workdir.mkdir()
            execution = runner.run(
                "openstego",
                ["extract", "-sf", "../openstego.png", "-xd", ".", "-xf", "payload.bin"],
                cwd=workdir,
            )
            require(execution.status == "completed", "OpenStego failed under default budget")
            require(
                (workdir / "payload.bin").read_bytes() == payload.read_bytes(),
                "OpenStego budget-repeat payload mismatch",
            )
        recovered = CTFService().solve(openstego_file, root / "openstego-job")
        require(recovered.status == "completed", f"OpenStego job: {recovered.error}")
        require(
            any(
                a.path and a.path.read_bytes() == payload.read_bytes() for a in recovered.artifacts
            ),
            "independent OpenStego payload mismatch",
        )
        print("PASS: pinned inventory, exact recovery, limits and redaction")
        for execution in external.tools:
            print(f"{execution.tool}: {execution.status}")
            if execution.status == "failed":
                print(execution.output[:400])


if __name__ == "__main__":
    main()
