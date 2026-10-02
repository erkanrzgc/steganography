"""Run inside the full image with read-only root and a writable /tmp tmpfs."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from core.ctf import CTFLimits, CTFService
from core.service import StegoService
from core.tools import ToolRunner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(os.getuid() != 0, "smoke must run as non-root")
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
        print("PASS: tool inventory, native, Steghide and OpenStego exact recovery, redaction")
        for execution in external.tools:
            print(f"{execution.tool}: {execution.status}")
            if execution.status == "failed":
                print(execution.output[:400])


if __name__ == "__main__":
    main()
