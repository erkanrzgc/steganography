"""The full-image launcher constrains JVM overhead without raising tool budgets."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX full-image shell launcher")
def test_openstego_launcher_bounds_native_overhead_and_preserves_arguments(tmp_path):
    java = tmp_path / "java"
    java.write_text('#!/bin/sh\nprintf "%s\\n" "$MALLOC_ARENA_MAX" "$@"\n')
    java.chmod(0o755)
    wrapper = Path(__file__).resolve().parents[1] / "scripts" / "openstego-headless.sh"
    args = ["extract", "-sf", "name with spaces.png", "literal;$(not-a-command)"]
    result = subprocess.run(  # noqa: S603 - local launcher and fake Java, no shell interpolation
        ["/bin/sh", str(wrapper), *args],
        env={"PATH": str(tmp_path), "MALLOC_ARENA_MAX": "999"},
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "2"
    assert "-XX:ActiveProcessorCount=2" in lines
    assert "-Xmx128m" in lines
    assert lines[-len(args) :] == args
