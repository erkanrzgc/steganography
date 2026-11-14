"""The external-tool guard starts safely from parallel analysis threads."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from modules import external_tool_worker as worker
from modules import external_tools as tools


def test_policy_limits_and_compatibility_entry(monkeypatch):
    import resource

    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, value: calls.append((kind, value)))
    tools._limits()
    assert calls == [
        (resource.RLIMIT_AS, (tools._MEMORY_BYTES, tools._MEMORY_BYTES)),
        (resource.RLIMIT_CPU, (20, 21)),
        (resource.RLIMIT_FSIZE, (tools._MAX_OUTPUT, tools._MAX_OUTPUT)),
        (resource.RLIMIT_CORE, (0, 0)),
        (resource.RLIMIT_NOFILE, (64, 64)),
    ]


@pytest.mark.parametrize("values", [(0, 20, 1), (1, 0, 1), (1, 20, 0),
                                    (True, 20, 1), (9 * 1024**3, 20, 1),
                                    (1, 3601, 1), (1, 20, 17 * 1024**2)])
def test_invalid_policy_precedes_any_resource_change(values):
    with pytest.raises(ValueError, match="policy"):
        worker.apply_limits(*values)


def test_launcher_preserves_arguments_and_applies_policy_before_exec(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "apply_limits", lambda *limits: calls.append(limits))
    monkeypatch.setattr(worker.os, "execv",
                        lambda executable, argv: calls.append((executable, argv)))
    monkeypatch.setattr(sys, "argv", ["worker", "1234", "5", "99", "--", "/fixed/tool", "a b;$(x)"])
    assert worker.main() == 0
    assert calls == [(1234, 5, 99), ("/fixed/tool", ["/fixed/tool", "a b;$(x)"])]
    assert tools._bounded_command(["tool", "argument"], posix=False) == ["tool", "argument"]


@pytest.mark.parametrize("args", [[], ["1", "2", "3", "--"],
                                  ["1", "2", "3", "wrong", "tool"],
                                  ["1", "2", "3", "--", ""],
                                  ["bad", "2", "3", "--", "/tool"],
                                  ["1", "2", "3", "--", "bad\0tool"],
                                  ["1", "2", "3", "--", "/tool"] + ["x"] * 252,
                                  ["1", "2", "3", "--", "/tool", "é" * 65536]])
def test_bad_launcher_arguments_are_redacted(args, capsys):
    assert worker.main(args) == 126
    assert capsys.readouterr().err == "external tool launcher unavailable\n"


def test_exec_error_is_redacted(monkeypatch, capsys):
    monkeypatch.setattr(worker, "apply_limits", lambda *_args: None)
    def fail(*_args):
        raise OSError("private argument")
    monkeypatch.setattr(worker.os, "execv", fail)
    assert worker.main(["1", "2", "3", "--", "/missing"]) == 126
    assert "private" not in capsys.readouterr().err


def test_parallel_real_launches_keep_limits_and_literal_arguments():
    # Real launches under the coverage collector exercise the path that
    # previously deadlocked in its traced preexec callback.
    source = ("import json,resource,sys; print(json.dumps({'args':sys.argv[1:],"
              "'as':resource.getrlimit(resource.RLIMIT_AS),"
              "'cpu':resource.getrlimit(resource.RLIMIT_CPU),"
              "'file':resource.getrlimit(resource.RLIMIT_FSIZE),"
              "'core':resource.getrlimit(resource.RLIMIT_CORE),"
              "'fds':resource.getrlimit(resource.RLIMIT_NOFILE)}))")
    literal = "literal space;$(not_a_command)"
    def launch(_index):
        return tools._run([sys.executable, "-I", "-c", source, literal])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(launch, range(12)))
    for code, output in results:
        assert code == 0
        value = json.loads(output)
        assert value == {"args": [literal], "as": [tools._MEMORY_BYTES] * 2,
                         "cpu": [20, 21], "file": [tools._MAX_OUTPUT] * 2,
                         "core": [0, 0], "fds": [64, 64]}


def test_real_launcher_propagates_exit_and_bounds_output(monkeypatch):
    assert tools._run([sys.executable, "-I", "-c", "raise SystemExit(37)"])[0] == 37
    monkeypatch.setattr(tools, "_MAX_OUTPUT", 64)
    code, output = tools._run([sys.executable, "-I", "-c", "print('x'*1000,flush=True)"])
    assert code != 0 and len(output.encode()) <= 64


def test_missing_executable_cannot_report_clean(tmp_path, monkeypatch):
    with pytest.raises(OSError, match="launcher unavailable"):
        tools._run([str(tmp_path / "missing-tool")])
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(tmp_path / "missing-tool"))
    result = tools.ExifToolAdapter().analyze(tmp_path / "input.png")
    assert result.status == "error" and result.error == "subprocess unavailable"
