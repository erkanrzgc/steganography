# Contributing

Contributions that improve defensive analysis, interoperability,
documentation, accessibility, or reproducibility are welcome. Do not submit
real confidential evidence, copyrighted model weights/datasets without clear
redistribution rights, credentials, or features designed primarily for abuse.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,api,dct]'
npm --prefix web ci
```

Research checks additionally need the relevant opt-in `research`, `jpeg-sim`,
or `jrm-reference` extras; review their [license boundaries](docs/LICENSING.md).
Missing dependencies or real datasets must be reported as skipped/unavailable,
not successful research validation. No training data or weights are downloaded
by these installation commands.

To reproduce CI's complete CPU-only regression and coverage environment:

```bash
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research-ci.txt
pytest
```

The lightweight Python 3.11–3.14 CI matrix uses `pytest --no-cov`, retaining
explicit optional-dependency skips. The separate complete research job owns
the unchanged total coverage gate; do not lower it to accommodate missing extras.

The complete provisioned suite can accumulate several GiB of temporary fixtures.
Check the temporary filesystem as well as the repository disk: `/tmp` may be
a small RAM-backed tmpfs even when the repository has free disk space. If so,
use a fresh, exclusive disk-backed scratch directory without changing any
coverage or parser limits:

```bash
mkdir -p .benchmark
steg_test_scratch=$(mktemp -d "$PWD/.benchmark/pytest-run.XXXXXXXX")
TMPDIR="$steg_test_scratch" pytest --basetemp "$steg_test_scratch/fixtures" \
  --cov-report=json:"$steg_test_scratch/coverage.json"
```

`--basetemp` clears its target; never point it at an existing corpus or caller
directory. Remove only the owned fixture directory after pytest and its children
exit, keeping logs/coverage and any failed research preparation for handoff.
An out-of-space invocation is a failed verification, not an optional-data skip.
Setting both locations keeps isolated workers' `tempfile` output off the small
tmpfs as well; `--basetemp` alone only relocates pytest-managed fixtures.

Before proposing a change, run:

```bash
ruff check .
mypy core modules report api steganography ui registry.py config.py cli.py
pytest
npm --prefix web test
npm --prefix web run build
python -m build
python scripts/check-identity.py dist
```

Tests must be deterministic and avoid network access. Use small synthetic or
independently sourced, redistributable golden fixtures; do not commit private
evidence or large corpora. Synthetic tests are regression evidence, not real
detection accuracy.
New detectors need clean and positive cases plus documented false-positive
limits. TUI changes need headless `run_test()`/Pilot coverage and keyboard-only
operation at 80×24. Maintain compatible imports, console commands, payload
readers, and `/v1` behavior unless a breaking release is explicitly planned.

Keep commits focused and explain the user-visible behavior, threat-model
impact, tests, and third-party licensing implications in the pull request.
Build release artifacts from a fresh checkout/build tree: stale `build/lib`
files can retain removed modules. Inspect the wheel for retired modules,
credentials, datasets, and model weights before publishing.
Security reports belong in the private channel described by [SECURITY.md](SECURITY.md).
Contributions must be compatible with the project's [MIT License](LICENSE);
identify any separately licensed material and retain its upstream notices.
