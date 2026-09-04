# Contributing

Contributions that improve defensive analysis, interoperability,
documentation, accessibility, or reproducibility are welcome. Do not submit
real confidential evidence, copyrighted model weights/datasets without clear
redistribution rights, credentials, or features designed primarily for abuse.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,dct]'
npm --prefix web ci
```

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

Tests must be deterministic, avoid network access, and use synthetic fixtures.
New detectors need clean and positive cases plus documented false-positive
limits. TUI changes need headless `run_test()`/Pilot coverage and keyboard-only
operation at 80×24. Maintain compatible imports, console commands, payload
readers, and `/v1` behavior unless a breaking release is explicitly planned.

Keep commits focused and explain the user-visible behavior, threat-model
impact, tests, and third-party licensing implications in the pull request.
Security reports belong in the private channel described by [SECURITY.md](SECURITY.md).
