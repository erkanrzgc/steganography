# steganography

**Inspect files. Recover hidden payloads. Preserve the evidence.**

[![CI](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-beta-64748B?style=flat-square)](docs/ROADMAP.md)
[![License](https://img.shields.io/badge/license-MIT-0F766E?style=flat-square)](LICENSE)

A local-first steganography and digital-forensics toolkit with a guided terminal
interface, bounded CTF recovery, encrypted evidence storage, and portable reports.
Analysis uses native detectors and optional local tools—not cloud AI services.

[Getting started](docs/GETTING_STARTED.md) · [Documentation](docs/README.md) ·
[Measured results](docs/BENCHMARK_RESULTS.md) · [Security](SECURITY.md)

## What you can do

- **Inspect suspicious files:** structure, metadata, known markers, statistical
  indicators, Unicode anomalies, and appended data, with explicit analyzer coverage.
- **Work through CTF challenges:** bounded decoding, carving, extraction, and
  recursive artifact inspection with provenance and user-supplied wordlists.
- **Hide and recover payloads:** compatible payload v2/v3 workflows, optional
  password encryption, compression, and error correction.
- **Preserve evidence:** encrypted local vault, cases, SHA-256 identities,
  custody events, and JSON, HTML, SARIF, or evidence-bundle exports.

Use the CLI, guided TUI, or optional browser/API workspace. They share the same
application services; optional executables are not required by the base package.

## Quick start

Requires Python **3.11+**. Install the current source checkout in a virtual environment:

```bash
git clone https://github.com/erkanrzgc/steganography.git
cd steganography
python -m venv .venv
source .venv/bin/activate
python -m pip install .
steganography tui
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1`. The package distribution
is `steganography-dfir`; the command and Python import are `steganography`.

For the optional browser workspace and API:

```bash
python -m pip install '.[api]'
steganography ui
```

Keep the service on loopback unless you have configured authentication and
deployment controls. See [Getting started](docs/GETTING_STARTED.md) for Docker,
vault setup, and first-run details.

## Common workflows

```bash
# Inspect one file and produce a structured report.
steganography analyze --in suspect.png --format json-v2 --profile balanced

# Run the bounded CTF playbook into a new output directory.
steganography ctf challenge.png --out ./ctf-result --report bundle

# Scan a directory for integration with security tooling.
steganography scan --dir ./evidence --report sarif --out findings.sarif

# Hide an encrypted payload; enter its password through stdin.
steganography embed --in secret.bin --carrier cover.png --out stego.png \
  --payload-version 3 --password-stdin --no-clobber

# Recover without overwriting an existing file.
steganography extract --in stego.png --out recovered.bin \
  --password-stdin --no-clobber
```

In the TUI, start with **Scan suspicious files**, **Hide data**, **Recover hidden
data**, or **Manage evidence**. Press `?` for help. The [CLI reference](docs/CLI.md)
covers limits, optional tools, cases, and advanced options.

## Format coverage

| Carrier | Hide / recover | Available analysis |
|---|---|---|
| PNG, BMP | RGB LSB, sequential or keyed | LSB statistics, bit planes, markers, trailing data |
| JPEG | EXIF and marker payloads; experimental DCT | Metadata, structure, markers; optional DCT/tools |
| TIFF | EXIF | Metadata and structure |
| WAV PCM16 | Sample LSB | Sample-plane indicators and markers |
| TXT, MD | Whitespace and zero-width encoding | Whitespace and Unicode anomalies |
| GIF, PDF | Checksummed structural trailer | Structure and trailing data |
| ZIP and supported archives | — | Bounded member and compression inspection |

This describes implemented features, **not validated detection accuracy**.
Missing optional tools are reported as unavailable; missing required native
coverage makes a low-score result inconclusive. See the
[detection guide](docs/DETECTION_GUIDE.md) and [coverage policy](docs/COVERAGE_POLICY.md).

## Know what a result means

`confirmed` requires a verified project marker or successful extraction.
`likely` and `suspicious` are evidence for further investigation.
`no_indicators` is not proof that a file is clean; `inconclusive` means the
required analysis could not be completed. Scores are not calibrated probabilities.

**Real-world statistical detection remains experimental.** No research model has
passed the project's cross-source support gates. Small development gains and
synthetic regression results are not an overall accuracy score. Research code
and failed experiments remain available for reproducibility, separate from
advertised capabilities; no model is downloaded or deployed automatically.

Read [Known limitations](docs/LIMITATIONS.md), [Measured results](docs/BENCHMARK_RESULTS.md),
and [Dataset sources and licenses](docs/DATASET_CATALOG.md) before relying on a verdict.

## Local by design

- Built-in analysis does not upload files or signals to cloud providers.
- External tools run with resource limits; extracted artifacts are never executed.
- Payload v3 supports Argon2id and AES-256-GCM. The evidence vault uses
  Argon2id and XChaCha20 secretstream. **Hiding is not encryption.**
- Model installation and dataset acquisition are explicit operations. The base
  wheel contains no model weights or datasets.

Third-party plug-ins and executables have their own trust boundaries. Review
the [threat model](docs/THREAT_MODEL.md) before handling sensitive evidence.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, and compatibility rules.
Report vulnerabilities privately through [SECURITY.md](SECURITY.md).
Changes and planned milestones are tracked in [CHANGELOG.md](CHANGELOG.md) and
the [roadmap](docs/ROADMAP.md).

## License

Project code is licensed under [MIT](LICENSE). Optional tools, research
dependencies, models, and datasets retain their own licenses and usage terms;
the project's license does not relicense them. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and [Licensing boundaries](docs/LICENSING.md).
