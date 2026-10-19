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

- **Inspect files:** metadata, markers, statistical indicators, Unicode and appended data.
- **Solve CTF challenges:** bounded decoding, carving and extraction with provenance.
- **Hide/recover payloads:** v2/v3 compatibility, password encryption, compression and error correction.
- **Preserve evidence:** encrypted vault, cases, SHA-256 identities, custody and portable reports.

CLI, TUI and optional browser/API share application services.

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

Windows: `.\.venv\Scripts\Activate.ps1`. Distribution: `steganography-dfir`;
command/import: `steganography`.

For the optional browser workspace and API:

```bash
python -m pip install '.[api]'
steganography ui
```

Keep the service on loopback unless authentication/deployment controls are configured.
See [Getting started](docs/GETTING_STARTED.md) for Docker and vault setup.

## Common workflows

```bash
# Inspect a file.
steganography analyze --in suspect.png --format json-v2 --profile balanced

# Run CTF recovery.
steganography ctf challenge.png --out ./ctf-result --report bundle

# Scan a directory.
steganography scan --dir ./evidence --report sarif --out findings.sarif

# Hide a payload; password via stdin.
steganography embed --in secret.bin --carrier cover.png --out stego.png \
  --payload-version 3 --password-stdin --no-clobber

# Recover without overwriting.
steganography extract --in stego.png --out recovered.bin \
  --password-stdin --no-clobber
```

Press `?` in the TUI for help. See the [CLI reference](docs/CLI.md)
for limits, optional tools and advanced options.

## Format coverage

| Carrier | Hide / recover | Available analysis |
|---|---|---|
| PNG, BMP | Sequential/keyed RGB LSB | Statistics, bit planes, markers, trailers |
| JPEG, TIFF | EXIF; experimental JPEG DCT | Metadata, structure; optional tools |
| WAV PCM16 | Sample LSB | Sample-plane indicators and markers |
| TXT, MD | Whitespace and zero-width encoding | Whitespace and Unicode anomalies |
| GIF, PDF | Checksummed trailer | Structure and trailers |
| ZIP and supported archives | — | Bounded member and compression inspection |

This describes implemented features, **not validated detection accuracy**.
Missing optional tools are reported as unavailable; missing required native
coverage makes a low-score result inconclusive. See the
[detection guide](docs/DETECTION_GUIDE.md) and [coverage policy](docs/COVERAGE_POLICY.md).

## Know what a result means

`confirmed` requires a verified marker or successful extraction.
`likely`/`suspicious` need investigation; `no_indicators` does not establish clean
content. `inconclusive` means required coverage is missing. Scores are uncalibrated.

**Detection remains experimental.** No research model has passed cross-source
support gates; synthetic regression results are not an overall accuracy score.
Models are never downloaded or deployed automatically. See
[limitations](docs/LIMITATIONS.md), [measurements](docs/BENCHMARK_RESULTS.md) and
[dataset licenses](docs/DATASET_CATALOG.md).

## Early training results

Development evaluations, **2026-10-14–16**, with controlled payloads.
**Balanced accuracy** averages recall and specificity; it equals accuracy here.
**ROC-AUC measures ranking, not accuracy**; 0.50 indicates chance-level ranking.

| Experiment / model | Development source | Balanced accuracy | ROC-AUC |
|---|---|---:|---:|
| JPEG SRNet-style, initial four arms | BOSSbase | 50.78–67.97% | 0.529–0.771 |
| JPEG SRNet-style, initial four arms | BOWS2 | 50.00–63.28% | 0.518–0.679 |
| JPEG refinement, all arms | BOSSbase | 50.00–76.56% | 0.502–0.848 |
| JPEG refinement, all arms | BOWS2 | 50.00–71.09% | 0.502–0.787 |
| PNG/BMP linear, all variants | BOWS2 | 50.00–100.00% | 0.609–1.000 |
| PNG/BMP linear, all variants | DIV2K | 50.00–93.75% | 0.547–1.000 |
| WAV linear, parity/residual | ESC-50 | 50.00% | 0.484–0.547 |
| WAV linear, parity/residual | FSDD | 50.00% | 0.359–0.656 |

Ranges cover both seeds and all evaluated arms/cells, not confidence intervals.
Native pilots have only **eight development originals/source** (two FSDD
speakers); JPEG has **64 pairs/source**. A 100% cell does not establish reliable
detection. No model qualified or deployed; native deep skeletons remain untrained.
AudioMNIST WAV has no accuracy yet; its generated GPU preflight passed.
See the [preparation report](docs/WAV_PILOT_PREPARATION_20261017.md).
Per-seed results and limitations: [initial JPEG](docs/JPEG_CONTROL_RESULTS.md),
[refinement](docs/JPEG_REFINE_NUMERICS_RESULTS.md),
[native pilot](docs/NONJPEG_PROCESS_PREFLIGHT_RESULTS.md),
[WAV diagnosis](docs/WAV_READONLY_RESULTS.md).

## Dataset preparation by format

| Extension(s) | Dataset state |
|---|---|
| `.jpg/.jpeg`, `.png`, `.bmp`, historical mono `.wav` | Controlled datasets; completed research fits |
| AudioMNIST `.wav` follow-up | 600 source-bound files; GPU preflight passed, no training accuracy yet |
| `.gif` | 13 licensed files; ancestry/pilot incomplete |
| `.pdf`, `.txt/.md`, `.zip` | Prepared local pilot; training roles unassigned |
| `.tif/.tiff`, `.mp3`, `.au`, `.eml`, `.csv/.log` | Representative datasets unavailable |
| `.tar`, `.tgz`, TAR `.gz/.bz2/.xz` | Generated fixtures; representative corpus unavailable |
| `.wave`, compound TAR aliases | Shared families; no independent data added |

The [complete inventory](docs/ALL_FORMAT_DATA_STATUS.md) covers every built-in
suffix. Fixtures/parser success do not establish dataset or detection quality.

## Local by design

- No cloud uploads; external tools are bounded and extracted artifacts never execute.
- Payload v3: Argon2id/AES-256-GCM. Vault: Argon2id/XChaCha20 secretstream.
  **Hiding is not encryption.**
- Explicit model/dataset acquisition; the base wheel includes neither.

Review the [threat model](docs/THREAT_MODEL.md) for third-party tool boundaries.

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
