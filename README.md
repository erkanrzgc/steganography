# Steganography

[![CI](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-beta-64748B?style=flat-square)](docs/ROADMAP.md)
[![License](https://img.shields.io/badge/license-MIT-0F766E?style=flat-square)](LICENSE)

A local Python toolkit for steganography analysis, payload recovery, CTF
investigation and digital forensics, with traceable evidence and portable reports.

CLI, terminal workbench and optional browser/API share application services.
Analysis runs locally, without cloud AI providers or uploads.

[Getting started](docs/GETTING_STARTED.md) · [Documentation](docs/README.md) ·
[Research status](docs/STATUS.md) · [Security](SECURITY.md)

## Capabilities

- **Analyze:** file structure, metadata, markers, bit planes, statistical indicators,
  Unicode anomalies, and appended data.
- **Recover:** bounded decoding, carving, extraction, and optional external tools,
  with artifact provenance and verified payload identities.
- **Embed/extract:** supported carriers, compatible v2/v3 payloads, password
  encryption, compression, and error correction.
- **Preserve:** encrypted vaults, cases, SHA-256 identities, custody records,
  and JSON, HTML, SARIF, or evidence-bundle exports.

## Installation

Requires **Python 3.11+**. Install from a source checkout:

```bash
git clone https://github.com/erkanrzgc/steganography.git
cd steganography
python -m venv .venv
source .venv/bin/activate
python -m pip install .
steganography tui
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1`. The distribution is
`steganography-dfir`; the command and import are `steganography`.

The base install ships no external tools, weights or datasets; optional extras
have separate setup and licenses, and downloads are explicit.

For the optional browser workspace and API:

```bash
python -m pip install '.[api]'
steganography ui
```

Open `http://127.0.0.1:8000`; [setup](docs/GETTING_STARTED.md)
covers authentication, Docker and vaults.

## Common workflows

```bash
# Analyze one file as structured JSON.
steganography analyze --in suspect.png --format json-v2 --profile balanced

# Recover candidates into a new directory and export an evidence bundle.
steganography ctf challenge.png --out ./ctf-result --report bundle

# Scan a directory and export SARIF.
steganography scan --dir ./evidence --report sarif --out findings.sarif

# Embed an encrypted v3 payload; read the password from stdin.
steganography embed --in secret.bin --carrier cover.png --out stego.png \
  --payload-version 3 --password-stdin --no-clobber
```

See the [CLI reference](docs/CLI.md); press `?` for TUI help.

## Format coverage

| Family | Embed / extract | Analysis |
|---|---|---|
| PNG, BMP | Sequential/keyed RGB LSB | Statistics, bit planes, markers, trailers |
| JPEG, TIFF | EXIF; experimental JPEG DCT | Metadata, structure, optional tools |
| WAV PCM16 | Sample LSB | Sample-plane indicators and markers |
| TXT, MD | Whitespace and zero-width encoding | Whitespace and Unicode anomalies |
| GIF, PDF | Checksummed trailer | Structure and trailers |
| ZIP and supported archives | — | Bounded member, comment and extra-field inspection |
| MP4, MOV, MKV, AVI | — | Container fields, unreferenced media data, lossless-frame LSB |
| PCAP, PCAPNG | — | DNS/ICMP tunnels and header covert channels |

These are implemented capabilities, **not validated detection accuracy**; see the
[format inventory](docs/ALL_FORMAT_DATA_STATUS.md).

Controlled research datasets exist for JPEG, PNG/BMP and mono WAV; other formats
rely on fixtures, rules or optional tools. Format support does not establish
benchmark coverage.

## Interpret results

`confirmed` requires a verified project marker or successful extraction.
`likely` and `suspicious` require investigation; `no_indicators` does not certify
clean content. Missing required coverage produces `inconclusive`. Optional-tool
absence is reported explicitly. Scores are uncalibrated.

No research model has passed the complete cross-source qualification gates;
synthetic regression results are not an overall accuracy score.
No research model is deployed automatically. See the
[detection guide](docs/DETECTION_GUIDE.md), [coverage policy](docs/COVERAGE_POLICY.md),
and [limitations](docs/LIMITATIONS.md).

## Training results

Controlled development evaluations, **2026-10-14–11-12**. Balanced accuracy averages
recall and specificity.
**ROC-AUC measures ranking, not accuracy.** Ranges span both seeds and all
reported arms/cells, not confidence intervals or expected real-world accuracy.

| Experiment | Development source | Balanced accuracy | ROC-AUC |
|---|---|---:|---:|
| JPEG SRNet-style, initial arms | BOSSbase | 50.78–67.97% | 0.529–0.771 |
| JPEG SRNet-style, initial arms | BOWS2 | 50.00–63.28% | 0.518–0.679 |
| JPEG refinement, all arms | BOSSbase | 50.00–76.56% | 0.502–0.848 |
| JPEG refinement, all arms | BOWS2 | 50.00–71.09% | 0.502–0.787 |
| PNG/BMP linear, all variants | BOWS2 | 50.00–100.00% | 0.609–1.000 |
| PNG/BMP linear, all variants | DIV2K | 50.00–93.75% | 0.547–1.000 |
| PNG/BMP RGB SRNet, native pilot | Commons | — | 0.518–0.528 |
| PNG/BMP RGB rich-model detector | DIV2K | 61.25–93.00% | 0.781–0.990 |
| PNG/BMP RGB rich-model detector | Commons | 57.50–72.50% | 0.630–0.755 |
| Gray LSB SRM, payload-specific | BOSSbase | 85.70–97.80% | 0.946–0.998 |
| Gray LSB SRM, unknown payload | BOSSbase | 72.10–94.50% | 0.877–0.994 |
| steghide, OutGuess, stegano, F5, OpenStego | DIV2K | 79.75–99.75% | 0.870–1.000 |
| WAV LSB trees | ESC-50 | 64.00–89.50% | 0.752–0.970 |
| WAV LSB trees | AudioMNIST | 57.75–99.75% | 0.628–1.000 |
| WAV LSB trees | FSDD | 49.20–58.00% | 0.496–0.603 |
| WAV linear, parity/residual | ESC-50 | 50.00% | 0.484–0.547 |
| WAV linear, parity/residual | FSDD | 50.00% | 0.359–0.656 |
| WAV standardized pair head | AudioMNIST | 47.75–78.75% | 0.467–0.933 |

Older pilots use 8–64 originals/source; newer rows 200–1,000 test items.
No model is qualified or deployed.

**80% per method:** most cells pass; misses are listed.
New rules catch network tunnels and video container hiding.
[Details](docs/DETECTION_EXPANSION_20261111.md),
[RGB rich model](docs/PNG_BMP_RGB_RICH_DETECTOR_20261111.md).

Evidence: [AudioMNIST pair head](docs/WAV_PAIR_NATIVE_RESULTS_20261023.md),
[JPEG controls](docs/JPEG_CONTROL_RESULTS.md),
[refinement](docs/JPEG_REFINE_NUMERICS_RESULTS.md),
[linear pilots](docs/NONJPEG_PROCESS_PREFLIGHT_RESULTS.md),
[WAV comparison](docs/WAV_PAIR_COMPARISON_RESULTS_20261022.md).

[PNG/BMP RGB Colab result](docs/PNG_BMP_COLAB_NATIVE_RGB_RESULTS_20261107.md):
development AUC **0.518 /0.528**; the learning rule failed.

## Development and licensing

Recorded CPU gate (2026-11-12): **4,578 tests passed**, two explicit CUDA skips,
**97.39% coverage**. [Verification](docs/DETECTION_EXPANSION_20261111.md#verification).
Setup, checks, and contribution rules: [CONTRIBUTING.md](CONTRIBUTING.md).
Report vulnerabilities through [SECURITY.md](SECURITY.md).

**Hiding is not encryption.** Review the [threat model](docs/THREAT_MODEL.md).
Project code is [MIT licensed](LICENSE); optional tools, models, and datasets
retain their own terms. See [licensing boundaries](docs/LICENSING.md),
[third-party notices](THIRD_PARTY_NOTICES.md), [changelog](CHANGELOG.md), and
[roadmap](docs/ROADMAP.md).
