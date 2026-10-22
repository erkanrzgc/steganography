# Steganography

[![CI](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/erkanrzgc/steganography/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-beta-64748B?style=flat-square)](docs/ROADMAP.md)
[![License](https://img.shields.io/badge/license-MIT-0F766E?style=flat-square)](LICENSE)

A local Python toolkit for steganography analysis, payload recovery, and digital
forensics. Inspect suspicious files, investigate CTF challenges, and preserve
findings with traceable evidence and portable reports.

Use the CLI, guided terminal workbench, or optional browser/API. All interfaces
share the same application services. Analysis runs locally, without cloud AI
providers or cloud uploads.

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

The base installation requires no external analysis tools and includes no model
weights or datasets. Optional tools and research dependencies have separate setup
and licensing requirements. Model and dataset acquisition is explicit.

For the optional browser workspace and API:

```bash
python -m pip install '.[api]'
steganography ui
```

Open `http://127.0.0.1:8000`. See
[Getting started](docs/GETTING_STARTED.md) for authentication, Docker, and vault setup.

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

# Extract without overwriting an existing file.
steganography extract --in stego.png --out recovered.bin \
  --password-stdin --no-clobber
```

Press `?` in the TUI for keyboard help. The [CLI reference](docs/CLI.md) covers
profiles, limits, cases, optional tools, and research commands.

## Format coverage

| Family | Embed / extract | Analysis |
|---|---|---|
| PNG, BMP | Sequential/keyed RGB LSB | Statistics, bit planes, markers, trailers |
| JPEG, TIFF | EXIF; experimental JPEG DCT | Metadata, structure, optional tools |
| WAV PCM16 | Sample LSB | Sample-plane indicators and markers |
| TXT, MD | Whitespace and zero-width encoding | Whitespace and Unicode anomalies |
| GIF, PDF | Checksummed trailer | Structure and trailers |
| ZIP and supported archives | — | Bounded member and compression inspection |

These are implemented capabilities, **not validated detection accuracy**.
The [complete format inventory](docs/ALL_FORMAT_DATA_STATUS.md) covers all
23 built-in suffixes, aliases, dataset states, and remaining gaps.

Controlled research datasets exist for JPEG, PNG/BMP, and mono WAV. GIF preparation
is incomplete; local PDF/TXT/MD/ZIP candidates have unassigned training roles.
Other extensions have parser fixtures or optional-tool coverage, with representative
datasets still unavailable. Format support does not establish benchmark coverage.

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

Controlled development evaluations, **2026-10-14–23**. Balanced accuracy averages
recall and specificity; these matched cells have equal cover/stego counts.
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
| WAV linear, parity/residual | ESC-50 | 50.00% | 0.484–0.547 |
| WAV linear, parity/residual | FSDD | 50.00% | 0.359–0.656 |
| WAV standardized pair head | AudioMNIST | 47.75–78.75% | 0.467–0.933 |

Older non-JPEG pilots use eight development originals/source; JPEG uses
64 pairs/source. AudioMNIST uses 200 development cuts grouped by ten speakers.
No model is qualified or deployed.

**Latest AudioMNIST pilot:** two seeds completed **2,280 total optimizer updates**.
The required 40%-plane-0 cells failed: AUC **0.467–0.487**, against **0.70**.
Pooled development AUC was **0.617 /0.623**; false-positive rates were **42–55%**.
The favorable plane-1 cells and successful generated control do not rescue
this native-data failure. [Full results and intervals](docs/WAV_PAIR_NATIVE_RESULTS_20261023.md).

<details>
<summary>Actual training curves and all development cells</summary>

![AudioMNIST training curves and development metrics](docs/assets/wav-pair-native-20261023.svg)

</details>

Protocols, source bindings, failures, and per-seed evidence:
[JPEG controls](docs/JPEG_CONTROL_RESULTS.md),
[refinement](docs/JPEG_REFINE_NUMERICS_RESULTS.md),
[native linear pilots](docs/NONJPEG_PROCESS_PREFLIGHT_RESULTS.md),
[generated comparison](docs/WAV_PAIR_COMPARISON_RESULTS_20261022.md).

## Development and licensing

Recorded CPU gate: **3,729 tests passed**, two explicit CUDA skips,
**97.00% coverage**. [Verification evidence](benchmarks/wav-pair-native-fit-verification-20261023.json).
Setup, checks, and contribution rules: [CONTRIBUTING.md](CONTRIBUTING.md).
Report vulnerabilities through [SECURITY.md](SECURITY.md).

**Hiding is not encryption.** Review the [threat model](docs/THREAT_MODEL.md).
Project code is [MIT licensed](LICENSE); optional tools, models, and datasets
retain their own terms. See [licensing boundaries](docs/LICENSING.md),
[third-party notices](THIRD_PARTY_NOTICES.md), [changelog](CHANGELOG.md), and
[roadmap](docs/ROADMAP.md).
