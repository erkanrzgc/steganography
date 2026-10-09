# steganography

Local-first steganography and DFIR workbench for guided file analysis,
encrypted evidence handling, payload hiding/recovery, and reproducible
research.

`steganography` helps an analyst inspect suspicious files, preserve evidence in
an encrypted local vault, hide and recover versioned payloads, and produce
repeatable reports. Deterministic detectors, bounded optional tools, and
optional signed ONNX models run on the machine you control.

It does **not** prove that a file is clean, recover arbitrary unknown stego
schemes, replace manual forensic validation, or make unencrypted hidden data
confidential. No default operation makes a network request.

Measured detection is still experimental: the published real-image baselines
failed, and no cross-source support gate has passed. See the
[results by method](docs/BENCHMARK_RESULTS.md) and
[dataset sources/licenses](docs/DATASET_CATALOG.md). Controlled CTF recovery
and synthetic regression scores are not overall steganalysis accuracy.

[Verified data expansion](docs/DATA_EXPANSION_RESULTS.md) adds 4,000 ALASKA2
JPEGs, 1,000 BOSS originals and 200 reserved WIFD images. Acquisition is not
accuracy. [Complete JPEG preparation](docs/JPEG_SCALE_PREPARATION_RESULTS.md)
now contains 7,380 train and 1,611 validation rows in bounded blocks; no new
model has been fitted. [Full-block streaming readiness](docs/SRNET_STREAM_READINESS_RESULTS.md)
now verifies every training row without a whole-corpus tensor allocation and
preserves the numerical trainer behavior. Adequate real fitting and blind-source
testing remain; engineering checks are not an accuracy improvement.
An [opt-in Windows/WSL2 CUDA workflow](docs/WINDOWS_GPU_TRAINING.md) is now
implemented. Its [first physical GPU preflight](docs/WSL_GPU_READINESS_RESULTS.md)
completed on an RTX 5060 Laptop in WSL2: one generated optimizer update and
same-weight CPU/GPU forward agreement. This is not real-data learning or
accuracy; the Kali development VM remains CPU-only.

The first real SRNet pilot also failed all six detection cells (AUC 0.502–0.510,
balanced accuracy 50%), despite passing numerical replay. See the
[complete pilot results](docs/SRNET_REAL_PILOT_RESULTS.md); no weights are deployed.

[Train-only BN diagnosis](docs/SRNET_TRAIN_DIAGNOSTIC_RESULTS.md) now exposes
normalization-mode sensitivity. Its in-sample paired results are not accuracy;
single-file detection remains unqualified and original failures remain published.

[Two-source/four-row real training](docs/SRNET_MULTIPAIR_REAL_RESULTS.md) completed:
all six detection cells still fail (AUC .501–.507, balanced accuracy 50–52%).
Overconfidence decreases but false alarms rise; no model is deployed.

Research models use byte-derived spatial/DCT signals, not filenames or dataset
identity as inputs. That does not prove generalization: compression/content
can still encode source bias. [Source/context diagnostics](docs/GENERALIZATION_PROTOCOL.md)
keep same-source development scores separate from independent-source evidence.
The [current audit](docs/GENERALIZATION_RESULTS.md) finds no unseen validation
sources for either research model; cross-source generalization is unproven.

## Choose a task

| Goal | Start here | What happens |
|---|---|---|
| Scan suspicious files | `steganography tui` → **Scan suspicious files** | Temporary multi-file analysis; nothing enters the vault unless you choose **Preserve in a case** |
| Hide data | `steganography tui` → **Hide data** | Capacity check and payload v3 embedding; encryption is optional but strongly recommended |
| Recover data | `steganography tui` → **Recover hidden data** | Automatic method detection, metadata preview, and no-overwrite extraction |
| Manage evidence | `steganography tui` → **Manage evidence** | Encrypted vault, cases, evidence custody, persistent scans, and report export |

## Three-minute start

Python 3.11 or newer is required. The distribution name differs from the
import and command names:

```bash
python -m pip install steganography-dfir
steganography tui
```

The TUI works without a local web server. It starts in Guided mode, uses a
Balanced scan profile, supports keyboard navigation, and displays a warning
below 80×24 terminals. Press `?` for help.

For the browser workspace and API:

```bash
python -m pip install 'steganography-dfir[api]'
steganography ui
```

Or run the hardened, non-root, read-only container:

```bash
docker compose up --build
docker compose exec steganography sh -c 'cat /state/v2-api.key'
```

Open `http://127.0.0.1:8000`. See [Getting started](docs/GETTING_STARTED.md)
for the complete first-run walkthrough.

## Formats and available features

| Format | Hide / recover | Built-in analysis |
|---|---|---|
| PNG, BMP | Sequential or keyed scattered RGB LSB | LSB statistics, bit planes, known payload markers, trailing data |
| JPEG | EXIF, marker payload; experimental DCT | EXIF/structure, marker and trailing data; optional external tools/model |
| TIFF | EXIF | Metadata and structure |
| WAV PCM16 | Sample LSB | Audio LSB indicators and markers |
| TXT, MD | Whitespace or zero-width encoding | Whitespace and Unicode anomaly analysis |
| GIF, PDF | Checksummed structural trailer | File structure and trailing-data analysis |
| ZIP and supported archives | — | Bounded member/path/compression anomaly inspection |

Optional `zsteg`, Stegseek, ExifTool, JPEG DCT support, ONNX Runtime, and AI
triage are reported as `unavailable` when absent. A low score is not proof of
a clean file; the pipeline's [minimum coverage policy](docs/COVERAGE_POLICY.md)
makes incomplete low-score scans inconclusive. Feature availability is not a validated detection
support claim. See the full [detection guide](docs/DETECTION_GUIDE.md).

## Steganography is not encryption

Steganography conceals that a payload exists. Encryption protects the content
when it is found. A passwordless payload is hidden but readable by anyone who
can identify the method. Guided Hide requires explicit acknowledgement before
creating one. Password-protected payload v3 uses Argon2id and AES-256-GCM;
case evidence uses an Argon2id-unlocked XChaCha20 secretstream vault.

## Reading verdicts

| Verdict | Meaning |
|---|---|
| `confirmed` | A verified project marker was found or extraction succeeded |
| `likely` | Strong evidence exists; validate with another method or extraction |
| `suspicious` | One or more heuristics deserve review and may be false positives |
| `no_indicators` | Supported detectors found nothing; this is not proof of absence |
| `inconclusive` | Required analyzers were unavailable, unsupported, or failed |

Scores are prioritization aids, not probabilities. Natural image noise,
editing software, metadata, appended signatures, and unusual Unicode can
produce false positives. Results therefore include analyzer names, evidence
strength, locations when known, unavailable tooling, and suggested next steps.

## Privacy and trust boundaries

- Analysis, vault, Studio, reports, and models are local by default.
- Passwords and placement keys remain in memory and are cleared from TUI fields;
  they are never written to the database, audit chain, or logs.
- AI is off by default. `--ai` sends signal text only. File upload additionally
  requires `--allow-ai-file-upload` and server-side opt-in.
- Models are not downloaded automatically. Installation requires a local,
  Ed25519-signed manifest and matching SHA-256.
- Optional external executables run with time, memory, descriptor, file-size,
  and output bounds. The full container documents their separate licenses.

The detailed boundaries are in the [threat model](docs/THREAT_MODEL.md),
[model guide](docs/MODELS.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

## CLI examples

```bash
# Existing compatible embed/extract defaults remain available.
steganography embed --in secret.bin --carrier cover.png --out stego.png
steganography extract --in stego.png --out recovered.bin --no-clobber

# Encrypted payload v3 with compression and Reed–Solomon protection.
steganography embed --payload-version 3 --compress --ecc-symbols 8 \
  --password-file ./password --in secret.bin --carrier cover.wav --out stego.wav

# One-file and directory analysis.
steganography analyze --in suspect.png --format json-v2 --profile balanced
steganography scan --dir ./evidence --report sarif --out findings.sarif --jobs 4

# Case workflow.
steganography case --state-dir ./state create --name "Case 2026-014"
steganography case --state-dir ./state add CASE_ID --file suspect.png --password-stdin
steganography case --state-dir ./state scan CASE_ID --profile balanced --password-stdin
steganography case --state-dir ./state export SCAN_ID --format html --out report.html
```

See the complete [CLI reference](docs/CLI.md) and [user guide](docs/USER_GUIDE.md).

## Python API

```python
from pathlib import Path

from core.pipeline import AnalysisPipeline

report = AnalysisPipeline().analyze(Path("suspect.png"))
print(report.verdict, report.confidence)
for finding in report.findings:
    print(finding.analyzer, finding.detail)
```

The compatible `/v1` endpoints and authenticated `/v2` case, Studio, model,
artifact, report, and audit APIs are documented in [API.md](docs/API.md).

## Development and release evidence

Real-data baselines, development experiments and controlled CTF recovery are
reported separately in [measured results](docs/BENCHMARK_RESULTS.md). The
[two-origin JPEG context experiment](docs/JPEG_CONTEXT_RESULTS.md) failed its
detection targets and is not deployed; no method has qualified cross-source support.
Its [residual/parity follow-up](docs/JPEG_RESIDUAL_RESULTS.md) lowers some false
alarms but also loses recall; it remains an undeployed development experiment.
The latest [nonlinear feature comparison](docs/JPEG_NONLINEAR_RESULTS.md)
improves ALASKA ranking/FPR, but detection still fails; BOSS/UERD and calibration
regress. It is research-only, not a deployed CNN or a global accuracy claim.
The optional [JRM + FLD reference](docs/JRM_REFERENCE_RESULTS.md) raises ALASKA/
UERD AUC to .698 and lowers FPR to 31.71%, but JUNIWARD recall regresses and
BOSS remains weak. All cells fail detection targets; research-only, not deployed.
The [two-way training-source exclusion test](docs/JRM_TRANSFER_RESULTS.md)
finds near-chance cross-origin AUC (.52–.55); source generalization remains unproven.
The [pixel-residual preparation](docs/PIXEL_RESIDUAL_PREPARATION_RESULTS.md)
verifies all 3,750 raw inputs and bounded CNN building blocks.
The [first real pixel-CNN trial](docs/PIXEL_CNN_TRAINING_RESULTS.md) completed
three fixed fits: all 12 cells fail detection (AUC .502–.509, recall 0%).
Independent numeric/ONNX replay passes; this model is not deployed.
The [controlled residual-domain follow-up](docs/PIXEL_CNN_SCALE_RESULTS.md)
also fails all cells (AUC .495–.509, zero recall); no accuracy gain claimed.
Next [SRNet-style preparation](docs/SRNET_PREPARATION.md) adds architectural
readiness only, not a trained detector or a reproduced published score.
[Unrounded JPEG preparation](docs/JPEG_FLOAT256_RESULTS.md) now verifies all
3,750 originals and 18 independent mathematical examples; accuracy still unavailable.
[Paired training schedules](docs/SRNET_SAMPLING_RESULTS.md) also pass all 30
scope/epoch integrity checks; no real SRNet fitting or accuracy gain yet.
[Paired CPU fitting](docs/SRNET_TRAINING_ENGINE.md) is now available explicitly
through `research srnet-fit`; generated tests do not establish real accuracy.
`research srnet-evaluate` verifies complete validation/cache/model provenance
and independent NumPy replay before publishing research predictions. See
[evaluation contract](docs/SRNET_EVALUATION.md); no new real accuracy claim.
[Isolated jobs and numerical readiness](docs/SRNET_NUMERICAL_READINESS.md)
add hard fitting limits and independent NumPy/ONNX checks, not a detection score.
Missing [required native coverage](docs/COVERAGE_POLICY.md) makes low-score
results inconclusive; completed analysis is still not proof that a file is clean.

```bash
python -m pip install '.[dev,dct]'
ruff check .
mypy core modules report api steganography ui registry.py config.py cli.py
pytest
npm --prefix web test
npm --prefix web run build
python -m build
python scripts/generate-sbom.py --out dist/steganography.cdx.json
python scripts/check-identity.py dist
```

The wheel is named `steganography_dfir-<version>-py3-none-any.whl`; the Python
import and console command remain `steganography`. Research benchmark gates,
payload compatibility, contribution rules, and disclosure policy are covered
by [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), and
[SECURITY.md](SECURITY.md).

Licensed under the [MIT License](LICENSE).
