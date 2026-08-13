<h1 align="center">steganography</h1>

<p align="center">
  <b>Embed, extract and analyze hidden data from one defensive Python toolkit.</b><br>
  Versioned payloads · AES-256-GCM · evidence-oriented reports · reproducible benchmarks.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-22C55E?style=flat-square" alt="license">
  <img src="https://img.shields.io/badge/tests-119%20passing-22C55E?style=flat-square" alt="tests">
  <img src="https://img.shields.io/badge/coverage-91%25-22C55E?style=flat-square" alt="coverage">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black&style=flat-square" alt="ruff">
</p>

`steganography` is a Python 3.11+ toolkit for authorized DFIR, red-team,
research and educational work. It supports deterministic local analysis,
optional NVIDIA NIM triage, resilient directory scans and carrier plug-ins.

## Install

The distribution is named `cyberm4fia-steganography`; the command and Python
package remain `steganography`. Install a published release with:

```bash
pip install cyberm4fia-steganography
```

Or install the base CLI from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Install optional capabilities only when needed:

```bash
pip install '.[ai]'          # NVIDIA NIM provider
pip install '.[api]'         # FastAPI analysis service
pip install '.[dct]'         # experimental JPEG DCT carrier
pip install '.[ai,api,dct]'  # everything
pip install '.[dev]'         # tests, typing, lint and build tools
```

The legacy `requirements.txt` and `requirements-dev.txt` workflows remain
available. Build and installable wheels are supported:

```bash
python -m build
pip install dist/cyberm4fia_steganography-*.whl
```

## CLI

```bash
# Existing default methods remain compatible.
steganography embed --in secret.bin --carrier cover.png --out stego.png
steganography extract --in stego.png --out recovered.bin

# Passwords can come from the environment, a file or stdin.
STEGANO_PASSWORD='...' steganography embed \
  --in secret.bin --carrier cover.wav --out stego.wav --no-clobber
steganography extract --in stego.wav --out recovered.bin --password-file ./password

# Explicit new methods.
steganography embed --method filestruct_trailer \
  --in secret.bin --carrier cover.pdf --out stego.pdf
steganography embed --method image_lsb_scatter --steg-key placement-key \
  --channels rgb --in secret.bin --carrier cover.png --out scatter.png
steganography embed --method image_jpeg_dct --steg-key placement-key \
  --in secret.bin --carrier cover.jpg --out dct.jpg

# Analysis and reports.
steganography analyze --in suspect.png --format json-v1 --profile sensitive
steganography scan --dir ./evidence --report html --out report.html --jobs 4
steganography scan --dir ./evidence --report json-v1 --out report.json \
  --fail-on high

steganography list-modules
```

The default analysis profile is `sensitive`; `balanced` and `strict` are also
available. Exit status is unchanged unless `--fail-on medium|high` is set.
Directory scans do not follow symlinks by default, skip their own report file,
and record per-module failures instead of aborting the scan.

## Reproducible corpus and benchmarks

Generate a labeled clean/stego corpus and evaluate every analysis profile:

```bash
steganography corpus --out .benchmark/corpus --seed 20260813
steganography benchmark \
  --corpus .benchmark/corpus \
  --out .benchmark/report.json \
  --html .benchmark/report.html \
  --baseline benchmarks/baseline.json \
  --min-recall 0.95 \
  --max-fpr 0.05 \
  --jobs 4
```

With the `dct` extra installed, the default corpus contains 66 paired samples
over 11 carrier/format recipes and low, medium and high payload densities. Its
covers, payload bytes and keyed placement salts are derived from the seed;
identical dependencies therefore produce the same manifest digest and file
hashes. CI applies `benchmarks/constraints.txt` to pin the codec/numeric
reproducibility boundary. The generator writes through a staging directory and
`--force` only replaces directories carrying its own marker.

Reports include confusion matrices, precision, recall, specificity, FPR, F1,
ROC-AUC and average precision, plus method, density and format breakdowns.
Absolute gates and baseline-delta gates return a non-zero exit status on a
regression. The committed 0.5.0 baseline has 66 samples, recall `1.000` and FPR
`0.000` for all three profiles.

Use repeatable `--method` flags to build a smaller corpus or `--exclude-dct`
when the optional JPEG dependency is unavailable.

### Payload compatibility

New embeds use payload v2: 64-bit length, validated flags and a SHA-256
integrity field. The extractor continues to read payload v1 files produced by
the earlier release. Encrypted data uses scrypt-derived AES-256-GCM keys.

### Methods

| Method | Formats | Embed / extract | Notes |
|---|---|---:|---|
| `image_lsb` | PNG, BMP | yes | Compatible sequential RGB LSB default |
| `image_lsb_scatter` | PNG, BMP | yes | Explicit, keyed ChaCha20 placement |
| `audio_wav` | WAV PCM16 | yes | Sample LSB |
| `text_whitespace` | TXT, MD | yes | Compatible default |
| `text_zerowidth` | TXT, MD | yes | Explicit selection recommended |
| `filestruct_exif` | JPEG, TIFF | yes | Compatible JPEG/TIFF default |
| `image_jpeg` | JPEG | yes | Marker-based appended payload |
| `filestruct_trailer` | PDF, GIF | yes | Explicit, checksummed structural trailer |
| `image_jpeg_dct` | JPEG | yes | Explicit, keyed and experimental |
| `filestruct_appended` | PNG/JPEG/GIF/PDF | analysis | Polyglot/trailing-data detector |

JPEG DCT work is executed in a time-limited subprocess so native codec
failures cannot terminate a directory scan. Unsupported progressive or
arithmetic-coded JPEGs return a structured `unsupported` result.

## Analysis and reports

The stable `json-v1` schema provides:

- tool/schema versions and UTC timestamps;
- file name, size, detected type, extension mismatch and SHA-256;
- one overall score/severity plus every module result;
- categorized signals with evidence strength;
- structured `ok`, `error`, `unsupported` and `unavailable` statuses.

Scores combine the strongest signal from each independent category using a
noisy-OR model. Validated tool/payload markers score at least 95. AI can raise
the deterministic verdict but never lower it. HTML output is self-contained,
filterable and escapes every dynamic field.

### AI privacy

AI is off by default. `--ai` sends only locally generated signal text. Sending
an image requires the separate `--allow-ai-file-upload` flag:

```bash
export NVIDIA_NIM_API_KEY='nvapi-...'
steganography analyze --in suspect.png --ai
steganography analyze --in suspect.png --ai --allow-ai-file-upload
```

Optional model and endpoint settings are `NVIDIA_NIM_MODEL` and
`NVIDIA_NIM_BASE_URL`. Provider errors become structured findings; scans keep
running.

## REST API

The API is analysis-only: it never exposes embed/extract or accepts server
filesystem paths. Start it on loopback with:

```bash
pip install '.[api]'
steganography serve --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /healthz`
- `GET /v1/modules`
- `POST /v1/analyze` — one multipart upload, synchronous result
- `POST /v1/scans` — multipart batch, persistent background job
- `GET /v1/scans/{job_id}`
- `DELETE /v1/scans/{job_id}`

Jobs and result JSON are stored in SQLite; uploaded files are removed after
analysis. Defaults are 50 MiB per file, 20 files/200 MiB per batch, two worker
threads and 30-day result retention. Environment overrides are documented in
`.env.example`.

Binding outside loopback is refused unless `STEGANO_API_KEY` is set. Clients
then send `Authorization: Bearer <key>`. CORS is not enabled. Internet-facing
deployment still requires a hardened reverse proxy, TLS, network controls and
operational monitoring.

API image upload to AI requires both request opt-in and
`STEGANO_ALLOW_AI_FILE_UPLOAD=1` on the server.

## Plug-ins

Built-ins are discovered from `modules/`. Installed distributions can publish
`Carrier` or `Analyzer` classes through the `steganography.carriers` and
`steganography.analyzers` entry-point groups. Carrier identifiers must be
unique; load failures appear in `list-modules` rather than preventing startup.

The supported Python surface is available from the distribution package:

```python
from pathlib import Path
from steganography import AnalysisService, StegoService

verdict = AnalysisService(profile="balanced").analyze(Path("evidence.png"))
print(verdict.overall_score, verdict.file.sha256)
```

## Development

```bash
ruff check .
mypy core modules report api steganography registry.py config.py cli.py
pytest
python -m build
```

CI runs Ruff, mypy, a Python 3.11–3.14 test matrix, a 90% coverage gate, wheel
construction, an installed-CLI smoke test and the full deterministic benchmark.

### Release process

Version tags matching `vX.Y.Z` run the release workflow. It verifies that the
tag and package versions match, repeats lint/type/test/benchmark gates, builds
and checks the wheel and source archive, creates GitHub build-provenance
attestations, publishes to PyPI through OpenID Connect, then creates the GitHub
Release. Third-party actions are pinned to full commit SHAs.

Before the first release, configure a PyPI Trusted Publisher for project
`cyberm4fia-steganography`, repository `erkanrzgc/steganography`, workflow
`release.yml` and GitHub environment `pypi`. Protecting that environment with
required reviewers is recommended. Once CI is green, a maintainer can publish
by pushing a version-matching tag; the workflow uses no long-lived PyPI token.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Ethical use

Use this project only on data and systems you own or are explicitly authorized
to test. It is intended for defensive investigation, sanctioned red-team work,
CTFs, research and education. The author assumes no responsibility for misuse.

## License

[MIT](LICENSE)
