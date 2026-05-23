# steganography

Dual-purpose steganography toolkit — **embed/extract** and **steganalysis**.

A standalone Python 3.11+ companion to [cyberm4fia-scanner](https://github.com/erkanrzgc/cyberm4fia-scanner). Shares the modular `modules/` plug-in style, `rich` CLI, and the cyberm4fia gradient banner.

## Features

- **Embed / extract** across image (PNG/BMP LSB, JPEG appended-data), audio (WAV LSB), text (zero-width unicode, trailing whitespace), and file-structure (EXIF UserComment) carriers
- **AES-256-GCM** encryption layer with scrypt password KDF
- **Steganalysis**: per-carrier statistical heuristics + known-tool signature detection + polyglot/appended-data detection
- **Pluggable AI triage** (any callable matching the `AIProvider` protocol)
- **Batch directory scan** with JSON and HTML reports
- Auto-discovered `modules/` — drop a new file in to add a technique

## Install

```bash
git clone <repo> steganography && cd steganography
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Usage

```bash
# Embed
python cli.py embed --in secret.bin --carrier cover.png --out stego.png [--password pw]

# Extract
python cli.py extract --in stego.png --out recovered.bin [--password pw]

# Analyze a single file
python cli.py analyze --in suspect.png [--json]

# Batch scan a directory
python cli.py scan --dir ./samples --report html --out report.html

# List discovered modules
python cli.py list-modules
```

Set `STEGANO_PASSWORD` env var to avoid passing `--password` on the command line.

## Configuration

All secrets — including any future AI provider keys (NVIDIA NIM, Anthropic, OpenAI) — are loaded through `config.get_secret(...)`, which reads from `.env` or the process environment. **No literal keys appear in source code.**

To set up locally:

```bash
cp .env.example .env
# edit .env and fill in the values you need
```

The toolkit works fully without any AI key — the `ai_triage` analyzer falls back to a heuristic-only score derived from other analyzers' signals when no provider is wired up.

### NVIDIA NIM (vision model) integration

A reference provider is shipped at `modules/ai_provider_nim.py`. It uses NIM's OpenAI-compatible Chat Completions API; for image files it sends the image base64-encoded alongside the heuristic signals, so a vision model can both look at the file and weigh prior findings.

```bash
# 1. Get an NVIDIA NIM API key from build.nvidia.com → copy to .env
# NVIDIA_NIM_API_KEY=nvapi-...
# NVIDIA_NIM_MODEL=meta/llama-3.2-90b-vision-instruct       (default)
# NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1   (default)

# 2. Use --ai on analyze or scan
python cli.py analyze --in suspect.png --ai --json
python cli.py scan    --dir ./samples --ai --report html --out report.html
```

If the key is missing, `--ai` prints a warning and falls back to heuristics. Network/parse errors return suspicion `0` with the error in the explanation — scans never crash on a flaky AI call.

## Architecture

See the [design spec](docs/superpowers/specs/2026-05-23-steganography-design.md) and [implementation plan](docs/superpowers/plans/2026-05-23-steganography.md).

Key idea: **repo root = import root**; `modules/` is auto-discovered by `registry.py`. Each module is one file implementing either `Carrier` (embed/extract/analyze) or `Analyzer` (analyze-only). Adding a new technique = drop a file into `modules/`; nothing else to edit.

```
steganography/
├── cli.py                     # entry point
├── config.py                  # central secret loader (.env via python-dotenv)
├── registry.py                # auto-discovery + dispatch
├── core/                      # Carrier ABC, Analyzer ABC, crypto, payload header, result types
├── modules/                   # one file = one capability
│   ├── image_lsb.py
│   ├── image_jpeg.py
│   ├── audio_wav.py
│   ├── text_zerowidth.py
│   ├── text_whitespace.py
│   ├── filestruct_exif.py
│   ├── filestruct_appended.py
│   ├── signatures.py
│   └── ai_triage.py
├── report/                    # JSON + HTML report writers
├── ui/                        # gradient banner
└── tests/                     # pytest, ≥80% coverage gate
```

## Testing

```bash
pytest                  # runs full suite + coverage gate
ruff check .            # lint
```

Coverage gate: **80%** (enforced via `pyproject.toml`). Current: 94%.

## Ethical use

This toolkit is for **authorized** red-team operations, blue-team / DFIR analysis, CTF play, security research, and education. Do not use it against systems or data you are not authorized to test. The author assumes no responsibility for misuse.

## License

MIT.
