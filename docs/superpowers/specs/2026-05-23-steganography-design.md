# steganography — Design Spec

**Date:** 2026-05-23
**Status:** Approved (brainstorming complete)
**Author:** brainstormed with the user
**Related project:** [cyberm4fia-scanner](https://github.com/erkanrzgc/cyberm4fia-scanner) (sibling, not a dependency)

---

## 1. Overview

`steganography` is a standalone, dual-purpose steganography toolkit:

- **Embedding side** — hide arbitrary data (optionally AES-encrypted) inside images, audio, text, and file-structure carriers.
- **Detection side (steganalysis)** — analyze files for statistical anomalies, known-tool signatures, and AI-triaged suspicion scoring.

It shares stylistic conventions with the user's `cyberm4fia-scanner` project (Python 3.11+, `rich` CLI, flat repo-root package, scanner-style `modules/` layout, gradient ASCII banner) but is fully independent — no runtime dependency on the scanner.

### Intended uses
Authorized red-team operations, blue-team / DFIR analysis, CTF solving, security research, education.

### Non-goals
- Not a turnkey C2 framework.
- Not an obfuscation/evasion-against-detection tool — detection is a first-class feature, not an adversary.
- No network exfiltration channel in v1 (carriers only; transport is the caller's problem).

---

## 2. Scope (MVP)

### Carriers (embed + extract + analyze)
- **Image:** PNG/BMP via LSB; JPEG via DCT and appended-data fallback
- **Audio:** WAV LSB
- **Text:** zero-width unicode characters; trailing-whitespace
- **File structure:** EXIF metadata; appended-data / polyglot detection

### Cross-cutting capabilities
- **AES-256-GCM encryption layer** (optional, password-derived key via `scrypt`)
- **Pluggable AI triage** for steganalysis interpretation (optional — toolkit fully functional without AI)
- **Known-tool signature detection** (steghide, OpenStego, zsteg-style patterns)
- **Batch directory scan + JSON/HTML report**

### Out of scope for v1
- Video carriers (MP4, AVI)
- Network steganography (covert channels in TCP/DNS/ICMP)
- Machine-learning–trained steganalysis classifiers (heuristics + AI-LLM triage only)
- GUI

---

## 3. Architecture

### Top-level principles
1. **One module = one file = one capability** (mirrors `cyberm4fia-scanner/modules/` convention).
2. **Registry-driven discovery** — `registry.py` scans `modules/` at import time, auto-registers any subclass of `Carrier` or `Analyzer`. Adding a new technique = drop a file into `modules/`; no other code edits. Carrier selection per file uses extension first, then magic-byte sniffing as a fallback (so text carriers and polyglots dispatch correctly even with ambiguous extensions).
3. **Common interface per role** — every carrier implements the same ABC; every analyzer implements the same ABC. Consumers depend on the interface, not concretes.
4. **Immutable result objects** — `EmbedResult`, `AnalysisResult` are frozen dataclasses (per project coding-style rule on immutability).
5. **Pluggable AI provider** — `analysis/ai_triage.py` defines an `AIProvider` protocol; concrete provider is injected. No mandatory LLM dependency.
6. **Flat repo-root package** — repo root *is* the import root, exactly like `cyberm4fia-scanner` (`from core.carrier import ...`, `from modules.image_lsb import ...`).

### Directory layout

```
steganography/                       # repo root = import root
├── cli.py                           # entry point: embed|extract|analyze|scan
├── ui/
│   └── banner.py                    # cyberm4fia gradient banner (ported from scanner utils/colors.py)
├── registry.py                      # auto-discovery of modules/
├── core/
│   ├── carrier.py                   # Carrier ABC
│   ├── analyzer.py                  # Analyzer ABC
│   ├── crypto.py                    # AES-256-GCM + scrypt KDF
│   ├── payload.py                   # header format (magic+version+len+flags+nonce+salt)
│   └── result.py                    # EmbedResult, AnalysisResult (frozen dataclasses)
├── modules/                         # one file per capability
│   ├── image_lsb.py                 # PNG/BMP LSB embed/extract/analyze
│   ├── image_jpeg.py                # JPEG DCT + appended-data
│   ├── audio_wav.py                 # WAV LSB embed/extract/analyze
│   ├── text_zerowidth.py            # zero-width unicode hiding
│   ├── text_whitespace.py           # trailing whitespace hiding
│   ├── filestruct_exif.py           # EXIF metadata embed/extract/analyze
│   ├── filestruct_appended.py       # data appended after EOF / polyglots
│   ├── signatures.py                # known stego-tool signature detection
│   └── ai_triage.py                 # AI-assisted suspicion scoring (pluggable)
├── report/
│   └── report.py                    # JSON + HTML report writer
├── tests/                           # pytest; round-trip + analysis regression
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 4. Core interfaces

```python
# core/carrier.py
class Carrier(ABC):
    name: str                              # short identifier (e.g. "image_lsb")
    extensions: tuple[str, ...]            # file extensions handled
    can_embed: bool = True
    can_extract: bool = True

    @abstractmethod
    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult: ...

    @abstractmethod
    def extract(self, src: Path) -> bytes: ...

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...

    @abstractmethod
    def capacity(self, src: Path) -> int: ...        # max embeddable bytes
```

```python
# core/analyzer.py
class Analyzer(ABC):
    name: str
    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...
```

Signature detection and AI triage are `Analyzer` subclasses (they don't embed/extract — analysis-only).

### Result types (immutable)

```python
@dataclass(frozen=True, slots=True)
class EmbedResult:
    carrier: str
    out_path: Path
    bytes_written: int
    encrypted: bool

@dataclass(frozen=True, slots=True)
class AnalysisResult:
    analyzer: str
    suspicion: int                  # 0-100
    signals: tuple[Signal, ...]     # individual heuristic findings
    explanation: str | None         # human-readable, may be AI-generated
```

---

## 5. Data flow

### Embed
```
input data
  → [optional: AES-256-GCM encrypt with password-derived key (scrypt)]
  → wrap in payload header (magic | version | flags | length | nonce | salt)
  → registry.select_carrier(out_path)
  → carrier.embed(src, payload, out)
  → EmbedResult
```

### Extract
```
src file
  → registry.select_carrier(src)
  → carrier.extract(src)
  → parse payload header; verify magic + version
  → [if encrypted flag: AES-GCM decrypt with provided password]
  → original bytes
```

### Analyze (single file)
```
src file
  → registry.select_carriers(src)               # one or more applicable
  → for each: carrier.analyze(src)              # statistical signals
  → analyzers.signatures.analyze(src)           # known-tool fingerprints
  → analyzers.ai_triage.analyze(src, signals)   # optional, if provider configured
  → merged AnalysisResult
```

### Batch scan
```
directory walk
  → for each file: analyze pipeline above
  → aggregate into report.write(results, format="json"|"html")
```

---

## 6. Payload header format

Fixed binary header prepended before embedding:

| Field    | Size    | Notes                                                          |
|----------|---------|----------------------------------------------------------------|
| magic    | 4 B     | `STEG` ASCII                                                   |
| version  | 1 B     | format version, starts at 1                                    |
| flags    | 1 B     | bit 0 = encrypted; bits 1–7 reserved                           |
| length   | 4 B     | payload length in bytes, big-endian                            |
| salt     | 16 B    | scrypt salt (zeros if not encrypted)                           |
| nonce    | 12 B    | AES-GCM nonce (zeros if not encrypted)                         |
| payload  | N B     | ciphertext + GCM tag, or plaintext                             |

Total fixed overhead: 38 bytes + payload. Header is part of what the carrier embeds, so `capacity()` accounts for it.

---

## 7. Cryptography

- Algorithm: **AES-256-GCM** via `cryptography>=41`.
- KDF: `scrypt` (n=2¹⁴, r=8, p=1) with a per-payload 16-byte random salt.
- Nonce: 12 bytes, randomly generated per embed.
- Password optional. No password → unencrypted embed; the `encrypted` flag in the header reflects this so `extract()` knows what to do.
- Decrypt failure (bad password / tampering) raises a typed exception; the CLI surfaces a clean error.

---

## 8. AI triage (pluggable, optional)

`modules/ai_triage.py` defines:

```python
class AIProvider(Protocol):
    def explain(self, file_info: dict, signals: list[Signal]) -> tuple[int, str]:
        """Return (suspicion 0-100, human-readable explanation)."""
```

Concrete providers (not in v1 — interface only, plus a stub that returns a heuristic score) live outside the toolkit. The user can wire in their `cyberm4fia-scanner` dual-model orchestrator, an Anthropic/OpenAI client, or anything else implementing the protocol.

If no provider is registered, `ai_triage` returns a heuristic-only score derived from the other analyzers' signals — the toolkit remains fully functional.

---

## 9. Signature detection

`modules/signatures.py` ships a curated set of byte/structural signatures for popular stego tools:

- `steghide` — JPEG/WAV/BMP/AU artifacts
- `OpenStego` — known marker bytes
- `outguess` — header markers
- `zsteg`-style PNG IDAT anomalies
- Generic appended-archive (ZIP/RAR/7z/PE) after image EOF (polyglot detection)

Signatures are declarative (a list of `Signature(name, applies_to_ext, matcher)` records) — adding one is a list entry, not a code change.

---

## 10. Reporting

`report/report.py` accepts a list of `AnalysisResult` (or `(path, AnalysisResult)` tuples) and writes:

- **JSON** — full structured output for tool chaining.
- **HTML** — `rich`-rendered, mirrors the visual style of `cyberm4fia-scanner/report.py`: dark theme, per-file collapsible findings, suspicion score badges, sortable.

---

## 11. CLI surface

Single entry point `cli.py`, subcommands modeled on the scanner's UX:

```
steganography embed   --in <data> --carrier <file> --out <file> [--password X]
steganography extract --in <file> --out <data> [--password X]
steganography analyze --in <file> [--ai] [--json]
steganography scan    --dir <path> [--ai] [--report html|json] [--out report.{html,json}]
steganography list-modules
```

Banner prints once on startup unless `--quiet`. Errors use `rich` traceback only in `--verbose`; otherwise a clean single-line message.

---

## 12. Error handling & security boundaries

- **Boundary validation:** every file is validated at the carrier boundary — size cap (configurable, default 200 MB), magic-byte check, format integrity. Bad input → typed exception, never silent.
- **Capacity overflow:** `embed()` checks `len(payload) ≤ capacity(src) - header_overhead`; rejects with `InsufficientCapacityError`.
- **Path safety:** output paths resolved + checked to not escape user-supplied root in `scan`.
- **No silent error swallowing** — exceptions either propagate or are caught at the CLI layer with a clean message and non-zero exit.
- **Ethical use note** in `README.md`: authorized testing, research, and CTF only.

---

## 13. Testing strategy (TDD, ≥80% coverage)

- **Per-carrier round-trip:** for each carrier module, generate a fixture file, embed random bytes, extract, assert equality. Parametrized over sizes (small, mid, at-capacity).
- **Crypto round-trip:** encrypt → decrypt with correct password (pass), wrong password (typed exception), tampered ciphertext (typed exception).
- **Analysis regression:** ship a small fixtures set of known-clean and known-stego files; assert suspicion thresholds remain in expected bands.
- **Registry discovery:** assert all expected modules are discovered; assert new test-only module gets picked up.
- **CLI smoke tests:** subprocess-invoke each subcommand on fixtures, assert exit codes and output shape.
- **Coverage gate:** `pytest --cov` ≥ 80%; enforced in CI.

---

## 14. Dependencies

```
# core
cryptography>=41
pillow>=10            # PNG/BMP/JPEG IO
numpy>=1.26           # LSB/DCT vectorized ops, statistical analysis
rich>=13              # CLI + report rendering
piexif>=1.1           # EXIF read/write
mutagen>=1.47         # WAV/audio metadata (optional, for richer audio analysis)

# dev
pytest>=8
pytest-cov>=5
ruff
mypy
```

No mandatory AI/LLM SDK — AI provider is plug-in.

---

## 15. Open questions deferred to plan

- Concrete AI provider wiring example (does it ship a thin Anthropic client as a reference impl, or stay protocol-only?)
- Whether `image_jpeg.py` ships full DCT-coefficient LSB in v1 or only the appended-data path.
- Exact HTML report theme — port `cyberm4fia-scanner/report.py` template directly, or rewrite minimally.

These are implementation details, not architectural; the writing-plans step will decide.

---

## 16. Success criteria

- `embed → extract` round-trip is byte-exact for every carrier in the fixtures suite.
- AES-encrypted embeds: wrong password fails cleanly, correct password recovers exactly.
- `analyze` on the known-stego fixture set scores >70 suspicion; on the known-clean set scores <30.
- Adding a new carrier requires only a new file in `modules/` — no edits elsewhere.
- Banner renders identically to `cyberm4fia-scanner`'s gradient on a 256-color terminal.
- Test coverage ≥ 80%.
