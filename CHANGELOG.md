# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases use
[Semantic Versioning](https://semver.org/).

## 0.5.0 (pending PyPI/GitHub release) - 2026-08-13

### Added

- Deterministic clean/stego corpus generation across image, audio, text and
  structural carrier methods.
- Benchmark reports with confusion matrices, precision, recall, F1, FPR,
  ROC-AUC, average precision and method/density/format breakdowns.
- Absolute quality gates and committed-baseline regression checks.
- Pinned benchmark codec/numeric dependencies for reproducible CI baselines.
- Tag-driven GitHub Release and PyPI Trusted Publishing workflow with build
  provenance attestations.
- Current Node.js 24 GitHub Actions generations pinned to immutable commits.

### Changed

- Distribution name is now `cyberm4fia-steganography` because the generic
  `steganography` name is already owned on PyPI. The CLI and Python import name
  remain `steganography`.
- Tool-produced whitespace and zero-width payloads are validated as complete
  versioned envelopes during analysis.

## 0.4.0 - 2026-08-13

### Added

- Versioned payload v2 with SHA-256 integrity and v1 read compatibility.
- Unified analysis service, stable JSON schema and escaped HTML reports.
- PDF/GIF trailers, keyed image LSB scattering and isolated JPEG DCT support.
- Analysis-only FastAPI service with SQLite-backed scan jobs.
- Installable wheel, public Python API, plug-in entry points and CI checks.
