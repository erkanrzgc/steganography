# Changelog

## 0.6.0 (unreleased)

- Added bounded, manifest-bound PNG/BMP spatial-summary feature extraction for
  train/validation; test image bytes are not accessed by this workflow.
- Training now rejects unbound NPZ inputs, mismatched feature/manifest hashes,
  invalid feature values and rows outside the exact train membership. Checkpoints
  retain provenance and identify the linear baseline accurately (not SRNet).
- Added training provenance, nonfinite loss guards, and safe ONNX export with model
  cards; verified CPU training reproducibility and ONNX parity against ONNX Runtime.
- Fixed raw 32-byte Ed25519 public key parsing in model verification when keys
  contain leading/trailing whitespace bytes.

- Added `research partition` with frozen-manifest overlap rejection, whole-source
  holdouts, connected lineage/camera/device grouping and revalidated split policy.
- Closed metadata-based lineage isolation bypasses and rejected research-file
  symlinks during integrity verification.
- Removed AI triage from aggregate suspicion, primary findings and usable
  detector coverage; raw triage output is retained for explanations.

- Added explicit range-bounded BOSSbase pilot acquisition, independent
  Steghide/OpenStego generation, lineage-aware detection metrics and exact CTF
  recovery evaluation. Published the failed single-source detection baseline
  separately from 30/30 controlled recovery; no support gate is claimed.
- Fixed OpenStego's headless preferences directory and added exact OpenStego
  recovery to the hardened Docker smoke test.

- Corrected external-tool input paths and bounded stdout collection, including
  process-group cleanup and remaining-job-budget checks between tools.
- Fixed full-image zsteg dependency installation and tool smoke commands.

- Added shared lazy analysis contexts, richer PNG/BMP bit-plane statistics, and
  additive JSON v2 coverage, rationale, calibration, score, and recommendation
  fields.
- Added the bounded recursive CTF playbook with decoder graph, carving, safe
  archive traversal, extraction provenance, centralized external-tool runner,
  JSON/HTML/SARIF/evidence bundles, CLI, API jobs/SSE/cancel/download, and TUI.
- Strengthened research catalogs with cover lineage and source metadata, and
  raised benchmark gates to the documented ROC-AUC, balanced accuracy, recall,
  FPR, ECE, and bootstrap confidence-interval protocol.

- Added the versioned case/evidence/analysis/finding/artifact/model/audit SQLite
  schema, SHA-256 deduplication and retention-aware case lifecycle.
- Added an Argon2id-unlocked libsodium secretstream evidence vault with
  authenticated chunking, tamper detection and in-memory key locking.
- Added authenticated `/v2` cases, evidence, scans, SSE jobs, reports, Studio,
  artifacts, models and audit endpoints while retaining `/v1` compatibility.
- Added JSON v2 verdicts, portable HTML, SARIF, correlation-aware ensemble
  helpers, signed model registry and reproducible dataset manifests.
- Added integrity-checked held-out research benchmarks with split isolation,
  source diversity, ROC-AUC and high-confidence false-positive quality gates.
- Added payload v3 with Argon2id/AES-GCM, metadata, compression and optional
  Reed–Solomon recovery; v1/v2 extraction remains supported.
- Added bit-plane, archive and text-anomaly analyzers plus bounded zsteg,
  Stegseek and ExifTool adapters.
- Added the React/Vite local workspace, core/full containers, Compose hardening,
  repository SARIF action, threat model and third-party notices.
- Added a keyboard-first Textual workbench with guided scan, Studio, evidence
  custody, persistent scans, report export, and an honest advanced workspace.

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

- Distribution name is now `steganography-dfir` because the generic
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
