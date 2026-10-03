# Implementation status — 2026-10-03

Source version is 0.6.0, unreleased and classified beta. Baseline 0.5.0 was
validated and committed as `1aaa747`. The complete v0.6–v1.0 roadmap has not
been delivered or validated by this slice.

## Implemented

- Shared analysis context with legacy analyzer compatibility; additional
  spatial-image, JPEG structure, GIF, WAV and text signals.
- Recursive CTF service, bounded decoder/archive candidates, artifact hashes
  and provenance, CLI, API jobs/SSE/cancel/download and guided TUI.
- Additive JSON v2, HTML, SARIF and portable evidence bundles.
- Explicit signed-model catalog installation and research train/calibrate/
  export commands; lineage-aware manifests and benchmark metrics/gates.
- Full-image upstream tool installation with checksummed Stegseek, zsteg and
  dependencies, and OpenStego; Debian package licenses recorded.
- Research partitioning with reserved-pilot exclusions, connected lineage/
  camera/device groups, explicit held-out sources and exclusive output creation.
- AI triage cannot change primary scores/findings or act as detector coverage.
  See `DEVELOPMENT_DATA.md` for next-experiment prerequisites and limitations.
- Manifest-bound, bounded PNG/BMP feature extraction for train/validation;
  training input validation rejects unbound NPZ and non-train or altered rows.
  The initial spatial-summary features are experimental, not a validated detector.
- Corrected duplicate SPA/RS adjacency evidence fusion; added PDF ASCII decoder
  preflight limits, complete Flate-stream checks and shared decode budgets.
- CTF candidate hygiene now excludes crash dumps and failed/unnamed tool output,
  validates BMP carving headers, avoids lossy binary URL decoding, and preserves
  native budget cancellation and recursion depth. Per-job pilot reports are saved.

## Verification

- Local checks use the provisioned `venv/bin/python` environment. The host's
  system Python has an older cryptography package without Argon2id and cannot
  collect the full suite; activate the environment before running check commands.
- Python 3.11: 332 tests pass; total coverage 92.69%. PDF analyzer coverage is
  97.69% and image-bitplane analyzer coverage is 98.68%. The two warnings concern
  deprecated ONNX export APIs, not test failures.
- Extended gradient corpus and spatial proxy tests are regression evidence,
  not calibration or real-world accuracy evidence. The two legacy SPA/RS names
  represent a single adjacency observation and are aggregated only once.
- Ruff and mypy (71 source files) pass; diff whitespace checks pass.
- PyTorch, ONNX, and ONNX Runtime are provisioned in the research environment.
  CPU optimizer execution, determinism/repeatability, nonfinite loss guards, and
  ONNX export parity against onnxruntime were verified end-to-end.
- Version 0.6.0 wheel and sdist build successfully; the wheel contains no models.
- Web: one test passes; TypeScript/Vite build passes.
- Full Docker build and non-root/read-only/network-disabled smoke have passed
  for native payload and independently generated Steghide and OpenStego exact recovery.
  `scripts/docker-ctf-smoke.py` reproduces these limited integration checks.
  OpenStego's writable-preferences initialization was fixed; tool completion
  alone is not evidence of a hidden payload or successful extraction.
- The 2026-10-02 CTF regression completed and exactly recovered all 30 frozen
  Kodak challenges in each of two final runs, without increasing budgets.
  OpenStego's native allocation/thread overhead was bounded; ten cold-start
  extractions passed the hardened Docker smoke. Failed intermediate runs remain
  recorded in `CTF_REGRESSION_20261002.md`. This used read-only current-source/
  launcher mounts on the existing image, not a freshly rebuilt full image.
- Python 3.12–3.14 are configured in CI but were not executed locally.

## Open acceptance gates and limitations

- BOSSbase single-source pilot: 1,000 covers + 1,000 independently embedded
  stegos, all stegos verified through upstream extraction. Base detection
  failed: recall 0% at threshold 70, ROC-AUC 0.499021. Controlled CTF recovery
  passed 30/30, including known-password image cases and simple decoders.
  See `PILOT_RESULTS.md`; these are different capabilities, not overall accuracy.
- ALASKA2/StegoAppDB and cross-source evaluation remain **unavailable**.
  No support cell has demonstrated the requested cross-source accuracy gates.
  Read-only checks on 2026-10-03 found an unauthenticated ALASKA2 file-list API
  response of 401 and a StegoAppDB endpoint response of 404. No new corpus was
  acquired; authorized local data is the next required input. See
  `DATA_ACCESS_STATUS.md` for observations, limits and the evaluation sequence.
- Small external-source check on all 24 Kodak images: 48 method-specific pairs,
  all stegos independently recovered. Base detection still failed (0/48 recall,
  AUC 0.49349). Controlled CTF payload recovery was 30/30, but only 25 jobs
  completed; five reported cancelled. See `KODAK_RESULTS.md`. This is a second
  source check, not the adequately sized cross-source release evaluation above.
  The subsequent published-challenge regression fixed these CTF cancellations
  (`CTF_REGRESSION_20261002.md`); it does not change the failed detection result.
- Blind 120-challenge recovery, top-three recommendations and latency: **unavailable**.
  The smoke examples are not evidence of 90% CTF recovery.
- Published, trained spatial/JPEG model packs: absent; model catalog is empty.
  Training/export plumbing is not an independently evaluated detector.
- Spatial RS/sample-pair/weighted signals are approximations. Full calibrated
  algorithms and JPEG recompression/family discrimination remain research work.
- New CTF code has not yet reached the 95% coverage target across every new file.
- Per-tool OS filesystem isolation, hard native-stage deadlines, exhaustive
  fuzzing and archive-bomb/process-escape gates are incomplete. A subprocess
  working directory does not restrict its filesystem permissions.
- Debian and Python transitive dependencies are not fully locked. Installation
  of every executable does not demonstrate successful recovery with every tool.
- CTF API jobs are process-local; durable restart/resume and a full live-output
  workflow need additional work.

Use the current release as a local exploratory/CTF assistant. Validate findings
with independent tools. Statistical scores are heuristic and uncalibrated;
they are not probabilities or forensic proof.
