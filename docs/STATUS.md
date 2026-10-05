# Implementation status — 2026-10-05

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
- Explicit ALASKA2 holdout acquisition: fixed-seed complete lineage selection,
  bounded authenticated ZIP64 range reads, credential isolation, CRC/JPEG/SHA
  checks and exclusive, provenance-bound resume. See `ALASKA2_ACQUISITION.md`.
- Preregistered ALASKA2 native-score evaluator: unchanged shared service,
  manifest integrity, complete paired lineages, failure-aware coverage, no
  held-out threshold search and per-method bootstrap intervals. Its protocol
  is frozen in `ALASKA2_PROTOCOL.md`; results are in `ALASKA2_RESULTS.md`.
- Separate ALASKA2 development import, bounded 968-feature JPEG extraction,
  class-balanced CPU training, train-only normalization, validation and ONNX
  export. The first fixed experiment completed, but is not deployed; see
  `JPEG_DEVELOPMENT_RESULTS.md` for inadequate scores and export-parity limits.
- Explicit pinned FSDD acquisition: 3,000 real WAV covers, audited source/
  license evidence, SHA/CRC and PCM data. The preregistered 7,000-file baseline
  completed with failed native detection; see `WAV_RESULTS.md`. Public
  source/usage inventory and method-level evidence are in `DATASET_CATALOG.md`
  and `BENCHMARK_RESULTS.md`; raw corpora remain local.
- Separate spatial development acquisition completed: 1,000 BOSSbase originals
  (816 train / 184 validation), all independently rehashed/CRC/decode checked,
  with no reserved hash/member overlap. The fixed 7,000-file controlled PNG/BMP
  development comparison now completed: richer spatial features improve AUC
  but low-rate scattered recall/FPR/calibration and strict ONNX parity still
  fail. Neither model is deployed; see `SPATIAL_DEVELOPMENT_RESULTS.md`.

## Verification

- Local checks use the provisioned `venv/bin/python` environment. The host's
  system Python has an older cryptography package without Argon2id and cannot
  collect the full suite; activate the environment before running check commands.
- Python 3.11: 515 tests pass; total coverage 93.30%. PDF analyzer coverage is
  97.69% and image-bitplane analyzer coverage is 98.68%. The two warnings concern
  deprecated ONNX export APIs (ten occurrences), not test failures.
- New spatial descriptor coverage is 100%; the explicit development runner is
  96.60%; shared feature/inference code is 100%. Independent scalar filter and
  extraction oracles, source mutations, symlinks, overwrite and budget tests
  are included. Ruff, mypy (78 application files) and diff checks pass.
- The standalone ALASKA2 downloader has 59 dedicated tests and 99.09% statement
  coverage in a separate script-coverage run. Tests cover credential/redirect
  isolation, ignored HTTP ranges, ZIP64 bounds, decompression limits, symlinks,
  overwrite/resume integrity, redacted errors and reserved-corpus overlap.
- ALASKA2 evaluation has seven new tests (including the real shared-service
  JPEG path), with 98.34% statement coverage of the evaluator. Missing required
  native coverage invalidates a cell rather than turning failures into negatives.
- Extended gradient corpus and spatial proxy tests are regression evidence,
  not calibration or real-world accuracy evidence. The two legacy SPA/RS names
  represent a single adjacency observation and are aggregated only once.
- New JPEG feature/model/research-feature/research-JPEG files have 98.82%,
  100%, 100% and 98.19% statement coverage respectively. Thirty standalone FSDD
  tests cover archive/PCM bounds, license evidence, symlinks, no overwrite,
  redirects and redacted failures; separate downloader coverage is 98.18%.
- Thirty-two WAV benchmark tests verify independent payload recovery, RIFF/PCM
  bounds, speaker reservations, immutable inputs, coverage failures and fixed
  scoring. New benchmark code has 97.67% statement coverage. Twelve BOSSbase
  development acquisition tests pass; standalone code coverage is 95.54%.
- Ruff and mypy (78 files including the current acquisition entrypoints)
  pass; diff whitespace checks pass.
- PyTorch, ONNX, and ONNX Runtime are provisioned in the research environment.
  CPU optimizer execution, determinism/repeatability, nonfinite loss guards, and
  ONNX export parity against onnxruntime were verified end-to-end.
  Small-fixture parity passes; the actual JPEG validation artifact exceeds a
  strict absolute 1e-6 score tolerance by 1.92e-7, with zero threshold-decision
  changes. That stricter artifact check is recorded as failed, not waived.
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
- ALASKA2 baseline completed on all 4,000 files, without analysis errors.
  At the frozen threshold 70, JMiPOD recall is 0.4% (4/1,000), JUNIWARD and
  UERD recall 0.1% each (1/1,000); AUC is 0.525617 / 0.505157 / 0.507181.
  The same 1,000 covers give FPR 0.2%. All three methods fail the detection
  targets. The preregistered identical-pair sensitivity does not change that
  conclusion. See `ALASKA2_RESULTS.md`; this is not a trained JPEG model,
  full-Docker verdict test or cross-source support qualification.
- StegoAppDB results and adequate cross-source evaluation remain **unavailable**.
  No support cell has demonstrated the requested cross-source accuracy gates.
- The separately trained JPEG development model has validation AUC
  0.648804 / 0.592481 / 0.585306 and FPR 41.46% on 205 shared covers. It is
  experimental, uncalibrated and local; this different subset is not evidence
  of improved old-test performance. Existing primary verdicts are unchanged.
- FSDD native WAV baseline: all six sequential/scattered LSB cells at 0.05,
  0.20 and 0.40 bits/sample have AUC 0.50, balanced accuracy 50%, recall/FPR 0%.
  All 7,000 files completed with required native coverage. Independent oracle
  extraction passed for all 6,000 generated payloads; this is ground truth,
  not a native recovery score. Training/validation speakers remain separate.
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
- The general analysis pipeline lacks declared per-format required-component
  coverage policy; low-score `no_indicators` is not a coverage-complete clean
  result. The ALASKA2 evaluator explicitly requires native JPEG components
  and marks failed/incomplete cells unavailable instead of counting clean files.
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
