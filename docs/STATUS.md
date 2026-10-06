# Implementation status — 2026-10-06

Source version is 0.6.0, unreleased and classified beta. Baseline 0.5.0 was
validated and committed as `1aaa747`. The complete v0.6–v1.0 roadmap has not
been delivered or validated by this slice.

## Implemented

- Full unrounded float256 preparation completed: 3,750 original hashes verified,
  18 preregistered independent mathematical examples exact; no accuracy claim.
  Separate SRNet numeric persistence validates all 183 weights/BN arrays and
  hostile NPY headers before allocation, preserves CPU RNG on reload.

- Explicit unrounded phase-aligned component-Y/256 float research caches and
  bounded SRNet fractional inference; all legacy pixel/cache defaults preserved.
  `JPEG_FLOAT256_PROTOCOL.md` freezes preparation/audit, not accuracy/training.

- Separate optional 4.78M-parameter SRNet-style structural block and explicit
  `research srnet-preflight` fixed resource-limited worker. Finite gradient/
  parameter-update and singleton/batch eval readiness, nested BN eval guards.
  No real fit, trained model or accuracy improvement; `SRNET_PREPARATION.md`
  records uint8/crop/paper differences and remaining preparation gates.

- Completed controlled residual-domain trial: three fixed fresh fits, all
  twelve cells still fail, AUC .495–.509, zero recall. Source/crop/optimizer
  controls and all previous failures retained; native/NumPy/ONNX replay passes.
  No deployment; see `PIXEL_CNN_SCALE_RESULTS.md`.

- Additive architecture-bound pixel-unit residual stem for a fixed controlled
  research follow-up; unchanged legacy default and learned layers. Numeric
  filters reject cross-domain cards/weights. `PIXEL_CNN_SCALE_PROTOCOL.md`
  freezes the experiment before real fitting; no detector improvement assumed.

- First real pixel-CNN trial completed: three frozen fits, all 12 cells fail
  detection (AUC .502–.509, zero recall). Full-source integrity, independent
  NumPy forward/metrics and ONNX batch 1/17/64 replay pass. No deployment or
  qualification; see `PIXEL_CNN_TRAINING_RESULTS.md` for every regression.

- Explicit CPU minibatch pixel-CNN train/predict services and shared CLI;
  source/class weighting on selected train rows only, fixed-scope provenance,
  bounded numeric-only model storage with array-header/finite/filter guards.
  First real fitting protocol frozen in `PIXEL_CNN_TRAINING_PROTOCOL.md`;
  no automatic model installation or primary verdict change.

- Explicit bounded JPEG uint8 center-crop caches, pinned decoder/crop provenance,
  framed isolated worker and shared `research pixel-cache` CLI. Custom optional
  residual CNN building block has independent filter/gradient smoke tests, but
  preparation alone is not training or deployed detector evidence.
  Actual preparation completed: all 3,750 original files rehashed, 18 bounded
  independent crop examples exact; see `PIXEL_RESIDUAL_PREPARATION_RESULTS.md`.

- Additive JRM training-scope provenance and a shared two-way source-exclusion
  diagnostic through the existing research CLI. Cache/split identities and both
  complete families validated before fitting; protocol in `JRM_TRANSFER_PROTOCOL.md`.
  Legacy all-source cards remain compatible; no deployed detector change.
  Completed `JRM_TRANSFER_RESULTS.md`: all eight cells fail, cross-origin AUC
  .52–.55; exact independent scalar votes for both models and all metric cells.

- Optional pinned sealwatch JRM/FLD research reference, bounded framed workers,
  separate checksum-bound float32 caches and immutable numeric inference without
  pickle. Explicit `research jrm-reference` CLI; protocol frozen in
  `JRM_REFERENCE_PROTOCOL.md`. Not deployed, not an installed ONNX model.
  Numeric model headers are checked before allocation, including adversarial
  tiny-archive/giant-array tests; score arithmetic remains unchanged.
  Full 3,750-file reference completed and independently audited: ALASKA/UERD
  AUC .698, FPR 31.71%, but JUNIWARD recall and BOSS ECE regress. All measurable
  numeric detection gates fail; see `JRM_REFERENCE_RESULTS.md`. No deployment.

- Explicit research-only residual-feature MLP64 with a linear skip, bounded
  architecture/domain validation, regularization provenance and shared export/
  inference. Legacy linear models remain compatible. The fixed comparison is
  preregistered in `JPEG_NONLINEAR_PROTOCOL.md`; not a deployed CNN.
  The completed run improves ALASKA AUC to .609/.603 and FPR to 38.54%, but
  BOSS/UERD and ALASKA calibration regress. Detection still fails; all cells
  and independent numerical audits are in `JPEG_NONLINEAR_RESULTS.md`.

- Research-only 2,066-feature JPEG block-DCT residual/parity contract and shared
  bounded worker dispatch. Two-origin weighting and old contracts preserved;
  fixed comparison is in `JPEG_RESIDUAL_PROTOCOL.md`. The actual 3,750-file run
  is complete: FPR 46.34% ALASKA / 40% BOSS, recall regressions retained,
  numerical export passes but detection fails. See `JPEG_RESIDUAL_RESULTS.md`.

- Versioned minimum native coverage by content-detected format: missing required
  components force low-score `inconclusive`, while independent positive findings
  stay visible. JSON v2 revision 2, HTML, SARIF and bundle views preserve the
  assessment. Failed analyzer signals cannot confirm. See `COVERAGE_POLICY.md`.

- Versioned measured JPEG context features, bounded local BOSS JPEG simulation,
  provenance-bound matched-family multi-origin preparation and train-only
  source/class-balanced objective. Fixed next experiment: `JPEG_CONTEXT_PROTOCOL.md`.
  The completed 3,750-file experiment and independent audit are published in
  `JPEG_CONTEXT_RESULTS.md`: source-specific FPR 50.24% / 48%, ALASKA AUC
  regression, stable ONNX parity, zero unseen validation sources. Not deployed.

- Shared cached-score source/context diagnostics with a `research diagnose`
  CLI: contract binding, source overlap, method/rate × format/declared-quality
  metrics, paired uncertainty, unknown metadata and single-label cell handling.
  Actual-artifact audits completed: 24 spatial / 9 JPEG cells independently
  checked. Both models have zero unseen validation sources and missing quality/
  camera/device metadata. See `GENERALIZATION_RESULTS.md`; no score/deployment changes.

- Explicit train-only controlled-lineage low-payload weighting and a
  checksum-bound cached-vector comparison runner. The fixed experiment is
  complete: sequential/scattered-5% recall rises to 67.93%/29.89%, but shared
  FPR doubles to 3.26%, higher-rate ranking regresses and calibration fails.
  No deployment; see `SPATIAL_WEIGHTING_RESULTS.md` for all cells and gates.

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
  `JPEG_DEVELOPMENT_RESULTS.md` for inadequate scores and the original export failure.
- Explicit pinned FSDD acquisition: 3,000 real WAV covers, audited source/
  license evidence, SHA/CRC and PCM data. The preregistered 7,000-file baseline
  completed with failed native detection; see `WAV_RESULTS.md`. Public
  source/usage inventory and method-level evidence are in `DATASET_CATALOG.md`
  and `BENCHMARK_RESULTS.md`; raw corpora remain local.
- Separate spatial development acquisition completed: 1,000 BOSSbase originals
  (816 train / 184 validation), all independently rehashed/CRC/decode checked,
  with no reserved hash/member overlap. The fixed 7,000-file controlled PNG/BMP
  development comparison now completed: richer spatial features improve AUC
  but low-rate scattered recall/FPR/calibration fail. The original export
  failure remains recorded; derived stable inference now passes numerical
  gates. Neither model is deployed; see `SPATIAL_DEVELOPMENT_RESULTS.md`.
- Explicit numerical precision repair now passes for both spatial models and
  the JPEG model at unchanged 1e-6 tolerance and unchanged decision outputs.
  A separated-role threshold experiment reduces assessment false alarms by
  one cover but worsens low-payload recall; it is not installed/calibrated.
  See `INFERENCE_PRECISION_RESULTS.md` for numerical and statistical boundaries.
- Fixed 684-feature parity/residual comparison completed on the same corpus:
  scattered-5% AUC 0.716801 → 0.906486, shared FPR 4.89% → 1.63%, but fixed-.5
  low-rate recall/ECE regress. Stable export passes; no model is deployed.
  All cells and failures are in `SPATIAL_PARITY_RESULTS.md`.

## Verification

- Float256 evidence and numeric persistence: Python 3.11.14, 955 tests pass,
  94.48% total coverage; decoder/network/cache/persistence 336/337 statements
  covered (99.70%). Ruff, mypy (103 source files), whitespace checks and
  wheel/sdist builds pass. Both packages contain the services but no models or
  corpora; Torch/jpeglib remain optional. Python 3.12–3.14 not rerun locally.

- Float256 implementation freeze: Python 3.11.14, 938 tests pass, 94.44%
  total coverage; decoder/SRNet/cache code 247/248 statements covered. Ruff,
  mypy (102 files) and diff check pass. Fixed real preparation/audit is still
  pending at freeze; no accuracy or actual SRNet training implied.

- SRNet-style preparation: Python 3.11.14, 907 tests pass, 94.38% total
  coverage; new architecture/readiness code 132/133 statements covered.
  Ruff, mypy (101 files), diff check and model-free wheel/sdist inspection pass.
  Real fixed-worker generated readiness passes; not a trained detector or
  accuracy test. Old CTF Pilot race fixed by waiting for the queued UI result.
  Python 3.12–3.14/fresh full Docker remain unverified.

- Residual-domain results: Python 3.11.14, 877 tests pass, 94.31% total coverage;
  CNN/model/training code 244/244 statements covered. Ruff, mypy (99 files),
  diff check and model-free wheel/sdist build/inspection pass. Complete real
  corpus and both previous references verified; all three NumPy/native/ONNX
  replays pass, all twelve detection cells fail. Python 3.12–3.14 and fresh
  full Docker remain unverified here.

- Residual-domain implementation freeze: Python 3.11.14, 876 tests pass,
  94.31% total coverage; new CNN/model/training code 244/244 statements covered.
  Ruff, mypy (99 files), diff check and fresh wheel/sdist build pass. Other
  Python versions/full Docker remain unverified; no real new scores at freeze.

- Pixel-CNN result publication: Python 3.11.14, 869 tests pass, 94.30% total
  coverage; new network/numeric model/training code 228/228 statements covered.
  Ruff, mypy (99 files), diff check and model-free wheel/sdist build/inspection
  pass. Real 3,750-file integrity, three complete NumPy/native replay audits and
  ONNX batches 1/17/64 pass; detection fails in all twelve cells. Python
  3.12–3.14 and fresh full Docker remain unverified in this environment.

- Pixel-CNN training implementation: Python 3.11, 861 tests at freeze verification,
  94.29% total coverage; network/numeric model/training code 227/228 statements
  covered. Ruff, mypy (99 files) and diff checks pass. No real trial results or
  ONNX qualification implied; other Python versions/full Docker remain unverified.

- Latest pixel-preparation slice: Python 3.11, 824 tests, 94.19% total coverage;
  new pixel/CNN/cache code 190/191 statements covered. Ruff, mypy (97 files),
  diff checks and fresh wheel/sdist pass; Torch optional, no raw cache/models
  bundled. Real-corpus CNN training/inference, Python 3.12–3.14 and fresh Docker
  remain unverified. Earlier counts below are historical slice evidence.

- Latest source-transfer slice: Python 3.11, 779 tests, 94.08% total coverage,
  JRM services 255/255 statements covered. Ruff, mypy (94 files), diff checks
  and fresh wheel/sdist build pass. Corpus rehashed; both 765-row scalar vote
  audits exact, numeric batches 1/17/765 exact. Other Python versions/full Docker
  unverified. Earlier test counts below remain historical slice evidence.

- Latest JRM slice: Python 3.11, 757 tests, total coverage 94.03%; new reference
  services 289/290 statements covered (99.66%). Ruff, mypy (93 application files),
  diff checks and fresh wheel/sdist build pass. All 3,750 corpus files rehashed;
  independent scalar FLD votes match all 765 predictions exactly. All 12 metric
  cells checked independently; numeric batches 1/17/765 exact. Eighteen upstream
  JRM boundary examples match, not independent algorithm qualification. Earlier
  counts below are historical; Python 3.12–3.14/fresh Docker remain unverified.

- Local checks use the provisioned `venv/bin/python` environment. The host's
  system Python has an older cryptography package without Argon2id and cannot
  collect the full suite; activate the environment before running check commands.
- Python 3.11: 692 tests pass; total coverage 93.84%. New JPEG residual descriptor
  coverage is 96.30%, shared JPEG worker 98.89%. Required-coverage assessment,
  shared pipeline and v2 report renderer coverage are 100%; 52 focused coverage
  cases include a real JPEG without optional DCT and recursive CTF gap handling.
  New JPEG context features
  are 100%, corpus service 95.60%,
  multi-origin preparation 98.10%, train weighting 100%. PDF analyzer coverage is
  97.69% and image-bitplane analyzer coverage is 98.68%. The two warnings concern
  deprecated ONNX export APIs (twenty-four occurrences), not test failures.
- New spatial descriptor coverage is 100%; the explicit development runner is
  96.60%; shared feature/checkpoint code is 99.40%, reconstruction is 100%,
  precision audit is 96.39% and operating-point service is 97.40%. Independent scalar filter and
  extraction oracles, source mutations, symlinks, overwrite and budget tests
  are included. New parity descriptor coverage is 100%, comparison runner
  97.33%. New weighting helper coverage is 100%, cached comparison runner
  98.55%; source/context diagnostic coverage is 100% with 36 dedicated tests.
  Ruff, mypy (90 application files) and diff checks pass.
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
- The earlier acquisition slice passed Ruff, mypy (78 then-current files) and
  diff whitespace checks; current full application verification is recorded above.
- PyTorch, ONNX, and ONNX Runtime are provisioned in the research environment.
  CPU optimizer execution, determinism/repeatability, nonfinite loss guards, and
  ONNX export parity against onnxruntime were verified end-to-end.
  Small-fixture parity passes; the actual JPEG validation artifact exceeds a
  strict absolute 1e-6 score tolerance by 1.92e-7, with zero threshold-decision
  changes. That stricter artifact check is recorded as failed, not waived.
- Version 0.6.0 wheel and sdist build successfully; the current build was verified
  to include new residual/context/coverage modules, no models or datasets, and simulation/
  DCT dependencies only behind optional extras. No package publication occurred.
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
- The shared pipeline now requires minimum native per-format execution; even
  coverage-complete `no_indicators` is not a clean-file certificate or qualified
  method support. Unknown/TIFF/generic formats lack complete format coverage.
  The historical ALASKA2 evaluator explicitly requires native JPEG components
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
