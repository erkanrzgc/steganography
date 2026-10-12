# Implementation status — 2026-10-12

GPU-deferred non-JPEG preparation: shared bounded spatial/audio generator and
generator-independent scalar audit implemented, with native historical roles,
fixed channel/plane/rate cells and compatible separate research manifests.
Initial focused checks passed 109 tests; additional binding/cancellation tests
and actual preparation/full regression remain pending at implementation freeze.
No GPU, fitting, scoring, download or detector improvement. GIF/text/container/
stereo/tool families keep their explicit native data/oracle gaps. See
`NONJPEG_PREPARATION_PROTOCOL.md` and `NONJPEG_METHOD_READINESS.md`.

Approved five-view revision (2026-10-12): preparation can explicitly select
`center-four-corners256` using a separate manifest/representation contract.
The same original identities and simulation remain fixed; unchanged individual
views are retained. Matched training cycles all views per pair and stored-BN
evaluation equally averages five singleton probabilities. A complete SciPy
audit checks every view. This implements the grayscale representation stage,
not a pretrained/color comparison or a measured accuracy increase. The own-GPU
SSH tunnel is still unavailable. See `JPEG_FIVE_VIEW_STUDY.md`.

Actual preparation completed: the unchanged 640-original identity selection
produced 1,280 JPEGs, 1,024 fit /256 development image rows and 6,400 views.
All views passed independent SciPy reconstruction (maximum absolute difference
5.684341886080802e-14, unchanged 1e-4 tolerance), native coefficient/quantization
checks and lineage-role separation. The historical blocking original remains
with exactly the same cover/stego JPEG hashes. Identical coefficient-view pairs
are retained: fit [1,4,11,1,4], development [0,0,1,0,1] in the fixed view order.
Total cache/JPEG bytes 1,724,097,458, preparation+audit 1,291.42 seconds.
No optimizer update occurred, no model was trained/deployed, camera independence
is unverified and WIFD remains unscored. The explicit local controls preflight
returned `unavailable: cuda_not_visible_or_cpu_torch`, with no CPU fallback.
See [portable preparation evidence](../benchmarks/jpeg-five-view-preparation-20261012.json),
bound to 49 source files at `fd2b0a4` and verified against its exact Git blobs.
Only this turn's regenerable test scratch (5.84 GiB) was removed; datasets,
failed preparations, coverage JSON and the model-free wheel are retained.

Five-view implementation verification: 165 focused tests passed; full provisioned
CPU suite 2,490 passed /one actual-CUDA unavailable skip, 32 existing ONNX
deprecation warnings, 694.88 seconds. Total coverage 96.37%; the eight study
modules aggregate 97.03%, each >=95%, new view decoder 99.04%. Ruff, 177-file
type checking, whitespace, identity checks and a 498,021-byte model/data-free
wheel pass. The all-optional-absent profile passed 119 with 46 skips, but it
does not represent partially provisioned environments. The preceding `b329fc9`
CI research/quality/package/benchmark jobs passed; all four lightweight Python
jobs failed on two missing-Torch test guards. Keep this red historical outcome.
These are regression checks, not real-GPU execution or learning qualification.

The two missing-Torch guards are now explicit in cancellation/kernel-resume
tests. This does not skip native JPEG/SciPy checks or reduce the complete
research coverage gate. The complete lightweight import-absence profile,
with native JPEG/SciPy present, now passes 1,836 tests with 234 explicit skips
in 103.56 seconds. Both repaired tests also pass with Torch present. This is
a local dependency profile, not actual Python 3.11–3.14 matrix qualification;
the all-optional-absent profile alone concealed the preceding CI failure.
Remote `4e36149` subsequently passed all eight CI jobs: the actual Python
3.11–3.14 matrix, quality, package, benchmark and complete CPU research gate
([run 38239679695](https://github.com/erkanrzgc/steganography/actions/runs/38239679695)).
Later documentation/evidence-only commits have their own CI runs; their success
is not inferred from this code checkpoint.

Causal JPEG controls (2026-10-12): `research jpeg-study --config FILE --out DIR`
implements bounded original-disjoint preparation, complete independent SciPy
IDCT audit, four source/initialization/order-matched optimizer/L2/batch arms,
two seeds, numeric resume snapshots and paired bootstrap learning gates.
Legacy `srnet_reference` and old trainers/checkpoint formats are unchanged.
An initial actual preparation stopped on an upstream iteration warning; direct
Shannon measurement found 0.07937% residual, inside the unchanged 0.1% entropy
bound. Both that run and a later actual excess-residual failure are retained.
An independently implemented bounded Gibbs/Shannon refinement now corrects
excess-residual probabilities without changing costs, seeds, tolerances or
selected originals. Native audit input is a verified private byte copy.
CUDA/SSH are unavailable and no new GPU fitting or accuracy
gain is claimed. Pretrained/color comparison, .2 adaptation and reserved-source
evaluation are not implemented yet; WIFD remains unscored. See
`JPEG_LEARNING_STUDY.md` for the complete first-slice contract and remaining work.

Actual center-only data gate: after 153 complete fit pairs, the next fixed BOSS
original failed unchanged-center detection. Independent SciPy reconstruction
confirms 646 full-image coefficient changes but **zero** in the center crop;
cover and stego inputs there are identical. The four fixed corners change
576/0/7488/5504 pixels. The incomplete corpus was retained, not fitted or filtered.
See [the portable single-original audit](../benchmarks/jpeg-study-center-signal-20261012.json).
This one case proves representation loss, not the cause of chance AUC on the
whole previous corpus. The label-independent five-view revision above was
explicitly approved; the center-only run stays incomplete and no new GPU trial
has occurred. Like the other
source-bound research runners, this command requires a source checkout with
protocol docs and explicit optional research dependencies, not only the wheel.

Verification of this slice: 127 focused tests; full provisioned CPU suite
2,452 passed /one actual-CUDA unavailable skip, 32 existing ONNX deprecation
warnings, 597.48 seconds. Total coverage 96.34%; the seven new executable study
modules have 96.59% aggregate coverage and each exceeds 95%. Ruff, 176-file
type checking, diff whitespace and a 493,882-byte model/data-free wheel pass.
An explicit import-absence profile passed 88 tests with 39 optional-dependency
skips; this is a regression profile, not actual GPU or Python-version qualification.
Malformed worker input no longer imports Torch to classify non-CUDA failures.
These checks do not qualify real-GPU execution or learning. Only this turn's
generated test scratch directories are removed after verification; datasets,
partial failed preparations, caller files and public evidence are retained.

Report parity (2026-10-11): HTML now shows CTF routes, required/optional
coverage and tool execution status. Escaped complete normalized JSON and SARIF
run properties preserve additive fields, limits and incomplete jobs; bundle
views round-trip the same evidence. No score, verdict or recovery change.
The prior `c077797` checkpoint passed both normal and full-image CI.

Report parity verification: seven new tests cover complete/failed/cancelled
jobs, empty/legacy views, additive fields, HTML injection escaping, bundle
round-trips and real service output without host paths. The focused integration
run passed 77 tests. Full suite: 2,325 passed, one actual-CUDA unavailable skip,
96.32% total coverage in 548.58 seconds; all seven changed executable lines in
`report.v2` were covered. Lint, 169-file type checking and a model-free wheel
build passed. These are content round-trip checks, not formal SARIF schema or
consumer interoperability qualification; detection/recovery scores are unchanged.

CTF inspection routes (2026-10-11): successful allowlisted native findings now
rank specific next steps without changing verdicts or extraction. Unknown,
informational and failed findings are excluded; raw plug-in/tool details are
not echoed. Required native coverage gaps are not disguised as optional tools.
The prior `0b2f2ce` checkpoint passed both normal and full-image CI; its local
full suite passed 2,282 tests with one explicit CUDA skip and 96.31% coverage.
Blind method discrimination and real-dataset accuracy gates remain open.

Triage verification: 36 new recommendation tests; 86 focused CTF/evaluator/matrix
tests passed. Full provisioned suite: 2,318 passed, one actual-CUDA unavailable
skip, 96.32% total coverage (789.07 seconds); all 20 changed executable lines
in `core.ctf` covered. Lint, type checking and model-free wheel build passed.
Fresh hardened full-image smoke and independent matrix passed (17 native plus
four upstream-tool fixtures). These counts do not qualify real-world accuracy.

Paired development recovery (2026-10-11): frozen native fixtures move 14/17 ->
17/17 completed exact payload recovery, with unchanged inputs/budgets. Fixes:
BMP trailer carving and per-channel stereo WAV LSB extraction, with invalid
BMP/PCM/read-limit tests. One generated group, no real-world/blind claim.
`CTF_MATRIX_RESULTS.md` preserves both runs. The preceding `1b54f0c` checkpoint
passed complete GitHub CI and full Docker.

CTF evaluation checkpoint (2026-10-11): the frozen-manifest runner verifies
exact non-root artifact bytes after solving, retains unavailable/failed cases,
and separates complete recovery from incomplete recovery and timing. Its two
encoding examples remain development evidence. Blind qualification is always
unavailable pending independent challenge custody and audit. See `CTF_EVALUATION.md`.

The preceding `b3ba459` checkpoint passed GitHub CI (Python 3.11–3.14, quality,
package, benchmark and full CPU research) and hardened full-image recovery.
Local complete suite: 2,207 passed, one actual-CUDA-unavailable skip, 96.17% coverage.

Latest checkpoint (2026-10-11): full-image direct tools and three base-image
digests are pinned. The checked 12-tool/four-gem inventory retains upstream
license identities and verifies executable/JAR/gem-tree hashes at runtime.
Local non-root/read-only/network-disabled smoke passed native, Steghide,
explicit-wordlist Stegseek and OpenStego exact recovery, cancellation/timeout,
symlink/overwrite/output limits and recovered-passphrase redaction. See
`FULL_IMAGE_VERIFICATION.md`; these are controlled fixtures, not blind accuracy.
Focused inventory/runner coverage is >=95%; complete regression/remote Docker
checks passed as recorded above. Old failed CI is not
rewritten: seven jobs passed, but its complete CPU suite had one strict
readiness-test assumption failure. The production numerical gate is unchanged.

Source version is 0.6.0, unreleased and classified beta. Baseline 0.5.0 was
validated and committed as `1aaa747`. The complete v0.6–v1.0 roadmap has not
been delivered or validated by this slice.

## Implemented

- Frozen train-only SRNet BN diagnosis completed: 32 selected training pairs,
  26 layers, original model unchanged. Native scores saturate in 56/64 inputs;
  the batch-stat contrast in 0/64. This reveals normalization-mode sensitivity,
  not held-out accuracy; diagnostic BN inference is never enabled for primary
  verdicts. See `SRNET_TRAIN_DIAGNOSTIC_RESULTS.md`; previous failures stand.

- First frozen real-data SRNet pilot completed: one BOSS-only epoch, 412
  updates, 229.29 seconds, all 765 validation rows evaluated and nine independent
  NumPy comparisons passed. Detection failed all six cells (AUC 0.502–0.510,
  balanced accuracy 50%, FPR 24–28.78%, ECE 0.491–0.500). Nothing deployed;
  source/sample-size qualification and full ONNX replay remain unavailable.
  See `SRNET_REAL_PILOT_RESULTS.md` and portable per-cell evidence.

- `research srnet-evaluate` binds complete validation inference to verified
  training/cache/schedule/card/model provenance and independent NumPy gates.
  Real pilot results are now recorded above. Full ONNX replay/qualification
  remain unavailable. Direct evaluation deadlines are cooperative; the pilot
  invocation additionally uses an external hard wall timeout.

- SRNet fitting CLI now uses hard-limit isolated jobs and artifact verification;
  independent NumPy forward and optional dynamic-batch ONNX have generated
  readiness checks. ONNX preflight rejects unsafe allocation/topology before
  runtime. Real NumPy replay is recorded above; full ONNX replay remains pending.

- Shared SRNet paired CPU training engine and `research srnet-fit`, bound to
  complete float-cache/plan provenance with finite learning/BN checks and
  pickle-free model cards. Real one-epoch fitting/NumPy evidence is now recorded;
  trained-real-model ONNX export evidence remains unavailable. Hard-limit jobs and generated
  numerical readiness are implemented as described above.

- SRNet train-only paired schedule service and shared `research srnet-plan` CLI:
  source-first quality/method balance, complete original pair coverage, bounded
  oversampling, independent RNG, per-epoch order hashes and source exclusions.
  Frozen `SRNET_SAMPLING_PROTOCOL.md`; all 30 real scope/epoch accounting checks
  pass, complete original pair coverage and 765 validation hashes excluded.
  See `SRNET_SAMPLING_RESULTS.md`; no fitting or detector score changes.

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

- Train-only SRNet diagnosis: Python 3.11.14, 1,139 tests pass, 94.86%
  total coverage; new core/service 125/125 statements covered. Ruff, mypy
  (115 files), whitespace and model/data-free wheel/sdist checks pass. Real
  32-pair probes verify unchanged model state/disk hash and no validation pixel
  loading. This is diagnosis, not better held-out accuracy. Python 3.12–3.14
  and fresh full Docker remain unverified locally.

- Frozen real SRNet pilot/results service: Python 3.11.14, 1,120 tests pass,
  94.80% total coverage; new diagnostic service 67/67 statements covered.
  Ruff, mypy (113 files), whitespace and model/data-free wheel/sdist checks
  pass. Both complete real evaluation runs reproduce all predictions/audits/
  metrics/intervals. All six detection cells fail; numerical success is not
  accuracy. Python 3.12–3.14 and fresh full Docker remain unverified locally.

- SRNet complete-validation service: Python 3.11.14, 1,093 tests pass,
  94.77% total coverage; evaluation/shared fit 149/149 statements covered.
  Ruff, mypy (112 files), whitespace and model/data-free wheel/sdist checks
  pass. Generated actual paired-fit/native/NumPy integration and distinct-row
  ordering are verified; no real-corpus fit/accuracy qualification. Python
  3.12–3.14, full Docker and hard-kill evaluation isolation remain unverified.

- SRNet isolated-job/numerical readiness: Python 3.11.14, 1,068 tests pass,
  94.72% total coverage; new runner/reference/ONNX/preflight 241/242 statements
  covered (99.59%). Ruff, mypy (111 files), whitespace and model/data-free
  wheel/sdist checks pass. Actual limited child fits generated fractional data;
  independent NumPy and dynamic batches 1/2/4 pass fixed logit/score/decision
  guards. No real-corpus training/replay or accuracy qualification. Python
  3.12–3.14 and fresh full Docker remain unverified here.

- SRNet fitting implementation: Python 3.11.14, 1,034 tests pass, 94.62%
  total coverage; training/fit/plan services 172/172 statements covered.
  Final strengthened source-index tests: 39 focused tests pass. Ruff, mypy
  (107 source files), whitespace and model/data-free wheel/sdist checks pass.
  Fitting uses generated fractional fixtures; forged plans, invalid gradients/
  weights and cooperative timeout cannot publish completed models. No actual
  real-corpus fit or accuracy audit. Other Python versions/full Docker unverified.

- Published SRNet schedule evidence: Python 3.11.14, 995 tests pass, 94.55%
  total coverage; sampler/plan service 116/116 statements covered. Ruff, mypy
  (105 source files), whitespace and model-free wheel/sdist checks pass.
  All 30 real schedule accounting checks pass; checkout-bootstrapped replay
  produces byte-identical records. No real fitting, accuracy or deployment.
  Python 3.12–3.14 and fresh full Docker not rerun locally.

- SRNet schedule implementation freeze: Python 3.11.14, 993 tests pass,
  94.55% total coverage; sampler/plan service 116/116 statements covered.
  Ruff, mypy (105 source files), whitespace and model-free wheel/sdist checks
  pass. Independent real schedule accounting remains pending at this freeze;
  no real fitting or accuracy evidence. Other Python versions not rerun here.

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
