# Changelog

- Recover and verify the original on-device state-check GPU report after local
  SSH restoration, without rerunning/overwriting. All 16 source hashes match;
  fixed five-epoch estimate 3130.26s remains ineligible against 1800s. Preserve
  retrieval history and earlier failures; no real fitting or accuracy gain.

- Add on-device full numeric-state validation for explicit CUDA, preserving
  exact keys/shapes/dtypes, finite values, BN variance/counter bounds and a final
  original NumPy check after CPU conversion. Preserve CPU validation/arithmetic.
  Freeze separate generated timing; equivalence tests are not real accuracy.

- Record separately preregistered physical CUDA timing after gradient-check
  consolidation: p95 0.095847s, fixed five-epoch estimate 3271.44s, still
  ineligible against 1800s. Preserve original evidence and all guards; no
  controlled speedup, real fitting or accuracy qualification claimed.

- Consolidate CUDA finite-gradient boolean synchronization without skipping any
  element or changing optimizer arithmetic. Preserve legacy CPU checking,
  per-update full numeric-state validation and resource limits. Generated
  exact-update equivalence is not a measured GPU speedup or accuracy gain.

- Publish exact physical 64-update CUDA timing with immutable source/protocol
  checks. The conservative five-epoch estimate is 3820.35s, above the unchanged
  1800s job limit; record ineligible without running an incomplete real fit,
  changing exposure/safety assumptions or claiming accuracy improvement.

- Add an explicit isolated generated CUDA timing service: 64 shared-engine
  updates, warmed inter-fetch intervals including numeric-state verification,
  source snapshots and fixed conservative five-epoch budget rule. Keep RAM,
  allocator and 1800s fit limits; timing is not learning or accuracy. Ignore
  the documented separate GPU virtual environment without publishing models.

- Record the first completed physical WSL2/RTX 5060 Laptop generated CUDA
  preflight, original-report checksum and matching execution-source hashes.
  Freeze local transfer/train-only readiness before checking the fresh copy;
  all 7,380 train rows and exact historical tensor/schedule hashes pass.
  Add immutable evidence/protocol regression checks; no raw data/weights
  published, real fit or improved accuracy implied.

- Add opt-in, no-fallback FP32 CUDA streaming execution and versioned GPU plan/
  card metadata, scoped deterministic/precision/RNG policy, allocator limits and
  required kernel resident RAM bounds. Add isolated generated one-update/parity
  preflight with explicit hardware unavailability. Preserve CPU defaults and
  published failures; emulated tests do not prove GPU or real detector accuracy.

- Add a versioned, checksum/audit-bound train-only block reader, full-source/
  quality/method four-row schedules and isolated research plan/check/fit jobs.
  Legacy whole-array limits and numerical optimizer behavior are preserved;
  both paths use one engine. Propagate a whole-job deadline through hash/read/
  optimizer boundaries, bind source hashes before data loading, and reject
  mutation before publication. Generated exact update equivalence is not a
  real learning or accuracy improvement. Full real train-only streaming reads
  all 7,380 unique rows and the complete planned epoch with <=1 MiB returned
  batches, without opening validation pixels. No real model is trained/deployed.

- Add frozen whole-expansion JPEG preparation in <=128-original blocks, with
  complete lineage/role/exclusion accounting and fresh unrounded-Y float caches.
  BOSS supports bounded explicit selection offsets without changing legacy
  defaults or increasing trainer caps. Add independent full JPEG/cache audit,
  PGM-to-JPEG cover replay, simulated DCT change/rate checks and sampled scalar
  IDCT oracles. No fitting, validation-score tuning or deployed detector change.

- Add frozen, explicit corpus expansion and pinned MIT-licensed WIFD acquisition
  with bounded fixed-origin downloads, SHA-256/Git blob identities and fresh
  outputs. Preserve the failed JPEG-only attempt; separately freeze opt-in
  primary-MPO acquisition without relabeling or converting files. Independent
  audits verify bytes, license evidence, splits/reserved identities and
  quarantine unchanged ALASKA cover/stego lineages. No detector/model change.

- Add the frozen optimizer/exposure-matched accumulation control: exact
  eight-row order, atomic two-microbatch gradient accumulation, independent
  full-network update equivalence, finite-gradient failure checks and preserved
  legacy training behavior. Complete real-train arms fail learning goals;
  original BA .50, amplified BA worsens, all 18 numeric oracles pass. Record
  full prepared data versus sanity-subset scope; no deployed detector change.

- Add the separately frozen, bounded eight-row training-context control:
  distinct source/lineage matching, unchanged legacy trainer callers, exact
  row-exposure/update accounting, serialized replay and independent oracles.
  Completed real-train arms reduce singleton loss but fail overall learning
  goals; original-input training BA stays .50, all 18 numeric oracles pass.
  This opt-in research experiment does not change the deployed detector.

## 0.6.0 (unreleased)

- Complete frozen train-only BN context/clone-refresh controls: both refresh
  goals fail, amplified context sensitivity remains diagnostic only; original
  weak-input training accuracy stays chance-level. Preserve source weights and
  all prior failures; numeric reload/oracles and parameter identity pass.

- Add preregistered real-delta signal-strength controls with separate original
  and derived identities, identical two-arm sampling, ordinary singleton eval,
  bounded numeric snapshots and per-arm independent reload/oracle verification.
  Artificial amplification and in-sample results never qualify real accuracy.

- Add frozen train-only SRNet gradient/null controls with state/RNG/flag/gradient
  preservation, bounded process execution and classifier central differences.
  Complete all six context cells in nine probes with exact repeated replay;
  publish descriptive weak/strong contrasts without detector accuracy claims.

- Complete preregistered generated strong-signal SRNet learning control: all
  three train-only objectives pass, with complete numeric reload and six NumPy
  forward oracles. Preserve real-data failures and unavailable qualification;
  no model installed or primary detector changes.

- Publish complete tiny train-only SRNet sanity: 50 epochs/400 updates, two
  learning objectives fail despite saved-model/NumPy replay and nonzero paired
  input differences. Keep loss decrease separate from actual discrimination;
  no model deployed and no validation-driven tuning.

- Add frozen, bounded 24-row train-only SRNet learning sanity: metadata-only
  complete lineage/quality selection, 50-epoch/BN accounting, singleton stored-BN
  evaluation and redacted failure handling. In-sample objectives never qualify
  detector accuracy; existing trainer/models/verdicts remain unchanged.

- Complete frozen two-source SRNet fitting and full evaluation: 1,580 updates,
  nine numerical replays and six independent metric audits. Publish every failed
  cell and prior-pilot regression; reduced overconfidence does not hide increased
  false positives or near-chance AUC. No model installed or deployed.

- Pin prior SRNet pilot evidence in the two-source comparison and reject extra
  duplicate cells; changed baselines cannot silently become a new comparator.

- Add frozen two-source real-run evaluation/publication audit and descriptive
  prior-pilot context comparison. Reject changed recipes/optimizer/counts and
  retain failed numerical gates without accuracy claims. Start the bounded
  real CPU fit; no new measured accuracy is claimed before it completes.

- Add explicit two-source/four-row SRNet training with exact v2 plan/card batch
  hashes and optimizer/BN accounting; preserve legacy v1 behavior and standard
  eval inference. Audit all 1,580 real batches and complete original coverage;
  actual multi-pair fitting/accuracy remain unavailable, no detector deployed.

- Add state-preserving, train-only SRNet BN diagnostics with provenance/cell/
  finite/deadline guards and a shared CLI. Publish real paired mode contrast;
  never promote in-sample scores or paired BN to deployed detector behavior.

- Complete the frozen real-data SRNet pilot: 412 paired updates, all 765
  validation rows and nine independent numerical replays. Publish all six
  failed detection cells and lineage intervals; no model installed/deployed.
- Add provenance-bound context diagnostics with complete-row/paired-lineage
  validation, finite-score and unchanged-gate guards; no threshold tuning.

- Freeze the first bounded real-data SRNet engineering pilot before fitting:
  one complete BOSS-only source-exclusion epoch, unchanged validation and gates.

- Add SRNet complete-validation replay bound to train/validation caches,
  fit plan/card/model and BN accounting. Failed independent numerical gates
  retain evidence but publish no predictions; no accuracy/deployment claim.

- Isolate explicit SRNet fitting under hard wall/CPU/memory/file limits and
  verify complete artifacts. Add independent NumPy math and optional bounded
  ONNX export/replay with allocation preflight; generated checks, no accuracy claim.

- Add explicit plan-bound paired SRNet CPU fitting, finite loss/gradient/BN-state
  guards and numeric provenance cards. Preserve RNG/threads on failure, reject
  validation/forged plans; generated regression only, no real fit or deployment.

- Complete 30 real SRNet schedule accounting checks across all-source and both
  source exclusions; retain exact oversampling counts/order hashes, exclude all
  validation rows. Fix audit entrypoint to use checkout modules, not stale wheels.

- Add checksum-bound SRNet training-plan preparation with deterministic matched
  pairs, hierarchical source/quality/method balance, bounded oversampling and
  per-epoch provenance. Reject non-train/incomplete rows; no real fit implied.

- Complete unchanged-corpus unrounded JPEG preparation: 3,750 source hashes,
  18 independent real IDCT examples exact. Add BN-safe bounded SRNet numeric
  weights/statistics persistence with hostile-header tests; no real model fit.

- Add explicit bounded unrounded JPEG Y/256 float preparation and shared
  phase-aligned cache/inference contracts. Respect actual component-table
  assignment and color-space names; preserve old uint8 cache/limits/verdicts.

- Add separate opt-in SRNet-style architectural preparation and explicit
  resource-limited synthetic readiness command; enforce all-stage eval BN and
  bounded uint8 inference. No real training, installed model or accuracy claim.

- Publish controlled residual-domain trial: three fixed fresh fits, all twelve
  detection cells fail again. Independent numeric/ONNX replay and paired
  comparisons against both previous CNN and JRM pass; no detector deployment.

- Add opt-in architecture-bound pixel-unit residual preprocessing for a frozen
  controlled trial. Preserve previous CNN defaults and reject cross-domain
  numeric weights/cards; no deployed detection change.

- Publish first real pixel-CNN three-fit experiment: all 12 detection cells
  fail, zero recall and near-chance AUC. Complete independent numeric/metric
  and ONNX replay, lineage uncertainty and prior-reference regressions retained.
  No detector deployment or accuracy qualification.

- Add bounded CPU pixel-CNN minibatch learning/prediction and numeric-only
  model persistence. Restore thread/RNG state, reject unsafe array headers and
  provenance/settings, preserve fixed filters; freeze three-source-scope trial.

- Publish complete real-corpus pixel preparation audit: 3,750 file hashes,
  decoder-bound uint8 caches and 18 isolated crop oracles. CNN filter/gradient
  smoke passes, but no actual trained model or accuracy/deployment claim.

- Add bounded decoded JPEG center-crop research caches and a custom opt-in
  pixel-residual CNN building block. Pin crop/codec contracts, keep generic limits
  and detector behavior unchanged; preparation is not training/accuracy evidence.

- Publish independently audited two-way JRM training-source exclusion: weak
  cross-origin AUC .52–.55, all eight detection cells fail. Keep within-origin
  controls, paired uncertainty and source-scope hashes; no detector deployment.

- Add explicit JRM train-origin exclusion, manifest-bound additive scope cards
  and shared two-way source-transfer diagnostic. Preserve legacy cards and
  detector behavior; freeze the eight-cell comparison before real cached fitting.

- Completed full-corpus JRM/FLD reference and replay audit: ALASKA/UERD AUC
  .603 → .698, FPR 38.54% → 31.71%, but JUNIWARD recall/BOSS ECE regress.
  All cells and failures published; exact independent scalar votes, unchanged
  primary detection, no supported methods or independent-source qualification.

- Preflight numeric FLD NPZ headers before allocation: reject gigantic declared
  shapes, invalid dtype/version and truncated payloads even in tiny archives.
  Add adversarial tests proving unsafe inputs never reach NumPy array loading.

- Added optional pinned upstream JRM/FLD local research reference: bounded batch
  workers, provenance-bound raw caches, numeric-only immutable ensemble weights
  and explicit CLI stages. Component research-use notices preserved; freeze
  comparison protocol before real-data processing. Deployed detection unchanged.

- Published the fixed nonlinear JPEG residual-feature experiment and independent
  audits: ALASKA AUC .609/.603 and FPR 38.54%, but BOSS/UERD and calibration
  regressions. All detection cells still fail; exact ONNX parity, no deployment.

- Added opt-in fixed residual-feature MLP64 with linear skip, architecture-bound
  research domains, train-only normalization and explicit weight-decay
  provenance. Legacy linear checkpoints and deployed detection unchanged.
  Freeze nonlinear comparison protocol before actual training/scoring.

- Published the independently audited same-file JPEG residual/parity experiment:
  FPR 50.24% → 46.34% on ALASKA and 48% → 40% on BOSS simulations, but several
  recall/ECE regressions and failed detection gates. All source/quality cells and
  paired change intervals retained; exact CPU ONNX parity, no model deployment.

- Added a custom 2,066-feature JPEG block-DCT residual/parity research descriptor,
  including DC statistics and the unchanged measured-context prefix. Reuse
  allowlisted bounded JPEG workers and require two-origin objective provenance.
  Fixed comparison protocol precedes actual feature extraction/training;
  old contracts, primary detection and model installation remain unchanged.

- Added minimum native format coverage policy: incomplete low-score pipeline
  results are inconclusive, not no-indicators. Preserve positive findings and
  unchanged scores; failed analyzer signals cannot confirm. Additive JSON v2
  revision 2, HTML and SARIF retain required-component execution assessment.

- Published the fixed two-origin JPEG context experiment and independent audits:
  3,750 files, ALASKA FPR 50.24% and BOSS-simulation FPR 48%; AUC/BA failures and
  all source/quality intervals retained. Strict CPU ONNX parity passes; detection
  regresses, no model installed and no cross-source qualification claim.

- Added experimental measured JPEG content/quantization interactions (1,098
  features), optional pinned BOSS JPEG simulation and checksum-bound two-origin
  preparation with matched JUNIWARD/UERD families and source/class-balanced
  training. Fixed protocol precedes real-data scoring; old contracts, primary
  analysis and installed models remain unchanged. JMiPOD is not substituted.

- Published independently verified 24-cell spatial and 9-cell JPEG context
  diagnostics: no unseen validation sources; missing declared quality/device
  metadata; format-specific low-rate/FPR failures retained. No retraining,
  score improvements, context-aware deployment or generalization claims.

- Preserve historical JPEG v1 prediction documents in source/context audits:
  missing additive format/rate fields come only from the already checksum-bound
  manifest after full identity validation; stored scores/documents stay unchanged.

- Added provenance-bound `research diagnose` source/context audits: source
  overlap, per-method/rate format and declared-quality strata, paired intervals,
  explicit missing metadata and unavailable single-label cells. These diagnostics
  do not change scores, train context-aware models or qualify generalization.

- Published the fixed weighted-objective comparison: low-rate sequential/
  scattered recall 38.04%/14.67% → 67.93%/29.89%, but shared FPR 1.63% → 3.26%
  and higher-rate/ECE regressions. All cells, paired intervals and independent
  numerical audits are retained; no deployment or cross-source support claim.

- Added explicit controlled-lineage low-payload training weights, weighted
  class balancing, persisted recipe provenance and a cached-vector comparison
  runner. Defaults and deployed detection are unchanged; the experiment is
  preregistered before training in `SPATIAL_WEIGHTING_PROTOCOL.md`.

- Published audited parity/residual results: scattered-5% development AUC
  0.716801 → 0.906486 and same-cohort FPR 4.89% → 1.63%, but low-rate fixed-.5
  recall and ECE regress. All six cells stay visible; strict stable ONNX parity
  passes and neither model nor threshold is deployed.

- Added a versioned 684-feature spatial parity/residual descriptor and fixed
  same-corpus development comparison. Only the new descriptor uses 1e-8
  quantization; legacy extraction and deployed primary detection are unchanged.
  Training can explicitly request stable double-accumulation inference.

- Published CPU numerical repair audits: all three explicit precision derivatives
  pass unchanged strict 1e-6 gates with original threshold decisions preserved.
  Cover-only operating-point assessment changes 3/90 false alarms to 2/90 but
  lowers low-rate recall; no calibration/deployment or cross-source claim.

- Added preregistered development-only spatial operating-point selection:
  whole-lineage fitting/assessment roles, cover-only empirical FPR threshold,
  exact tie handling, and per-cell recall/false-alarm uncertainty. Scores are
  not probability-calibrated, and no installed detector threshold is changed.

- Added explicit double-accumulation research inference with float32 I/O,
  preserving legacy arithmetic. Derived checkpoints retain original weights
  and provenance; bounded ZIP/weights-only loading protects research exports.
  Numerical repair audit is preregistered separately from detection accuracy.

- Published the audited 7,000-file PNG/BMP development comparison: richer
  features improve same-source AUC to 0.717–0.997, but 4.89% FPR, weak low-rate
  scattered recall, calibration and strict ONNX logit parity remain failed.
  Both models stay local and undeployed; frozen native baselines are unchanged.

- Added preregistered reserved-safe PNG/BMP controlled development comparison:
  independent grayscale LSB pairs, bounded 468-feature residual co-occurrences,
  shared validation inference, fair train-only normalization/class balancing,
  paired uncertainty and actual-artifact ONNX parity. Not deployed or qualified.

- Added separate BOSSbase development acquisition: reserved-member selection
  exclusion, SHA/lineage overlap checks, bounded shared ZIP/range helpers,
  CRC/PGM validation, exclusive outputs and provenance-bound resume. Originals
  remain local and this does not claim spatial detector accuracy improvement.

- Published the independently audited 7,000-file FSDD baseline: all six WAV
  sequential/scattered LSB method/rate cells have AUC 0.50 and recall 0%.
  All 6,000 generated payloads were exactly oracle-verified; native recovery
  was not measured. Test speakers remain reserved; detector support is unproven.

- Added preregistered, bounded FSDD WAV LSB baseline preparation/evaluation:
  whole-speaker reservations, six method/rate cells, exact independent scalar
  payload verification, original RIFF metadata preservation and shared-service
  scoring. Required coverage failures invalidate metrics; no threshold tuning.

- Published the first isolated JPEG development result: same-source validation
  AUC 0.648804 / 0.592481 / 0.585306, but 41.46% FPR. The model is not deployed
  and this is not a new held-out accuracy claim. Actual-artifact ONNX differences
  and the failed strict 1e-6 score-parity check are recorded without relaxing it.
- Added explicit pinned FSDD WAV acquisition with license evidence, ZIP/PCM
  limits, CRC/SHA verification, exclusive outputs and adversarial tests. Audited
  3,000 real recordings; audio detection has not yet been evaluated.
- Added a public dataset/license catalog and method-specific results index,
  preserving failed baselines and distinguishing validation, CTF and test results.

- Added explicit, reserved-lineage-safe ALASKA2 development acquisition and
  development-only import, separate from the frozen test subset. Contradictory
  source labels on identical bytes quarantine whole lineages before training.
- Added opt-in JPEG DCT summary features, bounded worker parsing, train-only
  normalization, class-balanced CPU linear training, validation predictions and
  normalization-preserving ONNX export. No automatic detector/model deployment.
- Added a preregistered ALASKA2 evaluator using the unchanged shared analysis
  service: fixed-threshold per-method metrics, paired-lineage confidence
  intervals, explicit coverage failures, persisted score evidence and declared
  sensitivity to ambiguous upstream labels. No held-out threshold search.
- Published the independently audited 4,000-file ALASKA2 baseline: JMiPOD /
  JUNIWARD / UERD recall 0.4% / 0.1% / 0.1% at threshold 70, AUC 0.525617 /
  0.505157 / 0.507181. All methods fail detection targets; no improvement,
  calibrated confidence, blind recovery or cross-source support is claimed.

- Added explicit, bounded ALASKA2 subset acquisition with pre-scoring paired
  selection, ZIP64 directory/range limits, credential isolation, CRC/JPEG/SHA
  integrity checks and provenance-bound incomplete-job resume. This reserves
  evaluation data; it does not train a detector or establish an accuracy result.
- Recorded the completed 4,000-file ALASKA2 acquisition and independent integrity
  audit. Three upstream UERD samples are byte-identical to their covers; separate
  file downloads confirmed this, and the frozen selection retains the ambiguity.

- Published repeated CTF regression evidence on the frozen Kodak challenges:
  two final runs achieved 30/30 completed exact recoveries with unchanged limits.
  Failed intermediate runs are retained; this is not a blind accuracy result or
  an improvement claim for automatic steganalysis.

- Bounded OpenStego's glibc allocator arenas and CPU-derived JVM thread pools
  within the unchanged 768 MiB process budget; added repeated cold-start exact
  recovery to the Docker smoke check. A failed JVM's output remains excluded.

- Fixed CTF candidate pollution: binary data is no longer lossily URL-decoded,
  BMP carving checks header consistency and declared extent, and carved files
  retain format suffixes for recursive native analysis. Boundary artifacts are
  analyzed without producing children beyond `max_depth`; native extraction
  limit exceptions propagate as cancellation instead of being swallowed.
- Disabled POSIX tool core dumps and restricted artifact adoption to a successful
  extractor's explicitly named, nonempty regular output. Failed/partial output,
  logs, symlinks and preexisting paths cannot become extraction evidence.
  Pilot CTF evaluation now preserves per-job reports, errors and output counts.

- Added explicit bounded Kodak-suite acquisition and pilot generation with both
  upstream methods on each original. Published negative external-source detection
  evidence separately from controlled payload recovery and job completion.

- Fixed spatial score inflation: legacy SPA/RS adjacency proxies now share one
  correlation group; their signal names are retained for existing consumers.
- Bounded PDF ASCIIHex/ASCII85 output before allocation, including ASCII85 zero
  runs, and rejected truncated Flate streams. Added shared per-parse decode
  budgets and passed the remaining CTF output allowance to PDF extraction.

- Added bounded, manifest-bound PNG/BMP spatial-summary feature extraction for
  train/validation; test image bytes are not accessed by this workflow.
- Training now rejects unbound NPZ inputs, mismatched feature/manifest hashes,
  invalid feature values and rows outside the exact train membership. Checkpoints
  retain provenance and identify the linear baseline accurately (not SRNet).
- Added training provenance, nonfinite loss guards, and safe ONNX export with model
  cards; verified CPU training reproducibility and ONNX parity against ONNX Runtime.
- Fixed raw 32-byte Ed25519 public key parsing in model verification when keys
  contain leading/trailing whitespace bytes.
- Added distinct chi-square and higher-plane-gated adjacency heuristics plus
  markerless synthetic gradient regression tests. These are uncalibrated
  proxies, not full SPA/RS algorithms or real-world accuracy evidence.
- Added smooth gradient cover generation and `EXTENDED_RECIPES` with `--extended`
  CLI flag to benchmarking corpus generation.
- Added JPEG DQT table analysis (flat unit quantization detection), Westfeld DCT
  Chi-Square PoV steganalysis, JSteg marker detection, native JSteg payload
  extraction, and automated CTF solving for JSteg carriers.
- Added WAV audio steganalysis (dynamic range-gated LSB bias, 8-bit PCM Pairs of
  Values Chi-Square, quiet-region LSB noise transition tests), structural RIFF
  trailer detection and carving, and raw LSB payload extraction in CTF analysis.
- Added GIF block-stream parsing (comment/trailer flag detection, palette duplicate
  entry analysis, animation frame-delay ASCII/binary steganalysis), exact structural
  GIF trailer carving, native GIF comment/delay extraction, and raw text
  whitespace/zero-width payload recovery in CTF solving.
- Added 4-pixel spatial calibration for JPEG DCT steganalysis to detect F5 matrix
  embedding shrinkage (excess zeros and ones depletion), and integrated external
  tool CTF flag pattern verification.
- Added MP3 audio carrier detection, ID3v2 metadata frame (`COMM`, `TXXX`, `PRIV`)
  and synchsafe parsing, ID3v2 padding area stego detection, MPEG audio frame
  private bit covert channel detection, exact structural stream end calculation
  with ID3v1 accounting, and automated CTF extraction for ID3 tags, padding,
  embedded APIC album artwork, and appended trailers.
- Added PDF carrier steganalysis (comment scanning, `/Info` metadata extraction,
  FlateDecode/ASCIIHex/ASCII85 object stream extraction, embedded file attachments,
  incremental revision checks, and structural `%%EOF` trailer carving), EXIF metadata
  flag extraction, and bounded BZ2/LZMA/raw deflate/hex/reversed CTF decoding.
- Added comprehensive unit and limit tests for `ArchiveSafeAnalyzer` (ZIP/TAR/EML
  traversal, resource limits, compression bombs), `ToolRunner` (cancellation,
  timeouts, secret redaction, output clipping), external tool adapters, and
  `ModelRegistry` verification guards.

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
