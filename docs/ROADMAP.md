# Roadmap and status

The frozen two-source real control completed (`SRNET_MULTIPAIR_REAL_RESULTS.md`):
1,580 updates, all 765 validation rows, nine independent numerical replays and
six independent scalar-metric audits. All six detection cells fail; AUC remains
near chance, overconfidence decreases but FPR rises. No model deployed. Next
freeze a train-only tiny-subset learning sanity control before longer compute;
reused development validation must not select settings or qualify support.

The opt-in four-row control is implemented and generated-fit/isolated-worker/
complete-eval tested. All 1,580 real batch groups pass full accounting
(`SRNET_MULTIPAIR_RESULTS.md`); all original stegos are covered, no validation
rows enter training. The real fit/eval is now complete as linked above; schedule
verification alone still does not imply improved accuracy.

`SRNET_MULTIPAIR_PROTOCOL.md` freezes the opt-in two-source/four-row control
before real accounting/fitting. Legacy v1 behavior remains unchanged; a new
plan/card v2 binds exact grouping and BN optimizer-update counts. No real
multi-pair fit or improved detector score is implied by plan verification.

The train-only diagnostic completed (`SRNET_TRAIN_DIAGNOSTIC_RESULTS.md`):
normalization-mode sensitivity and in-sample train/eval mismatch confirmed,
without model changes or validation loading. The separately frozen combined-source,
multi-lineage minibatch control is now implemented; paired BN is not deployed.

`SRNET_TRAIN_DIAGNOSTIC_PROTOCOL.md` freezes a train-only paired BN contrast
following the failed pilot. No model mutation, validation tuning or qualification;
then preregister a separate balanced learning experiment based on diagnosis.

`SRNET_REAL_PILOT_PROTOCOL.md` freezes a bounded one-epoch BOSS-only source
exclusion engineering fit before results. It is not multi-source qualification;
complete fitting/validation, numerical gates and honest failures are required.

The frozen first real BOSS-only pilot completed (`SRNET_REAL_PILOT_RESULTS.md`):
412 updates, complete validation and independent NumPy replay, all six detector
cells fail. Next diagnose train-only learning/BN/input behavior and freeze a
separate combined-source/longer-compute control; no weights are deployed.

Complete-validation SRNet evaluation is implemented (`SRNET_EVALUATION.md`):
full provenance/BN accounting, metadata-only oracle selection, no predictions
on failed numerical gates. Full ONNX replay and adequately sized independently
sourced evaluation remain pending after the failed first real pilot.
Hard-kill evaluation isolation remains pending; no new accuracy evidence.

SRNet isolated-job and independent numerical readiness are implemented
(`SRNET_NUMERICAL_READINESS.md`): hard CPU/wall/memory/file bounds, complete
artifact verification, separate NumPy math and optional dynamic-batch ONNX
with allocation preflight. Generated checks only; real compute/data/source
fitting controls and actual trained-model replay remain pending.

SRNet paired CPU fitting and provenance cards are implemented and generated-test
verified (`SRNET_TRAINING_ENGINE.md`). Exact shared plan reconstruction, finite
learning/BN state and caller RNG/thread restoration; no real-corpus fit yet.
Hard-limit jobs and independent forward/export now have generated readiness.
Next: bound evaluation and a frozen real optimizer/compute/data schedule.
No deployed model or new accuracy score.

`SRNET_SAMPLING_PROTOCOL.md` freezes checksum-bound train-only paired schedules,
balancing sources then quality/method contexts without dropping original pairs.
Explicit `research srnet-plan` and bounded sampler implemented; all 30 real
scope/epoch audits pass (`SRNET_SAMPLING_RESULTS.md`). Provenance-bound trainer/
card is now implemented; next: fixed optimizer/compute schedule and independent forward/export
parity. No real fitting or accuracy qualification.

`JPEG_FLOAT256_PROTOCOL.md` freezes versioned unrounded Y/256 preprocessing,
isolated framed workers, separate bounded float cache and independent IDCT
oracles before real preparation; no accuracy claim.
Preparation completed (`JPEG_FLOAT256_RESULTS.md`): all 3,750 original hashes,
18 independent real IDCT examples exact. BN-safe 183-array numeric persistence
also implemented and tested. Paired sampler accounting is complete as linked
above. Training provenance is implemented; independent export parity and a
frozen real schedule remain pending; untouched-source gates stay open.

SRNet-style structural preparation (`SRNET_PREPARATION.md`) replaces tiny-model
tweaks as the next research direction. Explicit resource-limited readiness,
learned front convolutions, no early pooling and eval-only BN implemented.
Unrounded/256 JPEG preprocessing and BN-safe numeric persistence are complete.
Next: independent parity and a frozen real-training protocol; no actual trained model
or untouched-source qualification yet. Prior failures remain unchanged.

`PIXEL_CNN_SCALE_PROTOCOL.md` freezes a controlled residual-domain follow-up:
same tiny network/data/optimizer, fixed pixel-unit residuals and clipping, both
source controls. Completed `PIXEL_CNN_SCALE_RESULTS.md`: all cells fail again;
do not pursue validation-driven tiny-network tweaks. Stronger learning and
untouched-source gates remain, no deployment.

First pixel-CNN trial completed in `PIXEL_CNN_TRAINING_RESULTS.md`: all twelve
cells fail, despite independent numeric/ONNX parity. Keep this failed baseline;
next research needs a stronger separately preregistered architecture and an
untouched external source, not validation-driven tuning or detector deployment.

`PIXEL_CNN_TRAINING_PROTOCOL.md` freezes the first three small raw-pixel fits
(all-source plus both source exclusions); bounded train/predict/numeric persistence
is implemented. Independent score/export audits and untouched-source gates stay
separate; no deployment or detector improvement inferred from training completion.

Pixel-residual preparation: `PIXEL_RESIDUAL_PREPARATION_PROTOCOL.md` fixes raw
center-crop caches and a custom small optional CNN before real-corpus extraction.
Training/persistence, fixed source-control comparisons and untouched-source
qualification remain separate pending gates; no detector score claim.
Completed `PIXEL_RESIDUAL_PREPARATION_RESULTS.md`: all 3,750 files rehashed,
18 bounded crop oracles exact, custom filter/gradient smoke passes. Bounded
minibatch training/persistence and all-source/source-exclusion runs are now
complete; the failed first CNN trial is linked above.

Before raw-residual learning, `JRM_TRANSFER_PROTOCOL.md` freezes a two-way
training-source exclusion diagnostic on existing bound caches. This measures
transfer versus within-source controls, not new untouched-source qualification.
Completed `JRM_TRANSFER_RESULTS.md`: cross-origin AUC .52–.55, all eight cells
fail; source-exclusion controls must accompany the next raw-residual experiment.

Completed implementation reference: fixed optional JRM + FLD protocol
in `JRM_REFERENCE_PROTOCOL.md`, all original files/splits, numeric-only caches
and model serialization. This changes several experimental choices together,
not a causal ablation or independent-source qualification. Raw CNN and untouched
source gates remain open; upstream research-use terms are recorded.
`JRM_REFERENCE_RESULTS.md` preserves improved ALASKA/UERD ranking (.698 AUC),
lower FPR (31.71% / 30%), JUNIWARD recall and BOSS ECE regressions. Every
measurable detection cell fails; no deployment. Next: fixed raw-residual learning
and licensed untouched-source qualification, not further validation tuning.

The next fixed development experiment (`JPEG_NONLINEAR_PROTOCOL.md`) tests a
64-unit nonlinear residual-feature network with a linear skip, not a CNN.
Untouched-source acquisition/evaluation and raw residual learning remain open;
no automatic model deployment or new supported methods are implied.
Completed `JPEG_NONLINEAR_RESULTS.md` retains modest ALASKA ranking gains,
BOSS/UERD and calibration regressions; every measurable detection cell fails.

The next JPEG slice is frozen in `JPEG_RESIDUAL_PROTOCOL.md`: a custom residual/
parity descriptor with DC support, same-file comparison and unchanged objective.
Completed results (`JPEG_RESIDUAL_RESULTS.md`) show small ranking/FPR changes
with recall/ECE regressions, no qualified detection. Move beyond coarse linear
summaries to a stronger residual/context architecture and untouched-source gates;
no new supported method or deployed detector implied.

Minimum native coverage policy now closes the low-score/missing-detector gap
(`COVERAGE_POLICY.md`); this changes conservative pipeline verdict handling,
not primary scores or the failed detector accuracy gates.

Next measured JPEG slice is frozen in `JPEG_CONTEXT_PROTOCOL.md`: optional local
BOSS JPEG simulations plus ALASKA2 development, matched JUNIWARD/UERD families,
byte-derived content/quantization interactions and source/class-balanced training.
The completed run (`JPEG_CONTEXT_RESULTS.md`) removes a single declared training
origin, not the qualification gap: FPR worsens and detection fails. Richer
residual/architecture work and untouched-source evaluation remain next.
JMiPOD and installed detector behavior are unchanged.

Generalization diagnostic slice: `research diagnose` audits source overlap
and method/rate scores by format and declared compression quality. It is not
a trained context-aware detector; independent-source qualification is pending.
Actual model audits (`GENERALIZATION_RESULTS.md`) confirm no unseen validation
sources and missing declared quality/device metadata for both current models.

Current research slice: explicit low-payload training weights and cached-vector
comparison completed. Low-rate sensitivity gains trade off against higher
false alarms and regressions; see `SPATIAL_WEIGHTING_RESULTS.md`. No deployed
detector changes or newly qualified methods are implied by this experiment.

Synthetic corpus results are smoke/regression evidence only. They are not a
real-world accuracy claim. A cell is `supported` only after the gates in
`BENCHMARK_PROTOCOL.md` pass; otherwise it is `experimental` or `unsupported`.

## v0.6 — in progress

Source version: 0.6.0 (beta classification; not published). See
`STATUS.md` for verification evidence and remaining release gates.

- [x] Validated and checkpointed the existing 0.5.0 platform.
- [x] Provider-independent project memory and architecture/benchmark docs.
- [x] Shared `AnalysisContext` and expanded PNG/BMP native analysis.
- [x] Bounded CTF service with CLI and common portable reports.
- [x] CTF API jobs, SSE/cancellation, artifact download, and guided TUI entry.
- [x] Correct duplicate spatial evidence fusion and bound PDF stream decoding.
- [x] Prevent crash-dump/binary URL/accidental BMP candidate pollution; enforce
  depth boundaries and propagate native extraction budget cancellation.
- [ ] Full-image tool inventory, pinned versions/checksums/licenses, and E2E.

## v0.7

The frozen BOSSbase pilot (`PILOT_RESULTS.md`) found chance-level discrimination
for the tested Steghide/OpenStego families despite 30/30 controlled CTF recovery.
Next detector work needs separate development/validation data and an untouched
second source; tuning on this published baseline is not a new held-out result.

Implemented `research partition` with frozen-corpus exclusions and whole-source,
lineage/camera/device isolation. AI is excluded from primary scores and findings.
Provenance-bound feature extraction/train-input validation is implemented, using
a simple experimental spatial-summary baseline. See `DEVELOPMENT_DATA.md`;
new independent data, measured feature/model improvements and held-out inference
remain the next steps. No detector accuracy improvement is claimed yet.

Current priority order: correctness/limit regressions, independent development
and untouched-source evaluation, then blind CTF recovery/latency measurement.
No new format expansion is needed to establish those gates.

The ALASKA2 evaluation subset now has an explicit range-bounded acquisition
workflow and frozen selection (`ALASKA2_ACQUISITION.md`). Completion/access
evidence is tracked in `DATA_ACCESS_STATUS.md`; acquiring this corpus is not
completion of JPEG inference, training, cross-source validation or model gates.
The initial native-score evaluator and pre-scoring policy are specified in
`ALASKA2_PROTOCOL.md`; this measures existing behavior without tuning it.
The completed 4,000-file result (`ALASKA2_RESULTS.md`) fails the detection
targets: recall 0.4% / 0.1% / 0.1% and AUC 0.526 / 0.505 / 0.507 for
JMiPOD / JUNIWARD / UERD. Keep these inspected images out of development;
separate development/validation data and measured JPEG model work remain next.
The first development slice adds a separately acquired, reserved-lineage-safe
selection and opt-in 968-feature JPEG CPU baseline. Train-only normalization,
validation prediction and ONNX parity are tested; this research model is not
automatically deployed. See `JPEG_DEVELOPMENT_PROTOCOL.md` for the fixed run.
The completed run (`JPEG_DEVELOPMENT_RESULTS.md`) retained 3,156 train and
820 validation images after predeclared ambiguity quarantine. AUC is
0.649 / 0.592 / 0.585, with 41.46% FPR: inadequate for deployment and not a
cross-source or old-test improvement claim. Stronger detector research remains.

The 2026-10-01 Kodak external-source check (`KODAK_RESULTS.md`) is complete:
24 original lineages, chance-level detection, 30/30 controlled payload recovery
but 25/30 completed CTF jobs. Dataset-scale accuracy and blind CTF gates remain
open; new feature/model work still requires separate development data.

The subsequent `CTF_REGRESSION_20261002.md` records candidate-integrity and
OpenStego native-memory corrections: two final runs completed and recovered all
30 published challenges under unchanged limits. Failed intermediate runs remain
visible. This closes the observed cancellation regression, not the blind CTF or
automatic steganalysis gates; magic-only carving, graph deduplication, tool
isolation and independent detector development remain work.

JPEG segment/DQT and calibrated DCT analyzers, signed spatial/JPEG ONNX model
packs, and tool-specific JSteg/F5/OutGuess/Steghide recovery.

## v0.8

GIF, WAV, text, and generic-container depth; complete API/TUI CTF controls and
cross-format recursive recovery.

Pinned FSDD acquisition now supplies 3,000 real WAV candidate covers with
speaker/recording identity and audited PCM hashes. `DATASET_CATALOG.md` records usage
conditions; `BENCHMARK_RESULTS.md` tracks the separate format/method work and
keeps missing evaluations unavailable rather than successful.
The preregistered WAV baseline (`WAV_PROTOCOL.md`) adds independent
sequential/scattered PCM LSB pairs and fixed-threshold method/rate evaluation,
with speaker-level reservations and coverage-aware shared-service scoring.
The completed `WAV_RESULTS.md` baseline fails all six cells (AUC 0.50,
recall 0%). All generated payloads were independently oracle-verified;
native CTF recovery was not measured. Separate spatial cover acquisition is
available in `SPATIAL_DEVELOPMENT_ACQUISITION.md`; models/features and fresh
cross-source qualification remain necessary for measured improvement.

## v1.0 gate

`SPATIAL_PARITY_PROTOCOL.md` fixes the next low-payload feature comparison on
existing development rows. The completed `SPATIAL_PARITY_RESULTS.md` records
better ranking/FPR but low-rate recall/calibration regressions. Parity-conditioned
histograms remain experimental; cross-source and named upstream-method gates
are still necessary. No automatic deployment or global accuracy claim.

The separated-role operating-point comparison is frozen in
`OPERATING_POINT_PROTOCOL.md`; it cannot change the installed verdict threshold
or claim fresh blind evidence from previously inspected validation images.

Opt-in double-accumulation inference repair is specified in
`INFERENCE_PRECISION_PROTOCOL.md`; it preserves original weights and old
reports. Passing numerical gates cannot promote a method to supported.

The spatial comparison frozen in `SPATIAL_DEVELOPMENT_PROTOCOL.md` completed;
see `SPATIAL_DEVELOPMENT_RESULTS.md`. Custom co-occurrences improve same-source
controlled LSB AUC over the same-data reference, but low-rate recall, false
alarms, calibration and cross-source evidence remain open. Opt-in precise
inference now passes strict CPU ONNX parity for all three development models;
cover-only threshold fitting exposes a sensitivity/false-alarm tradeoff, not a
model improvement. See `INFERENCE_PRECISION_RESULTS.md`.
No model is automatically deployed; installed native results remain unchanged.

Release only when every advertised support cell passes the real-dataset gates,
the blind CTF suite meets recovery/latency targets, and all sandbox/limit tests
pass. Until then the package remains beta and claims are per-cell.
