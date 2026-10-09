# Roadmap and status

Explicit isolated epoch jobs implemented with full source/protocol/schedule
binding, immutable parent cards, exact numeric continuation and before/after
train cache verification. Generated disk-resume parity and adversarial job
tests pass locally. Full regression and physical CUDA resume gates are pending
for this slice. The fixed five-epoch ALASKA+BOSS GPU pilot is preregistered in
`JPEG_EPOCH_LEARNING_PROTOCOL.md`; no real pilot result or accuracy gain yet.

Numeric whole-epoch checkpoint core implemented: optional shared-engine/block
fit segments preserve model/BN, Adamax slots/steps and Torch RNG, with full
schedule/settings/backend binding and strict non-pickle archive bounds. See
`SRNET_EPOCH_CHECKPOINTS.md`. All 54 generated exact-resume/adversarial tests
pass; full Kali regression: 1,719 passed, one live-CUDA skip, coverage 95.53%.
The three changed/new core files have 100% line coverage; lint/types/builds
pass. Isolated per-epoch CLI/controller, physical CUDA
resume parity and preregistered real-learning evaluation remain pending;
no new real fitting, changed per-job caps or accuracy gain.

On-device full numeric-state checks are implemented for explicit CUDA, with
unchanged original CPU checks and final NumPy verification after conversion.
Twenty generated acceptance/update/device/immutability tests pass. Full Kali
regression: 1663 pass, one live-CUDA skip, coverage 95.48%; changed training/model
files 100% line coverage. Lint/types/builds pass. Separate physical timing under
`CUDA_STATE_TIMING_PROTOCOL.md` completed on WSL; after user-restored SSH,
the original report was recovered on 2026-10-10 without rerunning. All 16
source hashes verify. p95 0.091553s, five-epoch estimate 3130.26s, still
ineligible against 1800s. See `CUDA_STATE_TIMING_RESULTS.md`; prior retrieval
incident preserved in `CUDA_STATE_VERIFICATION.md`. No controlled speedup,
new real fitting or accuracy gain. Next investigate remaining execution costs
or separately design bounded resumable training; preserve all current guards.

CUDA gradient-check consolidation is implemented: every gradient element is
still checked, with one aggregate host boolean rather than one per parameter.
CPU defaults and full per-update state validation remain unchanged. Fourteen
generated equivalence/fault/read-count tests pass. Full Kali regression:
1643 pass, one live-CUDA skip, coverage 95.47%; lint/types/builds pass.
Separate physical timing completed under `CUDA_GRADIENT_TIMING_PROTOCOL.md`:
p95 0.095847s, conservative five-epoch estimate 3271.44s, still ineligible
against 1800s. See `CUDA_GRADIENT_TIMING_RESULTS.md`. The descriptive reduction
is not a controlled speedup claim; no real fit or accuracy gain. Next isolate
state-transfer/check overhead while preserving all numerical/resource guards.

Physical CUDA timing completed under the frozen `CUDA_THROUGHPUT_PROTOCOL.md`:
64 generated updates, steady-interval p95 0.112541s, fixed five-epoch conservative
estimate 3820.35s >1800s. The prospective fit is ineligible; no real fit started,
epochs/safety factor/caps changed or improved accuracy claimed. See
`CUDA_THROUGHPUT_RESULTS.md`. Next investigate verified shared-engine execution
cost without weakening numerical/safety checks; adequate exposure and
stored-normalization learning remain separate gates. Any changed execution
requires a separately frozen timing protocol, not overwriting this result.

2026-10-09: the first physical WSL2/RTX 5060 Laptop generated CUDA preflight
completed. The original report was fetched over authenticated, pinned-host
local SSH and all 14 source hashes match this checkout. One generated update,
same-weight CPU/GPU forward agreement, bounded RAM/allocator; no real learning
or accuracy gain. A fresh local copy passed the recursive transport checksum
comparison; native train-only readiness now completes under the separately
frozen `WSL_TRANSFER_READINESS_PROTOCOL.md`: all 7,380 train rows, exact
historical tensor/schedule hashes, no native validation pixel reads or optimizer
execution. See `WSL_GPU_READINESS_RESULTS.md`. Next freeze adequate real GPU
exposure plus a stored-normalization learning gate, not validation-based tuning.

The following 2026-10-08 software-only status is historical; its explicit
Kali unavailable result is preserved, not relabeled as a GPU pass.

Opt-in FP32 CUDA streaming path and generated hardware preflight implemented:
`WINDOWS_GPU_TRAINING.md`. Reported Windows 11/RTX 5060 Laptop hardware is outside
the current Kali/VMware CPU-only environment. CPU/emulated policy/control-flow
tests pass; actual CUDA/WSL execution remains unavailable/unverified, not an
accuracy improvement. Next observe Windows driver/WSL, run bounded generated
hardware preflight, then freeze adequate real train-only exposure/learning.
Final verification: `CUDA_BACKEND_VERIFICATION.md`, 1,597 tests pass, actual
CUDA test explicitly skipped; the isolated Kali probe records unavailable.
CPU behavior/defaults and the 1800s ceiling remain unchanged; CUDA requires a
kernel resident RAM bound rather than an unsuitable CPU virtual-address cap.

Versioned streaming training infrastructure and full real train-only I/O
readiness completed: `SRNET_STREAM_READINESS_RESULTS.md`. All 7,380 train rows
read without whole-corpus tensor allocation; exact generated legacy/stream
model/loss/BN equivalence, hard isolated jobs, propagated deadlines and
start-of-job source snapshots verified. No new real fitting or detection score.
Next preregister adequate train-only exposure/learning, resolve compute/GPU
choice and implement/verify that bounded workflow before blind evaluation.
That historical readiness run is CPU-only; the separate generated physical
CUDA preflight is now completed as linked above. Real fitting is still pending;
no silent 1800s cap extension or cloud job.
Historical preparation-worker deadline/source-snapshot hardening remains pending.

Full expansion preparation and independent real audit completed:
`JPEG_SCALE_PREPARATION_RESULTS.md`, 16 whole-lineage blocks, 8,991 JPEG rows
(7,380 train / 1,611 validation), unrounded-Y caches, all 4,000 BOSS simulated
DCT change checks and nine scalar-IDCT contexts pass. No new model trained.
Versioned streaming implementation/readiness now follows as linked above;
adequate-exposure learning/GPU fitting still awaits its separate protocol.
Whole-array 4,000-row caps and prior failed scores remain.

2026-10-08 data scaling replaces further six-scene diagnostic repetitions:
an additional 4,000 ALASKA2 JPEGs and 1,000 BOSS originals are independently
byte/split/exclusion audited. Three unchanged ALASKA lineages are quarantined;
new eligible training originals: 816 ALASKA + 822 BOSS, with 359 validation
originals separate. The JPEG-only WIFD attempt remains unavailable; a separately
frozen retry completed: 200 WIFD files (120 JPEG, 80 primary-decoded MPO),
independently audited alongside all 5,000 development files. This adds no MPO
detector support; see `DATA_EXPANSION_RESULTS.md`. Whole WIFD origin
is reserved from training/calibration. Larger preparation/streaming and a
preregistered adequate-exposure fit are next; CPU-only host, GPU access pending.
These acquired files have not entered the existing train cache or detector.

Step/exposure-matched accumulation completed (`SRNET_ACCUMULATION_RESULTS.md`):
same 80 updates/640 presentations/exact group order, four-row microbatches;
both arms fail complete learning goals, original train BA .50. Amplified own
BA worsens (.6875 to .59375), all 18 numerical oracles pass, no deployment.
Next freeze training-only data/exposure scaling, not more unsupported accuracy
claims from the six-scene sanity subset. `JPEG_TRAINING_DATA_SCOPE.md` records
the full prepared train scope: 2,985 rows / 892 declared scene lineages.

Eight-row training-context control completed (`SRNET_WIDE_BATCH_RESULTS.md`):
both exposure-matched arms fail complete learning goals; singleton loss falls,
but original train BA stays .50 and own BA does not improve (.50 / .6875).
All 18 independent numeric oracles pass; no model deployed. Next freeze
four-row gradient accumulation with the exact wide-group order, matching
both optimizer updates and row presentations to separate context confounds.

BN context/refresh control completed (`SRNET_BN_REFRESH_RESULTS.md`): both
fixed clone-refresh goals fail; original training BA remains .50. Amplified
batch-stat diagnostics differ sharply from singleton eval but are not deployable.
All 18 numeric oracles and serialized parameter identity checks pass. Next
freeze larger/diverse training-batch context with explicit exposure accounting.

Signal-strength control completed (`SRNET_SIGNAL_STRENGTH_RESULTS.md`): two
fixed 20-epoch/160-update arms; amplification lowers batch loss but fails the
stored-BN own-input learning goal, original-input train BA stays .50. All 18
numeric oracles pass; neither model is deployed. Next preregister train-only
batch-context/stored-normalization controls, not validation-based tuning.

Train-only gradient/null controls completed (`SRNET_GRADIENT_RESULTS.md`):
all six cells covered in nine probes, classifier directional checks 9/9 pass,
complete replay matches. Real early-layer gradients are nonzero but smaller in
the failed tiny model. No unique cause or accuracy improvement established;
next freeze signal-strength/early-layer learning controls before longer fitting.

Generated positive learning control completed (`SRNET_POSITIVE_CONTROL_RESULTS.md`):
20 epochs/160 updates, all three in-sample objectives and six NumPy oracles pass.
The unchanged trainer can learn an obvious synthetic signal; real-data BA .50
and failed detector gates remain unchanged. Next preregister train-only gradient/
input and signal-strength diagnostics; no deployed model or running fit.

Train-only tiny sanity completed (`SRNET_TINY_SANITY_RESULTS.md`): 24 rows,
50 epochs/400 updates; two of three sanity objectives fail, stored-BN train
balanced accuracy .50. Saved-model/NumPy replay passes and all input pairs differ.
Next freeze strong-signal generated learning/gradient diagnostics before longer
real fits. No accuracy improvement, deployed model or running training job.

`SRNET_TINY_SANITY_PROTOCOL.md` freezes the next train-only control before
selection/fitting: 24 rows, 50 epochs/400 updates, metadata-only selection and
stored-BN singleton evaluation. In-sample learning is not accuracy qualification.

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
