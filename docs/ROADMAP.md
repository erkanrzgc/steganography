# Roadmap and status

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
existing development rows. Parity-conditioned histograms are experimental;
cross-source and named upstream-method gates remain necessary.

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
