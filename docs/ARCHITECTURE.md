# Architecture

`core/srnet_multibatch` groups the unchanged deterministic paired schedule into
explicit two-source/four-row batches. Plan/card v2 bind exact grouping hashes
and optimizer-update counts; fit and complete eval reconstruct this contract.
Absent `batch_recipe` preserves v1 two-row behavior. No inference BN-mode change
or detector integration. See `SRNET_MULTIPAIR_RESULTS.md` for real accounting,
which is not real fitting or accuracy evidence.

`core/srnet_diagnostics` owns bounded read-only BN-mode contrasts and restores
numeric state/flags/hooks on failure. `research_srnet_diagnose` binds training
cache/scope/card/plan, selects metadata-only train probes and restores threads;
`research srnet-diagnose` is a thin CLI adapter. This batch-dependent diagnostic
cannot feed primary verdicts or change saved models. Portable aggregation lives
in the fixed train-diagnostic audit script, not in UI/API detector code.

`research_srnet_results` verifies prediction/card/plan/cache provenance,
complete validation identities and fixed independent numerical audit fields
before calculating same-source/quality/method diagnostics. It reuses paired
lineage intervals, never searches thresholds or promotes small/reused cells to
qualified support. `scripts/audit-srnet-real-pilot.py` applies frozen pilot
anchors and publishes aggregate evidence, without raw data or model artifacts.

`research_srnet_evaluate` shares `bound_training` and `card_contract` with fit,
verifies both complete caches/card/model/BN counts and performs complete native
validation plus metadata-selected independent NumPy replay. CLI is a thin
adapter; numerical failures retain audits without usable predictions. These
research outputs do not affect primary analyzers, calibration or deployment.

`research_srnet_job` executes fitting in a fixed hard-limit child, checks config
identity, bounded stdout and complete numeric model/card hashes. `run_fit`/CLI
use it; direct fitting retains cooperative checks. Missing limits are unavailable.
`core/srnet_reference` independently evaluates float64 NumPy math from weights.
`srnet_onnx` exports/replays bounded optional CPU graphs; `srnet_onnx_guard`
rejects unsafe topology/weights/operators and propagates bounded activation shapes.
Generated numeric readiness is separate from trained-model/accuracy qualification.

`core/srnet_training` owns optional CPU matched-pair learning and finite-state/
cooperative deadline guards. Shared `research_srnet_fit` verifies the complete
plan via `research_srnet_plan.schedule_record` before any update, then writes
numeric state and provenance card. `research srnet-fit` is a thin adapter.
No validation inputs or primary analyzer changes; hard-limit jobs and independent
numerical forward/export now have generated readiness, not real accuracy evidence.

`core/srnet_sampling` creates deterministic train-only matched JPEG pairs with
hierarchical source/quality/method balance and bounded oversampling. The shared
`research_srnet_plan` service binds manifest, complete float-cache integrity,
scope and per-epoch order hashes; `research srnet-plan` is a thin CLI adapter.
No validation cache, optimizer, trained model or detector behavior is involved.

`core/srnet_model` stores an exact 183-array numeric state, including float BN
running means/variances and scalar int64 counters. Fixed shape/header/expanded
size checks precede NumPy allocation; no pickle. All-stage eval snapshots,
32MiB limit, checksum, symlink/overwrite protection and CPU RNG-safe reload.
This is not a model card, installed model or training qualification.

`core/jpeg_float256` provides isolated component-Y float64 IDCT to unrounded
phase-aligned float32/256 crops. Shared pixel cache service explicitly selects
the new schema/1GiB research limit; legacy uint8/64MiB defaults are unchanged.
SRNet float inference preserves fractions and requires all-stage eval.

`core/srnet.py` is a separate optional SRNet-style structural research block,
not an extension of the tiny CNN's nine-array model contract. All-stage eval
is mandatory for bounded inference to prohibit batch-dependent BN predictions.
`research_srnet` runs explicit generated readiness in a resource-limited fixed
worker; no datasets, weights, model installation or detector changes.

`scripts/audit-pixel-cnn.py` replays the explicit local research artifacts with
an explicit architecture-bound residual domain. Two allowlisted pixel stems
share exact learned layer shapes; fixed numeric filters prevent cross-domain
model loading. Legacy defaults/weights remain unchanged.
The audit replays independent NumPy arithmetic/metrics and ONNX CPU batches, separate from detection
gates. Both complete trials are published without deploying the failed models.

`research_pixel_cnn` adds explicit CPU minibatch train/predict stages, using the
unchanged pixel cache/architecture and train-only source/class weighting. Numeric
`jpeg_cnn_model` persistence preflights nine fixed float32 array headers before
allocation; no pickle/symlink/overwrite. Scope/settings bind before inference,
thread/RNG state restore after training. No API trainer or installed model change.

Explicit pixel research owns a separate bounded uint8 JPEG cache and versioned
center-crop/decoder contract (`research_pixels`, `core/jpeg_pixels`). It does not
weaken generic feature/model limits or change analyzer inputs. `core/jpeg_cnn`
provides an optional custom residual network and bounded in-memory inference;
no model loader, installed detector or full-image coverage claim. CLI delegates
cache preparation; no independent extraction logic in presentation layers.

JRM research has optional train-source selection with additive scope provenance:
opaque included/excluded IDs, row count and ordered identity hash. Validate scope
against train-only manifest rows before model loading; legacy cards remain valid.
Shared `research_jrm_transfer` runs two fixed single-source fits on complete bound
caches. CLI dispatch only; inspected validation is diagnostic, not blind evidence.

`core/jpeg_jrm.py` isolates optional pinned upstream JRM extraction with framed
bounded workers. `core/fld_reference.py` owns immutable numeric-only FLD
inference; `research_jrm` binds explicit raw caches, training/card provenance
and validation. The CLI is a thin adapter. No upstream pickle/code/model is
bundled; generic feature/checkpoint limits and deployed detection are unchanged.
See `JRM_REFERENCE_PROTOCOL.md` and upstream component usage restrictions.
Numeric NPZ loading checks exact NPY shapes, dtypes, header versions and payload
lengths before allocation; a small archive cannot advertise an enormous array.

`core/feature_model.py` reconstructs fixed bounded linear and opt-in residual
MLP64 feature networks. Research contracts bind architecture to domain across
training, prediction, export and diagnostics; unknown combinations fail closed.
Legacy checkpoints default to linear. This is not raw-residual CNN inference;
see `JPEG_NONLINEAR_PROTOCOL.md`.

`core/jpeg_residual.py` adds a versioned research-only block-DCT residual/parity
descriptor, retaining the existing measured context prefix. It shares the fixed
allowlisted native worker and original resource limits; old feature contracts
are unchanged. Two-origin weighting remains mandatory for the new contract.
See `JPEG_RESIDUAL_PROTOCOL.md`; this is not DCTR/JRM or deployed model inference.

`core/coverage.py` defines versioned minimum native execution requirements by
content-detected format. The shared pipeline preserves positive evidence but
cannot emit a low-score `no_indicators` result when required coverage is missing.
JSON/HTML/SARIF carry the same assessment; see `COVERAGE_POLICY.md`. Execution
completion is not a method-support or accuracy qualification.

`core/jpeg_context.py` defines a versioned 1,098-feature measured DCT/content/
quantization interaction contract, not a source-ID feature or deployed model.
Local-only `research_jpeg_corpus` uses bounded optional upstream simulation
workers; `research_jpeg_multisource` binds cached vectors and two declared
acquisition origins with matched method families. Train-only source/class
weighting rejects missing provenance or method-source shortcuts. Legacy feature
contracts and primary analysis remain unchanged; see `JPEG_CONTEXT_PROTOCOL.md`.

`research_generalization.py` is the shared cached-score domain/context
diagnostic service; the `research diagnose` CLI only passes declared inputs.
It binds training/validation contracts, reports source overlap and stratifies
scores without changing them. Source/identity metadata is not a model feature.

Research-only low-payload weighting is an explicit versioned training recipe
in `research_weighting.py`; it accepts complete controlled training lineages,
not validation-derived weights. Weighted class masses and reduction semantics
are persisted in checkpoint provenance. `research_weighted_comparison.py`
reuses checksum-bound parity vectors; deployed shared analysis is unchanged.

`core/spatial_parity.py` extends the shared spatial histogram descriptor with
joint center-bit/neighbor-residual statistics. Its new version declares 1e-8
feature quantization; old feature versions are unchanged. The explicit
`research_parity` service binds an audited development corpus and previous
predictions for same-row comparison; CLI is only an adapter to shared research.

`research_operating_point` is an explicit research service, not deployed
threshold logic. It uses manifest-bound validation features and independent
whole-lineage fitting/assessment roles. Fractional research cutoffs compare
unrounded probabilities directly; CLI/primary integer thresholds are unchanged.

Feature checkpoint inference defaults to legacy float32. Explicit research
derivatives may declare `inference_arithmetic: float64` for normalization and
accumulation while preserving float32 input/output; no automatic conversion
or installation occurs. Shared research loading bounds modern ZIP checkpoint
bytes, member count, expanded bytes and feature dimensions before reconstruction.

Spatial development shares manifest-bound feature extraction/training/inference
with JPEG. `core/spatial_cooccurrence.py` defines a custom experimental 468-bin
descriptor (not SRM/SPAM); `steganography/research_spatial.py` orchestrates explicit,
bounded local research only. It cannot install a model or change primary verdicts.

The project is a local-first steganalysis and CTF workbench. Its interfaces are
thin adapters around application services:

```text
CLI / API / TUI
       |
       +-- AnalysisService -> Analyzer and Carrier plug-ins
       +-- CTFService ------> AnalysisContext, ToolRunner, decoder graph
       +-- Case/Scan/Studio -> encrypted Vault and SQLite audit chain
       |
       +-- JSON v2 / HTML / SARIF / evidence bundle
```

`AnalysisContext` lazily caches file identity, bytes, decoded image arrays,
entropy, JPEG metadata, and PCM WAV samples for one input. Context-aware native
components share it; legacy and third-party analyzers retain the compatible
`analyze(Path)` adapter.

`CTFService` owns the bounded playbook. Artifacts form a parent-child graph and
carry SHA-256 plus extraction provenance. `ToolRunner` invokes optional upstream
programs without a shell, with time/resource/output limits and redacted report
commands. An unavailable tool is coverage metadata and can make a result
inconclusive; it is not negative evidence.

External extraction adoption is invocation-specific: only the explicitly named
output of a zero-exit, completed extractor is eligible. Unnamed files (including
logs and crash dumps), failed/partial outputs, symlinks, empty outputs and paths
that existed before invocation are excluded. POSIX children have `RLIMIT_CORE=0`.
This is evidence hygiene, not per-tool filesystem isolation; rejected incidental
files can remain in the job directory but are not included in artifact bundles.
The full-image OpenStego launcher fixes allocator arenas and JVM active processors
at two, with the existing 128 MiB Java heap, rather than enlarging `ToolRunner`'s
768 MiB address-space ceiling. Other tools and the base wheel are unchanged.

The depth limit is checked before any child-producing stage. Boundary artifacts
still receive analysis; native extraction budget exceptions propagate to an
inconclusive, cancelled job. URL decoding requires a printable ASCII envelope
and a valid percent escape, and preserves decoded bytes exactly. BMP signature
carving checks header/size/dimension/plane consistency, considers at most 64
header hits, and retains the declared file extent. Other signatures remain
heuristic candidates. Carved files retain format suffixes for native analysis;
neither a filename extension nor a carving signature confirms steganography.

Managed job artifacts are stored beneath a newly created output directory. Artifacts are
regular files, names are generated rather than trusted, archive members are
streamed with count/depth/size/ratio limits, and existing files are not
overwritten. Extracted content is analyzed, never executed.

The tool working directory and resource limits are not an operating-system
filesystem sandbox. Native parsers run in-process and the job deadline is
cooperative between stages. Use a non-root, read-only container with only the
job's writable mount for hostile inputs; stronger per-tool isolation and hard
native-stage deadlines remain release work.

JSON v2 remains additive through `schema_revision`. HTML, SARIF, and bundles are
views of the same normalized report. Host paths and secret-bearing arguments
are redacted before serialization.

`steganography.benchmarking.pilot` evaluates the same analysis and CTF services;
it does not implement a second detector. Independent upstream tools generate
cover/stego pairs, with hashes and lineage recorded before evaluation. Solver
job directories exclude expected payloads. Aggregate evidence is versioned,
while downloaded images and full local runs remain in ignored `.benchmark/`.
Each CTF job also retains its portable report; aggregates include failure reasons,
artifact counts and generated-output bytes (excluding the copied input).
Its opt-in `--both-methods` generation mode uses both upstream methods per cover
without splitting or multiplying original lineages. Explicit Kodak acquisition
records provenance and rejects exact hash overlap with a reserved manifest.

`scripts/fetch-alaska2-pilot.py` is an explicit, standalone research acquisition
utility, not an analyzer or automatic application download. It samples complete
original/stego groups before scoring and obtains only bounded ZIP/ZIP64 ranges
from an authenticated, fixed-origin archive. Private credentials are never sent
to the storage redirect target. CRC/JPEG checks, local SHA-256 re-reading,
exclusive outputs and provenance-bound resume precede a success manifest.
See `ALASKA2_ACQUISITION.md`; acquisition does not establish detector accuracy.

`steganography.benchmarking.alaska2` evaluates those original JPEGs through the
unchanged `AnalysisService`. It validates frozen manifest/selection hashes and
four-way lineages, persists portable per-file JSONL, and reuses the covers in
each method's paired summary without multiplying independent observations.
Required-native failures invalidate cell metrics rather than becoming negatives;
optional coverage remains explicit. Fixed-threshold metrics disable threshold
search, and uncertainty resamples whole lineages. See `ALASKA2_PROTOCOL.md`.

`core.dataset` defines shared identity and connected-component rules. Research
partitioning uses these to keep lineage/camera/device groups together, reserve
whole test sources and reject overlap with frozen manifests. Partition policy
is rechecked by research manifest validation. AI triage stays in raw analysis
results but cannot contribute to aggregate scores or primary pipeline findings.

`core.features` owns the bounded `spatial-summary-v1` preprocessing contract.
`steganography.research_features` binds feature rows to verified partition
manifests and validates complete train-only inputs for the research trainer.
The CLI delegates extraction to that shared workflow. Artifacts and checkpoint
provenance use hashes, not absolute dataset paths. No feature/model downloads or
automatic deployment occur.

The legacy `sample_pair_balance` and `rs_regular_singular_balance` image signals
are two names for the same adjacent-LSB parity observation, not implementations
of full SPA/RS algorithms. Both use `image_lsb_adjacency`, so service and hybrid
ensemble aggregation count them once. Scores remain uncalibrated heuristics.

The opt-in `jpeg-dct-summary-v1` research contract adds 968 luminance DCT
histogram/co-occurrence/quantization features. Native parsing runs in a
15-second, resource-limited worker with bounded input/output, not in the CLI.
Train/validation artifacts use the same manifest binding as spatial features.
`core.feature_model` preserves train-only normalization as an explicit operation
in PyTorch/ONNX; folding near-constant features into weights caused numerical
cancellation and is deliberately avoided. This is a CPU linear research
baseline, never automatically used by the primary analysis service.

Development-only manifests have a distinct policy and cannot contain test
samples or claim whole-source holdout. The explicit ALASKA2 development importer
checks acquisition purpose, original membership, reserved identities and split
assignment, quarantining entire contradictory-label lineages before features.
The old test corpus is never converted into training data.

`scripts/fetch-fsdd-pilot.py` is another explicit research-only acquisition
entrypoint: pinned public archive, no redirects, bounded ZIP central directory,
compressed/expanded bytes and PCM frames, verified source license evidence,
exclusive outputs and per-recording ancestry/speaker metadata. It downloads
only candidate WAV covers, never models or stegos; upstream Python files are
not extracted or executed. It assigns no splits or accuracy/support status.
Public aggregate evidence and license/source links live under `benchmarks/`
and `docs/`; raw data remains outside Git. See `DATASET_CATALOG.md`.

`steganography.benchmarking.wav` prepares marker-free research LSB replacements
with exact scalar-oracle recovery, retaining original RIFF bytes outside PCM.
Whole-speaker roles precede generation; only frozen test speakers are scored
through `AnalysisService`. Native coverage failures invalidate method/rate cells.
Shared bounded-file/fingerprint utilities are reused from the earlier benchmark;
the application has no separate detection or extraction implementation here.
The generator/oracle are experiment ground truth, not a native CTF recovery claim.
See `WAV_PROTOCOL.md`; artifacts and their provenance remain bounded and local.

The separate BOSSbase development downloader reuses the tested standalone
range/catalog/member helpers with an explicit anonymous fixed-origin validator.
No credentials or Kaggle requests occur. Reserved archive names are excluded
before random selection, content/lineage hashes checked on download, and
resume bound to selection/archive/reserved provenance. See
`SPATIAL_DEVELOPMENT_ACQUISITION.md`; PGM originals are development covers,
not a scored detector corpus or a new independent source.

PDF stream decoding enforces a per-stream 16 MiB output limit and a shared
32 MiB decoded-output budget per parse. CTF extraction further restricts that
budget to the remaining job output allowance. ASCII decoder input is bounded
before whitespace normalization; ASCII85 zero-run expansion is preflighted.
Truncated/invalid streams remain raw candidates with `decode_status=unavailable`,
never successful decodes. These limits do not make the regex-based PDF parser a
complete PDF implementation or provide hard CPU deadlines/native OS isolation.
