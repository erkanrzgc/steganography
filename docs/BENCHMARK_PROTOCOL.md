# Benchmark protocol

`COVERAGE_POLICY.md` defines minimum pipeline execution coverage. Missing required
components are inconclusive at low scores, never clean negatives. Completion
alone cannot qualify a method or turn the frozen failed baselines into passes.

The measured JPEG context development slice follows `JPEG_CONTEXT_PROTOCOL.md`:
two declared acquisition origins, matched method families, fixed objective and
measured content/quantization interactions. Previously used BOSS originals and
ALASKA validation remain development; no independent held-out claim is implied.
Completed failures and unchanged numerical tolerances are retained in
`JPEG_CONTEXT_RESULTS.md`, with all source/quality cells and paired intervals.

Cached-score source/context audits follow `GENERALIZATION_PROTOCOL.md`.
Pooled or same-source development metrics never count as independent-source
support; missing quality/device metadata remains explicitly unavailable.
Initial actual-artifact audits and legacy JPEG compatibility handling are
recorded in `GENERALIZATION_RESULTS.md` without modifying the frozen protocol.

The fixed train-objective comparison is preregistered in
`SPATIAL_WEIGHTING_PROTOCOL.md`: cached checksum-bound parity features, train-only
5% payload weighting, fixed threshold, all six cells and paired uncertainty.
Results are in `SPATIAL_WEIGHTING_RESULTS.md`. Already inspected validation
remains development evidence, never blind evidence.

The initial single-source experiment is specified in `PILOT_PROTOCOL.md` and
reported in `PILOT_RESULTS.md` / `benchmarks/pilot-20260918.json`. Its failed
detection baseline and controlled recovery results do not satisfy the gates below.

The subsequent small Kodak source check is frozen in `KODAK_PROTOCOL.md` and
`KODAK_RESULTS.md` / `benchmarks/kodak-20261001.json`. Both methods use the same
24 originals; bootstrap sampling retains all variants of a lineage together.
Report recovered payload counts separately from fully completed CTF jobs.
Published challenges may subsequently be rerun as regression evidence, with the
same manifest, expected payload hashes, passwords and limits. Preserve prior
reports and write each new run to an exclusive output directory. Record the
implementation revision, per-job status/error, artifact counts and output bytes.
Fixing published failures is not a fresh held-out or blind accuracy result; do
not tune thresholds or expand limits to turn a cancelled job into a pass.
The first such CTF regression is recorded in `CTF_REGRESSION_20261002.md` and
`benchmarks/ctf-regression-20261002.json`, including unsuccessful intermediate
runs and the full-image launcher configuration needed to reproduce the result.

## Catalog and provenance

`DATASET_CATALOG.md` inventories actual acquisition, usage conditions and
publication boundaries. `BENCHMARK_RESULTS.md` indexes results without combining
development scores, native tests and CTF recovery into one accuracy claim.
FSDD cover acquisition is not an audio benchmark: freeze speaker/recording
ancestry, splits, embedding methods/rates and thresholds before scoring; add
independently verified stegos and another source before claiming support.
Citation alone does not authorize raw corpus or trained-model redistribution.

The first controlled WAV evaluation is preregistered in `WAV_PROTOCOL.md`:
whole-speaker roles, marker-free sequential/scattered replacement, three payload
rates, independent scalar extraction verification, unchanged shared-service
balanced/70 scoring, required-native coverage and paired bootstrap intervals.
Generation ground truth uses known ordering; it does not measure native recovery.

Each sample records source URL/import origin, license, checksum, format,
cover-lineage, camera/device/app group, method family, payload rate, and lossy
quality factor where relevant. Permissively licensed data is downloaded only by
an explicit command; restricted corpora are imported locally and never placed
in the repository. Initial references are BOSSBase, ALASKA2, and StegoAppDB;
Aletheia is an independent parity/reference measurement.

The ALASKA2 holdout selection is specified in `ALASKA2_ACQUISITION.md`: select
complete cover/three-method groups with a fixed seed before examining scores,
retain all selected failures and never use this subset for training/calibration.
Acquisition provenance and JPEG/CRC/SHA checks are not stego extraction or
detection evidence. Unknown camera/device/scene metadata must remain unknown;
method folders cannot be counted as independent data sources.
The first native-score evaluation is preregistered in `ALASKA2_PROTOCOL.md`:
balanced/70, no training or threshold search, shared covers per method,
paired-lineage bootstrap, explicit missing coverage and a declared UERD
label-ambiguity sensitivity calculation. Score-based metrics are not proof of
a payload, calibrated probabilities or complete deployed-verdict accuracy.
The completed baseline is published in `ALASKA2_RESULTS.md` and
`benchmarks/alaska2-baseline-20261004.json`. All three methods fail the numeric
detection targets. Its inspected holdout must not be reused for tuning or
represented as a newly untouched evaluation after subsequent changes.

All derivatives of a cover stay in the same split. Grouping uses lineage and
camera/device/source metadata, not just file hashes, to prevent cover–stego
leakage.

Use `research partition` for subsequent experiments; see `DEVELOPMENT_DATA.md`.
Reserve published pilot identities, explicitly hold out whole test sources and
retain camera/device groups. Generated partition metadata is validated again
by benchmark/calibration manifest checks. Legacy manifests remain readable;
their acceptance alone is not evidence that training was cross-source isolated.

`research features` extracts train or validation only. The trainer requires a
matching manifest hash, feature-artifact hash and exact ordered train membership;
unbound NPZ input is rejected. Preserve the feature version, model/checkpoint,
seed and training provenance before validation/calibration and final evaluation.
Feature smoke tests do not establish detection accuracy or cross-source support.

Correlation regressions check that identical LSB-adjacency proxies cannot
increase aggregate suspicion through separate category names. PDF decoder
regressions cover exact byte boundaries, ASCII85 zero runs, malformed/truncated
streams and shared multi-stream budgets. A lower score after removing duplicate
evidence is an intended correction, not grounds for tuning thresholds on the
frozen pilot or relaxing the benchmark gates.

CTF candidate-integrity regressions use independent generated fixtures for
accidental binary signatures, byte-preserving percent decoding, successful vs.
partial tool output, crash dumps, symlinks/preexisting outputs, and depth/count/
byte boundaries. External process tests verify that child core dumps are disabled
without modifying the parent's resource limits. These are correctness/security
regressions, not evidence that OS sandbox escape or all decoder-bomb gates pass.
The Docker smoke check includes ten cold OpenStego extractions under the default
tool memory limit and a five-second deadline. Every process must complete and
recover exact bytes; a failed process's output and retries cannot count as passes.

## Support-cell gates

`JPEG_DEVELOPMENT_PROTOCOL.md` is a separate development experiment with a
new lineage-disjoint ALASKA2 selection. Its same-source validation metrics must
not be relabeled as final-test results or an improvement on the frozen baseline.
Development-only partition policy refuses test rows; source-label ambiguity
quarantines entire groups before features. Models remain unpublished/local
until license, calibration and independently held-out support gates are met.

A format/method/payload-rate cell needs at least 1,000 covers and 1,000 stegos
from at least two independent source groups. Cross-source held-out evaluation
must publish bootstrap 95% confidence intervals and meet all of:

- ROC-AUC >= 0.90
- balanced accuracy >= 0.85
- recall >= 0.80
- deployed false-positive rate <= 0.03
- expected calibration error <= 0.05

Failed cells remain `experimental` or `unsupported`; metrics are not collapsed
into a single global “accuracy” claim. Cloud AI output is excluded.

## CTF and performance gates

The fixed `SPATIAL_PARITY_PROTOCOL.md` comparison is iterative development on
the already inspected corpus, not an untouched test. New feature/model choices
must report every method/rate cell and false positives, with original reports
retained and no automatic deployment even if selected-cell scores improve.

`OPERATING_POINT_PROTOCOL.md` specifies an exploratory spatial threshold
development assessment, with cover-only fitting and separate lineage roles.
Already inspected validation remains inspected; lower empirical FPR cannot be
reported without its recall tradeoff or relabeled independent-source evidence.

The separate `INFERENCE_PRECISION_PROTOCOL.md` freezes numerical repair tests
on inspected development artifacts. Such export audits are not fresh blind
tests, threshold/calibration changes or cross-source detector qualification.

Spatial development uses the preregistered `SPATIAL_DEVELOPMENT_PROTOCOL.md`:
same-source validation only, reserved original lineage exclusions, independent
marker-free gray-plane pairs and fixed normalized/class-balanced baselines.
Actual validation-artifact ONNX parity is separate from small fixture parity.
Neither a good development score nor perfect generator-oracle recovery passes
the cross-source detector or blind CTF gates.

The blind suite contains 120 supported challenges. At least 90% require exact
payload/flag recovery; remaining challenges must place the correct method in
the first three recommendations. For typical inputs below 10 MiB, publish the
reference hardware and require median <= 60 seconds and p95 <= 180 seconds.

PR CI uses independent golden fixtures, synthetic regression, unit/integration,
TUI Pilot, schema/SARIF, fuzz/property tests, and Python 3.11–3.14. Scheduled and
release jobs use real dataset caches; absent data is `skipped`/`unavailable`.
Full Docker E2E verifies exact tool versions/licenses, non-root/read-only mode,
timeout/cancel, exact recovery, provenance, and archive/symlink/overwrite limits.
