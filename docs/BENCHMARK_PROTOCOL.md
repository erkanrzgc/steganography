# Benchmark protocol

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

The blind suite contains 120 supported challenges. At least 90% require exact
payload/flag recovery; remaining challenges must place the correct method in
the first three recommendations. For typical inputs below 10 MiB, publish the
reference hardware and require median <= 60 seconds and p95 <= 180 seconds.

PR CI uses independent golden fixtures, synthetic regression, unit/integration,
TUI Pilot, schema/SARIF, fuzz/property tests, and Python 3.11–3.14. Scheduled and
release jobs use real dataset caches; absent data is `skipped`/`unavailable`.
Full Docker E2E verifies exact tool versions/licenses, non-root/read-only mode,
timeout/cancel, exact recovery, provenance, and archive/symlink/overwrite limits.
