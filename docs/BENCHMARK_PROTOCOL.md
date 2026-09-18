# Benchmark protocol

The initial single-source experiment is specified in `PILOT_PROTOCOL.md` and
reported in `PILOT_RESULTS.md` / `benchmarks/pilot-20260918.json`. Its failed
detection baseline and controlled recovery results do not satisfy the gates below.

## Catalog and provenance

Each sample records source URL/import origin, license, checksum, format,
cover-lineage, camera/device/app group, method family, payload rate, and lossy
quality factor where relevant. Permissively licensed data is downloaded only by
an explicit command; restricted corpora are imported locally and never placed
in the repository. Initial references are BOSSBase, ALASKA2, and StegoAppDB;
Aletheia is an independent parity/reference measurement.

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
