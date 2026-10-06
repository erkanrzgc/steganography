# Train-only tiny learning control

Checkout-only frozen research control, not a general-purpose trainer or a
deployed detector. See immutable `SRNET_TINY_SANITY_PROTOCOL.md` (commit
`64584b0`) for data, selection, optimization and sanity objectives. The shared
service rejects additional config fields, including validation inputs and
optimizer/selection overrides. Missing dependencies/resource limits fail closed.

Explicit command from the repository root:

```sh
timeout --signal=TERM --kill-after=5s 1920s \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python -m steganography.research_srnet_sanity \
  --config .benchmark/srnet-tiny-sanity-20261007/config.json \
  --out .benchmark/srnet-tiny-sanity-20261007/result
```

Config has exactly `manifest`, `manifest_sha256`, `cache`, `cache_sha256`,
matching the frozen original training cache/manifest. Outputs must be fresh,
regular/non-symlink. This control verifies the complete train cache before
selecting metadata; it never opens validation cache/image bytes. Selection
retains original train-row order and every derivative of six selected declared
lineages (eight quality groups), not six independent sources.

The unchanged shared training engine creates a fresh seeded model, validates
all included/excluded rows and recomputes every balanced paired epoch. The
sanity service records 50 pair/batch schedules, all losses and exact 400 BN
updates. Final metrics use one image at a time in ordinary all-stage eval;
there is no live/batch-stat BN inference. Caller threads/RNG are preserved.
Batch-loss trajectories use balanced paired presentations; singleton metrics
use 24 distinct rows (eight covers, 16 stegos). Balanced accuracy accounts for
that class imbalance. Do not directly equate these differently weighted losses
or attribute a mode difference solely to BN without a separate matched control.

Read-only post-fit audit under a separate hard wall:

```sh
timeout --signal=TERM --kill-after=5s 300s \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  venv/bin/python scripts/audit-srnet-tiny-sanity.py \
  --config .benchmark/srnet-tiny-sanity-20261007/config.json \
  --result .benchmark/srnet-tiny-sanity-20261007/result \
  --out .benchmark/srnet-tiny-sanity-20261007/audit.json
```

This reconstructs all metadata/schedules, reloads numeric weights, verifies BN
counts and all singleton scores, then separately compares nine train-only
examples with independent NumPy forward math using the unchanged gates. It
also measures decoded pixel differences for all 16 selected cover/stego pairs.
Failed numerical gates retain audit evidence; neither an audit nor an in-sample
objective pass implies a detector qualification or accuracy improvement.

Exit 0 means the three in-sample sanity objectives passed, **not** detector
qualification. Exit 2 can mean completed-but-failed sanity objectives or
unavailable/incomplete execution; only a complete `sanity.json` with status
`completed` proves completion. Exceptions/timeout never produce usable partial
results; incomplete files may remain and must not be used or overwritten.
The entrypoint redacts exception text; no host paths or secrets in reports.
Linux resource bounds plus external timeout are not a filesystem/network sandbox.

Do not run over the existing real result. For a deliberate separately authorized
replay, use a different fresh output. Never install the numeric snapshot or
advertise the selected training scores as validation accuracy. A failed sanity
control motivates train-only investigation before expensive full-corpus fits.
