# SRNet paired-schedule audit — 2026-10-06

All three preregistered scopes and ten epochs each pass independent accounting
on the unchanged real development training split. Every original cover/stego
pair appears, labels and source/lineage/quality match, exact hierarchical
counts and ordered SHA-256 agree, and replay is deterministic. None of the
765 validation hashes appears. **No model was fitted and no accuracy measured.**

[Frozen protocol](SRNET_SAMPLING_PROTOCOL.md), implementation `d2e3132`,
[portable evidence](../benchmarks/srnet-sampling-20261006.json).

| Scope | Original train rows | Pairs per epoch | Input rows per epoch |
|---|---:|---:|---:|
| Both declared origins | 2,985 | 3,160 | 6,320 |
| ALASKA only | 2,367 | 1,578 | 3,156 |
| BOSS simulations only | 618 | 412 | 824 |

In the combined scope each source contributes 1,580 pairs. ALASKA's two
unknown-quality method cells contribute 790 apiece (789 original pairs each).
BOSS's four Q75/Q95 × JUNIWARD/UERD cells contribute 395 apiece (103 original
pairs each). Repeated covers/positives are deliberate oversampling, not new
independent data, scenes or evidence. Single-source scopes need no oversampling.
All ten epoch hashes and exact cell counts for each scope are published.

The complete train float-cache byte/SHA contract was rechecked for every scope,
including excluded sources. No validation cache or scores were loaded. Original
JPEG hashes were verified by the earlier float preparation; this audit uses
those bound caches and does not reparse all originals. Unknown quality remains
unknown, rather than being inferred from class labels. Primary analyzers,
model catalog and deployed inference are unchanged.

This is schedule integrity, not an independent forward computation or unseen
camera/device qualification. The accounting oracle independently regroups
metadata, recomputes counts and hashes; generation/replay uses the actual
sampler, not a second implementation of its PRNG. Native decoder proof,
provenance-bound trainer/card, fixed optimizer/compute schedule, numerical
forward/export parity and real fitting are separate gates still pending.
Previously inspected validation is not blind, and BOSS positives are simulated.

Verification: 995 tests pass on Python 3.11.14, 94.55% total coverage;
sampler/plan service 116/116 statements covered. Lint, type checking and
whitespace checks pass; inspected wheel/sdist remain model/data-free. A fresh
checkout-bootstrapped audit replay is byte-identical to the initial audit.
Other Python versions and a fresh full Docker run remain unverified here.

Replay with fresh output; the auditor explicitly loads the checkout rather
than a potentially stale installed package:

```sh
venv/bin/python scripts/audit-srnet-sampling.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --cache .benchmark/srnet-float-preparation-20261006/train/cache.json \
  --out .benchmark/srnet-sampling-replay
```

The shared CLI prepares a schedule from a bounded JSON config:

```sh
steganography research srnet-plan --config CONFIG.json --out FRESH_PLAN.json
```

Required keys: `manifest`, `manifest_sha256`, train `cache`, `cache_sha256`.
Optional: `epochs`, `seed`, opaque `training_source_id`. All other keys are
rejected, including validation inputs. This command does not start training.
