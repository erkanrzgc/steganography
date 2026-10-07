# Eight-row context learning control — 2026-10-07

**Completed, not detector-qualified. Both arms fail the complete learning
objectives; original-input training balanced accuracy remains .50. No model
is deployed or validation pixels loaded.**

Frozen protocol `SRNET_WIDE_BATCH_PROTOCOL.md`, commit `c78a9eb`, SHA-256
`f4aede48cb279a2745c0979b20615f07125bfb8193a0047da00f023d95b7fe9d`.
Executed implementation commit `b89a90a`; report records all execution source
hashes. Historical baseline `srnet-signal-strength-20261007.json`, SHA-256
`a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf`.
No result-selected rerun, settings, checkpoint, clipping or BN repair.

Keep the same 24 metadata-selected training rows (eight cover, 16 stego;
six original scenes/eight quality groups), preprocessing, sampler identities,
tensor hashes, fresh seed 20261012 and Adamax settings. Each eight-row batch
contains four matched pairs, two per declared source, with four distinct
(source, lineage) keys; no rows dropped/padded. Bounded first-complete matching
reorders unchanged four-row groups. All 20 real schedules pass accounting.
Source/camera/perceptual independence is still not proven. Factor 32 remains
artificial amplification, not actual embedding or payload-rate evidence.

Twenty epochs per arm: **80 updates, 640 presented rows**, versus the historical
four-row baseline's **160 updates, 640 rows**. Row exposure is matched;
optimizer steps and batch order are not. Each stored model independently
reloads with exactly 26 BN counters of 80. Both old source models' checksums
remain unchanged. No weights or tensors are published.

Ordinary stored-BN singleton inference on **24 unique training rows**:

| Measurement | Factor 1 | Factor 32, artificial |
| --- | ---: | ---: |
| First epoch mean batch CE | 1.048317 | 1.174654 |
| Last epoch mean batch CE | .693116 | .257156 |
| Relative batch loss reduction | 33.88% | 78.11% |
| Own-input singleton CE | .694339 | .464312 |
| Own-input singleton BA | .50 | .6875 |
| Own-input recall / FPR | .25 / .25 | .75 / .375 |
| Original-input singleton CE | .694339 | 2.250566 |
| Original-input singleton BA | .50 | .50 |
| Final batch CE <= .35 | Fail | Pass |
| Loss reduction >= 25% | Pass | Pass |
| Own singleton BA >= .90 | Fail | Fail |

Compared with the historical baseline, own singleton CE falls from .711237
to .694339 (factor 1) and 2.833542 to .464312 (factor 32). Neither own BA
improves (.50 / .6875 in both). Factor 32 original-input CE falls from
4.776955 to 2.250566, still with chance-level BA and no real accuracy gain.
Correlated training examples and repeated matched covers cannot qualify
generalization, support thresholds, or camera/device independence. FPR here
is training-only, not the deployed FPR acceptance gate.

All 48 own/original singleton logits/metrics replay per model, and all 18
independent NumPy float64 forward oracles pass; no tolerance changes.
Maximum logit/score errors: factor 1, 1.952177e-7 / 5.408997e-8;
factor 32, 8.862839e-7 / 9.184839e-8. Learning failures remain separate.
CLI exit 2 records failed objectives in a complete job, not an unavailable run.

Portable evidence `benchmarks/srnet-wide-batch-20261007.json`, SHA-256
`ae828ea18337fcff82f40506717c7fe5f33d664e380537632a4ac1ca4dcf9427`.
Local model hashes: factor 1
`04abea7b70c64ac0536f23d4a81f824552a567832d06539b8953b0189cf9a1fb`;
factor 32 `4c56d7d13e27a056c3aecfc8c98adda565e3ce76081c575a3baddbb6ecd6d136`.
Complete post-preparation runtime 466.28s, Linux 8 vCPU/15 GiB, two Torch
threads/OpenBLAS one. Concurrent QA makes this engineering timing, not a
controlled performance comparison or challenge-latency qualification.

Interpretation: fixed eight-row grouping reduces singleton loss but does not
establish weak-signal learning or improve detection accuracy. Update count
and group order confounds prevent a unique normalization-cause conclusion.
Next preregister four-row gradient accumulation using the exact eight-row
group order, matching both 80 optimizer steps and 640 row presentations.
This isolates microbatch context more carefully without tuning on validation.

Explicit checkout-only replay with optional research dependencies, fresh output:

```text
OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 3840s venv/bin/python -m steganography.research_srnet_widebatch --config .benchmark/srnet-tiny-sanity-20261007/config.json --out .benchmark/wide-batch-fresh
```

The CLI applies CPU/address/file/core limits; external wall timeout is required.
Shared-service callers provide process isolation. Incomplete runs can leave
earlier-arm artifacts but never a complete two-arm report. No download occurs.

Verification on Python 3.11.14: 1,302 tests pass, total coverage 95.20%; the
two new implementation modules cover 153/154 statements (99.35%), and the
shared trainer has 100% full-suite statement coverage. Ruff, full mypy
(129 source files) and `git diff --check` pass. Local `--no-isolation` wheel
and sdist builds include both new modules and no model/corpus artifacts.
Python 3.12–3.14 and full Docker were not verified locally.
