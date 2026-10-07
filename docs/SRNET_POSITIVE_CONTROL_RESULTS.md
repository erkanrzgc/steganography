# Generated strong-signal learning control — 2026-10-07

This is an intentionally obvious checkerboard training control, not JPEG
steganography or a real detector benchmark. The protocol was committed as
`ebd8916` before generation/fitting. Actual executed source is retained in
`3d95213`; a subsequent list/ndarray variable rename fixes type checking only,
without changing numeric operations or rewriting recorded execution hashes.

Twenty complete epochs, 24 generated training tensors, 160 unchanged four-row
optimizer updates. Fresh SRNet, ordinary stored-BN singleton inference, no real
dataset/validation loaded. Runtime 182.16 seconds on Linux, 8 vCPU/15 GiB,
two Torch threads; this is a local engineering measurement, not a throughput
benchmark. No weights installed or detector behavior changed.

| Frozen train-only objective | Measured | Result |
| --- | ---: | --- |
| Final batch loss <= .35 | .0003578252 | pass |
| Loss reduction >=25% | 99.6705% | pass |
| Stored-BN training balanced accuracy >=.90 | 1.00 | pass |

First epoch loss .1085919484; singleton cross-entropy .0182024338,
recall 1.00, FPR 0.00, **all on the same generated training rows**. No confidence
interval/generalization claim or supported-method status is appropriate.
Method/format tags are sampler compatibility labels, NOT actual JUNIWARD,
UERD or JPEG generation. SHA-256 binds raw little-endian float32 tensor bytes.

Read-only reload reproduced all 24 singleton logits/metrics and every BN
counter was 160. All six independent NumPy float64 forward oracles pass:
max logit difference 1.629103e-6; max score difference 2.480131e-8;
all decisions agree. Numerical and learning gates are separate.

Evidence: `benchmarks/srnet-positive-control-20261007.json` and
`benchmarks/srnet-positive-audit-20261007.json`. Control SHA-256
`605f31b19a0440234114dc71835ff5dba18f21a3c53a72fda72bac137acb2c63`;
model SHA-256 `b4b7204c0851f4bc95baba0336887f0d7ddf4eb672bb27979b4540085cb13cbb`.
Weights/tensors stay local and are not included in wheels or Git.

Interpretation: the unchanged pipeline can learn this strong signal; it is
not wholly incapable of updating a classifier. This does NOT establish that
weak real stego gradients, optimization, normalization or preprocessing are
correct. The previous real tiny sanity still has BA .50; real held-out cells
still fail. Next preregister train-only gradient/input and signal-strength
diagnostics before longer real-corpus fits. Do not tune on reused validation.

Explicit checkout-only replay (fresh output directories; optional Torch needed):

```text
timeout --signal=TERM --kill-after=5s 1920s venv/bin/python -m steganography.research_srnet_positive --out .benchmark/positive-fresh
timeout --signal=TERM --kill-after=5s 180s venv/bin/python scripts/audit-srnet-positive-control.py --result .benchmark/positive-fresh --out .benchmark/positive-fresh/audit.json
```

The training module applies hard CPU/address-space/file/core limits; the audit
is read-only and requires an external timeout. Failed/incomplete fits or audits
exit 2 and never qualify detection. A kill may leave incomplete local files;
no incomplete output is a complete usable result. Output paths cannot already
exist or traverse symlinks. No downloaded/extracted code is executed.

Verification: Python 3.11.14, 1,205 tests pass; total coverage 94.98%, new shared
control/audit and entrypoint 126/127 statements covered (99.21%). Ruff, mypy
(121 source files), diff checks and model/corpus-free wheel/sdist checks pass.
The post-type-fix numeric audit is byte-identical to the original audit.
Python 3.12–3.14 and fresh full-Docker checks were not run for this slice.
