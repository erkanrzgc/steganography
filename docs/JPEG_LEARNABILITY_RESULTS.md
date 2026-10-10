# Real-pair learnability results — 2026-10-10

Both frozen arms pass the four-row in-sample criterion. This establishes that
the existing optimizer/input/model path can learn these two real original pairs.
It does **not** improve or replace the historical 50% inspected-probe results,
prove generalization, establish upstream parity, or qualify a deployed detector.

## Fixed comparison

Followed [the preregistered protocol](JPEG_LEARNABILITY_PROTOCOL.md), implementation
checkpoint `2e8f153`. Own RTX 5060 Laptop GPU, CUDA FP32, deterministic algorithms,
no AMP/TF32/CPU fallback/cloud job. Both arms used fresh seed 20261010, identical
initial learned tensors, the same two BOWS2 Q75 JUNIWARD original pairs, and
1,024 repeated four-row updates in eight epochs. No augmentation or best-epoch
selection. Both selected cover/stego crops differ; only those four train pixels
were read. No probe, validation or test pixels; models discarded, no weights
published or deployed.
The covers are real BOWS2 images; stego is labeled upstream JUNIWARD embedding
simulation at .2 bpnzAC, not a naturally collected or blind challenge corpus.

| Arm | Final singleton correct | Final cross-entropy | Seconds | Peak GPU reserved |
| --- | --- | --- | --- | --- |
| Unchanged legacy numerics | 4/4 | 0.0007014005 | 156.27 | 1,319,108,608 bytes |
| SAME pooling / BN epsilon .001 | 4/4 | 0.0006875885 | 202.11 | 1,321,205,760 bytes |

Both initially predicted cover for every row (2/4). Initial large evaluation
logits reflect the untrained running-statistics state, not calibrated scores.
Final singleton/four-row evaluation parity passed at atol/rtol .0001.
Host resident cap 8 GiB; allocator cap 4 GiB; no cap relaxation or failed-result
filtering. All 31 implementation/protocol source hashes stayed unchanged.

Reference-inspired epoch-average train loss was .5501 in epoch two versus
legacy .6784, and .01379 versus .11798 in epoch three. This is an observation
from one seed and two original pairs, **not** evidence of statistically reliable
speed or generalization superiority. Both numerical changes were bundled; their
separate causal effects remain unknown. The reference-inspired arm took longer.

## Portable evidence and independent checks

- [Legacy original report](../benchmarks/jpeg-learnability-legacy-20261010.json).
- [Reference-inspired original report](../benchmarks/jpeg-learnability-reference-20261010.json).
- [Independent scalar/metadata audit](../benchmarks/jpeg-learnability-audit-20261010.json).
- [Reproducible independent logsumexp replay](../tests/test_learnability_evidence.py).

The independent audit checks raw singleton logits with Python logsumexp rather
than the training margin helper, correct-count/criterion, identical initialization
and exposure, both original report SHA-256 values, all source hashes, and fixed
selection against the hash-bound private manifest. The selected first two sorted
BOWS lineages and repeated schedule match. This verifies scalar accounting, not
an independent whole-model forward/optimizer oracle. Private images stay out of
the repository; reports contain no host paths or secrets.

## Diagnosis and next experiment

The blanket explanation "the model cannot learn real JPEG inputs" is contradicted
for this subset. A fatal path-wide optimizer/input failure is less plausible;
broader training may still have optimization, signal, exposure and source-context
problems. Memorizing four rows cannot distinguish learning stego from learning
image-specific differences. These results do not prove insufficient data is the
sole cause, nor that adding more images or epochs alone will fix the detector.

Next, separately freeze a larger train-only Q75/high-payload curriculum, paired
D4 augmentation and sufficient per-original exposure, followed by .2 adaptation
and untouched original/source evaluation. Preserve context-specific quality/method
metrics and all failures. Do not run an unfrozen broad epoch extension or unlock
deployment. See [the primary-source research register](JPEG_RESEARCH_FINDINGS.md).

Verification: full local disk-backed regression 2,100 passed /one actual-CUDA
skip, plus three later evidence tests separately; total coverage 95.92%.
New core services 100%, frontend 96%; lint, types, dependencies and fresh
model-free wheel/sdist checks pass. Physical CUDA fitting is evidenced by the
two original reports, not inferred from the local skipped hardware test.
