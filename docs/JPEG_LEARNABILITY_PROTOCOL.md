# Frozen real-pair learnability and reference-numerics control — 2026-10-10

Freeze before fitting; no deployment or generalization claim. The failed
BN/GN experiments and source-pinned legacy engines remain unchanged.

Hypothesis: a fixed real four-row task may reveal optimization failure, and
reference-inspired pooling/BN numerics may change learnability. Neither outcome
isolates one numerical difference as the cause. This is not a full reproduction
of the original TensorFlow recipe, nor the promised final accuracy improvement.

Use the same hash-bound, independently audited 480-JPEG train kit. Select only
BOWS2 /Q75 /JUNIWARD: first two sorted original hashes among the first 24 fit
originals, keeping matched cover/stego rows. Verify both selected unrounded
256-square Y crops differ. No selection by prediction, no probes/validation/test
pixels, no filtering of historical failed results. Different inputs are not
proof of adequate signal strength; this test attempts memorization only.

Fresh matched seed 20261010, identical learned initial parameters and four-row
order [cover, stego, cover, stego]. Two arms: unchanged legacy numerics versus
separately tagged reference-inspired numerics. The latter changes only all 26
BatchNorm eps values from 1e-5 to 1e-3 and the four 3x3/stride2 average pools to
TensorFlow SAME geometry with valid-sample divisors. Keep weights, classifier,
optimizer and preprocessing fixed. Original classifier-bias, regularization,
batch size, augmentation and training schedule differences remain; do not call
this exact upstream SRNet parity. Independent NumPy pooling checks include odd
and even dimensions and prove the legacy difference; generated BN equations
check eps without using the model as its own oracle.

Eight fixed epochs of 128 repeated four-row updates (1,024 total) per arm,
Adamax lr .001 /weight decay .0001 /eps 1e-8 /betas .9,.999 /foreach false.
No data augmentation in this fixed-subset memorization test. Initial and final
ordinary singleton logits and final four-row parity are reported. Learnability
criterion: all four final singleton labels correct AND mean cross-entropy <.2.
Report train losses and failed criterion; never select a best epoch or tune a
threshold. Passing means in-sample memorization only, not detector support.

RTX 5060 own GPU only, deterministic FP32/no AMP/TF32/CPU fallback; host cap
8 GiB, allocator cap 4 GiB, two threads, 1800s inner/1830s outer timeout per arm,
CPU cap 3600s. Source/cache identities verified before/after. Exclusive report
outputs only; discard these models, publish no weights, never load the new
architecture through the legacy contract. No cloud spending/model downloads.

Sources inspected (no upstream code copied):
- [Author SRNet source](https://dde.binghamton.edu/download/feature_extractors/),
  `SRNet.zip` SHA-256
  `79b31cfa4177a9dee62639b5c9d2a9d154d82d75b5216878607360df62e1dfb3`.
- [TensorFlow 1.15 contrib BatchNorm default eps](https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/contrib/layers/python/layers/layers.py).
- [SRNet paper](https://ws.binghamton.edu/fridrich/research/SRNet.pdf): easier
  Q75 initialization for Q95 and high-to-low payload curriculum are later,
  separately frozen experiments, not changes to this control.
