# Frozen train-only SRNet BatchNorm diagnosis — 2026-10-07

Targeted post-hoc diagnosis after the failed first pilot, not an independent
benchmark, acceptance gate or new trained detector. Freeze before executing
these probes. Do not load validation pixels/scores, select a checkpoint, tune
a threshold, change weights or update stored BN statistics.

## Bound inputs and selection

Reuse only the completed BOSS-only pilot's original model/card/plan and complete
training float cache. Manifest SHA
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`,
train cache SHA
`828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`,
plan SHA `27634b6576a1428669d3d8c7c490544f86a59a3449c5b7e304fbbd2dac38bdda`,
model SHA `b48faf464c9116fa1778d9cc411e03d154f54f3d31c2fc260cce79424b6acd51`,
card SHA `1e4db84ee3b905c73f04bd7de19691ef34c491af38d5d52929c870fc510dc99f`.
Reconstruct the full fit contract and BN update counts before probing.

Use seed 20261012, epoch-zero declared sampler order; select the first eight
distinct training stegos per declared source/quality/method cell. This gives
32 paired probes over the four BOSS Q75/Q95 JUNIWARD/UERD cells. No sample is
selected by observed scores, filename, training loss or validation behavior.
The generic helper caps probes at 64, never silently truncates cells. Tiny
generated regression cells may contain fewer than eight original pairs.

## Paired diagnostic contrast

Each input is exactly a matched cover/stego pair, unchanged unrounded float32
Y pixels, labels [0,1]. Compare ordinary all-stage eval against a diagnostic
forward using batch statistics in BN only. Other modules stay eval, gradients
disabled. Disable BN running-stat tracking for the contrast and restore all
flags/buffers even on failure. Preserve exact numeric model state, input,
RNG/thread state and original disk artifacts. The diagnostic contrast is
batch-dependent and **must never become the primary inference mode**.

For all 26 BN layers in both passes, record observed channel means/unbiased
variances against the saved running mean/variance: median/max standardized
mean shift and variance ratios. Record logits, class-one scores, paired cross
entropy, decisions and input cover/stego RMS difference. Keep finite checks,
threshold 0.5 (ties stego) and fixed score saturation cutoffs .001/.999.
No learned parameters/statistics are saved; no recalibrated model is produced.

CPU float32, two Torch threads, cooperative 180-second probe deadline; launch
the fixed audit script under an external 300-second hard wall timeout, with
OPENBLAS one thread. Direct service checks are not an OS/filesystem sandbox.
Missing dependencies/failed probes are unavailable/failed, never successful.
Store raw per-pair identity diagnostics locally. Publish portable aggregate
layer/cell summaries and hashes only; no corpus, weights or host paths.

## Interpretation

Contrast can show a sensitivity to BN mode or running-stat mismatch on this
training subset; it cannot prove that BN is the sole cause or demonstrate
generalization. These are correlated, in-sample probes, not accuracy gates.
After diagnosis, freeze any new combined-source learning/BN experiment
separately. Do not rewrite this protocol or the failed original pilot.
