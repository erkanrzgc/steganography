# Nonlinear residual-feature results — 2026-10-05

**Development improvement on ALASKA, mixed BOSS results, still failed detection.**
The 64-unit ReLU network with linear skip learns interactions in pooled custom
DCT/content/quantization statistics. It is not raw-residual CNN/SRNet or a
context-invariant detector. Implementation and [protocol](JPEG_NONLINEAR_PROTOCOL.md)
were frozen in `1c9a407` before training; [portable evidence](../benchmarks/jpeg-nonlinear-development-20261005.json)
retains all 14 cells, paired intervals, provenance and failures. No deployment.

## Fixed comparison

Reuse all 3,750 JPEGs and the exact cached 2,066-feature vectors: 2,985 train,
765 validation. CPU seed 20261007, 300 full-batch Adam epochs, LR .001,
weight decay .001, train-only scaling and source/class-balanced objective.
Cutoff .5, no calibration, early stopping, tuning or best-run selection.
Architecture, learning rate and regularization change together; these results
are not a causal architecture ablation. Final training loss .6836586.

Both sources are shared with training; validation was inspected in earlier
experiments. **Zero unseen sources**, not a blind or cross-source test. BOSS
has 25 original validation scenes with Q75/Q95 derivatives, not 50 independent
scenes. Its stegos are simulations, not payload-recovery evidence. JMiPOD is
excluded; ALASKA quality/payload and camera/device/app metadata are unknown.

| Cohort | Method | Covers/stegos each | AUC | BA | Recall | FP / covers (FPR) | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALASKA | JUNIWARD | 205 | .608923 | .602439 | .590244 | 79 / 205 (.385366) | .075311 |
| ALASKA | UERD | 205 | .602546 | .582927 | .551220 | 79 / 205 (.385366) | .060300 |
| BOSS, both qualities | JUNIWARD | 50 | .554400 | .530000 | .460000 | 20 / 50 (.400000) | .006748 |
| BOSS, both qualities | UERD | 50 | .510400 | .500000 | .400000 | 20 / 50 (.400000) | .032597 |
| BOSS Q75 | JUNIWARD | 25 | .505600 | .500000 | .400000 | 10 / 25 (.400000) | .034127 |
| BOSS Q75 | UERD | 25 | .512000 | .520000 | .440000 | 10 / 25 (.400000) | .012546 |
| BOSS Q95 | JUNIWARD | 25 | .611200 | .560000 | .520000 | 10 / 25 (.400000) | .033967 |
| BOSS Q95 | UERD | 25 | .515200 | .480000 | .360000 | 10 / 25 (.400000) | .063136 |

Each method shares cohort covers; repeated FP counts are not new evidence.
All pooled/format and unknown-quality cells remain in JSON; unknown quality is
unavailable. Every measurable cell fails AUC/BA/recall/FPR targets; several also
fail ECE. No method qualifies as supported.

Against the immediate [linear residual reference](JPEG_RESIDUAL_RESULTS.md):

- ALASKA FPR .463415 → .385366, delta CI [-.156098, -.009512]. JUNIWARD
  AUC rises .035978, CI [.020319, .053098]; UERD rises .032409, CI
  [.015176, .048150]. Ranking improves even against the older single-origin
  JUNIWARD/UERD model (.592481/.585306), but remains far below .90.
- ALASKA JUNIWARD recall .551220 → .590244; UERD .556098 → .551220
  regresses. Both recall change intervals cross zero. ECE worsens from
  .021644/.025980 to .075311/.060300; both ECE change intervals are positive.
- BOSS JUNIWARD AUC .525600 → .554400 and recall .44 → .46; UERD
  AUC .521600 → .510400 and recall .44 → .40 regress. Both AUC change
  intervals cross zero. Aggregate FPR stays .40, with Q75 .36 → .40 and
  Q95 .44 → .40; unchanged aggregate counts conceal changed error identities.

Paired 200-resample, original-lineage 95% intervals retain both quality variants
together. These are uncorrected iterative-development intervals, not independent
confirmatory evidence or support qualification.

## Audit and reproduction

Rehashed all 3,750 JPEGs and bound both caches; checked all 765 ordered
prediction identities against the manifest/reference. Independently reproduced
confusion, pairwise AUC, BA, recall, FPR and ten-bin ECE in all 12 measurable
cells. NumPy train mean/scale deviations: 1.21e-7 / 1.48e-8. Independent
NumPy64 skip/ReLU logits match native outputs exactly; saved sigmoid difference
≤5.42e-8. CPU ONNX batches 1/17/765 agree exactly with identical .5 decisions
at abs 1e-6, rtol 0. Numerical correctness does not mean useful detection.

Python 3.11: 710 tests, total coverage 93.85%, shared feature model 100%;
Ruff, mypy (90 files), diff checks pass. Other Python versions and fresh Docker
E2E are unverified locally. Optional legacy ONNX exporter emits deprecation
warnings. Model weights, caches and raw data stay local; wheel remains model-free.
Fresh wheel/sdist builds pass; the nonlinear module is included, Torch stays
opt-in and no weights or corpus are bundled.

For reproduction after explicit prior dataset preparation, bind the original
manifest and cache checksums in fresh train/validation config files. Add
`architecture: residual-mlp64-v1`, `weight_decay: .001` and the fixed settings
above to the existing training configuration. Call shared `train_model`,
`predict_validation`, `export_onnx` and `diagnose_validation` with fresh outputs;
the existing `research train`/`export` commands use the same implementation.
No implicit acquisition or model installation occurs. Training/prediction/export/
diagnosis took 6.59 seconds with two math threads, excluding cache creation and
independent audit; not a CTF or inference latency claim.

Next: preserve this failed development model; implement raw residual/quantization
learning rather than claiming more pooled-feature capacity solves context
awareness. Qualification still needs licensed untouched sources and adequate
original-lineage counts. Never tune on the frozen native holdout.
