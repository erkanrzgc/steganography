# Frozen train-only cumulative BN refresh control — 2026-10-07

Freeze before probes or buffer refresh. No optimizer step, learned-parameter
change, validation loading, threshold selection, model installation or primary
verdict change. Preserve both original signal-strength models and all failures.
Use the frozen 24-row tiny train selection and manifest/cache checksums from
`SRNET_SIGNAL_STRENGTH_PROTOCOL.md`. Pin its portable evidence SHA-256
a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf.
Model SHA-256: factor 1
14dd74b8d27ac046438563376ef70a3b7d27b65e1f87bb1acd960d5e85d88413;
factor 32 5bfcf4a9b319a4e33ae22a562e9763bd15c06a186b6a4fa8aa3b006e7ac1895b.
Execute both arms in order 1/32; no result-selected repetitions or settings.

Reconstruct unchanged float32 factor 1/32 tensors and original sampler identities.
Bind source/tensor/order hashes and replay all baseline own/original singleton
metrics against pinned evidence before mutation. Use the original final epoch
19 four-row schedule (seed 20261012, eight batches). Read-only contrast on all
32 presented rows: ordinary stored-BN eval versus BN-only batch-stat forwards,
tracking disabled, gradients off and all other modules eval. Report balanced
accuracy, loss, recall/FPR and per-batch logits. This context-dependent contrast
is a training diagnostic, NEVER deployable inference or held-out evidence.

Create a separate clone; never mutate the caller/source model. Reset the clone's
26 BN running statistics/counters. Set BN only to training, tracking true,
momentum None; run exactly one complete eight-batch pass on its OWN training
inputs without gradients or optimization. Cumulative averaging follows installed
Torch semantics. Restore momentum/training flags to ordinary eval afterwards.
All learned parameters, including BN affine scale/bias, must remain bit-identical;
only BN running mean/variance/counters may differ. All clone counters must be 8,
source counters remain 160. This is a cumulative minibatch-stat refresh, NOT
an exact population variance estimator: averaging within-batch variances omits
between-batch mean variance. No larger batches, extra passes or alternate data.

Measure refreshed ordinary stored-BN singleton behavior on all 24 own and all
24 original training inputs, fixed .5 (ties positive). In-sample objective per
arm: own-input BA >=.90 AND own-input CE below the pinned baseline. Publish each
goal separately, including failures; original-input metrics are descriptive.
Do not use results to select any checkpoint, factor, threshold or deployed model.

Save fresh bounded numeric clone snapshots locally only. Reload and reproduce
all refreshed own/original singleton logits/metrics; independently check nine
metadata-first NumPy float64 own-input forward oracles per clone under existing
logit/score/decision tolerances. Verify source model/RNG/threads/inputs/hooks/
existing gradients unchanged; failures restore transient mode flags. Emit a
complete two-arm report only after both arms finish. No weights/tensors published.

CPU float32, two Torch threads, OpenBLAS one thread; cooperative job limit 300s,
external wall 420s plus five-second kill grace; CPU 820/821s, address space 8 GiB,
file 32 MiB, core dumps disabled. Missing tools, incomplete coverage, timeouts or
failed checks are unavailable/failed, never success. Direct service callers must
provide process isolation. Portable reports contain no host paths/secrets.

Any improvement diagnoses sensitivity to this specific refresh on correlated
training data; it does not prove BN is the sole cause, real generalization or
qualification. A failed control remains evidence and motivates a separately
preregistered batch-context/optimization experiment, not automatic deployment.
