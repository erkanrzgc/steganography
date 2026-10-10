# Frozen fresh BN versus GN learning pilot — 2026-10-10

Preregister before optimizer execution. Hypothesis: removing batch-dependent
normalization may improve ordinary singleton discrimination, not just confidence.
GN is batch-size-independent, not a promised steganalysis remedy; see
[Group Normalization](https://arxiv.org/abs/1803.08494) and
[PyTorch 2.14 GroupNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html).
No upstream implementation copied; retain the existing local SRNet-style network.

Use the fixed, independently audited 480-JPEG ALASKA/BOSS/BOWS train-role kit.
Reuse the preregistered per-origin sorted original hashes: first 24 originals
for fit, last eight for probe. 360 fit /120 probe rows, 72 /24 disjoint original
groups. This previously inspected training-scope probe is development diagnosis,
NOT blind validation or accuracy qualification. The new fresh models have not
fitted these probe rows; historical ALASKA/BOSS originals were used by older
models and BOWS is a newer train-role acquisition. No untouched WIFD/test access.

Two arms, from scratch, identical seed 20261010 and initial learned parameters:
unchanged BN architecture/control versus replacing all 26 BN layers with
affine eight-group GroupNorm, eps 1e-5. Keep convolutions, residuals, pooling,
classifier, crop, input units and initialization unchanged. GN gets a distinct
architecture identifier; never load it through the legacy BN model contract.

Both arms use eight fixed epochs, 144 four-row source/quality/method-balanced
updates per epoch, 1,152 total updates; identical ordered schedules and targets
cover/stego/cover/stego. Complete fit-original pair exposure each epoch.
Adamax: lr .001, weight decay .0001, betas .9/.999, eps 1e-8, foreach false.
BN delegates to the unchanged shared numerical engine. The versioned GN adapter
keeps optimizer arithmetic/settings, finite-gradient/state checks and threading;
test its BN-mode arithmetic against that engine on generated data before fitting.

Evaluate only the final fixed epoch, never select an epoch or tune a threshold
using probe scores. Each arm reports all ten method/context confusion cells,
cross-entropy, per-epoch train losses, exact update/schedule hashes, ordinary
singleton logits and singleton-versus-four-row parity (atol/rtol 1e-4).
Report paired probe changes including regressions; these small, already inspected
probes cannot pass real support gates. Private snapshots are numeric NPZ only;
GN keys/architecture are different, no pickle, extraction execution or deployment.

Own RTX 5060 only; no paid rental. Per arm: strict deterministic IEEE FP32,
no AMP/TF32 or CPU fallback, actual host cgroup <=8 GiB, allocator <=4 GiB,
two CPU threads, job <=1800s, outer timeout 1830s, CPU <=3600s, file <=32 MiB.
Independent arms use fresh exclusive output directories; retain every failure.
Bind and reverify source/data identities before/after. Never relax caps, reuse
old weights, truncate epochs, overwrite a checkpoint or tune the protocol after
observing results. No full-corpus readiness or generalization is implied.
