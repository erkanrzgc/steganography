# Frozen generated SRNet positive learning control — 2026-10-07

Freeze before generating tensors or fitting. This is an intentionally obvious
synthetic train-only signal, NOT steganography, JPEG decoding, independent-source
evaluation, detector accuracy, calibration or a release gate. Keep all previous
real-data failures unchanged. No real dataset or validation cache is accessed.

Generate exactly 24 float32 tensors of shape 1x256x256: two artificial source
groups, four lineages per group, three rows per lineage. Local NumPy PCG64 seed
20261013; in source-major/lineage-major order generate one uniform [-1,1) noise
plane per lineage. Cover = float32(96 + 16*source + 4*lineage + noise).
Positive variants add float32 checkerboard amplitudes 48 and 56 respectively,
where checkerboard = 2*((row+column)%2)-1. Both positives share the same cover
noise; each row SHA-256 hashes its little-endian contiguous float32 bytes.
Sampler labels JUNIWARD/UERD and format JPEG are compatibility tags ONLY:
no such embedding or encoded JPEG is produced. Quality is unknown/null.

Use the unchanged two-source/four-row sampler, seed 20261012 and fresh
srnet-gray12-cpu-v1 model. Twenty complete epochs, eight updates each (160
total), CPU two threads, Adamax LR .001, weight decay .0001, betas [.9,.999],
epsilon 1e-8, foreach false. No scaling, augmentation, BN repair, early stopping,
optimizer changes or reruns selected by results. Optimizer wall limit 1,800s;
external hard wall 1,920s plus five-second kill grace; address space 8 GiB,
CPU 3,660/3,661s, file size 32 MiB, core dumps disabled. Incomplete work fails.

Require every BN counter to equal 160. Ordinary stored-BN singleton evaluation
on all 24 generated training rows, fixed .5 threshold (ties positive). Record
all epoch pair/batch hashes, losses, row content hashes, logits and metrics.
Preregister objectives separately: final batch cross-entropy <= .35, loss
reduction >=25% from first epoch, stored-BN balanced accuracy >=.90. Failed
objectives remain published. Zero initial loss means zero relative reduction.

Save only a fresh non-symlink bounded numeric model locally; publish no weights
or tensors. Reload and independently replay the first cover and both positive
variants of lineage zero in each source (six NumPy float64 forward oracles),
using existing logit/score/decision tolerances. Numerical failure is independent
of learning failure. Preserve caller Torch RNG/threads. Record protocol, source
and model hashes. Passing establishes only ability to learn an obvious generated
signal; it does not prove correct weak-signal optimization or generalization.
