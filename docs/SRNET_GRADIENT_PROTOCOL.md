# Frozen train-only gradient diagnostic — 2026-10-07

Freeze before probing. No fitting, optimizer step, model export, validation
pixels, threshold selection or detector change. Keep all historical failures.
Use the manifest/train cache and 24-row selection from the frozen tiny sanity:
manifest SHA 0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14,
cache SHA 828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028.
Probe the first epoch-zero four-row batch touching each of the six declared
source/quality/method cells; deduplicate batches in original sampler order.
Seed 20261012. Compare a fresh seeded network and the failed 400-update tiny
model SHA 78dfff3e6da836fcabd3a6e65dc103b0a7508f4fcd9d30ca15ce961984c4b16e.
Also probe the first generated-positive epoch-zero batch on a fresh seeded
network, using the already frozen checkerboard generator. No real source claim.

For each probe run the exact four-row training forward, labels [0,1,0,1],
with BN batch statistics but running-stat tracking disabled. This is a training
diagnostic only, NEVER deployed inference. Compute cross-entropy gradients via
autograd.grad for all learned parameters and input; do not assign .grad.
Repeat with positive rows replaced by their paired covers, leaving targets
unchanged. This deliberately contradictory null control is not valid training.
Record true/null loss, every parameter gradient L2 norm, global norms, gradient
difference/relative difference and cosine, input gradient L2, input difference
RMS, and input directional derivative along the actual positive-cover delta.
Use float64 aggregation; finite checks and shape bounds apply everywhere.

Check the classifier's analytic gradient with central differences in the
direction of its normalized analytic gradient, epsilon .001. Keep all other
parameters fixed, restore the classifier after each probe. Gate absolute
derivative error <= .002 + .02*abs(analytic derivative). Zero classifier
gradient is unavailable, not a pass. This checks one parameter direction,
not every layer's gradient and not input finite differences. No gradient-norm
or alignment threshold is used as an accuracy gate.

Restore exact model parameters/buffers, training/tracking flags, caller RNG and
threads, existing gradients and hooks even on failure. No learned/statistical
state saved; only fresh non-symlink portable JSON. CPU float32, two Torch
threads, cooperative 180-second probe limit, external hard wall 300 seconds
with five-second kill grace, 8 GiB address space, CPU 580/581 seconds, file
32 MiB, core dumps disabled. Timeout/absence/incomplete coverage fails.

Interpretation: nonzero gradients or a passed classifier directional check do
not establish useful weak-signal learning, generalization or a unique cause.
Publish all failures and source/model/content/order hashes. Strong versus weak
contrast is descriptive and confounded by generated versus real content.
