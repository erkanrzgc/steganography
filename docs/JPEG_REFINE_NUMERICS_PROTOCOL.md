# Separately frozen numerical preflight revision

This amendment follows the failed zero-training physical attempt and two
generated-only diagnoses. The original 31 source files, frozen protocol,
failed receipt and 16 historical checkpoints remain immutable. The original
preflight still fails; its result is never relabeled as passed.

The owned-device diagnosis measured a maximum initial logit difference of
2.7567148208618164e-7. Independently rounded gradients led to post-Adamax
weight differences near 2e-4. Adamax's near-zero gradient normalization makes
post-update state comparison a different condition from forward/backward
numerical parity. This is a measured explanation, not proof of model accuracy.
[PyTorch's numerical accuracy documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
also describes CPU/GPU rounding differences; deterministic execution applies
within the same hardware and software environment.

The replacement preflight binds a new source closure and checks, before any
real-data update:

- CPU/CUDA logits, objective, gradients and pre-optimizer state at unchanged
  rtol 1e-3 /atol 1e-4, before applying the independently rounded gradients.
- Author-rule Adamax parameters and all slots against independent float64
  scalar equations over four steps on each backend, with zero, near-zero and
  sign-changing gradients; rtol 1e-6 /atol 1e-8.
- Exact model, loss and optimizer save/resume equality on the GPU over two
  further steps. The numeric checkpoint format and counter scaffold stay fixed.
- Eight genuine batch-16 GPU update timings, the existing 1.25 safety factor,
  startup/checkpoint overhead and 900-second evaluation reserve. A matrix that
  does not fit the remaining allowance never starts fitting.

No optimizer, training tensor, initialization, seed, batch, architecture,
schedule, exposure, checkpoint, endpoint, threshold or learning rule changes.
Training and evaluation continue to call the original frozen kernels. The new
adapter uses the original controller's injected source/launcher/clock contracts.
Handled unavailable worker responses return valid JSON with allowlisted reason
codes and process status zero; real process exits/signals remain separate.
Raw exception text, stderr, paths and secrets never enter public reports.

The amendment JSON binds both original and additional sources. Configurations
must bind the complete new closure. No old snapshot can be imported into this
revision, and its adapter rejects `resume`. There have been zero real-data
updates in the earlier attempt. Only one amended physical attempt is permitted;
there is no automatic retry or training intervention after a failed matrix.

The six-hour /ten-GiB allowance never resets. Debit the original controller's
34.309597279 seconds plus three conservative 190-second diagnostic/validation
reservations before the amended controller starts. Its charged clock returns
an explicitly earlier start once, then the unmodified monotonic clock, so every
deadline, estimate and reported elapsed second includes that expenditure.
`external-accounting.json` records the debit; the original controller's
`prior_seconds` field remains zero because no same-source snapshot was resumed.
The enclosing host-local watchdog uses only the remaining allowance plus at
most 60 seconds. Conservatively reserve 256 MiB for prior staging/diagnostics;
before every worker reserve another 97 MiB against the cumulative ten-GiB cap.
Software CI preparation does not run the experiment and cannot supply learning
evidence. Actual diagnostic timings and conservative debits are reported apart.

Freeze sources before generated GPU validation. Require focused tests,
full unchanged coverage gate, Ruff, mypy, portable receipt checks and old-source
identity checks before real fitting. Run generated validation and fitting in
fresh owned disk-backed directories under <=8-GiB kernel resident memory,
zero swap, <=4-GiB CUDA allocator and a host-local service watchdog. Keep the
physical UUID binding, IEEE FP32 and deterministic CUBLAS policy. Downloads,
CPU training fallback, deployment and production qualification remain absent.
