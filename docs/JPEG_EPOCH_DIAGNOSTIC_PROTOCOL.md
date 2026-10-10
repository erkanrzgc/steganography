# Frozen train-only epoch diagnosis — 2026-10-10

This read-only follow-up is frozen after the failed final development report,
before inspecting any predictions from this diagnostic. It is not new fitting,
validation, calibration or single-file detection. Do not rewrite failed scores.

Use only the final five-epoch model, original full plan and audited TrainBlocks.
Bind plan SHA-256 `13ea2f8d97e382e5a4c1ed2df8d84aeebe70c656948219dfed2407adeaf0ed90`,
final card `b48686d96f9302b95380a830763c550f238ba9f378878ac4ccf5c6ad2b1e22c7`,
model `71f6f5fe8730944c38d406362bf26e382f348f302b03c7f30809991db73e44b6`.
Require the recorded 16 training source hashes; additionally bind diagnostic
core and the read-only execution script before execution. No validation pixels.

Selection is metadata-only: epoch-0 full four-row schedule, seed 20261008;
flatten consecutive cover/stego pairs, keeping the first eight unique stego
identities per source/quality/method cell. Require six cells and 48 pairs.
No score-dependent selection, retries, alternative model or threshold search.

Invoke the existing tested `core.srnet_diagnostics.paired_probe` unchanged,
on CPU with two threads. Compare original stored-BN versus diagnostic paired
batch statistics, preserving all parameters, buffers and module flags. No
gradients, optimizer updates or BN refresh. Record pair input difference,
scores/loss and all 26 BN-layer deviations; hash source/model and reverify
train bytes after execution. Publish only portable identities and statistics.

Whole diagnostic deadline 180 seconds, outer timeout 240 seconds plus kill
grace, actual host MemoryMax 8 GiB, report cap 1 MiB, no overwrite or symlinks.
If a guard fails, record unavailable/failed; never relax caps or publish a pass.

Paired batch-statistics inference sees a known cover plus its stego derivative,
information unavailable when investigating one unknown file. Its in-sample
accuracy is NOT detector accuracy. A difference supports normalization-mode
sensitivity, not a sole causal explanation. Any training intervention requires
a new preregistered protocol; do not tune on the scored validation rows.
