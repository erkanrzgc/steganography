# Layerwise population-BN control — no useful discrimination

The frozen `JPEG_POPULATION_BN_PROTOCOL.md` and implementation commit `1ec3036`
ran on the user's RTX 5060 Laptop GPU in 60.27475 seconds. The actual cgroup
RAM limit was 8 GiB and GPU allocator cap 4 GiB; strict deterministic FP32 model
operations, FP64 central moments, no AMP/TF32/CPU fallback. No RunPod rental.

The final five-epoch model was cloned, not retrained. All 26 layers received
sequential population statistics from 360 rows /72 original groups. All 120
probe rows /24 disjoint original groups were scored singly, never with a cover
partner. Both baseline and refreshed singleton-versus-four-row logits pass
atol/rtol 1e-4. Learned weights and original model bytes remained unchanged;
zero optimizer updates, no validation/test access or deployment.

These probe originals are from historical training acquisitions; the tiny
partition is NOT untouched held-out validation, independent-source support
or a detector accuracy benchmark. Cover rows are shared across two method
cells, so the cell denominators must not be summed as independent samples.
The report's `probe_is_historical_training_data` means the kit's declared
train-only acquisition role, not that every original was used by the old
optimizer: that prior model fitted ALASKA/BOSS, not BOWS. BOWS is newly acquired
training-scope data; its calibration originals participate in this correction.
This distinction still does not create a blind cross-source qualification.

| Origin | Q | Method | Probe balanced accuracy (%) | Probe cross-entropy |
| --- | --- | --- | --- | --- |
| ALASKA2 | unknown | JUNIWARD | 50.00 → 50.00 | 15.710 → 2.251 |
| ALASKA2 | unknown | UERD | 43.75 → 50.00 | 15.718 → 2.254 |
| BOSSbase-1.01 | 75 | JUNIWARD | 50.00 → 50.00 | 2.742 → 1.151 |
| BOSSbase-1.01 | 75 | UERD | 50.00 → 50.00 | 2.740 → 1.145 |
| BOSSbase-1.01 | 95 | JUNIWARD | 50.00 → 50.00 | 1.733 → 1.034 |
| BOSSbase-1.01 | 95 | UERD | 50.00 → 50.00 | 1.731 → 1.033 |
| BOWS2 | 75 | JUNIWARD | 50.00 → 50.00 | 18.345 → 1.849 |
| BOWS2 | 75 | UERD | 50.00 → 50.00 | 18.342 → 1.881 |
| BOWS2 | 95 | JUNIWARD | 50.00 → 50.00 | 14.753 → 1.842 |
| BOWS2 | 95 | UERD | 50.00 → 50.00 | 14.753 → 1.840 |

All ten refreshed cells remain at 50% balanced accuracy. Loss decreases in
every cell but remains above binary uniform loss (log(2)). Reduced erroneous
overconfidence is not useful steganalysis. This rules out this specific
layerwise statistics-only intervention as a sufficient fix on these selected
rows; it does not establish the sole cause of the failed model.

The original simultaneous cumulative minibatch refresh and failed full
five-epoch development results remain unchanged. Do not tune another variant
against these probe scores or silently promote this private clone. Next
learning should be a separately preregistered fresh-model normalization control
with matched exposure/optimizer settings and ordinary single-file inference,
followed by a genuinely untouched evaluation policy. More epochs on the failed
recipe or a larger GPU alone are not a demonstrated remedy.

## Evidence and verification

`benchmarks/jpeg-layerwise-population-bn-20261010.json` contains all ten cells,
layer sample counts, hashes, memory peaks and exact execution-source closure.
All source hashes were checked locally after authenticated retrieval. The
private numeric clone stays on WSL; no model weights or raw media are published.
The public application and old numerical/checkpoint sources were not modified.

29 generated unit/security tests pass, including real Torch 26-layer toy
calibration versus independent NumPy population moments, unequal-count central
moment merging, source immutability and singleton checks. New services/frontend
coverage 97% /97% /98%; this is engineering coverage, not detection accuracy.
Full disk-backed regression passes 2,009 tests /one local actual-CUDA skip
(not a pass), 32 existing ONNX warnings, in 584.30 seconds; total coverage
95.83%. The three physical portable-evidence tests were added after full-run
collection and pass separately alongside all 29 new controls (32 focused).
Ruff, types, dependency and whitespace checks pass. Local suite Python 3.11;
physical CUDA control on WSL Python 3.12.3. Real whole-model independent
mathematical replay and cross-source accuracy qualification remain unavailable.
Fresh wheel/sdist build and member inspection pass (165/334 members), with
no private corpus, tensors, model weights or provider/SSH directories.

