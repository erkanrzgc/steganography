# Train-only JPEG normalization-mode diagnosis

This is a **post-hoc read-only diagnostic**, not another fitting arm, a new
learning gate, normalization recalibration or a deployment mode. It follows
the completed eight-control matrix and full fixed-statistics fit/development
comparison. It uses no development or reserved pixels and does not select
thresholds. Source files and numeric snapshots of the completed matrix remain
unchanged.

## Protocol fixed before the diagnostic

- Evaluate all four arms and both seeds; no favorable-model selection.
- Select eight fit originals from each of the two source groups using a fixed
  NumPy RNG seed 20261014 and sample without replacement. Keep paired cover/
  stego rows, alternate source pairs and use the same selection for every model.
- Use only the existing center view, without D4 augmentation. Compare identical
  inputs and learned state, with each arm's genuine batch size (4 or 16 images).
- Contrast stored BatchNorm statistics with paired-batch statistics under
  `no_grad`. For the latter, enable only BatchNorm batch-statistic computation
  and disable running-stat tracking. Restore all flags afterwards; parameter
  and buffer digests must remain exactly equal. No optimizer is created.
- Record cross entropy, absolute logit margins and cover/stego probability
  summaries, not accuracy, ROC-AUC or a threshold recommendation. Paired batch
  statistics depend on the known-cover composition and cannot be substituted
  for single-file production inference.
- Require the complete source-bound matrix and fit diagnostic, all 26 expected
  BatchNorm layers, numeric-only verified snapshots and unchanged fitting /
  controller source hashes. Reject nonfinite output, missing state, unexpected
  flags or any state mutation.
- Own GPU only, FP32/IEEE, no TF32/AMP or CPU fallback, 4 GiB allocator and
  8 GiB kernel RAM limits. Each model is bounded to 120 seconds, the diagnosis
  to at most 1,200 seconds, and all added time consumes the **remaining original
  six-hour controls envelope**, not a new budget. Timeout/OOM is unavailable.

## Interpretation limits

An immediate loss/margin change with fixed inputs and learned state isolates
the normalization-mode effect on these selected fitting blocks. It does not
prove that BatchNorm alone explains the full development failure, identify a
safe production fix, or establish sufficient training exposure/data diversity.
Only 16 original pairs are used; these are not an independent-source benchmark.
Historical matrices, failed learning gates and production acceptance remain
unchanged. Any proposed normalization intervention requires a separate matched
training/evaluation protocol before it can change shipped detection.

Results and exact receipt/source hashes will be recorded after execution; a
pending diagnostic is not a passed measurement.
