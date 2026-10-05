# Parity-residual development protocol — 2026-10-05

Freeze before real feature extraction/training/scoring. This is iterative
development on previously inspected BOSSbase validation, not a blind test.

- Reuse the exact audited 7,000-file controlled gray-plane LSB corpus, original
  manifest SHA `707db7debce75b2a241e1ea3b1a5e3cb9b89c287d7396f95abad5c3364e5eda7`.
  Preserve 816 train / 184 validation lineages, seven variants per original.
  No old frozen-test images, corpus regeneration or new dataset download.
- New custom `spatial-parity-residual-v1` contains the previous 468 full/top
  co-occurrence features plus 216 joint parity/residual probabilities: four
  neighbor directions, bit planes 0/1/2, center parity 0/1, signed neighboring
  difference clipped to [-4,4]. Pool channels; grayscale and identically
  replicated RGB agree. No interchannel shortcut, marker or resize.
- Round descriptor values to 1e-8 under this new version only, keeping the
  full feature artifact within the existing 64 MiB limit. Legacy extraction
  remains unchanged. No SRM/SPAM or real mobile-app accuracy claim.
- Train one normalized/class-balanced linear CPU model: seed 20261005, Adam
  .01, 300 epochs, train-only std floor 1e-4. Float32 training, explicitly
  float64 inference accumulation with float32 I/O. Fixed threshold .5; no
  hyperparameter, threshold, calibration or best-run selection.
- Compare every validation row against the previously completed 468-feature
  model on the identical corpus/split, binding original report and prediction
  SHA hashes and ordered row identities. Show all six cells, not just the
  targeted scattered 5% cell. Per-cell paired bootstrap 200, seed 20261005.
- Record ROC-AUC, balanced accuracy, recall, FPR, ECE and confusion counts.
  Better ranking with worse false alarms, or regressions elsewhere, stays
  visible. Cross-source support remains unavailable; no model installation.
- Export and compare ALL validation logits and probabilities in CPU ONNX and
  PyTorch: absolute 1e-6, relative zero; unchanged threshold decisions. Preserve
  all original models/reports; exclusive nonsymlink outputs and bounded inputs.

Next qualification still requires frozen calibration/operating point and
newly untouched independent-source data, including named upstream methods.
