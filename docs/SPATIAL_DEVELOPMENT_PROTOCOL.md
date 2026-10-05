# Spatial development protocol — 2026-10-05

Preregister before generating derivatives or inspecting model scores. This is
same-source development, not a blind test and not a deployment gate.

- Source: the already acquired 1,000 BOSSbase 1.01 PGM originals, source SHA-256
  `eea495733f0243b420850ca00372877e30c89836738ebc9175ec4b2bbe571d57`.
  Preserve acquisition splits (816 train / 184 validation), bind selection and
  all eight reserved manifests, reject overlap and changed files. Old test
  image bytes are not read. Unknown camera/device identity remains a limitation.
- One lossless output format per original: final SHA hex digit parity chooses
  PNG (even) or BMP (odd). All derivatives inherit it. Keep grayscale pixels;
  do not add independent RGB-channel noise or resize.
- Independent marker-free grayscale LSB replacement: sequential or scattered,
  5%, 20%, 40% of pixel samples, rounded down to whole payload bytes. SHAKE-256
  payload and scatter seed bind public seed 20261005, original SHA, method and
  rate. Verify exact extracted payload independently and decoded distortion
  and nonselected pixels. Seven files per lineage, no duplicate file hashes.
- Bound originals to 1,000; inputs to 16 MiB and 4 million pixels each; corpus
  output to 2 GiB. Reject symlinks/overwrites. Raw data are local only: source
  redistribution permission is unspecified. Do not execute generated files.
- Compare spatial-summary-v1 (12 features) and experimental
  spatial-cooccurrence-v1 (468 features) on identical splits. The latter pools
  sign/reversal-symmetrized clipped three-residual histograms of six filters,
  whole image and top quarter. This custom descriptor is NOT full SRM/SPAM.
- Both models: train-only normalization (std floor 1e-4), class-balanced BCE,
  linear sigmoid, full-batch CPU Adam, learning rate .01, 300 epochs, seed
  20261005. No hyperparameter/threshold search, score threshold .5. Never train
  on validation/test. Report all six method/rate cells, paired-lineage bootstrap
  200 replicates with fixed seed, ROC-AUC, balanced accuracy, recall, FPR, ECE.
- Export both ONNX models and independently compare all validation logits and
  probabilities with CPU PyTorch at absolute tolerance 1e-6, relative zero;
  record failures without relaxing the gate. No model install/autodeployment.
- Cross-source evidence, calibration and >=1,000 cover + >=1,000 stego per cell
  remain unavailable. Acquisition is real imagery; the embedding recipes are
  controlled generated stego, not naturally occurring or unseen-app samples.

Reference taxonomy: [Binghamton feature extractors](https://dde.binghamton.edu/download/feature_extractors/).
No upstream research-only feature implementation is copied into this project.
