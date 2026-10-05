# Inference precision repair — 2026-10-05

Freeze before running new exports. This is numerical engineering on already
inspected development data, not fresh detection evidence or retraining.

- Keep originals, old exports and published failed statuses unchanged.
- Explicitly derive a checkpoint, preserving weights and train-only scaling,
  with additive preprocessing `inference_arithmetic: float64`. Convert float32
  feature input to double for normalization/GEMM; round output once to float32.
  Legacy checkpoints keep their original float32 arithmetic and I/O contracts.
- Audit the existing spatial reference/residual and JPEG development models.
  Only train/validation feature artifacts are accessible; no test image reads.
- Compare CPU ONNX Runtime against PyTorch for every validation row, batch size
  one, bounded batches of 17 and the full batch. Logits AND sigmoid scores must
  meet absolute 1e-6, relative zero; all threshold-.5 decisions must match.
- Compare new versus original PyTorch probabilities at absolute 1e-6 and
  require identical threshold decisions. Record drift/failures; no tolerance
  relaxation, score search, recalibration, model installation or deployment.
- Bind source checkpoint SHA, derived checkpoint SHA, validation feature
  hash, manifest, exported ONNX and protocol hashes. Use exclusive nonsymlink
  outputs and bounded weights-only checkpoint/ZIP input. No raw data/model
  redistribution; publish portable numerical audit only.

Later calibration/sensitivity work must use development data and disclose
inspected validation. Solving export parity does not solve false positives or
establish independent-source accuracy.
