# JPEG residual/parity comparison — frozen before feature extraction

Date: 2026-10-05. This fixed follow-up tests a richer **custom block-DCT**
descriptor, not DCTR, JRM, SRNet, a deep model or an installed detector. Freeze
code/protocol before extracting the actual development corpus or training.
Unit-test fixtures are not accuracy evidence. No best-run selection or tuning.

## Inputs and isolation

Reuse exactly the preceding two-origin JPEG development experiment, documented
in `JPEG_CONTEXT_RESULTS.md`. No downloads, regeneration, relabeling, new
quality selection, exclusions or test-image access. Previously inspected
validation is iterative development, not a blind or cross-source test.

- Portable reference SHA-256:
  `eafbc99fa5bfebfa26e6582f285267991eb18e57d1e39e2db564c84acffea7fb`.
- Shared 3,750-file manifest SHA-256:
  `0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
- Previous prediction SHA-256:
  `fcb7af1f419cb33982de276dbc9ff883a45be35b42face147e74ebdce82d0c21`.
- Context train/validation feature hashes:
  `bcdc6254f8821cda4b337f88eab1c35e3eb1a52f834372c79f209117f4ce1f91` /
  `ed126653ad0f69a5138aebb34ac6e0a9e99a3a76640295170a4ed56d6e1adc30`.

Keep 2,985 training / 765 validation JPEGs, including both Q75/Q95 BOSS
derivatives in their original split/lineage. Both declared sources remain shared
between training/validation. JMiPOD remains excluded: no matching simulator.
ALASKA quality/payload and all camera/device/app metadata remain unknown.
The upstream BOSS stegos are simulations, not message-recovery evidence.

## Versioned representation

`jpeg-dct-residual-parity-v1` has **2,066** features, rounded to 1e-8:

1. Preserve the prior 1,098 measured context features as an exact prefix.
2. Select 32 DCT frequency modes, including DC, sorted by `(u + v, u, v)`.
   For each mode and horizontal/vertical adjacent block pair, use
   `clip(second - first, -2, 2)`: normalized five-bin signed residual histogram.
   This contributes 320 values.
3. For those same pairs, record the normalized 10-bin joint histogram of
   `first & 1` and clipped residual, parity-major ordering: 640 values.
4. Add eight absolute DC magnitude bins clipped at seven.

Subtract signed coefficients in int64 to avoid overflow. Retain the existing
signed-16-bit, quantization, pixel/dimension and JPEG-byte guards. The prefix must
match every bound cached context vector before training. Residual/parity uses
actual coefficients; no source names, filenames, labels, declared quality-factor,
cover references or expected payloads are prediction inputs. These handcrafted
statistics do not imply camera invariance, calibration or generalization.

Reuse the fixed allowlisted research worker mechanism: 2 MiB JPEG input,
4 million pixels, 128 KiB output, 15-second timeout, 1 GiB address space,
CPU/file/core limits. Four workers maximum; absence/failure is not a pass.
Train/validation artifact limits remain 64 MiB each. Do not relax limits after
seeing a failure, truncate rows or overwrite historical artifacts.

## Fixed training and evaluation

Keep the same objective and settings as the preceding comparison: CPU linear,
seed 20261006, 300 Adam epochs, learning rate .01, train-only standardization
(scale floor 1e-4), source/class-balanced recipe, float32 training, explicit
float64 accumulation with float32 I/O inference. Both classes, identical method
sets and distinct origin/license records remain mandatory. Cutoff stays .5;
no calibration or threshold fitting. Two math threads and four extraction workers.

Extract through `research_features.extract_features`; use shared `train_model`,
`predict_validation`, `export_onnx` and `diagnose_validation`. Publish every source,
method, pooled and declared-quality cell with paired-lineage 95% intervals;
unknown-quality cells stay unavailable. Compare reference/new metrics on exactly
the same ordered rows; raw corpus, weights and full feature caches stay local.

Independently check all copied JPEG hashes, all feature prefixes, a scalar
residual/parity oracle on actual JPEGs, train-only normalization, confusion/
pairwise AUC/ECE calculations and NumPy64/PyTorch/CPU ONNX outputs. Batch sizes
1, 17 and all validation rows must pass unchanged 1e-6 absolute tolerance,
zero relative tolerance and threshold decisions. Keep all failures/regressions.
Numerical export success never qualifies detection. Cross-source support is
unavailable without untouched sources and adequate original-lineage counts;
no automatic model installation or primary verdict change is permitted.
