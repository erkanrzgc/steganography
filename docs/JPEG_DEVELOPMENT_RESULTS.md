# JPEG development result — 2026-10-04

The first trained baseline is **not suitable for deployment**. It shows some
same-source ranking signal, but produces 85 false alarms among 205 validation
covers (41.46%). It is not a new test-set or cross-source accuracy result, and
the primary analysis service has not changed to use it.

## Data isolation and fixed experiment

The [development protocol](JPEG_DEVELOPMENT_PROTOCOL.md) and implementation
were committed in `b1d6050` before the completed training run. A separate
1,000-lineage ALASKA2 acquisition, seed 20261005, contains 4,000 original JPEGs
(421,459,421 bytes). The previous ALASKA2, BOSSbase and Kodak identities and
lineages are reserved, not train/validation inputs. Acquiring a second subset
of ALASKA2 does not create a second independent dataset source.

Six source-labeled stegos are byte-identical to their own covers: UERD
08136, 25034, 25364, 72724 and 73579; JUNIWARD 30862. The predeclared rule
quarantined all four variants of each affected lineage before feature
extraction. The acquisition retains the originals. The resulting development
manifest contains 3,156 training images (789 complete lineages) and 820
validation images (205 lineages). No score-based exclusions occurred.

One fixed CPU run used `jpeg-dct-summary-v1`: 968 luminance DCT histogram,
co-occurrence and quantization features; train-only standardization; a
class-balanced linear logistic model; seed 20261005; Adam 0.01; 300 epochs.
This is not SRNet or DCTR. No final-test prediction or calibration was run.

## Same-source validation, not held-out-source performance

Each method below uses 205 stegos and the **same** 205 covers. There are 820
unique validation files, not 1,230 independent files.

| Method | TP / FN | Balanced accuracy | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| JMiPOD | 139 / 66 | 63.17% | 67.80% | 0.648804 |
| JUNIWARD | 108 / 97 | 55.61% | 52.68% | 0.592481 |
| UERD | 110 / 95 | 56.10% | 53.66% | 0.585306 |

All comparisons share TN=120, FP=85, FPR=41.46%, sigmoid threshold 0.5.
The machine-readable metrics express scores on a 0–100 scale with threshold
50. Sigmoid outputs are uncalibrated scores, not validated probabilities.
Confidence intervals were not computed for this development run. The final
support gate still requires a sufficiently sized independent test and 95% CIs.

These numbers cannot be subtracted from the frozen native-detector test scores
to claim an accuracy improvement: the model, image selection and decision
threshold differ. All measured validation metrics miss their corresponding
numeric acceptance targets, and cross-source qualification remains unavailable.

## Verification and reproducibility

Independent read-only recomputation checked reserved-corpus and train/validation
identity isolation, feature-row/prediction membership, exact train-only
normalization and PyTorch predictions. Pairwise positive/negative comparisons
independently reproduce AUC and all confusion counts.

ONNX Runtime on the actual 820 validation rows differs from PyTorch by at most
`4.76837158203125e-6` in logits and `1.1920928955078125e-6` in sigmoid scores.
The strict absolute score tolerance `1e-6` with zero relative tolerance **fails**;
all 820 threshold decisions agree. The small-fixture ONNX tests pass, but do
not substitute for this actual-artifact check. No tolerance was relaxed and
this export is not promoted to a signed or deployed model pack.

The complete runner took 129.65 seconds on a VMware guest exposing eight
Ryzen 9 8945HX vCPUs, without CUDA. Four feature workers and one CPU math
thread were used. This includes import, features, training, validation and
export; it is not an inference-latency benchmark. Package versions and artifact
hashes are in [the portable summary](../benchmarks/jpeg-development-20261004.json).

The acquisition was interrupted after 3,456 files and resumed with the same
selection and integrity checks. The first experiment invocation rejected a
mistyped source checksum before reading image bodies or training. Its empty
output directory remains local; `run2` denotes the first **completed** training
run, not a selected best model among multiple experiments.

After explicit authorized development acquisition, reproduce using a fresh output:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m steganography.research_jpeg run-development \
  --source .benchmark/alaska2-development-20261004 \
  --source-sha256 f54f1f66c366a5e57c4ab65214a211e807ac4a0d0a5a1a549e736b34c2bbe288 \
  --reserved-manifest .benchmark/alaska2-holdout-20261004/source.json \
  --reserved-manifest .benchmark/boss-pilot-1000/source.json \
  --reserved-manifest .benchmark/pilot-v1/manifest.json \
  --reserved-manifest .benchmark/kodak-20261001/source.json \
  --reserved-manifest .benchmark/kodak-pilot-20261001-serial/manifest.json \
  --out .benchmark/jpeg-development-reproduction
```

Keep raw images, feature matrices and checkpoints local pending their separate
license/distribution review. A stronger feature/model experiment needs a new
documented development decision, not another threshold on the inspected test.

The subsequent [opt-in numerical repair](INFERENCE_PRECISION_RESULTS.md) passes
the unchanged strict gate on all 820 validation rows for a derived checkpoint.
Original weights/training data remain unchanged; this does not improve detection
accuracy or overwrite the original failed export and published report.
