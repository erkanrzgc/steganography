# Frozen SRNet real-data engineering pilot — 2026-10-06

Freeze and commit this protocol before fitting or inspecting its predictions.
It is a compute/learning/forward pilot, not a paper reproduction, qualified
detector, hyperparameter selection or claim of real-world generalization.
Do not revise parameters or numerical tolerances after seeing results.

## Inputs and source exclusion

- Manifest SHA-256:
  `0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
- Complete train float256 cache SHA-256:
  `828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`;
  data `fd27c6da9869f4bea8bf7b97edeffd8ebae519824cd23d15b029aba25ceaf392`.
- Validation cache SHA-256:
  `1c8c25244f204aaa0cad981d032241c06f7535b31bd03d7c547818dc046ae3a6`;
  data `81eecbd2b6bb2c7a54593dd86b89a1b7493b967955f844d1c4d843671913438a`.
- Architecture `srnet-gray12-cpu-v1`, exact unrounded component-Y float32
  units/crop from the existing frozen float256 protocol, no normalization.
- Select only `source-6a23b26e6f51f8cc` (BOSS-derived train). Exclude ALASKA
  source `source-d2a1dfbee30c5891` from training, without removing it from the
  complete verified cache. This is a bounded initial source-exclusion control,
  not a substitute for subsequent combined-source and ALASKA-only training.
- Selected 618 rows, 103 original train scenes, declared Q75/Q95, cover plus
  JUNIWARD/UERD. One complete epoch: 412 matched pairs, each original stego
  paired once, 103 updates per quality/method cell. No dropped tail/oversampling.

## Fixed fit and operational limits

Prepare a new one-epoch checksum-bound plan with seed 20261012. Its epoch-zero
pair order must equal the earlier unchanged BOSS schedule:
`57affc332b551ff55afe45bfeff890daf53c7f20201d2a0b5fcd29276fbe8122`.
Use CPU float32, two threads, Adamax lr 0.001, weight decay 0.0001,
betas [0.9,0.999], epsilon 1e-8, foreach false, constant learning rate.
The isolated fit has a 1,800-second optimizer deadline, 1,920-second hard wall
deadline including preparation, 8 GiB address limit, CPU limit 3,660/3,661
seconds, 32 MiB per file and no core dumps. No early stopping, checkpoint
selection, restart, augmentation, validation training or parameter changes.
Timeout/failure is a failed pilot with no usable model, not a truncated epoch.
Record actual runtime/package versions; reference environment is the current
8-vCPU/15-GiB Linux VM, not a portable performance promise.

## Validation, numerical gates and reporting

Only a complete fit can enter evaluation. Verify full training provenance,
numeric model hash and all BN counters (412). Evaluate all 765 unchanged
validation rows (ALASKA 615, BOSS 150), native batches at most four. Independently
replay the first manifest row per declared source/quality/label/method cell
with NumPy float64, following `SRNET_EVALUATION.md`. Logit abs/relative limits
1e-4/1e-5, softmax difference 1e-6 and identical decisions remain unchanged.
A failed gate retains audit evidence but publishes no eligible predictions.
Evaluation is launched with an additional external 1,800-second hard wall
timeout; direct service checks alone remain cooperative. No forced success.

If native/NumPy gates pass, summarize the six source/quality/method cells using
their same-source/quality covers, threshold 0.5, ties stego, uncalibrated
class-one softmax. Publish AUC, balanced accuracy, recall, FPR and 10-bin ECE;
use the existing 200 lineage-group bootstrap replicates and fixed seed from
`research_spatial.paired_intervals`, without recommending/tuning thresholds.
Preserve all failing cells. Sample-size/source-independence release gates
cannot pass this pilot, even if point estimates look good.

Optional ONNX export and complete validation replay are separate readiness
checks under the same fixed comparisons. Absence/failure is unavailable/failed,
not passed, and cannot qualify a model. Raw corpus, model and per-file
predictions stay local/ignored; publish portable provenance, counts, failures
and aggregate results only. Models remain non-installed/non-deployed.

## Known limitations and next decision

Both source datasets and validation results have already been inspected during
development. ALASKA training exclusion is not untouched/blind test evidence.
BOSS positives are simulated; original scenes are small, camera/device/app
independence is unverified; ALASKA quality/payload rates are unknown. One epoch
is an engineering pilot and may badly underfit; do not treat it as the final
SRNet baseline. Retain all previous failed baselines. Report numerical or
runtime failure honestly, then preregister the next compute/training experiment
separately. No automatic dataset/model downloads or primary verdict changes.
