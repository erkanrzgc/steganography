# Frozen residual-domain intervention — 2026-10-06

Commit implementation and this protocol before any new real fitting/scoring.
The previous three-fit CNN failed all twelve cells; retain its complete public
record. This follow-up was chosen after inspecting those failures and is an
exploratory development hypothesis, not an untouched confirmatory test.

## Hypothesis and sources

The [normalization sensitivity study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8444093/)
reports that image/filter normalization changes learning on spatial BOSS/WOW
experiments. This motivates testing the residual domain, not borrowing its
scores or claiming the finding transfers to our JPEG task. No upstream code,
weights or trained architecture copied. This custom network is not SRNet/YeNet.

Change **only the fixed residual stem contract**: architecture
`jpeg-center128-pixel-residual-cnn8-v1` receives unchanged float32 raw/255 inputs,
but stores three fixed original high-pass filters multiplied by 255 and clips
their responses to [-3,3]. This approximates pixel-unit residual convolution
subject to float32 arithmetic. Original architecture stores unscaled filters
and clips [-1,1]. Scaling and clipping change together; this cannot isolate
which of these two choices causes an effect. Do not change crop, decoded color,
learned layers, pooling, classifier capacity, loss, source weights or optimizer.

Same 3,729 learned parameters, no batch norm, metadata, reference cover,
augmentation, pretrained weights, architecture search or trainable high-pass.
Same seed gives identical initial learned weights in both architectures.
Fixed filter values are persisted and bound to architecture before inference:
neither old nor new weights can load under the other's contract. Legacy default
architecture/inference/model arrays remain unchanged and compatible.

## Fixed runs

Same manifest SHA
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`,
same frozen uint8 caches and original split/order/lineages as the first protocol.
Exactly three independent fresh fits: all (2,985 rows), BOSS (618), ALASKA
(2,367). Seed 20261011, ten epochs, batch32, two CPU threads, Adam LR .001,
weight decay .0001, per-fit deadline 1,800 s; previous source/class weighting.
No warm-start, tuning, thresholds, calibration, retries or best-run selection.
Publish failures/timeouts as failures. Raw caches/models remain local/ignored.

Train via existing explicit `research pixel-cnn` service with `architecture`
set to the new allowlisted ID. All original parsing/size/symlink/deadline/RNG
guards remain. Base wheel model-free, Torch optional; no auto downloads.

## Complete reporting and numeric replay

All 12 fit × validation-origin × JUNIWARD/UERD cells at inclusive cutoff .5.
Full confusion/AUC/BA/recall/FPR/ECE, original-lineage bootstrap200 with seed
20261005 and correlated BOSS quality variants retained. Same-fit comparisons
against the first CNN plus the previous all-source JRM reference; paired
intervals and every regression retained, not a causal JRM-vs-CNN comparison.
Bind prior public records and all prior predictions by SHA before comparing.
First CNN public record SHA:
`a10b916f815238e167cc9c166f1d851aea2a98b70a8c14bc6195c46f4e25b5d1`.

Rehash all 3,750 originals, replay all 765 rows per fit through independent
NumPy convolution/pooling, persisted native weights and ONNX CPU batches
1/17/64. Frozen tolerance 1e-6 absolute / zero relative and exact cutoff
decisions; missing dependencies unavailable, failed parity not waived. Metric
audit separate from detection gates (.90 AUC/.85 BA/.80 recall/.03 FPR/.05 ECE).

No new external source has been acquired. Reused inspected validation, unknown
camera/device/app/ALASKA quality/payload, BOSS simulation and only 25 validation
scenes, excluded JMiPOD: qualification stays unavailable regardless of scores.
Both source exclusions diagnostic only; counts/optimization steps differ.
No model catalog, supported method, primary detector or API behavior change.
Untouched licensed external-source evaluation remains required before claims.
