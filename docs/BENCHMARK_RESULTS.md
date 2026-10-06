# Measured results, not a global accuracy score

[First real pixel-CNN trial](PIXEL_CNN_TRAINING_RESULTS.md): three fixed fits,
12/12 detection cells fail (AUC .502–.509, zero recall). Complete independent
numeric/ONNX replay passes; model not deployed, qualification unavailable.

[Pixel-residual preparation](PIXEL_RESIDUAL_PREPARATION_RESULTS.md): all 3,750
unchanged files rehashed, 18 isolated crop oracles exact and custom CNN gradient
smoke passes. Preparation alone has no accuracy result; qualification unavailable.

The [two-way JRM training-source exclusion diagnostic](JRM_TRANSFER_RESULTS.md)
finds ALASKA → BOSS AUC .534/.522 and BOSS → ALASKA .548/.554. All eight
cells, including within-origin controls, fail. Source transfer remains weak;
inspected development data are not blind/untouched-source qualification.

The optional [JRM + FLD reference](JRM_REFERENCE_RESULTS.md) improves ALASKA/
UERD AUC .603 → .698 and FPR 38.54% → 31.71%. JUNIWARD recall falls to
47.32%; BOSS ECE worsens and detection fails in all measurable cells. Numeric
vote parity passes, independent-source evidence is unavailable; not deployed.

The [nonlinear residual-feature comparison](JPEG_NONLINEAR_RESULTS.md) improves
ALASKA AUC to .609/.603 and FPR to 38.54%, but still fails detection.
BOSS/UERD and ALASKA calibration regress; no deployment or independent-source
qualification. All cells and paired changes remain published, not cherry-picked.

The [same-file JPEG residual/parity comparison](JPEG_RESIDUAL_RESULTS.md) adds
DC/neighbor statistics, but still fails: ALASKA FPR 46.34%, BOSS simulations
40%, AUC .503–.573. Recall regresses in several cells; all intervals and failures
are published. Numeric export passes; no deployment or independent-source support.

As of 2026-10-05, **no method has passed the cross-source support gates**.
The application is an exploratory inspection/extraction tool, not a reliable
certificate that a file is clean. More formats, datasets or passing unit tests
do not imply better detection.

The [two-origin measured JPEG context experiment](JPEG_CONTEXT_RESULTS.md) is
complete: 3,750 development JPEGs, 1,098 features, matched JUNIWARD/UERD methods.
It **regresses** ALASKA AUC and raises FPR to 50.24%; BOSS simulations have 48%
FPR and near-chance AUC. Numerical export passes, detection does not. Both sources
are shared with training; JMiPOD is excluded and no model is deployed.

[Source/context audit](GENERALIZATION_RESULTS.md): both current model
validation sets contain zero unseen source groups. PNG/BMP low-rate failures
remain visible by format; JPEG reproduces its previous weak ALASKA2-only
development scores. Missing quality/camera/device metadata cannot demonstrate
context awareness. This adds diagnostic controls, not detection improvement.

The fixed low-payload training-weight comparison raises same-source 5% recall
from 38.04% to 67.93% (sequential) and 14.67% to 29.89% (scattered), while
shared false alarms rise from 1.63% to 3.26%. Higher-rate ranking and most ECE
values regress. All six cells remain experimental and undeployed; see
[full results](SPATIAL_WEIGHTING_RESULTS.md) and
[portable evidence](../benchmarks/spatial-weighted-development-20261005.json).

| Scope | Evidence | Outcome |
| --- | --- | --- |
| PNG/OpenStego and BMP/Steghide native detection | [Frozen BOSSbase pilot](PILOT_RESULTS.md): 1,000 pairs, 500 per family | Combined AUC 0.499021; recall 0%, FPR 0% at threshold 70; failed |
| PNG/BMP small second-source check | [Kodak](KODAK_RESULTS.md): both methods on 24 originals | AUC 0.49349; recall 0%; insufficient independent sample size and failed detection |
| JPEG JMiPOD / JUNIWARD / UERD native detection | [Frozen ALASKA2 baseline](ALASKA2_RESULTS.md): 1,000 pairs per family, shared covers | AUC 0.525617 / 0.505157 / 0.507181; recall 0.4% / 0.1% / 0.1%; failed |
| Experimental trained JPEG model | [Separate development validation](JPEG_DEVELOPMENT_RESULTS.md): 205 pairs per family, shared covers | AUC 0.648804 / 0.592481 / 0.585306; FPR 41.46%; not deployed; not an independent test |
| Experimental trained spatial model | [Controlled LSB development](SPATIAL_DEVELOPMENT_RESULTS.md): 184 pairs/cell, six method/rate cells | AUC 0.716801–0.997460 versus same-data reference 0.516422–0.648423; FPR 4.89%, low-rate scattered recall 19.57%; not deployed; support/export gates failed |
| Experimental spatial parity/residual model | [Same-row iterative comparison](SPATIAL_PARITY_RESULTS.md): all six cells, 184 pairs each | Scattered-5% AUC 0.906486, FPR 1.63%; low-rate recall regresses to 14.67%, calibration fails; stable export passes; not deployed or qualified |
| WAV sample-LSB replacement | [FSDD baseline](WAV_RESULTS.md): 1,000 test covers, 6,000 stegos; sequential/scattered at three rates | All six cells: AUC 0.50, recall 0%, FPR 0%; failed. Generation oracle verified every payload; native CTF recovery unmeasured |
| GIF, text, PDF, containers, MP3/TIFF-specific claims | Unit/integration and generated examples | Representative labeled corpus evaluation **unavailable**; no real-world score |
| Controlled CTF extraction | [Kodak regression](CTF_REGRESSION_20261002.md): two final 30/30 completed exact recoveries | Published-challenge regression, not blind recovery or automatic detector accuracy |
| Blind CTF and release qualification | Planned 120 challenges and independent-source gates | **Unavailable** |

Native baselines and the trained JPEG validation use different samples/models/
thresholds. Their numbers are **not** a before/after accuracy comparison. Refer
to each report for environment, confusion counts, uncertainty where computed,
coverage and limitations. Scores are not calibrated probabilities.

## Next measured development slices

The [numerical repair and operating-point follow-up](INFERENCE_PRECISION_RESULTS.md)
passes export gates for opt-in derived checkpoints; historical failures above
remain accurate for the originals. Threshold fitting on 94 covers and separate
assessment on 90 previously inspected lineages changes FPR from 3/90 to 2/90,
but lowers weak-cell recall. It is not probability calibration or deployed
detection improvement; old primary detector results are unchanged.

1. PNG/BMP: the [parity/residual comparison](SPATIAL_PARITY_RESULTS.md)
   improves ranking and false alarms on the same 7,000 controlled files but
   loses fixed-threshold low-rate recall. Next improve low-rate sensitivity,
   false positives and calibration, qualify stable exports on target runtimes and evaluate a newly
   untouched independent source. Named upstream methods require separate tests;
   do not train on the published pilot or claim this generic LSB result for them.
2. JPEG: use the new development split for stronger residual/co-occurrence
   features or a suitable trained architecture. Freeze calibration and model
   before a newly untouched source; keep this failed linear baseline visible.
3. WAV: use the separately reserved training/validation speakers for richer
   features/models; preserve the failed six-cell baseline and inspect false
   positives on validation. Acquire another untouched audio source for testing.
4. GIF/text/container: first establish reproducible independent method fixtures
   and representative benign corpora, then method-specific exact recovery and
   false-positive measurements. A decoder solving base64 is not generic detection.

These are remaining gates, not deployed improvements. The same rules apply
to each new method: publish failures, preserve frozen tests, and promote support
only after the [acceptance gates](BENCHMARK_PROTOCOL.md) actually pass.

## Stegseek availability is environment-specific

The host has no `stegseek` executable, which explains the frozen native baseline's
`unavailable` coverage. Existing full Docker image `e2c255ae0709` contains
StegSeek 0.6. A 2026-10-04 network-disabled, non-root, read-only-root check used
upstream Steghide on a generated BMP, a single known fixture-password wordlist,
and Stegseek cracking; exit status was zero and payload recovery was exact.

That verifies one controlled Steghide integration, not blind cracking or all
JPEG methods. [Stegseek's upstream scope](https://github.com/RickdeJager/stegseek)
is Steghide; installing it does not repair the JMiPOD/JUNIWARD/UERD classifier.
No real wordlist was bundled, no host installation occurred, and the full image
was not rebuilt in this slice.
