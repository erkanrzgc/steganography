# ALASKA2 native-score baseline — 2026-10-04

**Outcome: the current automatic detector fails the requested accuracy gates.**
The full 4,000-file run completed without sample or analyzer errors, but it
missed almost every source-labeled stego at the frozen threshold. Acquisition
and evaluation infrastructure work; that is not successful steganalysis.

## Fixed evaluation

`ALASKA2_PROTOCOL.md` was committed in `4bf003a` before scoring. The analyzer
code is unchanged from `f929429`; the detector-source fingerprint matches the
preregistered value. Original 512×512 JPEGs were passed to the shared service,
balanced profile, threshold 70, AI off. No training, calibration, threshold
search or post-result detector adjustment occurred.

One thousand original lineages contain a cover and JMiPOD, JUNIWARD and UERD
variant each. Each method is compared with the same 1,000 covers. This is
4,000 files, not 6,000 independent observations or three independent sources.
Ground truth is the source's labels, not independently recovered hidden payloads.

## Detection results

| Method | TP / FN | Recall | Balanced accuracy | ROC-AUC (paired-bootstrap 95% CI) |
| --- | ---: | ---: | ---: | ---: |
| JMiPOD | 4 / 996 | 0.4% | 50.10% | 0.525617 (0.519404–0.531758) |
| JUNIWARD | 1 / 999 | 0.1% | 49.95% | 0.505157 (0.501588–0.509276) |
| UERD | 1 / 999 | 0.1% | 49.95% | 0.507181 (0.503577–0.511692) |

All three comparisons share TN=998, FP=2 and FPR=0.2%. A low false-positive
rate does not compensate for 99.6–99.9% of stegos being missed. The small
above-chance rank separation is far below the 0.90 ROC-AUC target; balanced
accuracy and recall also fail their 0.85 and 0.80 targets. No method is supported.

Intervals use the preregistered 200 whole-lineage bootstrap draws, seed 20261004.
They describe variation within this selection, not uncertainty across cameras
or new datasets. Complete confusion counts and intervals for AUC, balanced
accuracy, recall and FPR are in `benchmarks/alaska2-baseline-20261004.json`.
ECE remains unavailable: the scores are uncalibrated heuristics, not probabilities.

### Upstream ambiguity sensitivity

The three previously identified UERD/cover byte-identical pairs were retained
in primary results, with identical predictions. The declared sensitivity run
excludes both files of those three lineages: 997 pairs, TP=1, FN=996, TN=995,
FP=2, recall=0.1003%, balanced accuracy=49.9498%, AUC=0.507203
(95% CI 0.504095–0.511296). The ambiguity does not explain the failed detector.

## Coverage, timing and independent verification

All six preregistered native components returned `ok` on all 4,000 files.
ExifTool 12.76 was available and completed. Stegseek was unavailable; zsteg and
the non-JPEG analyzers were not applicable. Thus these are available native-score
metrics, not full-Docker or coverage-complete deployed-verdict accuracy. The
preexisting pipeline lacks an explicit per-format required-component policy;
its low-score `no_indicators` output must not be treated as proof of a clean file.

Wall time was 191.112 seconds with four workers. Per-file analysis median was
0.188 seconds and p95 0.211 seconds. Hardware: VMware guest exposing eight
vCPUs, AMD Ryzen 9 8945HX; Python 3.11.14, jpeglib 1.0.2, NumPy 2.4.6,
Pillow 12.2.0, CPU math threads one. This measures analysis on small JPEGs,
not CTF recovery latency or a dedicated reference-hardware performance gate.

A separate audit rechecked all 4,000 score identities against the source,
unique path membership, source/selection/protocol/runner/score hashes and the
three identical-pair predictions. It independently recomputed the four metric
sets using score-histogram cross-products (rather than the evaluator's rank
formula), plus all sixteen bootstrap intervals with explicit percentile
interpolation. They match. There was one complete evaluation attempt; no
samples, failed attempts or results were selectively removed.

Local evidence, not redistributed image data:

- `.benchmark/alaska2-baseline-20261004/run.json`: configuration and provenance.
- `.benchmark/alaska2-baseline-20261004/scores.jsonl`: all 4,000 per-file scores,
  status and signal codes; SHA-256
  `4fb057e0b9e5617cc97656ca18874a2f837af094ea6f2166d4c8a5cecf4db328`.
- `.benchmark/alaska2-baseline-20261004/report.json`: aggregate report; SHA-256
  `91e24a425b783df4096f77f1ed1c00aa5ae3ffbdfa6456b7ca03f63f9fa0b513`.

The versioned portable summary adds hardware and independent-audit notes; it
is not byte-identical to the local aggregate. Source images remain ignored.

## What follows

Keep this selection out of development, model selection and threshold tuning;
it is now an inspected baseline, not a fresh blind test. Separate, provenance-
checked development/validation images and a validated JPEG preprocessing/model
workflow are needed before claiming improvement. Freeze any trained model and
calibration before evaluation on a newly untouched source. Unknown payload
rates and camera/device ancestry prevent a per-payload or cross-source support
claim from these results alone. BOSSbase/Kodak and CTF reports remain unchanged.

Use the tool as an exploratory inspection/extraction assistant, not to certify
that a file contains no hidden data. This result is a clear measurement of the
current limitation, not evidence that more downloads alone improved detection.
