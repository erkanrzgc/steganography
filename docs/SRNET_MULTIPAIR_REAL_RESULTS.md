# Two-source SRNet real control — detection gates failed

Frozen [protocol](SRNET_MULTIPAIR_PROTOCOL.md), commit `837cb79`, unchanged.
[All cells, intervals and provenance](../benchmarks/srnet-multipair-real-20261007.json)
and [independent scalar metric audit](../benchmarks/srnet-multipair-metric-audit-20261007.json)
are public; images, float caches, numeric weights and per-file scores stay local.
No model is installed or deployed. The previous failed BOSS-only pilot is unchanged.

## Complete fitting and verification

Both declared sources, 2,985 train rows, one complete epoch, seed 20261012.
All 3,160 source-balanced pairs / 6,320 presented rows / 1,580 four-row updates
completed. Fit took 1,787.13 seconds (29m47s) on the 8-vCPU Linux VM, two math
threads, within the frozen budget. Mean training batch loss: 0.694231; not
accuracy and not directly comparable to the prior differently scoped pilot.
Some regression tests ran concurrently; this is not an isolated throughput
benchmark or a CTF latency claim. Actual worker CPU/address/file/core limits
were independently observed read-only while fitting.

All 765 reused development-validation rows were evaluated in standard stored-BN
eval mode, in 77.16 seconds. Nine independent NumPy replays passed unchanged
logit/score/decision gates: maximum logit difference 2.94e-7, maximum probability
difference 1.07e-7. Model/card/plan/cache hashes and BN update counts verified.
Independent pairwise AUC/direct confusion/bin arithmetic also agrees for all
six cells: maximum rounded-metric difference 4.70e-7, tolerance 1e-6, exact
confusion counts. Audit correctness is not detection qualification.

Independent scalar replay (use a fresh output):

```sh
venv/bin/python scripts/verify-srnet-multipair-results.py \
  --evidence benchmarks/srnet-multipair-real-20261007.json \
  --predictions .benchmark/srnet-multipair-real-20261007/evaluation/predictions.json \
  --out .benchmark/srnet-multipair-real-20261007/metric-replay.json
```

This reads local per-file scores, not weights or images; no fitting/deployment.

## Fixed-threshold cells

Threshold .5; 200 cover-lineage bootstrap resamples and all 95% intervals are
in JSON. All metric gates fail. BOSS quality variants are correlated views,
not additional independent scenes; ALASKA quality/payload remain unknown.

| Source | Q | Method | Cover/stego | AUC | Balanced accuracy | Recall | FPR | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOSS-derived | 75 | JUNIWARD | 25/25 | .507200 | .520000 | .760000 | .720000 | .060205 |
| BOSS-derived | 75 | UERD | 25/25 | .505600 | .520000 | .760000 | .720000 | .060348 |
| BOSS-derived | 95 | JUNIWARD | 25/25 | .500800 | .520000 | .760000 | .720000 | .054325 |
| BOSS-derived | 95 | UERD | 25/25 | .500800 | .500000 | .720000 | .720000 | .063662 |
| ALASKA | unknown | JUNIWARD | 205/205 | .503343 | .502439 | .409756 | .404878 | .067736 |
| ALASKA | unknown | UERD | 205/205 | .502391 | .502439 | .409756 | .404878 | .067855 |

## Interpretation and next gate

Ranking remains near chance; **no meaningful stego discrimination gain**.
Recall rises alongside false positives. BOSS FPR rises from .24 to .72;
ALASKA FPR from .287805 to .404878. Do not headline recall alone as improvement.
Overconfidence decreases (ECE about .49–.50 before, .054–.068 now), but the
scores are uncalibrated and still fail the fixed gates. Training sources, BN
context and update counts all changed, so this is not an isolated causal BN
comparison. Slight point changes on tiny/reused cells are not blind evidence.

All sample-size/untouched-source gates remain unmet; camera/device independence
is unverified. Full trained-model ONNX replay is still unavailable. Keep the
wheel model-free and the original failures public. Do not tune thresholds or
choose new settings on this development validation.

Next freeze a train-only, metadata-selected tiny-subset learning/overfit sanity
control before spending on longer training. Check whether the same objective
can learn matched pairs and preserve ordinary single-file eval; in-sample
sanity figures cannot count as accuracy. A positive sanity check must still be
followed by separately frozen adequate training and genuinely unseen sources.
GPU availability/longer compute are separate constraints, not guarantees.

Verification: Python 3.11, 1,165 tests passed, total coverage 94.89%; Ruff,
mypy (116 files), diff checks and wheel/sdist builds passed. Published evidence
tests bind original protocol/card/prior/metric-audit hashes and keep failed
gates visible. Other Python versions/full Docker were not rerun here.
