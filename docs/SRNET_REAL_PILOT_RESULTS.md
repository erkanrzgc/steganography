# First real-data SRNet pilot — failed detection gates

Frozen protocol: `SRNET_REAL_PILOT_PROTOCOL.md`, committed as `ff24d10` before
fitting. Portable evidence: [all cells and intervals](../benchmarks/srnet-real-pilot-20261006.json).
Raw corpus, numeric weights and per-file predictions remain local, not shipped
or installed. This is a completed engineering pilot, not a qualified detector.

## What actually ran

Unchanged ALASKA/BOSS-derived manifest and unrounded Y/256 float caches. One
complete BOSS-only epoch, 618 selected train rows / 103 original scenes,
412 matched updates, seed 20261012 and fixed Adamax parameters. ALASKA was
excluded from fitting. Fit completed in 229.29 seconds on the 8-vCPU/15-GiB
Linux VM with two math threads. Mean training-pair loss was 0.635814; this is
not held-out accuracy. Persisted BN counters and full model/card/plan hashes
passed verification.

All 765 validation rows received native inference. Nine metadata-selected
independent NumPy float64 forward comparisons passed the unchanged logit,
probability and decision gates. Maximum first-run logit difference 1.458e-5;
maximum probability difference 1.024e-11. Numerical correctness did **not**
imply useful classification. Full trained-model ONNX replay remains unavailable.

Two complete evaluation runs took 84.39 and 78.34 seconds. All 765 per-file
predictions, nine numerical audits, six cell metrics and bootstrap intervals
were identical across runs. This is repeatability, not accuracy or a CTF
latency claim; the published evidence retains both report identities.

## Measured context cells

Threshold fixed at 0.5; no validation threshold search/calibration. Every cell
has balanced accuracy **50%**, and every metric gate fails. Same-quality/source
covers are reused across method cells; these are not extra independent images.
ECE values are uncalibrated diagnostics, not meaningful probability guarantees.

| Validation source | Q | Method | Cover/stego | AUC | Recall | FPR | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BOSS-derived | 75 | JUNIWARD | 25/25 | 0.509600 | 24% | 24% | 0.499882 |
| BOSS-derived | 75 | UERD | 25/25 | 0.508000 | 24% | 24% | 0.499900 |
| BOSS-derived | 95 | JUNIWARD | 25/25 | 0.504800 | 24% | 24% | 0.499940 |
| BOSS-derived | 95 | UERD | 25/25 | 0.503200 | 24% | 24% | 0.499941 |
| ALASKA | unknown | JUNIWARD | 205/205 | 0.501927 | 28.78% | 28.78% | 0.491355 |
| ALASKA | unknown | UERD | 205/205 | 0.502035 | 28.78% | 28.78% | 0.491413 |

All bootstrap 95% intervals are published in the JSON (200 cover-lineage
resamples, fixed seed 20261005; all derivatives stay together). The tiny BOSS
cells have only 25 original validation scenes each; correlated quality variants
cannot be counted as independent scenes. No cell reaches the 1,000+1,000
sample-size gate. Previously inspected ALASKA exclusion is not blind/untouched
source validation. Camera/device independence and unknown payload/quality
metadata remain unavailable. No aggregate headline accuracy or supported-cell
claim is made, and previous failed baselines remain unchanged.

## Interpretation and next work

This one-epoch model is near chance and badly uncalibrated. Low training loss
did not translate into validation discrimination. The observations suggest
image/content-dependent behavior rather than useful stego separation; its cause
is not yet proven. A single deliberately short source-exclusion fit does not
establish that SRNet-style learning cannot work. It also cannot justify shipping
these weights or declaring the tool improved at detection.

Post-hoc diagnostics (not preregistered acceptance gates): all 255 declared
source/quality/lineage groups assign the same threshold decision to their cover
and both stegos. Of 765 scores, 526 are below 0.001 and 199 above 0.999. These
group counts include correlated quality variants, not 255 independent scenes.
This supports investigating score saturation/content-dependent behavior; it
does not prove that BatchNorm or any single implementation component caused it.

Next: inspect train-only learning curves and BN/input-scale behavior, then
freeze a new complete combined-source/compute control separately. Do not alter
this pilot, relax numerical tolerances, tune thresholds on these results, or
call reused validation an untouched test. A longer balanced fit, full ONNX
replay and truly independent licensed sources remain required.

## Reproduce locally

The matching existing float caches and completed numeric fit must be present.
Use the fixed checkout script (no downloads or fitting):

```sh
timeout --signal=TERM --kill-after=5s 1800s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python scripts/audit-srnet-real-pilot.py \
  --job .benchmark/srnet-real-pilot-20261006 --out FRESH_LOCAL_DIRECTORY
```

The script verifies the frozen protocol/plan/optimizer, runs full validation,
retains numerical failures, and writes portable aggregate evidence. Its local
evaluation config/per-file scores must not be copied into public reports.
Direct evaluation remains cooperative; this invocation adds an external hard
wall timeout, not a filesystem/network sandbox. Never execute extracted files.

Local verification: Python 3.11.14, 1,120 tests pass; total coverage 94.80%,
new diagnostic service 67/67 statements covered. Ruff, mypy (113 files),
whitespace and model/data-free wheel/sdist checks pass. The other supported
Python versions and fresh full Docker were not rerun locally. Regression
coverage is not detection accuracy, and the real pilot remains failed.
