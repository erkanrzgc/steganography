# PCM pair-control read-only diagnosis — 2026-10-22

**The cache contains useful plane-0 ranking signal, while the trained head
fails to exploit it.** The strongest supported optimization hypothesis is
unequal signal scale and competition between planes in the shared objective.
It is **not a proven causal diagnosis of the complete training trajectory**:
trained weights, score arrays, loss history and update-gradient logs were not
retained. No refit was performed to recreate them.

The [post-hoc contract](WAV_PAIR_DIAGNOSIS_PROTOCOL_20261022.md) and
[frozen inputs/audit hash](../benchmarks/wav-pair-diagnosis-protocol-20261022.json)
precede train-cache analysis. [All numeric findings](../benchmarks/wav-pair-diagnosis-results-20261022.json)
include all ten features /eight train cells and both initializations. The
[original failed experiment](WAV_PAIR_CONTROL_RESULTS_20261022.md) and all its
source bindings, thresholds, metrics and decisions remain unchanged.
[The exact frozen audit source](../benchmarks/wav-pair-diagnosis-audit-20261022.txt)
is archived for review; it adds no application service or training permission.

## What the measurements establish

1. **This is not absence of plane-0 signal in the feature cache.** The fixed
   first-quarter plane-0 feature has train AUC 0.909031 /0.7530625 on the
   sequential/scattered 40%-plane-0 cells. Its paired shift is positive in
   all 400 originals for both cells. The global plane-0 feature has train
   AUC 0.744294 /0.7038625; every corresponding original paired shift is also
   positive. The trained heads give only 0.540769 /0.517663 and
   0.545819 /0.495488 in those same train cells. Thus useful train ranking
   information survived extraction and FP32 caching; this does not prove
   sufficiency for native speech or a development pass.
2. **Raw signal scales differ materially.** In q0 the plane-1 mean paired
   shifts for 40% payload are 0.318052 /0.203449, versus plane-0
   0.085794 /0.055161: about 3.7×. Cover stds are 0.236286 versus 0.074592.
   Both inputs being in [0,1] does not equalize their variance or the relative
   optimization signal. After division by cover std, these illustrative
   shift/std values are 1.346/0.861 versus 1.150/0.740. This division is a
   descriptive calculation; no normalizer or transformed model was fitted.
3. **The shared objective is already plane-1 dominated at initialization.**
   The audit measures each balanced cell's gradient minus a duplicated-cover
   balanced null gradient. Plane-1 cells account for 93.67% /89.57% of the
   sum-direction projection at the two original initializations. Cross-plane
   contrast cosines include negative alignments (11/16 and 10/16 pairs).
   These are properties of the initial, unchanged heads, not percentages of
   training updates, gradient-energy shares or observations of final weights.
   They support a competition/scale hypothesis without establishing that it
   persisted throughout fitting.
4. **No source/split/order or exposure discrepancy was found.** Train cache
   and row hashes match; all 3,600 rows agree with the bound cover/recipe/source
   order. Each replayed seed has 1,140 batches, 72,000 presentations, 8,000
   presentations of each cover/recipe category and equal weighted epoch mass
   (1,800 cover /1,800 stego). Last batches contain 16 rows as specified.
   Four/five batches have no cover; all are last batches. This is fixed
   random-batch composition, not missing class exposure or proof of the cause.
5. **A time cutoff does not explain this pair-head failure.** Both actual
   training schedules completed and the original independent charged interval
   is 30.556292 seconds. Neither a slower GPU, extending the time limit nor
   restarting the old unavailable v3 run is supported as a remedy by this
   evidence. The original cached train/development metrics are close within
   each cell; they do not show a large observed train–development ranking gap.
   They do not establish natural-speech generalization.

The generator's unchanged `speech_like` construction applies a central
speech envelope plus Gaussian noise of 0.6–3 integer sample units before PCM16
rounding. The stronger bit-1 equalization signal is consistent with this
small-noise synthetic construction and its observed cover distributions.
That consistency is an interpretation, not an isolated generator ablation.
Likewise, untouched q2/q3 features for sequential payloads correctly produce
identical paired values and AUC 0.5; they are not an extraction malfunction.

## All train feature AUCs, fixed higher = stego

These are descriptive untrained feature scores, not newly qualified models.
All features and cells are retained; no best feature is selected for deployment.
No new development feature, prediction, direction or threshold was examined.

| Feature / historical head | Seq 5 p0 | Seq 5 p1 | Seq 40 p0 | Seq 40 p1 | Scatter 5 p0 | Scatter 5 p1 | Scatter 40 p0 | Scatter 40 p1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global.pair_equalization.p0 | 0.557672 | 0.513306 | 0.744294 | 0.535844 | 0.526563 | 0.507400 | 0.703862 | 0.532181 |
| global.pair_equalization.p1 | 0.503181 | 0.578944 | 0.508519 | 0.850838 | 0.501575 | 0.535450 | 0.508219 | 0.781469 |
| q0.pair_equalization.p0 | 0.644381 | 0.527100 | 0.909031 | 0.554409 | 0.531666 | 0.508706 | 0.753062 | 0.539863 |
| q0.pair_equalization.p1 | 0.506088 | 0.686362 | 0.512487 | 0.999700 | 0.502131 | 0.540456 | 0.508644 | 0.807037 |
| q1.pair_equalization.p0 | 0.500000 | 0.500000 | 0.512806 | 0.500456 | 0.501900 | 0.499856 | 0.509694 | 0.500075 |
| q1.pair_equalization.p1 | 0.500000 | 0.500000 | 0.499431 | 0.543306 | 0.499825 | 0.504519 | 0.501131 | 0.527613 |
| q2.pair_equalization.p0 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500850 | 0.500822 | 0.507637 | 0.501106 |
| q2.pair_equalization.p1 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.499800 | 0.504331 | 0.502806 | 0.529631 |
| q3.pair_equalization.p0 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.532444 | 0.507975 | 0.751469 | 0.539587 |
| q3.pair_equalization.p1 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.502094 | 0.539931 | 0.509156 | 0.811612 |
| historical head 20261016 | 0.517219 | 0.738537 | 0.540769 | 0.999781 | 0.502881 | 0.537769 | 0.517663 | 0.790031 |
| historical head 20261017 | 0.520238 | 0.714256 | 0.545819 | 0.996506 | 0.499213 | 0.537169 | 0.495488 | 0.799175 |

The q0 example is a post-hoc illustration within this complete table. Its
0.909 figure is **training AUC**, not accuracy or a new held-out result.
The historical fixed global-p0 development comparator remains exactly
0.728675 /0.695350 on the two 40%-p0 cells; the latter is below the old 0.70
criterion. It does not rescue either failed head. The q0 feature was not newly
scored on development data.

## Cover scale and sample-length association

Pearson r uses original covers only. Length affects histogram occupancy and
may act as a nuisance; correlation alone does not establish causality. Paired
cover/stego comparisons share exactly the same original length.

| Feature | Cover mean | Cover std | Cover/frames Pearson r |
|---|---:|---:|---:|
| global.pair_equalization.p0 | 0.906176 | 0.042945 | 0.181375 |
| global.pair_equalization.p1 | 0.787364 | 0.124083 | 0.067715 |
| q0.pair_equalization.p0 | 0.886842 | 0.074592 | 0.051972 |
| q0.pair_equalization.p1 | 0.654430 | 0.236286 | 0.015352 |
| q1.pair_equalization.p0 | 0.817583 | 0.064677 | 0.370624 |
| q1.pair_equalization.p1 | 0.809982 | 0.067198 | 0.369116 |
| q2.pair_equalization.p0 | 0.817597 | 0.066234 | 0.366126 |
| q2.pair_equalization.p1 | 0.809477 | 0.067883 | 0.368721 |
| q3.pair_equalization.p0 | 0.886491 | 0.074958 | 0.037790 |
| q3.pair_equalization.p1 | 0.654629 | 0.233500 | 0.016278 |

## Zero-update checks and limitations

The audit opens only the bound train NPY and metadata; development NPY,
all raw/native/reserved media and all model checkpoints remain closed.
Optimizer construction and fitting entry points are explicitly blocked.
All 418 parameters in both reconstructed initializations remain identical
under `torch.equal` checks against their pre-analysis copies.
The 80 feature/cell AUCs agree exactly between independent all-pairs half-tie
counts and independent average-rank sums, including tied scalar fixtures.
Every historical global-p0 train comparator value also replays exactly.
An additional 100 tied-score random fixtures agree with independent scalar
Python pair counting; four empty/nonfinite cases are rejected. Two Pearson
fixtures pass with a 1e-15 absolute tolerance. The initial exact-equality
Pearson fixture returned 0.9999999999999998; its failure was retained before
correcting the fixture tolerance. The frozen audit and its results were not
changed or rerun.

The frozen weighted full-cache gradient equals the mean of the eight
balanced cell gradients, with maximum absolute errors
1.32742571e-07 /4.95929271e-08.
This checks the objective's mathematical class/recipe accounting. It does not
reconstruct actual historical minibatch gradients or prove that AdamW should
converge within the frozen exposure.

The audit completed in 1.635411 worker seconds /2.042527 independent systemd seconds, with 238,497,792 kernel cgroup peak bytes and zero swap. The unit enforced 2 GiB, 200% CPU, 120 seconds and no restart. No optimizer update, download, checkpoint, new learning result or qualification occurred.

The unchanged application sources retain their prior software gate:
3,598 passed /two explicit CUDA skips, 96.96847% total coverage. This diagnosis
does not claim a new full test run; its numerical checks are the independent
AUC/gradient/order/hash identities above. The audit source and every input are
bound before and after execution. Missing trained-state observability remains
unavailable; logs were not manufactured from a refit.

## Next action

The [next-intervention draft](WAV_PAIR_NEXT_INTERVENTION_DRAFT_20261022.md)
proposes a matched raw-versus-train-standardized head comparison with complete
numeric learning telemetry, preserving representation, architecture, seeds,
exposure and acceptance. It is a hypothesis, **not executable permission**.
Changing multiple model/data/loss variables or increasing epochs now would
prevent this diagnosis from identifying the effective intervention.

The pair statistic is still lossy and can merge distinct one-LSB inputs when
pair counts swap. The generated family is not native speech. Any next fit
requires a separately frozen complete protocol; real AudioMNIST additionally
requires explicit approval and an accepted full generated control. The old
raw-transfer ledger/parent-recording ancestry limitations are not repaired or
reset by this cache-only diagnosis.
