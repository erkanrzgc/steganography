# nsF5 calibrated coefficient relations (2026-11-15)

The user's current research/fix request permits this fresh CPU intervention
on existing first-3,000 BOSS training JPEGs. No consumed fit is restarted,
payload changed, media/model downloaded, GPU used, native role altered or
locked ALASKA2 accessed. All prior misses and failures remain unchanged.

The [fit/calibration diagnosis](LOW_SIGNAL_DIAGNOSIS_PROTOCOL_20261115.md)
replays eleven old feature rows and every sampled nsF5 change count exactly.
At .05/.1 bpnzac the original simulator changes only about .56/1.30% of
nonzero AC coefficients. DCTR-only classifier changes did not meet 80%.
This intervention tests coefficient-domain rather than spatial-residual
relations, motivated by [JRM](https://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf)
and [calibration](https://ws2.binghamton.edu/fridrich/Research/calibration-color.pdf).
It is a compact adaptation, not a reproduction of full JRM or published
accuracy. Use already-installed sealwatch 2024.12; no package install.

## Features and source roles

Read only bounded regular existing Q75 gray JPEG covers (<=2 MiB,
512x512 /64x64 DCT blocks), preserving all original source bytes. Replay
unchanged nsF5 seeds [2026111206,2 or 3,role,index] and .05/.1 payloads.
Every changed count must match the existing cache; DC and preexisting zero
coefficients stay unchanged and decrements reduce absolute value by one.
Keep first-3,000 roles: 2,400 fit rows (index mod 5 !=0), 600 calibration,
with pixel-disjoint 1,000 historical test sources checked from the old split.
Test byte integrity may be bound early; no test features precede choice freeze.

Per file, extract the installed JRM absolute-value subset (2,512 values)
and each of 63 AC-mode absolute histograms (eight bins, values 0..6 and >=7),
504 more. A separate calibration reference uses the installed source-bound
libjpeg decode/crop/recompress procedure, cropping four pixels from all four
edges, with identical quantization table. It yields 63x63 reference blocks,
avoiding the previous 508-pixel edge padding. Store 3,016 own plus 3,016
reference features and 64 signed least-squares decrement estimates.
For each AC mode, use reference histogram direction [h1,h2-h1,...,h5-h4]
and the own-minus-reference probabilities in bins 0..4; estimate their
projection divided by squared direction norm, zero for norm <=1e-12.
The last estimate pools numerators/denominators over all modes. Thus 6,096
float32 values; no paired clean image, key or true payload is a query input.

## Fixed models and selection

For each unchanged payload, fit ten candidates: three Fisher ridges
.03/.3/3 on the best 1,024 fit-only Fisher-score columns; six Gaussian
ridge-.01 models (width factors .25/1/4) on plain square-root 3,016 features
without whitening or signed-square-root calibrated differences/estimates
with exact float32 StandardScaler arithmetic; and a 300-iteration /15-leaf
/lr=.05 /l2=1 /seed-zero histogram tree on 256 fit-only Fisher-score columns.
Feature-score ties use original index; columns remain sorted for inference.
Kernel width uses 256 fixed fit covers, seed 2026111523, direct squared
pairwise distances, zero <1e-12 excluded. Cholesky residual <=1e-6; exported
scores must replay within absolute/relative 1e-7, tree probabilities 1e-10.

Choose the new model on calibration BA/AUC/order. Compare it against the
already-frozen fair-fit DCTR kernel, and fixed new/reference score mixtures
with new weight .25/.5/.75, standardized using calibration covers only.
Select one BA threshold with exact-count closest-to-zero/lower tie breaks.
Freeze both payload choices globally. Only a selected calibration BA >=.80
permits new historical test feature construction/inference; otherwise record
`not_evaluated_calibration_gate` and retain the old measured test miss.
If the old reference wins, reuse its retained test result without inference.
No test-selected substitution or post-test adjustment is allowed. Report all
calibration misses, test metrics, TPR/FPR and paired-source BA/gain intervals
(2,000 draws, seed 2026111524); repeated historical testing is adaptive.

## Execution and closing

One exclusive fresh parent, total 2,400 seconds including a five-second
source-preparation debit, at most four independent-file workers and one
BLAS/OpenMP thread each. Summed sampled descendant RSS <=6 GiB, combined
private output <=256 MiB, fresh disk-backed TMPDIR. Small compressed numeric
feature/state files may persist; no copied raw media or Gram matrix persists.
Transient calibration JPEGs are bounded and removed by the temporary-file
scope; they are not a new acquired corpus. Parent remaining-time guards apply
to every phase. Preserve every failure without automatic retry.

Generated controls independently check histogram/decrement calculations,
feature dimensions/finite values and numeric Fisher/tree/kernel exports
before real fitting. A separate audit replays all calibration choices,
source/model hashes, selected test metrics/intervals and sampled predictions.
No model is deployed. Verify all unchanged 570 application/test hashes against
the actual passed 4,646-test gate; publish fresh document checks separately.
