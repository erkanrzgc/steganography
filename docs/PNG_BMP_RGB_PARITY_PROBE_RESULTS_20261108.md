# Native RGB: parity-input diagnosis — 2026-11-08

**Final result (2026-11-09): the native RGB gate passes on both seeds and on
the closed reserved originals (AUC .87).** Root cause: raw-intensity SRNet
cannot derive LSB parity. Three parity input channels fix fitting; fresh
per-epoch payloads fix memorization; lr 1e-3 with a step decay fixes
under-training. The sections below record each step in order; the first
parity run alone still failed on generalization (development AUC .62).

## Question

The 2026-11-07 fit failed with train loss ≈ ln 2. A training-free check on the
same 64 train pairs showed the parameter-free Weighted Stego (WS) estimator
separates them perfectly, while a generic high-pass residual-energy feature
reproduces the CNN's failure signature (correct paired ordering, near-chance
global AUC, content offsets far above the embedding response):

| Feature (64 train pairs) | Global AUC | Correct pairs | Source SD / pair RMS |
| --- | ---: | ---: | ---: |
| Raw mean intensity | .5005 | 34/64 | 3116 |
| High-pass residual energy | .5137 | 64/64 | 520 |
| WS, unweighted | .9431 | 64/64 | 0.467 |
| WS, weighted | 1.0000 | 64/64 | 0.135 |

Hypothesis: a ReLU CNN on raw 0–255 intensities cannot easily derive sample
parity, so it falls back to a content-dominated noise-energy cue. LSB
replacement is content-invariant only through parity (even values move up, odd
values move down).

## One-variable test

`research/spatial_parity_probe.py` keeps the seed-2026102301 recipe: 64/20 split,
payload seed 2026102805, identical initial RGB weights (first convolution
widened, all other tensors bit-identical), schedule, batch of four pairs,
scalar AdamW 1e-4, twenty epochs /320 updates and the fixed three-part gate.
The only change appends `±64 · LSB sign` as three input channels; the extra
first-layer weights use a separate fixed generator (seed 2026110801).

The user approved one run on Colab T4. Browser control was unavailable, so the
same single attempt ran on the local CPU instead (no GPU): Python 3.11.14,
NumPy 2.4.6, Torch 2.14.0+cpu, four threads, 679.5 seconds. Device numerics
differ from the CUDA control; the comparison is qualitative. The executed source was then at `core/spatial_parity_probe.py`
(SHA-256 `a692e3e62201d3b0e431c9150c3c03f8adcf416732b4df7c07754fdd02fed3e2`); it was
later moved, with only import paths changed, out of the frozen `core/` closure. No checkpoint was
written; reserved 21 RGB /13 L sources stayed closed. Private result:
`.benchmark/png-bmp-rgb-parity-probe-20261108/rgb-parity-probe-result-20261108-cpu.json`
(SHA-256 `917d51c48122fbaa686bff641e9045fea4b7241b21bb71e57700ff3960da99e4`).

| Seed 2026102301 | Raw input (2026-11-07, CUDA) | Parity input (CPU) |
| --- | ---: | ---: |
| First /last epoch loss | .7023 /.6944 | .629 /.144 |
| Train AUC | .5056 | **.9836** |
| Train correct pairs | 51/64 | 64/64 |
| Train source SD / pair RMS | 105.48 | 0.171 |
| Development AUC | .5175 | .6200 |
| Development correct pairs | 15/20 | 10/20 |
| Gate (train ≥.85, dev ≥.70, loss ≤.8×) | fail | fail (dev) |

WS on the same development pairs: AUC **.9525** with no fitted parameters.

## What is established

- Parity information is the missing ingredient for fitting: with the same
  data, initialization, order and update budget, train loss falls far below
  ln 2 from the first epoch and content offsets no longer dominate.
- "Too little data" did not cause the original failure: the raw model could
  not fit 64 pairs, while the parity model fits them within 320 updates.
- The new failure is overfitting. Development covers average .654 stego
  probability and development pairs are ordered at chance (10/20), so the
  model memorized the 64 fixed training pairs rather than the embedding rule.

## Open hypothesis and smallest next test

Each training original receives the same payload and positions in every
epoch, so the network sees only 64 distinct stego images twenty times. The
smallest discriminating change is to re-embed a fresh payload and fresh
positions on every presentation (everything else unchanged): if development
pair ordering and AUC rise, fixed-stego memorization is the cause. That is a
new fit and needs its own explicit approval. One seed, one development set
already inspected many times (twenty originals /twelve ancestry components)
and a CPU-versus-CUDA device change limit every conclusion here.

## Follow-up runs (same day, user-approved, local CPU)

Each row is one consumed attempt on the same 64/20 split and fixed scoring
stego (payload seed 2026102805). "Fresh payload" re-embeds a new payload and new
positions every epoch (seed 2026102806 + epoch index); scoring is unchanged, so
"train AUC" then measures train covers with a payload the model never saw.

| Run | Train AUC | Dev AUC | Train pairs | Dev pairs | Loss ratio | Gate | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Parity, fixed payload, 1e-4, seed 1 | 0.9836 | 0.6200 | 64/64 | 10/20 | 0.229 | fail | 679.5 |
| Parity, fresh payload, 1e-4, seed 1 | 0.7449 | 0.6150 | 63/64 | 17/20 | 0.591 | fail | 514.1 |
| Parity, fresh payload, 1e-3, seed 1 | 0.8645 | 0.7375 | 54/64 | 14/20 | 0.573 | pass | 564.9 |
| Parity, fresh payload, 1e-3, seed 2 | 0.7908 | 0.7675 | 60/64 | 19/20 | 0.575 | fail | 1422.3 |

- Fresh payloads remove memorization: development pair ordering rises from
  10/20 to 17/20 at the same learning rate.
- Learning rate 1e-3 (the original SRNet rate) lifts development AUC above .70
  for both seeds. Seed 1 passes all three conditions; seed 2 misses only train
  AUC (.79 < .85), so the two-seed recipe does **not** yet pass the gate.
- Development AUC still swings by epoch (.54–.80), consistent with a high
  final learning rate; SRNet's original schedule decays 1e-3 to 1e-4.
- These four decisions were made sequentially on the same twenty development
  originals, so development AUC is optimistically biased; only the closed
  reserved originals can give an unbiased estimate.

Executed sources (private copies, SHA-256): fresh
`8850871d20bda1743da8a1b442d8b5d6b704916032d06dfd16ff9be97e5fca50`, lr 1e-3
`1186ae83088971eb0517e3f5faf8abd272a88dc75d018f648d4726e8382121b4`, seed 2
`d8fb62b4e1833df79a0d10fe4a862960dca1a3ab1b4f6a158048b6576d202788`. Results:
`c83c8fa2fdb9db15a3012b02719c53a1ff3c3bc53f441a06e0466e7f8c8c182d`,
`beae8fe19408212c6f561c39f90538a95a4ea26afd82627f9527faa13ade3e51`,
`1f3d3c9e60ab290755e9455e8a3f2e7311f024e87de0fc74ca7a7ddbb1b24d7f`.

## Two-seed pass: 40 epochs with step decay

Same recipe (parity input, fresh payload per epoch, AdamW 1e-3) for forty
epochs, decaying the rate to 1e-4 after epoch 30 as in the original SRNet
schedule; 640 updates per seed. The executed source equals
`research/spatial_parity_probe.py` (SHA-256
`376846dffdcc9766b59b85281fee08c1756b213ed1d44b10cbefba5c3523894e`). Final CPU
states were saved privately for a later reserved evaluation.

| Seed | Train AUC | Dev AUC | Train pairs | Dev pairs | Loss first → last | Gate | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 2026102301 | 0.9097 | 0.8175 | 58/64 | 18/20 | .692 → .262 | pass | 3882 |
| 2026102302 | 0.9753 | 0.8125 | 64/64 | 17/20 | .740 → .108 | pass | 4844 |

**Both original seeds pass the fixed three-part gate for the first time.**
After the decay, development AUC stays within .78–.83 for both seeds instead of
swinging between .54 and .80. The CNN still trails the parameter-free WS
estimator (development AUC .9525), and development AUC remains optimistically
biased by five sequential decisions on the same twenty originals.

Results (SHA-256): seed 1
`49c00bfb9698af01b5fe2f4b585168de83c281f7958d0a234fb6bb912f91a93b`, seed 2
`12dd0794a93fb8071cb40fb730bfc67f570c4625ff799d609556c00e4e308393`; states
`eca75b069795333816325bb69b677d2b625efcd6a6cc88f97dc918c1218feef9` and
`8580dfdacd905adcb68bd21ba46c60f396e28a7f70f732661eb763946e849059`. An unbiased
estimate needs one evaluation of these frozen states on the 21 closed reserved
RGB originals; that one-time use requires an explicit user decision.

## One-time reserved evaluation (2026-11-09)

At the user's go-ahead, the two frozen decay40 states were scored once on the
21 closed reserved RGB originals (cover vs 40% scattered plane-0 control,
payload seed 2026102805), with no training, tuning or retry. Metrics and the
pass rule (both seeds AUC ≥ .70) were frozen before scoring in a private
preregistration (SHA-256
`3d8104ecc8792554a0ef92f52ed7c30884b9b6d0cdd8946606522c4fbb39e945`). The
script was first checked on the already-used development set, where it
reproduced the recorded AUCs (.8175 /.8125) and WS (.9525) exactly.

| Model (21 reserved pairs) | AUC | 95% bootstrap CI | Correct pairs | Accuracy @ .5 |
| --- | ---: | --- | ---: | ---: |
| SRNet + parity, seed 2026102301 | **0.8685** | .760–.968 | 20/21 | .810 |
| SRNet + parity, seed 2026102302 | **0.8662** | .753–.971 | 20/21 | .810 |
| Two-seed probability mean | 0.8776 | .773–.973 | 20/21 | .810 |
| Weighted Stego (no fitted parameters) | 1.0000 | 1.000–1.000 | 21/21 | — |

**The pre-registered rule passes: both seeds generalize to unseen originals
(AUC .87), slightly above their development AUC (.82 /.81).** WS remains the
stronger detector for this LSB-replacement setting. With 21 sources the
intervals are wide; the lower bounds stay above .75. The reserved set is now
consumed for this recipe: further changes evaluated on it would be biased.
CPU scoring took 5.3 seconds. Private result SHA-256
`63ae70cdf2cd672898f57353a40166953ecda2a9b338b1026f7af3d97bff6d9b`.

## Harder embeddings: frozen CNN vs Weighted Stego (2026-11-09)

The two frozen decay40 states (trained only on 40% LSB replacement) and WS
were scored, with no training, on the twenty development originals under new
random-position embeddings (fresh seeds from 2026110903). The reserved
originals were not used. "Matching" is ±1 LSB matching (boundary values move
inward). Changed-sample fraction is half the rate.

| Embedding | Rate | WS AUC | CNN seed 1 | CNN seed 2 | CNN mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Replacement | 40% | **1.000** | .810 | .813 | .830 |
| Replacement | 20% | **.953** | .685 | .680 | .688 |
| Replacement | 10% | **.925** | .565 | .590 | .578 |
| Replacement | 5% | **.908** | .535 | .555 | .545 |
| Matching | 40% | .525 | .820 | .835 | **.845** |
| Matching | 20% | .558 | .675 | .690 | **.680** |
| Matching | 10% | .505 | .573 | .603 | **.570** |

- For LSB replacement WS dominates at every rate; the CNN degrades quickly
  below its 40% training rate.
- WS fails on LSB matching (chance), as expected, while the CNN transfers
  unchanged (.85 at 40%) although it never saw matching. It has learned a
  general change-energy cue, not only the replacement parity asymmetry.
- The CNN's value is therefore in matching-type embeddings, but it is weak at
  low rates. Training on matching and lower rates is the next learning
  question. The 40% replacement row (.81 /.81) is consistent with the recorded
  development AUCs; twenty sources and prior development reuse limit precision.
  CPU time 21 seconds; private result in `hard-conditions-20261109`.

## LSB-matching fine-tune (2026-11-09)

`research/spatial_matching_probe.py` (executed SHA-256
`bcad06dfdb4b74fe3be3a46ce28a1c5a6aff973f05a3f0c1b3f286f1f671fd9a`) warm-starts
each frozen decay40 state and trains on ±1 LSB matching only: forty epochs,
curriculum 40% → {20, 40}% → {10, 20, 40}% (epochs 10 /20), pixel-exact D4
augmentation, AdamW 1e-3 → 1e-4 after epoch 30, the same 64 train originals.
Development scoring reuses the fixed hard-condition stegos; epoch 0 reproduced
the frozen comparison exactly. Pre-set gate: matching 20% ≥ .80 and 10% ≥ .65
on both seeds. Local CPU, two seeds in parallel, 1,330 seconds each.

| Development AUC | Seed 1 before → after | Seed 2 before → after |
| --- | --- | --- |
| Matching 40% | .820 → .818 | .835 → .763 |
| Matching 20% | .675 → **.783** | .690 → **.738** |
| Matching 10% | .573 → **.663** | .603 → **.695** |
| Replacement 40% | .810 → .820 | .813 → .795 |
| Replacement 20% | .685 → .785 | .680 → .768 |
| Replacement 10% | .565 → .655 | .590 → .683 |
| Replacement 5% | .535 → .585 | .555 → .593 |

**Gate: fails** (matching 20% .78 /.74 < .80; 10% passes). Low-rate detection
improves by 7–11 AUC points on both seeds without losing the 40% replacement
level, but sixty-four training originals cap progress: development AUC keeps
swinging by epoch. The next intervention is more cover data. Private results:
`cae3f00d…` /`fc4b13ec…`; states `ea31d5f0…` /`3874fc38…` in
`png-bmp-rgb-matching-probe-20261109`. Reserved originals were not opened.
