# Native RGB: parity-input diagnosis — 2026-11-08

**Result: the representation diagnosis is confirmed; the learning gate still
fails, now on generalization.** Adding three LSB-parity input channels lets the
unchanged SRNet recipe fit the training set (train AUC .51 → .98), but
development AUC is only .62 and development pairs are ordered at chance.

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
