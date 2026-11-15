# WAV modular residual and re-embedding intervention (2026-11-15)

The user's current research/fix request permits this distinct bounded CPU
study on existing admitted PCM16 WAVs. The preceding source-bound diagnosis
finds heavy saturation under old residual quantizers and a bit-depth shortcut
in the nicolas training speaker. These are hypotheses, not proof of a data
accuracy ceiling. No old candidate is retried, payload raised, source deleted,
media/model downloaded, GPU used or detector deployed.

## Inputs and fixed features

Retain the corrected ESC 1,200/398/402 recording-group roles and AudioMNIST
300/100/200 clips from 15/5/10 speakers. For this fresh FSDD fit only, omit
nicolas's 500 upscaled-eight-bit training clips: 1,000 fit clips from george
and jackson, 500 calibration clips from lucas, and the unchanged 1,000 test
clips from theo/yweweler. Exclusion is a new training overlay, not deletion
or reassignment; all original manifests, roles and results stay unchanged.
This source change and feature change occur together, so no single-cause gain
is claimed. No reserved AudioMNIST speaker, native data or ALASKA2 is accessed.

Require regular existing WAVs <=4 MiB, PCM16, <=500,000 frames, <=two channels;
use the unchanged original first-channel rule. Preserve original per-file
embedding seeds [2026111201,dataset,condition,index] and .05/.1/.2/.4 sample
rates, with the same replacement/matching operations. Fit/calibration decode
precedes choice; no test features are constructed until global choice freeze.
Only sixteen formerly failing dedicated conditions are compared: six ESC,
two AudioMNIST matching and all eight FSDD. Old passing dedicated and mixed
rows remain separate records, not rerun or relabelled.

Each analyzed file independently yields 982 base descriptors. On the complete
signal and its quietest quarter of contiguous 1,024-sample frames, for orders
0..3 retain all eight modular residues and 64 adjacent residue-pair frequencies
(modulo eight, no clipping). For differences of orders 1..3, quantize at
1/2/4 and retain seventeen bins clipped to +/-8 instead of the old +/-3.
For three existing polynomial predictors, retain parity-weighted WS estimate,
prediction RMS and coverage under neighbor variance <=1/4/16/64 or unbounded.
Empty contexts return zero. Add log sample count, eight low-bit occupancies
and mean absolute amplitude. This preserves modular structure when absolute
residuals saturate; it does not manufacture information in uniform noise.

Construct one independent full-rate matching reference with fixed seed
2026111527, not a hidden key or filename/class-dependent seed. Store base
and base-minus-reference descriptors, 1,964 float32 values. This is inspired
by [audio re-embedding calibration](https://arxiv.org/abs/1701.05614), not a
claim to reproduce that paper's frequency features or published accuracy.
Each query uses its own signal and a declared synthetic reference only.

## Models, selection and evidence

Per condition, eight fresh models: Fisher ridges .03/.3/3 on 512 fit-only
Fisher-score columns, four Gaussian ridge-.01 models on standardized signed
square-root base or full descriptors (width factors 1/4), and one fixed
300-iteration /15-leaf /lr=.05 /l2=1 /seed-zero histogram tree on 256 fit-only
columns. Stable index ties, sorted columns. Kernel width uses 256 fit covers,
seed 2026111528 and positive direct squared distances >=1e-12. Reproduce
StandardScaler's float32 arithmetic exactly; Cholesky residual <=1e-6,
exported score parity absolute/relative 1e-7, tree probability parity 1e-10.

Select on calibration BA/AUC/original order and one exact-count BA threshold
(closest to zero, then lower). Freeze all sixteen choices before any new test
features. Only calibration BA >=.80 permits selected historical test inference;
other new test cells are `not_evaluated_calibration_gate`, with old measured
misses retained as context. The gate is not a test pass. No test-selected
fallback, payload change or post-test adjustment follows. Retain all measured
confusion counts, AUCs, source-group intervals (2,000 draws, seed 2026111529)
and adaptive historical-test limitations. FSDD has only two test speakers.

One fresh exclusive parent, 1,800 seconds including five preparation seconds,
<=six-GiB summed sampled descendant RSS and 256-MiB private output. At most
two independent-file workers, one BLAS/OpenMP thread each, fresh disk-backed
TMPDIR. Persist only small compressed numeric features/states/receipts, no raw
media or Gram matrix. Apply remaining-wall guards to every phase; preserve
any limit breach or unavailable phase without automatic retry.

Generated scalar oracles independently verify modular/quantized histograms,
context statistics, reference determinism and numeric exports before actual
fit. Closing audit replays every calibration choice, sampled state predictions,
selected test counts/AUC/intervals, one real feature replay and all bindings.
All 570 application/test hashes must still match the passed 4,646-test CPU
gate; this private experiment changes no application/test code. No research
model is installed or promoted by this study.
