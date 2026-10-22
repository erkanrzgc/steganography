# Full-exposure generated WAV control acceptance — 2026-10-21

This is the source-bound acceptance contract for the next generated WAV
learning control, not permission to fit AudioMNIST. The full-exposure runner
has not yet been implemented or executed. The old reduced-exposure diagnostic
script is preserved as historical evidence; it cannot produce a valid result
under this contract. A runner needs its own frozen source binding before a
GPU attempt. No attempt or compute allowance is consumed by this preparation.

The 2026-10-19 receipt amendment passed its execution-capacity check at its
historical commit. Adding pluggable CNN recipes subsequently changed the
trainer and its tests. That old pass remains valid for its original source
closure and is not a pass for the current source. Do not overwrite either
failed attempt, the successful amended receipt, or the reduced learning
control from 2026-10-20.

## Planned exposure and rule

Use only synthetic quiet speech-like mono PCM16 at 48 kHz, with 400 training
and 200 development originals. Use the observed frame lengths from the pinned
preparation receipt as metadata only. Generate 20 training and ten development
groups, 20 originals each. No source WAV or reserved speaker is opened. Keep
each original and its eight controlled recipes together in their generated
group. The generator, deterministic independent corpus seed and corpus hashes
must be frozen in the future runner protocol; the reduced script's random
length sampling must not silently substitute for the exact length schedule.

Use the unchanged draft v3 recipe, both seeds `20261016` and `20261017`,
20 epochs, virtual batches of 64 (final batch 16), 3,600 training rows,
72,000 presentations and 1,140 optimizer updates per seed. Input transforms,
GroupNorm, AdamW and weight decay come from `core/wav_pilot_v3.py`. Reuse the
shared trainer and fixed evaluation threshold; no hyperparameter, recipe or
validation-threshold search, checkpoint selection or ablation is part of this
control. Synthetic development rows are distinct from synthetic training rows.

Both seeds must separately achieve training AUC across all eight recipes of
at least 0.60, and development AUC of at least 0.70 in **both** sequential-40%-p0
and scattered-40%-p0 cells. Record all eight distinct development cells, even
though only two participate in this previously chosen rule. Do not pool seeds,
invert score direction or lower the thresholds after observing a result.

The planned run uses the owned RTX 5060 Laptop GPU, FP32 with AMP and TF32 off,
and the shared deterministic backend policy. A single 3,600-second wall ceiling
includes generation, both complete fits, training/development scoring and
receipt assembly. CUDA allocation is capped at four GiB, host resident memory
at eight GiB, swap at zero. A future external supervisor must enforce and
measure these bounds, retain partial-update evidence on failure, and prevent
an automatic restart. This proposed generated-control allowance does not
extend JPEG's six-hour budget or authorize real-data fitting.

## Read-only verifier

`core.wav_control_gate` verifies the exact current source closure, both complete
seed schedules, finite bounded AUCs, unique cells, declared generated-only
provenance and measured resource fields. Its explicit adapter never imports
Torch or starts fitting:

```text
python -m steganography.research_wav_control \
  --protocol benchmarks/wav-full-control-protocol-20261021.json \
  --receipt path/to/future-full-control-result.json
```

A complete below-threshold run is `failed_learning_gate`. Reduced exposure,
missing seeds/cells, stale sources or incomplete resource evidence is
`unavailable`. Only both complete passing seeds produce `passed_generated_only`.
Every decision keeps `real_fit_authorized=false` and
`production_qualified=false`. The verifier checks evidence consistency; it
does not independently establish that a producer's declarations are truthful.
Do not treat hand-written test receipts as experimental evidence.

The future result uses schema `wav-full-control-result-v1`: protocol hash,
the full source closure, recipe ID, generated/real/reserved/checkpoint flags,
zero real optimizer updates, exact CUDA device name, total wall seconds,
host/CUDA/swap peaks, and two seed records. Its `corpus` evidence must record
400/200 files in 20/10 groups, 20 files per group, zero group/decoded-PCM
overlap, 4,800 payload oracles and `length_schedule=exact_metadata_order`. Each seed records all exposure
counts, `train_auc_all`, and eight `development_cells` containing `method`,
`rate_percent`, `plane` and `roc_auc`. Missing evidence is never a learning pass.
