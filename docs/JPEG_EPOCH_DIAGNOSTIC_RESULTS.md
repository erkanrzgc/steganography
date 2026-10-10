# Final-model train-only normalization diagnosis — 2026-10-10

The frozen `JPEG_EPOCH_DIAGNOSTIC_PROTOCOL.md` completed in 70.33 seconds,
with actual host MemoryMax 8 GiB, two CPU threads, 48 train pairs and no
validation pixels, optimizer updates or model/BN mutation. This is a read-only
diagnosis of the GPU-trained model, not more GPU fitting or a detector pass.

| Source / quality / method | Stored-BN correct rows | Paired batch-statistics correct rows | Stored-BN mean loss | Paired mean loss |
| --- | ---: | ---: | ---: | ---: |
| ALASKA2 / unknown / UERD | 7/16 | 13/16 | 1.387297 | .512807 |
| ALASKA2 / unknown / JUNIWARD | 8/16 | 11/16 | 4.499787 | .567017 |
| BOSS / 75 / UERD | 7/16 | 15/16 | 12.288058 | .586966 |
| BOSS / 75 / JUNIWARD | 8/16 | 11/16 | 1.472119 | .608798 |
| BOSS / 95 / UERD | 8/16 | 14/16 | 3.433176 | .642631 |
| BOSS / 95 / JUNIWARD | 8/16 | 12/16 | 5.246575 | .682427 |

These are selected training-pair row counts, not independent held-out accuracy.
Paired inference sees a known original alongside its stego derivative, which
single-suspect analysis does not have. Do not deploy this diagnostic mode or
report its 76/96 correct rows as detector accuracy. Stored-BN scored 46/96;
large normalization sensitivity is established, but a sole causal explanation
or a successful corrective intervention is not. The failed full-development
scores in `JPEG_EPOCH_LEARNING_RESULTS.md` remain unchanged.

Next meaningful experiment: a separately frozen train-only normalization
intervention that does not require a cover partner at inference. Include an
unchanged-architecture control, the same data/exposure and single-row inference
checks; do not simply change BN mode or extend epochs against this validation
set. New held-out claims need an explicit untouched evaluation policy. No
follow-up training job is currently running.

## Provenance

`benchmarks/jpeg-epoch-train-normalization-summary-20261010.json` is a derived
portable aggregate, SHA-256
`590a5b315a8ff10df349d87694c61ac5a935bc165843951b2f22a4b1a3374d6d`.
It binds the private original detailed report
`8675c3fe838a2f68c4bb7499d12d55f8c0394d0c43dcc9db1e550f3c36f8ce5a`,
all 16 training sources, original plan/card/model, diagnostic core and protocol.
The original 48-pair/26-layer-per-mode details remain in the private job sandbox.

The unchanged read-only execution source is published as text evidence at
`benchmarks/jpeg-epoch-diagnostic-script-20261010.py.txt`, SHA-256
`999ac0fea03c35beba42f64bafd421c6f21159406816adb49c7243efe5f2592a`.
It ran as `diagnose.py` inside the original private epoch-job directory, under
a separate limited systemd scope and 240-second outer timeout. This text is
an execution artifact, not a new installed CLI. No weights/raw corpus or host
paths are published. No retry, threshold change or deadline relaxation occurred.

App regression before evidence publication: 1,793 passed / one actual-CUDA skip,
95.61% coverage. After the static evidence additions: the focused diagnostic
and pinned-evidence tests pass; lint/types and fresh model-free wheel/sdist
inspection also pass. The full app suite was not repeated for static evidence.
