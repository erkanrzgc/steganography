# Tiny train-only learning sanity — objectives failed

Immutable [protocol](SRNET_TINY_SANITY_PROTOCOL.md), commit `64584b0`, frozen
before selection. [Full loss/schedules/selected identities](../benchmarks/srnet-tiny-sanity-20261007.json)
and [read-only numeric/input audit](../benchmarks/srnet-tiny-sanity-audit-20261007.json)
are published. This is an in-sample learning check, **not detector accuracy**.
Original models/data/failures remain unchanged; no model is installed.

Exactly 24 train rows: four ALASKA lineages, two BOSS lineages with both Q75/Q95,
selected by metadata only. Six declared original scenes/eight quality groups,
not independent sources. All 50 epochs completed, eight four-row updates each,
400 total; fit plus selected-row evaluation took 487.91 seconds (about 8m08s).
Some regression checks ran concurrently; not an isolated throughput benchmark.
No validation cache or validation image bytes were opened.

| Frozen objective | Actual result | Outcome |
| --- | --- | --- |
| Final training batch loss <= .35 | .693085 (first epoch 1.020972) | Failed |
| Relative loss reduction >= 25% | 32.12% | Passed |
| Stored-BN train balanced accuracy >= .90 | .50 | Failed |

Selected-row singleton recall and FPR are both .625; mean cross-entropy .684087.
Loss reduction alone is misleading: the final batch loss remains near log(2),
and the saved model does not discriminate the selected training classes.
Batch loss uses balanced pair presentations while singleton CE uses eight
covers/16 stegos; they are not directly comparable mode-only measurements.

Read-only audit reloads checksum-bound numeric weights, reconstructs every
schedule/selected identity, verifies all BN counters equal 400 and repeats all
24 singleton predictions. Nine independent NumPy examples pass unchanged
forward logit/score/decision gates. All 16 decoded cover/stego crop differences
are nonzero (RMS .01953–1.18322 pixel units); identical selected crops do not
explain this failure. This is not proof of correct gradients or a sole root cause.
The completed worker and audit exited; no training job remains active.

Do not spend blindly on a longer full-corpus fit or ship these weights. Next
preregister a generated strong-label-signal positive control and train-only
gradient/input diagnostics to distinguish wiring/optimization problems from
insufficient learning of weak real stego signals. Generated controls remain
smoke/regression, never real-dataset accuracy. Any subsequent real fit needs
its own frozen protocol and unchanged honest held-out/independent-source gates.

Verification: Python 3.11, 1,184 tests passed, total coverage 94.93%; new
core/service statements 116/117 covered (99.15%). Ruff, mypy (118 files), diff
checks and model/data-free wheel/sdist builds passed. Other Python versions
and fresh full Docker were not rerun locally.
