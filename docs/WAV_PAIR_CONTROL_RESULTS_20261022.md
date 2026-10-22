# Fresh generated PCM pair-control results — 2026-10-22

The separate CPU control is **failed_learning_gate**. The previous v3
unavailable attempt and its failed complete first seed are preserved.
Real AudioMNIST fitting, native accuracy and production qualification remain
unavailable; no original/reserved media, download or checkpoint was involved.

See the [frozen recipe](WAV_PAIR_CONTROL_PROTOCOL_20261022.md),
[protocol receipt](../benchmarks/wav-pair-control-protocol-20261022.json),
[pre-launch binding](../benchmarks/wav-pair-control-launch-20261022.json) and
[complete numeric evidence](../benchmarks/wav-pair-control-results-20261022.json).

## Data and acceptance

Generator seed 2026102201 is separate from the old control. The frozen native
length schedule is metadata only: 400 train /200 development generated sources,
20/10 disjoint groups, nine rows per source. Full-alphabet global/quarter
plane-0/1 pair statistics produce ten finite FP32 inputs. Identifiers, group,
recipe and oracle data do not enter the model. This is the same speech-like
generator family, not independent natural speech or a representative benchmark.

Before any metric, the recipe fixes a 418-parameter 10→32→2 FP32 head,
two seeds, 20 epochs and 1,140 updates /72,000 row presentations per seed.
Class-balanced AdamW settings, data order and score direction are unchanged
throughout the attempt. The independent acceptance rule requires training
AUC >=0.60 and both 40%-plane-0 development AUCs >=0.70 in **each** seed.
All eight cells and both complete schedules are required. No threshold,
direction, normalization, model selection or early stop is fitted to development.

## Zero-update profile

All **4,800** payload recoveries passed; all 600 raw/decoded hashes are unique
and disjoint across splits and from the old corpus. Generated WAVs total
**36,335,714 bytes**; both role matrices contain 5,400×10
features. The [manifest](../benchmarks/wav-pair-fresh-manifest-20261022.json),
profile and cache bytes/order were bound before fitting by the
[fit binding](../benchmarks/wav-pair-control-fit-binding-20261022.json).

The [profile](../benchmarks/wav-pair-control-profile-20261022.json) took
24.503 worker seconds, including imports;
extraction took 22.889 seconds and 20 backward
batches took 0.005877 seconds. It constructed no optimizer,
performed **zero updates** and left all parameters identical. Its conservative
full-control projection was 91.203
seconds; this was a projection, not measured fitting throughput.

## Complete learning results

| Seed | Updates | Presentations | Train AUC | Development AUC | Sequential 40% p0 | Scattered 40% p0 |
|---|---:|---:|---:|---:|---:|---:|
| 20261016 | 1,140 | 72,000 | 0.643081 | 0.643962 | 0.541750 | 0.520375 |
| 20261017 | 1,140 | 72,000 | 0.638483 | 0.638870 | 0.546025 | 0.495800 |

All 2,280 updates /144,000 row presentations completed. Numeric completed-fit and scored-seed receipts were retained before the next seed began. No weights were saved. AUC is ranking performance, **not accuracy**.

| Method | Payload | Plane | Seed 20261016 train/dev AUC | Seed 20261017 train/dev AUC | Fixed statistic train/dev AUC |
|---|---:|---:|---:|---:|---:|
| sequential | 5% | 0 | 0.517219 / 0.518950 | 0.520238 / 0.520375 | 0.557672 / 0.556825 |
| sequential | 5% | 1 | 0.738537 / 0.738775 | 0.714256 / 0.715050 | 0.513306 / 0.513250 |
| sequential | 40% | 0 | 0.540769 / 0.541750 | 0.545819 / 0.546025 | 0.744294 / 0.728675 |
| sequential | 40% | 1 | 0.999781 / 1.000000 | 0.996506 / 0.996625 | 0.535844 / 0.534775 |
| scattered | 5% | 0 | 0.502881 / 0.503775 | 0.499213 / 0.499038 | 0.526563 / 0.526950 |
| scattered | 5% | 1 | 0.537769 / 0.537175 | 0.537169 / 0.536700 | 0.507400 / 0.506625 |
| scattered | 40% | 0 | 0.517663 / 0.520375 | 0.495488 / 0.495800 | 0.703862 / 0.695350 |
| scattered | 40% | 1 | 0.790031 / 0.790900 | 0.799175 / 0.801350 | 0.532181 / 0.530425 |

The comparator is cached FP32 global plane-0 pair equalization, higher = stego. Its direction and column were fixed in advance; it cannot rescue a failed head. The old raw CNN used another corpus, so these results are not a matched model comparison.

## Resources and independent closeout

Independent systemd monotonic durations: profile
25.028681 seconds, conservatively
debited as 27 seconds; fit
3.556292 seconds. The authoritative
combined charged interval is **30.556292 /1,800
seconds**. The worker reported 3.034337 fit-phase seconds.

Both units enforced 2-GiB memory, zero swap, 200% CPU, no restart and bounded
watchdogs. Worker-read kernel cgroup memory peaks were
278,994,944 /235,089,920
bytes for profile/fit, with zero swap. Systemd summary peaks are unavailable for at least one phase; worker-read kernel peaks are retained.
Independent systemd timestamps establish elapsed time.

Retained experiment artifacts/receipts measured at closeout total
678,720 bytes. The conservative storage-peak
upper bound, including the full generated corpus and a 3-MiB reserve for
remaining receipts/finalization, is
**40,160,162 /67,108,864 bytes**.
Individual experiment artifacts/logs remain <=1 MiB. Generated temporary WAVs
were removed normally; no historical dataset/checkpoint was deleted. Final
34-source and cache bindings passed. This CPU intervention adds no JPEG time
and does not reset the incomplete legacy raw-transfer ledger.

## Software verification and limits

The unchanged full gate passes **3598 tests /2 explicit
CUDA skips**, 96.96847% total statement coverage.
The 44 new focused tests cover scalar class-weight/last-batch gradients,
full schedule accounting without optimizer mutation, hash/domain/split and
array corruption, symlinks/overwrite, deadlines, memory/swap and retained
first-seed evidence on second-seed failure. New service/adapter coverage is
99.29078% /
96.55172%.
Ruff and provisioned-environment mypy (221 source files) pass. The first
global-environment mypy attempt lacked compatible Torch/ONNX/NumPy definitions;
its error log is retained separately. Python 3.11 was checked locally;
3.12–3.14 remain CI requirements. Verification scratch is recorded separately
from experiment output and historical experiment ledgers.

The pair statistic loses within-quarter ordering and can merge a one-LSB
change when pair counts swap. Fresh generated data does not remove the
same-family/adaptive research caveat. No representative native accuracy,
production qualification or real fitting authorization follows. Preserve this
outcome; another fit needs a separately frozen intervention. Real AudioMNIST
also requires explicit approval and an accepted complete generated control.
