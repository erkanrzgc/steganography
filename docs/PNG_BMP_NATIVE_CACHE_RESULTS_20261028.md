# Local native cache results — 2026-10-28

**Cache contents verified; final memory peak unavailable.** The one data-only
preparation decoded the 118 already admitted canonical PNG crops, wrote a
private unassigned cache and completed its independent control/batch replay.
Its closing observer then failed because systemd returned `[not set]` for
`MemoryPeak`. Preserve that failure: the strict protocol acceptance is
`unavailable_final_memory_peak`, not an unconditional resource pass. The cache
was not regenerated or retried. No model, GPU, optimizer, network request,
full original image or reserved corpus was used.

[Frozen protocol](PNG_BMP_NATIVE_CACHE_PROTOCOL_20261028.md),
[worker evidence](../benchmarks/png-bmp-native-cache-preparation-20261028.json),
[independent replay](../benchmarks/png-bmp-native-cache-independent-20261028.json),
[closeout](../benchmarks/png-bmp-native-cache-closeout-20261028.json),
[source bindings](../benchmarks/png-bmp-native-cache-source-bindings-20261028.json),
and [readable tools/failures](../benchmarks/png-bmp-native-adapter-tools-20261028.json).

Freeze SHA-256:
`92ddef59034f0a03678fcdbcb03ac2f30df5209f9672693bf4b57a85216a7dc4`.
All 244 source/document bindings remain unchanged, including the original
closing observer. Cache manifest SHA-256:
`8da0ac4ac98b7b0e9e3613b02b58c3cb076adb9456c74ef29c90f6a70ee1dfd2`.
The [public manifest](../benchmarks/png-bmp-native-cache-manifest-20261028.json)
contains attribution and hashes only; raw pixels remain private under
`.benchmark/png-bmp-native-adapter-20261028/native-cache` on Kali.

## Content and execution

- 13 L /105 RGB originals, 119 files: 118 raw HWC uint8 arrays and one manifest.
  Raw pixels total 21,495,808 bytes; the cache totals **21,591,572 bytes**.
  All source, canonical-cover and decoded-crop hashes verify. PNG/BMP variants
  still represent one original. Every role remains unassigned.
- Every source was checked directly and in one of **31 FP32 batches**, with
  **236 scalar forty-percent plane-0 payload recoveries**. The independent
  checker reconstructs all **118 controls** using scalar arithmetic, replays
  all 31 batch hashes and checks every cache hash/extent against snapshot 016.
  These new controls do not replace the existing 32-byte admission fixtures.
- The worker took 7.072901 seconds before its independent audit; the audit
  took 2.385772 seconds. Retained kernel journal start/stop timestamps give a
  conservative **9.769115-second unit window**, below the separate 180-second
  preparation limit. These are CPU data-preparation costs, not integrated
  model-step timing or a native full-training budget.
- Runtime guards verified a kernel memory limit at most 512 MiB and zero swap
  before decoding. The worker observed **85,831,680 bytes** of peak charged
  memory before the independent audit; this is a lower bound for the entire
  job, not its final peak. The retained final systemd peak is unavailable.
  After stop, the unloaded unit reports no historical limits/counters; those
  current defaults must not be interpreted as its execution envelope.

The unit `png-bmp-native-cache-preparation-20261028.service` is inactive,
MainPID zero, with no restart. Its start record, outputs, failed observer,
private journal and source identities are preserved. The subsequent closeout
only rechecks existing hashes/receipts and journal timestamps: it does not
decode images, regenerate controls, construct models or relaunch the unit.
No remaining preparation time grants another attempt or a native fit.

## Software verification

The full unchanged application/test run completed **4,044 passed /one failed /
two explicit CUDA skips**, with **97.150080%** total coverage and 100% new-module
coverage. Its sole failure was the existing README's 923-word length against
the unchanged 900-word limit. Preserve that run and its exit code 1.
Only README wording changed: 898 words, unchanged required scope statements.
The complete README/cloud-removal test file then passed **15/15**; application
and test source hashes are unchanged. All **4,045 unique tests** are thereby
verified, recorded as `passed_after_documentation_replay`, not a claim that
the original full command passed. Ruff and 238-file mypy pass; all 37 new
focused tests have 100% module coverage.
See [the exact software receipt](../benchmarks/png-bmp-native-adapter-verification-20261028.json).

The worker/auditor also completed an end-to-end generated-only 118-source /
31-batch preflight. Preserve its first failure on empty Python namespace
files: the artifact reader correctly rejects empty data; only the worker's
source-hashing routine was corrected. No application/carrier safety check
was relaxed. Those fixtures are separate from actual native cache execution.

## Remaining work

Cache contents are ready, while strict final-peak resource acceptance remains
unavailable. Native corpus/group feasibility, a role-bound optimizer and
numeric-continuation supervisor, integrated I/O/GPU costs and a separately
approved full-fit allowance are still missing. The old target lacks 115 L /
23 RGB originals and ten L groups; the historical RGB ten-group train plan
cannot supply eighty originals. Unknown ancestry/cover cleanliness remain
unproven. All old holds, exclusions, reserved data, consumed attempts and
transfer/metadata ledgers stay unchanged. No fitting or acquisition follows.
