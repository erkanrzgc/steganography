# JPEG refinement numerical amendment outcome

The original failed preflight and historical snapshots remain immutable.

Controller status: **completed**. Verified real-data updates: 28672.
Learning gate: **false**. Production qualification: **unavailable**. No deployment.

See [portable results](../benchmarks/jpeg-refine-numerics-results-20261015.json) and [the frozen amendment](JPEG_REFINE_NUMERICS_PROTOCOL.md).

All 56 segment receipts and six evaluation endpoints were verified. The cumulative run used 11,903.32 seconds including the conservative prior debit, and 3,490,924,619 retained bytes including its prior storage debit. Both remain within the frozen limits.

| Arm | Seed | BOSSbase AUC | BOWS2 AUC | Combined selection |
| --- | ---: | ---: | ---: | --- |
| warmup | 20261012 | 0.770508 | 0.679199 | unavailable |
| paired | 20261012 | 0.848145 | 0.786865 | unavailable |
| unpaired | 20261012 | 0.672363 | 0.654053 | unavailable |
| warmup | 20261013 | 0.613037 | 0.574951 | unavailable |
| paired | 20261013 | 0.748291 | 0.688477 | unavailable |
| unpaired | 20261013 | 0.501953 | 0.501953 | unavailable |

The paired continuation improves both sources in the first seed, but second-seed BOWS2 AUC is 0.688477, below the frozen 0.70 rule. No arm passes every source and seed. The unpaired continuation does not establish superiority. These are development AUCs, not production accuracy. No further fitting was started.
