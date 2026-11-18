# EBS 0.4 bpnzac with a corrected payload size (2026-11-18)

**EBS at 0.4 bpnzac on gray BOSSbase Q75 reaches 93.30% balanced accuracy**
(paired 95% interval 92.20–94.30%, AUC .979) on the 1,000-pair historical
test, with one frozen calibration choice and a single test look. EBS was
recorded as "unavailable (invalid simulator)". The cause was the payload size,
not the EBS cost. [Numeric receipt](../benchmarks/ebs-20261118.json).

## What was wrong with the earlier EBS run

conseal 2025.11 `ebs.simulate_single_channel(alpha=0.4)` sizes the message as
alpha times **all** DCT coefficients, excluding only four wet modes per block.
It does not use the nonzero AC coefficients (nzAC). At Q75 only 17–23% of
coefficients are nonzero AC, so the requested "0.4 bpnzac" was in fact about
1.6–2.2 bits per nzAC. On five training covers it changed 50–71% of the nzAC, which
matches the recorded ~71%.

The block-entropy division by zero is not the defect. conseal turns the
resulting infinite cost of an all-zero block into its wet cost, as intended.

This study keeps conseal's EBS cost (`compute_cost_adjusted`, `EBS_ORIGINAL`)
and its ternary simulator unchanged and only sets the message size to
n = nzAC. This is the bpnzac definition used for J-UNIWARD, UERD and nsF5.

A check shows the change is only that size. With n set to conseal's own count,
the corrected function gives a byte-identical stego to
`simulate_single_channel` for the same seed. The corrected embedding changes
8.49% (fit) and 8.46% (test) of nzAC, next to J-UNIWARD's 8.36% and UERD's 7.94%.

## Data, roles and candidates

- Covers: the same 3,000 training and 1,000 test Q75 JPEGs as the academic
  JPEG study (`boss-jpeg-academic-repair-20261113`). Cover features come from
  its legacy-rounding calibrated-DCTR cache. Two covers per role were
  recomputed and match it exactly.
- Roles: 2,400 fit (index mod 5 ≠ 0), 600 calibration (mod 5 = 0) and
  1,000 test. The image hashes are disjoint across all three roles.
- Embedding seed: `[2026111206, 5, role == test, index]`, one stego per cover.
- Eleven candidates, fixed before fitting:
  - ridge Fisher on DCTR (8,000) and on calibrated DCTR (16,000), each at
    ridge .01/.03/.1/.3/1;
  - a 51-learner FLD ensemble (d_sub 1,000) on square-root DCTR.
- Selection: calibration BA with a threshold chosen on calibration, then AUC,
  then declared order. Gate .80. Test features were built only after the
  choice was frozen.

| Candidate | Calibration BA | AUC |
|---|---:|---:|
| DCTR ridge .01 | **93.25%** | .981 |
| DCTR ridge .03 | 92.25% | .979 |
| DCTR ridge .1 | 90.58% | .967 |
| DCTR ridge .3 | 88.58% | .951 |
| DCTR ridge 1 | 84.92% | .921 |
| Calibrated DCTR ridge .01 | 82.92% | .896 |
| Calibrated DCTR ridge .03 | 85.92% | .921 |
| Calibrated DCTR ridge .1 | 88.00% | .945 |
| Calibrated DCTR ridge .3 | 90.17% | .962 |
| Calibrated DCTR ridge 1 | 91.25% | .971 |
| FLD ensemble, d_sub 1,000 | 92.92% | .981 |

## Test result

| Method | Payload | Test BA | 95% interval | AUC | TPR | FPR |
|---|---|---:|---:|---:|---:|---:|
| EBS | 0.4 bpnzac | **93.30%** | 92.20–94.30% | .979 | 92.5% | 5.9% |

The selected DCTR ridge .01 sits at the smallest ridge in the grid. No ridge
was added after this result.

The 1,000 test covers have been reused by earlier JPEG studies of other
methods, so this is historical-test evidence. It is not untouched
qualification. It is, however, the first and only EBS test look. The interval
is a paired bootstrap over the 1,000 test sources (2,000 resamples).

## Verification and resources

- Four study checks pass: parity with conseal apart from the message size,
  a ±1 changed fraction of 5–12% of nzAC, role and seed definitions, and exact
  replay of the cover cache.
- Stego features: 1,295 s for the 3,000 training covers and 442 s for the
  1,000 test covers, on six CPU workers. No GPU.
- conseal 2025.11, sealwatch 2024.12, numpy 2.4.6, jpeglib 1.0.2.
  No `core/` or `steganography/` source changed.
- Stego features and models stay private under `.benchmark/ebs-20261118/`
  (gitignored).
