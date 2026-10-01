# External-source check and assessment — 2026-10-01

Outcome: current base steganalysis again failed to discriminate the tested
Steghide/OpenStego variants. CTF extraction recovered all controlled payloads,
but five jobs stopped before completing all analysis stages.

## Evidence

The [Kodak suite curator](https://r0k.us/graphics/kodak/) provides the 24 color
images used here. Its usage statement is a curator's understanding, not a formal
license independently verified here. Images and payloads remain local.

All 24 originals were tested with both methods: 48 cover/stego pairs, 96 files,
but only **24 original lineages**. All 48 stegos round-tripped exactly through
the original upstream extraction tools. Original file hashes did not overlap
the frozen BOSSbase pilot; camera independence is unknown.

| Base detection, balanced threshold 70, AI off | Result |
|---|---:|
| TP / FN / FP / TN | 0 / 48 / 0 / 48 |
| Recall / FPR | 0% / 0% |
| Balanced accuracy | 50% |
| ROC-AUC (paired-lineage bootstrap 95% CI) | 0.49349 (0.46784–0.51956) |
| Steghide/BMP AUC (24 pairs) | 0.47830 |
| OpenStego/PNG AUC (24 pairs) | 0.50868 |
| Analysis errors | 0 |

ExifTool was available; zsteg and Stegseek were not. These results describe the
base analysis configuration, not full-Docker detector accuracy. The zero FPR is
not useful success when every stego is missed. Uncalibrated scores are not
probabilities. No thresholds were adjusted in response to these observations.

## CTF recovery and completeness

Thirty generated cases used five examples each of Steghide, OpenStego, base64,
gzip, ZIP and base64-wrapped ZIP. Steghide passwords were supplied. Exact payload
hash recovery was **30/30**, but completed exact recovery was **25/30 (83.3%)**.
All five Steghide cases reported `cancelled` after recovering their payloads.
One representative re-run confirmed `artifact byte limit reached`; therefore
do not present these results as 30 fully completed jobs. The configured output
budget was 16 MiB, excluding the input copy. This is not a blind suite.

CTF median was 0.192 seconds and p95 7.355 seconds, dominated by 20 simple decoder
cases. Tests and evaluations overlapped on the 8-vCPU machine; these figures
are observations, not a final reference-hardware performance qualification.

## Subjective engineering rating, not an accuracy percentage

- Overall local exploratory/CTF tool: **6/10**. Useful integrations, repeatable
  artifacts/reports and extensive tests, but incomplete isolation and recovery
  jobs consuming the artifact budget prevent production-grade confidence.
- Automatic steganalysis, on current evidence: **2/10**. Two real-image-source
  checks have not demonstrated reliable discrimination for the tested families.
  Synthetic success and broader format coverage do not overturn that evidence.

Use it as an assistant to inspect/extract, not to certify that a file is clean.
Next work should prioritize measured feature/model development on separate
data, then a newly untouched source, and reducing unproductive CTF candidate
expansion without raising limits to make these test results look better.

Protocol: `KODAK_PROTOCOL.md`. Aggregate evidence and local-report hashes:
`benchmarks/kodak-20261001.json`. Local full reports:
`.benchmark/kodak-detection-20261001/report.json` and
`.benchmark/kodak-ctf-20261001/report.json`. The original BOSSbase report and
manifest were not overwritten. No source images are committed.
