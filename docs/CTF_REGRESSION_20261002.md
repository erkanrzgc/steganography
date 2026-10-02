# CTF candidate integrity and resource-limit regression — 2026-10-02

This reruns the **published**, non-blind Kodak CTF challenges. It is regression
evidence, not a new held-out accuracy result. The original detection report and
its failed 0/48 recall remain unchanged. No detector, threshold, calibration,
challenge, expected payload or password was tuned.

## Findings and corrections

The original five cancelled Steghide jobs had already recovered their expected
payloads. Inspection of one diagnostic job found two ELF `ET_CORE` crash dumps
being adopted as payloads, a lossy URL decoder discarding 405,020 non-ASCII
bytes from the BMP, and a false `BM` signature at offset 21,675 spawning large
suffix artifacts. These are orchestration/candidate-integrity bugs, not evidence
of additional hidden files.

`bacae39` restricts external adoption to the current completed, zero-exit
extractor's explicitly named nonempty regular output. Preexisting files,
symlinks, incidental logs and failed/partial outputs are excluded; POSIX child
core dumps are disabled. Printable-ASCII percent decoding preserves bytes;
BMP carving validates headers and declared lengths. Depth-boundary artifacts
are analyzed without generating children, and native extraction resource-limit
exceptions now cancel the job instead of disappearing inside parser handlers.
Full per-job portable reports and output counters are retained by the runner.

A verification run after that fix completed all 30 jobs but accepted only 29
payloads: OpenStego extracted a file and then its JVM failed a native allocation
with exit status 1. The failed invocation's output was correctly excluded.

`2cc988a` bounds OpenStego's allocator arenas and active JVM processors at two,
without increasing the existing 128 MiB Java heap or 768 MiB tool address-space
limit. CPU-derived thread pools and allocator arenas consume memory outside the
Java heap; this was the working explanation for the observed native-allocation
failure, not a claim that physical RAM was exhausted. See the
[glibc allocation tunables](https://www.gnu.org/software/libc/manual/html_node/Memory-Allocation-Tunables.html)
and [Java 17 runtime options](https://docs.oracle.com/en/java/javase/17/docs/specs/man/java.html).
Ten independent cold JVM extractions now form part of the Docker smoke check;
every invocation must complete and recover exact bytes, with no retry credit.

## Evidence

Machine-readable hashes and measurements are in
[`benchmarks/ctf-regression-20261002.json`](../benchmarks/ctf-regression-20261002.json).
All successful regression runs use the frozen manifest
`fdf9edf658a323d09b3adf0d783f84183b0ba2e17a41b3dbd86451ec28ca6341`.
Limits remain depth 3, 64 artifacts, 16 MiB generated output, 5 seconds/tool,
60 seconds/job, with the known fixture password supplied for Steghide.

| Run | Exact payloads | Completed jobs | Cancelled jobs |
| --- | ---: | ---: | ---: |
| Published 2026-10-01 baseline | 30/30 | 25/30 | 5 |
| Candidate fix, original JVM launcher | 29/30 | 30/30 | 0 |
| Candidate fix + bounded JVM launcher | 30/30 | 30/30 | 0 |
| Same fixed revision, second full run | 30/30 | 30/30 | 0 |

Each fixed run contains five cases each of Steghide, OpenStego, base64, gzip, ZIP
and base64+ZIP. Total managed artifacts: 82; generated output: 9,633,982 bytes.
The largest job creates eight artifacts and 5,390,462 output bytes. The copied
input is not included in generated-output accounting. No core dumps were found.
Full-image smoke also passed native, Steghide and OpenStego exact recovery,
password redaction, and ten cold OpenStego extractions under the default budget.

An earlier uncommitted candidate-fix run recovered 30/30. Its subsequent 29/30
verification is retained above rather than selecting only the successful run.
A launcher-bind diagnostic recovered 25/30 because a non-executable mounted
script caused PATH lookup to select the stock launcher with a 1 GiB heap. That
run is retained separately as an invalid harness configuration; the script now
has executable mode, matching Dockerfile.full's existing chmod step.

## Environment and limitations

The existing full image
`sha256:e2c255ae0709058e4a320f278e41d193b61223d6fbabbcaf279a51a0da835575`
was used with current source and the updated executable launcher mounted
read-only. The full image was not rebuilt in this slice. Jobs ran as UID/GID
1001, root filesystem read-only, network disabled, all capabilities dropped,
256-PID ceiling and 256 MiB `/tmp` tmpfs. Python was 3.11.13 in the container;
the host tests used 3.11.14. Eight logical CPUs were available.

The first fixed run took 39.973 s, median 0.136 s and p95 6.657 s; the repeat took
38.199 s, median 0.158 s and p95 6.469 s. These are the same 30 cases, not 60
independent challenges. Workloads overlapped during the first fixed run;
these are descriptive timings, not a dedicated performance-gate measurement.
Simple decoder cases dominate the median. Zsteg failed in ten image cases and
Stegseek seed scanning failed in five under these limits; their failures remain
visible and are not counted as successful recovery. The other extractor paths
produced the expected payloads.

JPEG and other magic-only carvings can still produce unverified candidates;
graph-wide artifact deduplication and stronger structural validation remain
work. This slice does not establish hard native deadlines, complete OS tool
isolation, the blind 120-challenge gate, trained detector accuracy, or v1.0
readiness. Independent fixtures and adversarial unit tests passed alongside the
full 332-test suite (92.69% total coverage), Ruff and mypy. Python 3.12–3.14 were
not executed locally.
