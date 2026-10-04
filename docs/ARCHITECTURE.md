# Architecture

The project is a local-first steganalysis and CTF workbench. Its interfaces are
thin adapters around application services:

```text
CLI / API / TUI
       |
       +-- AnalysisService -> Analyzer and Carrier plug-ins
       +-- CTFService ------> AnalysisContext, ToolRunner, decoder graph
       +-- Case/Scan/Studio -> encrypted Vault and SQLite audit chain
       |
       +-- JSON v2 / HTML / SARIF / evidence bundle
```

`AnalysisContext` lazily caches file identity, bytes, decoded image arrays,
entropy, JPEG metadata, and PCM WAV samples for one input. Context-aware native
components share it; legacy and third-party analyzers retain the compatible
`analyze(Path)` adapter.

`CTFService` owns the bounded playbook. Artifacts form a parent-child graph and
carry SHA-256 plus extraction provenance. `ToolRunner` invokes optional upstream
programs without a shell, with time/resource/output limits and redacted report
commands. An unavailable tool is coverage metadata and can make a result
inconclusive; it is not negative evidence.

External extraction adoption is invocation-specific: only the explicitly named
output of a zero-exit, completed extractor is eligible. Unnamed files (including
logs and crash dumps), failed/partial outputs, symlinks, empty outputs and paths
that existed before invocation are excluded. POSIX children have `RLIMIT_CORE=0`.
This is evidence hygiene, not per-tool filesystem isolation; rejected incidental
files can remain in the job directory but are not included in artifact bundles.
The full-image OpenStego launcher fixes allocator arenas and JVM active processors
at two, with the existing 128 MiB Java heap, rather than enlarging `ToolRunner`'s
768 MiB address-space ceiling. Other tools and the base wheel are unchanged.

The depth limit is checked before any child-producing stage. Boundary artifacts
still receive analysis; native extraction budget exceptions propagate to an
inconclusive, cancelled job. URL decoding requires a printable ASCII envelope
and a valid percent escape, and preserves decoded bytes exactly. BMP signature
carving checks header/size/dimension/plane consistency, considers at most 64
header hits, and retains the declared file extent. Other signatures remain
heuristic candidates. Carved files retain format suffixes for native analysis;
neither a filename extension nor a carving signature confirms steganography.

Managed job artifacts are stored beneath a newly created output directory. Artifacts are
regular files, names are generated rather than trusted, archive members are
streamed with count/depth/size/ratio limits, and existing files are not
overwritten. Extracted content is analyzed, never executed.

The tool working directory and resource limits are not an operating-system
filesystem sandbox. Native parsers run in-process and the job deadline is
cooperative between stages. Use a non-root, read-only container with only the
job's writable mount for hostile inputs; stronger per-tool isolation and hard
native-stage deadlines remain release work.

JSON v2 remains additive through `schema_revision`. HTML, SARIF, and bundles are
views of the same normalized report. Host paths and secret-bearing arguments
are redacted before serialization.

`steganography.benchmarking.pilot` evaluates the same analysis and CTF services;
it does not implement a second detector. Independent upstream tools generate
cover/stego pairs, with hashes and lineage recorded before evaluation. Solver
job directories exclude expected payloads. Aggregate evidence is versioned,
while downloaded images and full local runs remain in ignored `.benchmark/`.
Each CTF job also retains its portable report; aggregates include failure reasons,
artifact counts and generated-output bytes (excluding the copied input).
Its opt-in `--both-methods` generation mode uses both upstream methods per cover
without splitting or multiplying original lineages. Explicit Kodak acquisition
records provenance and rejects exact hash overlap with a reserved manifest.

`scripts/fetch-alaska2-pilot.py` is an explicit, standalone research acquisition
utility, not an analyzer or automatic application download. It samples complete
original/stego groups before scoring and obtains only bounded ZIP/ZIP64 ranges
from an authenticated, fixed-origin archive. Private credentials are never sent
to the storage redirect target. CRC/JPEG checks, local SHA-256 re-reading,
exclusive outputs and provenance-bound resume precede a success manifest.
See `ALASKA2_ACQUISITION.md`; acquisition does not establish detector accuracy.

`steganography.benchmarking.alaska2` evaluates those original JPEGs through the
unchanged `AnalysisService`. It validates frozen manifest/selection hashes and
four-way lineages, persists portable per-file JSONL, and reuses the covers in
each method's paired summary without multiplying independent observations.
Required-native failures invalidate cell metrics rather than becoming negatives;
optional coverage remains explicit. Fixed-threshold metrics disable threshold
search, and uncertainty resamples whole lineages. See `ALASKA2_PROTOCOL.md`.

`core.dataset` defines shared identity and connected-component rules. Research
partitioning uses these to keep lineage/camera/device groups together, reserve
whole test sources and reject overlap with frozen manifests. Partition policy
is rechecked by research manifest validation. AI triage stays in raw analysis
results but cannot contribute to aggregate scores or primary pipeline findings.

`core.features` owns the bounded `spatial-summary-v1` preprocessing contract.
`steganography.research_features` binds feature rows to verified partition
manifests and validates complete train-only inputs for the research trainer.
The CLI delegates extraction to that shared workflow. Artifacts and checkpoint
provenance use hashes, not absolute dataset paths. No feature/model downloads or
automatic deployment occur.

The legacy `sample_pair_balance` and `rs_regular_singular_balance` image signals
are two names for the same adjacent-LSB parity observation, not implementations
of full SPA/RS algorithms. Both use `image_lsb_adjacency`, so service and hybrid
ensemble aggregation count them once. Scores remain uncalibrated heuristics.

PDF stream decoding enforces a per-stream 16 MiB output limit and a shared
32 MiB decoded-output budget per parse. CTF extraction further restricts that
budget to the remaining job output allowance. ASCII decoder input is bounded
before whitespace normalization; ASCII85 zero-run expansion is preflighted.
Truncated/invalid streams remain raw candidates with `decode_status=unavailable`,
never successful decodes. These limits do not make the regex-based PDF parser a
complete PDF implementation or provide hard CPU deadlines/native OS isolation.
