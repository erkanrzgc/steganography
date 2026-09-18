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
