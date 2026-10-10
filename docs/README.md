# Documentation

Start with the workflow you need. Research evidence is separate from operational
instructions so that implementation readiness is not mistaken for accuracy.

## Use the toolkit

| Guide | Contents |
|---|---|
| [Getting started](GETTING_STARTED.md) | Installation, interfaces, and first-run workflow |
| [User guide](USER_GUIDE.md) | Scanning, hiding, recovery, and cases |
| [CLI reference](CLI.md) | Commands, profiles, and limits |
| [API guide](API.md) | Local API, authentication, and jobs |
| [Detection guide](DETECTION_GUIDE.md) | Analyzer behavior and optional tools |
| [Coverage policy](COVERAGE_POLICY.md) | Required execution and inconclusive verdicts |
| [Known limitations](LIMITATIONS.md) | False positives, recovery scope, and research status |

## Evaluate the evidence

- [Measured results](BENCHMARK_RESULTS.md): method-specific results, including failures.
- [Benchmark protocol](BENCHMARK_PROTOCOL.md): splits, qualification gates, and reproducibility.
- [Dataset catalog](DATASET_CATALOG.md): sources, provenance, and acquisition conditions.
- [Latest JPEG curriculum results](JPEG_CURRICULUM_RESULTS.md): small inspected-development
  changes, not qualified real-world accuracy.
- [Model guide](MODELS.md): explicit installation and verification of optional local models.

## Maintain and deploy

- [Full-image verification](FULL_IMAGE_VERIFICATION.md): pinned tools and hardened recovery smoke.

- [Architecture](ARCHITECTURE.md) and [roadmap](ROADMAP.md).
- [Threat model](THREAT_MODEL.md) and [security reporting](../SECURITY.md).
- [Contribution guide](../CONTRIBUTING.md) and [changelog](../CHANGELOG.md).
- [Licensing boundaries](LICENSING.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
- [Cloud AI removal](MIGRATION.md): retired options and compatibility behavior.

Historical protocols and reports remain in this directory and `benchmarks/`.
Their presence does not imply that an experiment passed, a model is deployed,
or a format has qualified detection support.
