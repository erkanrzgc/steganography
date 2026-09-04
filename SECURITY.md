# Security policy

## Reporting a vulnerability

Please use GitHub private vulnerability reporting for this repository. Do not
open a public issue containing exploit details, sensitive evidence, passwords,
keys, or unredacted forensic material. Include the affected version, component,
reproduction steps, expected impact, and any safe mitigation you identified.

Maintainers will acknowledge a report when available, validate scope, and
coordinate remediation and disclosure. This volunteer project cannot promise a
specific response SLA.

## Supported code

Security fixes target the latest released version and the current default
branch. Older versions may be asked to upgrade. Verify release artifacts and
review the [threat model](docs/THREAT_MODEL.md) before deployment.

## Operational safety

Use a strong, unique vault password; preserve original evidence separately;
keep the state directory access-controlled and backed up; do not expose the API
directly to an untrusted network; and treat every analyzed file, archive,
external-tool output, model, and dataset as untrusted.

The audit hash chain detects database event modification or reordering but is
not an external timestamp or immutable ledger. An administrator or compromised
kernel is outside the local single-user threat model.
