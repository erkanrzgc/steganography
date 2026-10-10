# Known limitations

The toolkit supports file inspection, controlled payload recovery, and evidence
workflows. It is not a universal hidden-data detector or an autonomous forensic
decision maker. Validate consequential findings independently.

## Detection is not proof of absence

Native detectors inspect known markers, structure, statistics, and anomalies.
Natural noise, image processing, metadata editors, ordinary application data,
and unusual Unicode can trigger false positives. Low payload rates and unknown
embedding schemes can evade these checks. A score is a prioritization aid, not
the probability that a file contains hidden data.

`confirmed` is reserved for a verified project marker or successful extraction;
it does not prove attribution or malicious intent. `no_indicators` is not a
clean-file certificate. Missing required native coverage yields `inconclusive`
for low-score results; optional tool availability is reported separately.

## Recovery has a defined scope

Project payload readers preserve v2/v3 compatibility. The bounded CTF playbook
also uses supported decoders, carving, archive inspection, and optional tools.
Decoder output remains a candidate unless extraction or a marker verifies it.
Not every candidate is meaningful, and encrypted payloads may require a
user-supplied password or wordlist. Arbitrary unknown stego schemes are not
guaranteed to be recoverable.

Job limits can prevent deeper recovery. Limits are safeguards, not findings;
retain the original evidence and inspect the report's coverage and provenance.

## Research is not a deployed accuracy claim

No research detector has passed the project's cross-source qualification gates.
Latest [JPEG curriculum results](JPEG_CURRICULUM_RESULTS.md) have two tiny
development cells at balanced accuracy 56.25%, eight at 50%, and false-positive
rates of 12.5–50%. Each cell has only eight covers and eight stegos and uses
previously inspected development groups. This is not a reliable broad gain.

Synthetic regression, numerical parity, training loss reduction, dataset
acquisition, and GPU capacity checks each establish different engineering facts.
None establishes real-world accuracy. Retain failed results; do not publish a
single overall percentage or choose a favorable cell as the project's score.
See [measured results](BENCHMARK_RESULTS.md) and [benchmark protocol](BENCHMARK_PROTOCOL.md).

## Operational boundaries

Cloud AI triage and file-upload integrations have been removed. Optional local
models remain explicit and experimental; weights and datasets are not bundled.
Third-party analyzers and executables must be reviewed separately and can have
their own behavior, vulnerabilities, and licenses.

The local audit hash chain is not an immutable ledger or trusted timestamp.
A compromised administrator, OS, or kernel is outside the local single-user
threat model. Preserve originals separately and protect vault backups.
