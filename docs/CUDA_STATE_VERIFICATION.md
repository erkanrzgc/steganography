# On-device state validation verification — 2026-10-09

Implementation/protocol committed and pushed at `c9a6f94` before execution.
The [separate frozen protocol](CUDA_STATE_TIMING_PROTOCOL.md) SHA-256 is
`78798edde84b0aeb39c0d7e0e281cce9241ab74bd44b043768224dd226ac8727`.
Earlier physical timing reports and their ineligible decisions are unchanged.

## Verified software

20 generated acceptance/equivalence tests pass: complete NumPy versus tensor
state contracts, NaN/infinities, variance/counter bounds, metadata/device/layout
errors, noncontiguous values, no mutation/CPU transfer, one host scalar decision
and exact loss/state agreement across two real CPU optimizer updates. These
are not CUDA hardware kernel equivalence or independent mathematical oracles.

Kali/Python 3.11.14 full suite: **1663 passed, one live-CUDA test explicitly
skipped**, 32 warnings, 243.38s, total coverage **95.48%**. Changed training/model
files: **100% line coverage**. Ruff, mypy (140 files), diff checks and fresh
wheel/sdist pass (150/301 members). Archive checks exclude raw data/numeric
weights, credentials and provider-specific memory. Fixed model-state storage
is 19,139,920 bytes; temporary concatenations are bounded by exact shapes.
The existing 4 GiB allocator /8 GiB resident RAM limits remain enforced.

## Physical measurement remains unverified

WSL fast-forwarded to the frozen implementation and the bounded profiler
returned `completed`. Before the original JSON report could be fetched,
key-authenticated SSH connections began resetting before key exchange. The
loopback reverse listener remains present, but the remote service cannot be
inspected. Its configured four-hour expiry is a plausible, unverified cause.

No original report bytes/checksum, matching-source validation, new interval
estimate or fit eligibility has been established in this checkout. Do not
infer a successful timing/learning gate from the worker completion message.
No real training or accuracy gain; no CPU fallback or resource-cap relaxation.

Next: user restores the existing dedicated loopback SSH service without
changing host keys, password policy or expiry; fetch the existing original
report first. Do not rerun/overwrite the measurement or bootstrap keys. Then
verify bytes/source hashes/budget before publishing any outcome. Python
3.12 full-suite, Python 3.13–3.14 and full Docker remain unverified.
