# Threat model

`steganography` 0.6.0 is a single-user, local-first DFIR application. Evidence,
uploaded files, archives, image parsers, optional models and external-tool
output are untrusted.

Security boundaries:

- Evidence is encrypted per object with libsodium XChaCha20 secretstream. The
  master key is derived with Argon2id and only retained in process memory while
  the vault is unlocked.
- V2 routes require an API key or an HttpOnly, SameSite session. Cookie writes
  additionally require a per-session CSRF token. CORS is not enabled.
- Archive inspection does not extract members and enforces count, depth,
  aggregate-size and compression-ratio limits.
- Optional tools have time, address-space, output, descriptor and file-size
  limits. Their absence yields `unavailable`, never a crash or a clean verdict.
- Model installation is opt-in and requires an Ed25519-signed manifest plus a
  matching SHA-256. Models and user datasets are never downloaded by default.
- Audit events form an append-only SHA-256 chain. This detects modification or
  reordering; it is not a substitute for an external timestamp/notary.

Out of scope for v1: hostile local administrators, memory inspection while the
vault is unlocked, multi-user authorization, cloud tenancy and protection from
a fully compromised host kernel.

Current limitation: a subprocess working directory is not filesystem isolation.
External executables retain the invoking user's filesystem permissions; native
analysis deadlines are cooperative. Container read-only mounts reduce exposure
but do not establish the plan's complete per-tool sandbox acceptance gate.
