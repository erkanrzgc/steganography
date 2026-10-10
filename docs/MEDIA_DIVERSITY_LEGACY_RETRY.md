# Legacy-manifest compatible acquisition — 2026-10-10

The initial three acquisition invocations failed the reserved-identity guard
before creating outputs or downloading any media. The old frozen BOSS pilot
and Kodak manifests contain original byte SHA-256 identities but no lineage
field. This was our incorrect legacy-schema assumption, not an upstream failure.
Their historical metadata and the first protocol remain unchanged.

Accept omitted lineage only for these two exact previously audited manifest
hashes: `a2412f69124b3c2cb84907d8f276dd7a2f21a0aacb6eac212a123c889822f832`
and `a81fb50acdf989fe3cb37c6ac88316140e611d5f1da3046962426498ac1a7728`.
Use the original byte hash as its exclusion identity. Every other manifest
still requires both valid identities. All ten manifests and all existing
download/parser/geometry/role/license/time/byte guards remain required.

Separately run explicit fresh `*-verified-diversity-20261010` acquisitions.
Preserve the original logs and failure record; no automatic retry, relabeled
success, threshold tuning, fitting or paid cloud provisioning.
