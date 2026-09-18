"""Conservative identity and grouping rules for research dataset isolation."""

from __future__ import annotations

from typing import Any


def identity_keys(sample: dict[str, Any]) -> set[tuple[str, ...]]:
    """Metadata changes must not separate a known cover from its derivatives.

    Content-hash lineages are global identities; human-assigned lineages are
    namespaced by source. Camera/device metadata is deliberately not identity.
    This cannot infer undocumented ancestry or detect perceptual duplicates.
    """
    keys: set[tuple[str, ...]] = set()
    digest = str(sample.get("sha256") or "").lower()
    if digest:
        keys.add(("hash", digest))
    lineage = str(sample.get("lineage") or "")
    if lineage:
        if len(lineage) == 64 and all(c in "0123456789abcdef" for c in lineage.lower()):
            keys.add(("hash", lineage.lower()))
        else:
            keys.add(("lineage", str(sample.get("source_group") or "unspecified"), lineage))
    return keys


def grouped_indices(samples: list[dict[str, Any]]) -> list[list[int]]:
    """Connected components sharing identity, camera or device within a source."""
    parents = list(range(len(samples)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[str, ...], int] = {}
    for index, sample in enumerate(samples):
        keys = identity_keys(sample)
        for field in ("camera", "device"):
            if sample.get(field):
                keys.add((field, str(sample["source_group"]), str(sample[field])))
        for key in sorted(keys):
            previous = owners.setdefault(key, index)
            parents[root(index)] = root(previous)
    groups: dict[int, list[int]] = {}
    for index in range(len(samples)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())
