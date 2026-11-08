"""Offline native ancestry review and bounded, unassigned spatial pair caches.

This service has no acquisition, role assignment, optimizer or GPU entry point.
Native execution needs its own source/data/exposure contract; fixtures suffice
to verify the adapter without opening the actual corpus.
"""

from __future__ import annotations

import io
import re
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core import colab_artifacts as a
from core import spatial_colab_data as admission
from core import spatial_commons_native as native
from core import spatial_training_data as scattered

RECIPE = "native-spatial-scattered-40p0-v1"
MAX_SOURCES = 256
MAX_MANIFEST = 2 * a.MIB
MAX_OUTPUT = 64 * a.MIB


@dataclass(frozen=True)
class Source:
    id: str
    mode: str
    group: str
    author: str
    page: str
    original_sha256: str
    crop_sha256: str
    license: str
    license_url: str
    evidence_sha256: str
    role: str = "unassigned"

    def __post_init__(self) -> None:
        for name in (self.id, self.group):
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9-]{1,80}", name):
                raise ValueError("native source/group identity")
        for checksum in (self.original_sha256, self.crop_sha256, self.evidence_sha256):
            a.identity(checksum)
        if (
            not isinstance(self.mode, str)
            or not isinstance(self.license, str)
            or not isinstance(self.license_url, str)
            or self.mode not in ("L", "RGB")
            or self.role != "unassigned"
            or not isinstance(self.author, str)
            or not 1 <= len(self.author) <= 1024
            or not self.author.strip()
            or self.license not in native.LICENSES
            or self.license_url != native.LICENSES[self.license]
            or not isinstance(self.page, str)
            or len(self.page) > 4096
        ):
            raise ValueError("native mode/license/unassigned provenance required")
        page = urllib.parse.urlsplit(self.page)
        if (
            page.scheme != "https"
            or page.netloc != "commons.wikimedia.org"
            or not page.path.startswith("/wiki/File:")
            or page.query
            or page.fragment
        ):
            raise ValueError("native source page allowlist")


def sources(rows: tuple[Source, ...]) -> tuple[Source, ...]:
    if (
        not isinstance(rows, tuple)
        or not 1 <= len(rows) <= MAX_SOURCES
        or any(type(row) is not Source for row in rows)
    ):
        raise ValueError("bounded immutable native catalog required")
    for field in ("id", "original_sha256", "crop_sha256"):
        if len({getattr(row, field) for row in rows}) != len(rows):
            raise ValueError("duplicate native original or crop")
    if max(Counter(row.mode for row in rows).values()) > scattered.MAX_ORIGINALS:
        raise ValueError("native per-mode source limit")
    return rows


def components(rows: tuple[Source, ...]) -> tuple[tuple[str, ...], ...]:
    """Merge all known group/author/page connections, including across modes."""
    sources(rows)
    parents = list(range(len(rows)))

    def root(i: int) -> int:
        while parents[i] != i:
            i = parents[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        author = " ".join(unicodedata.normalize("NFKC", row.author).casefold().split())
        page = unicodedata.normalize("NFC", urllib.parse.unquote(row.page)).replace("_", " ")
        for key in (("group", row.group), ("author", author), ("page", page)):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    merged: dict[int, list[str]] = {}
    for i, row in enumerate(rows):
        merged.setdefault(root(i), []).append(row.id)
    return tuple(sorted(tuple(sorted(ids)) for ids in merged.values()))


def proposal(rows: tuple[Source, ...], roles: dict[str, str]) -> None:
    """Validate a hypothetical complete grouping; do not change actual roles."""
    groups = components(rows)
    if (
        not isinstance(roles, dict)
        or set(roles) != {row.id for row in rows}
        or any(role not in ("train", "development", "future_reserved") for role in roles.values())
    ):
        raise ValueError("complete native role proposal required")
    if any(len({roles[source_id] for source_id in group}) != 1 for group in groups):
        raise ValueError("native common-ancestry role leakage")


def feasibility(rows: tuple[Source, ...]) -> dict:
    """Capacity bounds, not a role assignment or proof of unknown ancestry."""
    groups = components(rows)
    lookup = {row.id: row for row in rows}
    domains = {}
    for mode in ("L", "RGB"):
        sizes = sorted(
            (sum(lookup[source_id].mode == mode for source_id in group) for group in groups),
            reverse=True,
        )
        sizes = [size for size in sizes if size]
        minimum = next((i for i in range(1, len(sizes) + 1) if sum(sizes[:i]) >= 80), None)
        domains[mode] = {
            "originals": sum(sizes),
            "known_ancestry_groups": len(sizes),
            "group_sizes_descending": sizes,
            "missing_originals_to_128": max(0, 128 - sum(sizes)),
            "missing_groups_to_16": max(0, 16 - len(sizes)),
            "ten_group_train_capacity_upper_bound": sum(sizes[:10]),
            "minimum_known_groups_covering_80": minimum,
            "target_128_inventory_count_met": sum(sizes) >= 128,
            "ten_group_80_train_capacity_met": sum(sizes[:10]) >= 80,
        }
    return {
        "schema": "spatial-native-group-feasibility-v1",
        "status": "inventory_bounds_only",
        "roles": "unassigned",
        "known_components": [list(group) for group in groups],
        "domains": domains,
        "shared_mode_components": sum(
            len({lookup[source_id].mode for source_id in group}) > 1 for group in groups
        ),
        "unknown_common_ancestry_ruled_out": False,
        "joint_partition_feasibility_proven": False,
        "optimizer_updates": 0,
    }


def reviewed_snapshot(path: Path, checksum: str) -> tuple[Source, ...]:
    import json

    a.identity(checksum)
    raw = a.read(path, MAX_MANIFEST)
    if a.digest(raw) != checksum:
        raise ValueError("native snapshot hash")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "spatial-native-expansion-snapshot-v1"
        or value.get("status") != "ready_prepared_subset_only"
        or value.get("roles") != "unassigned"
        or not isinstance(value.get("originals"), list)
        or not 1 <= len(value["originals"]) <= MAX_SOURCES
    ):
        raise ValueError("reviewed unassigned native snapshot required")
    rows = []
    for row in value["originals"]:
        if not isinstance(row, dict) or row.get("role") != "unassigned":
            raise ValueError("native snapshot source role")
        try:
            rows.append(
                Source(
                    id=row["id"],
                    mode=row["native_mode"],
                    group=row["group"],
                    author=row["author"],
                    page=row["page"],
                    original_sha256=row["original_sha256"],
                    crop_sha256=row["crop_sha256"],
                    license=row["license"],
                    license_url=row["license_url"],
                    evidence_sha256=row["evidence_sha256"],
                )
            )
        except KeyError as exc:
            raise ValueError("native snapshot source fields") from exc
    return sources(tuple(rows))


@dataclass(frozen=True)
class Input:
    source: Source
    cover: Path
    sha256: str

    def __post_init__(self) -> None:
        a.identity(self.sha256)
        if (
            type(self.source) is not Source
            or not isinstance(self.cover, Path)
            or self.cover.name not in (self.source.id + "-cover.png", self.source.id + "-cover.bmp")
        ):
            raise ValueError("explicit native cover reference required")


def prepare(inputs: tuple[Input, ...], out: Path) -> dict:
    if not isinstance(inputs, tuple) or any(type(item) is not Input for item in inputs):
        raise ValueError("immutable native cover references required")
    rows = sources(tuple(item.source for item in inputs))
    result: dict = {
        "schema": "spatial-native-cache-v1",
        "recipe": RECIPE,
        "roles": "unassigned",
        "sources": [
            {"source": asdict(item.source), "input_sha256": item.sha256} for item in inputs
        ],
        "pixel_bytes": sum(256 * 256 * (1 if row.mode == "L" else 3) for row in rows),
        "optimizer_updates": 0,
    }
    raw_manifest = a.encode(result)
    if result["pixel_bytes"] + len(raw_manifest) > MAX_OUTPUT or len(raw_manifest) > MAX_MANIFEST:
        raise ValueError("native cache output limit")
    a.directory(out)
    for item in inputs:
        raw = a.read(item.cover, admission.MAX_FILE)
        if a.digest(raw) != item.sha256:
            raise ValueError("native cover hash")
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (256, 256):
                raise ValueError("fixed native prepared crop required")
        if item.cover.suffix == ".png":
            values = native.native_pixels(raw, item.source.mode)
        else:
            if raw[:2] != b"BM":
                raise ValueError("native BMP signature required")
            values = admission.crop(raw)
        values = values[:, :, None] if values.ndim == 2 else values
        values = scattered.pixels(values)
        if values.shape[2] != (1 if item.source.mode == "L" else 3):
            raise ValueError("native cache mode mismatch")
        encoded = values.tobytes()
        if a.digest(encoded) != item.source.crop_sha256:
            raise ValueError("native crop hash")
        name = item.source.id + ".u8"
        a.put(out / name, encoded, 256 * 256 * 3)
    a.put(out / "cache.json", raw_manifest, MAX_MANIFEST)
    return result


@dataclass(frozen=True)
class Cache:
    root: Path
    rows: tuple[Source, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        sources(self.rows)
        a.identity(self.manifest_sha256)
        if not isinstance(self.root, Path):
            raise ValueError("native cache directory required")


def load(root: Path, checksum: str) -> Cache:
    import json

    a.identity(checksum)
    raw = a.read(root / "cache.json", MAX_MANIFEST)
    if a.digest(raw) != checksum:
        raise ValueError("native cache manifest hash")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "recipe", "roles", "sources", "pixel_bytes", "optimizer_updates"}
        or value["schema"] != "spatial-native-cache-v1"
        or value["recipe"] != RECIPE
        or value["roles"] != "unassigned"
        or type(value["optimizer_updates"]) is not int
        or value["optimizer_updates"] != 0
        or not isinstance(value["sources"], list)
        or not 1 <= len(value["sources"]) <= MAX_SOURCES
    ):
        raise ValueError("native cache schema")
    rows = []
    for record in value["sources"]:
        if not isinstance(record, dict) or set(record) != {"source", "input_sha256"}:
            raise ValueError("native cache source record")
        a.identity(record["input_sha256"])
        if not isinstance(record["source"], dict) or set(record["source"]) != set(
            Source.__dataclass_fields__
        ):
            raise ValueError("native cache source fields")
        rows.append(Source(**record["source"]))
    catalog = sources(tuple(rows))
    total = sum(256 * 256 * (1 if row.mode == "L" else 3) for row in catalog)
    if (
        type(value["pixel_bytes"]) is not int
        or value["pixel_bytes"] != total
        or total + len(raw) > MAX_OUTPUT
    ):
        raise ValueError("native cache byte accounting")
    return Cache(root, catalog, checksum)


def pair(cache: Cache, index: int, payload_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    scattered.seed(payload_seed)
    if type(cache) is not Cache or type(index) is not int or not 0 <= index < len(cache.rows):
        raise ValueError("native cache source/index required")
    row = cache.rows[index]
    channels = 1 if row.mode == "L" else 3
    raw = a.read(cache.root / (row.id + ".u8"), 256 * 256 * channels)
    if len(raw) != 256 * 256 * channels or a.digest(raw) != row.crop_sha256:
        raise ValueError("native cached pixels hash/extent")
    cover = np.frombuffer(raw, dtype="u1").reshape(256, 256, channels).copy()
    identity = a.digest(
        a.encode(
            {
                "recipe": RECIPE,
                "original": row.original_sha256,
                "mode": row.mode,
                "seed": payload_seed,
            }
        )
    )
    rng = np.random.Generator(np.random.PCG64(int(identity[:16], 16)))
    payload = rng.integers(0, 256, (cover.size * 2 // 5) // 8, dtype="u1").tobytes()
    control, selected = scattered.embed(cover, payload, int(identity[16:24], 16))
    if scattered.recover(control, selected) != payload:
        raise ValueError("native scattered payload recovery failed")
    return (
        cover,
        control,
        {
            "id": row.id,
            "group": row.group,
            "role": "unassigned",
            "cover_sha256": row.crop_sha256,
            "control_sha256": a.digest(control.tobytes()),
            "payload_sha256": a.digest(payload),
            "payload_bytes": len(payload),
            "positions_sha256": a.digest(selected.astype("<i8").tobytes()),
        },
    )


def batch(
    cache: Cache, mode: str, indices: tuple[int, ...], payload_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(cache) is not Cache
        or mode not in ("L", "RGB")
        or not isinstance(indices, tuple)
        or not 1 <= len(indices) <= 4
        or any(
            type(i) is not int or not 0 <= i < len(cache.rows) or cache.rows[i].mode != mode
            for i in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("one to four distinct native pairs of one mode required")
    values: list[np.ndarray] = []
    for index in indices:
        cover, control, _ = pair(cache, index, payload_seed)
        values.extend((cover.transpose(2, 0, 1), control.transpose(2, 0, 1)))
    return np.stack(values).astype("<f4"), np.tile(np.array([0, 1], dtype="<i8"), len(indices))
