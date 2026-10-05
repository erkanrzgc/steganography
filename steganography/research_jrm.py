"""Explicit, manifest-bound local JRM/FLD reference, separate from deployed ML."""

from __future__ import annotations

import hashlib
import io
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from core import jpeg_jrm as jrm
from core.fld_reference import LEARNERS, SUBSPACE, FLDReference
from core.jpeg_features import MAX_IMAGE_BYTES
from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples
from steganography.research_jpeg import write_json
from steganography.research_weighting import _jpeg_source_weights

MAX_CACHE_BYTES = 192 * 1024**2
MAX_MODEL_BYTES = 1024**2
SEED = 20261008
MAX_EXTRACTION_SECONDS = 1800


def _regular(path: Path) -> None:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ResearchManifestError("reference inputs must be regular non-symlink files")


def _fresh(out: Path) -> None:
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("reference output exists or uses a symlink")


def extract_cache(
    manifest_path: Path, out: Path, *, source: Path, split: str, workers: int = 1
) -> dict[str, Any]:
    _fresh(out)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ResearchManifestError("reference workers must be 1..4")
    manifest, manifest_hash = read_document(manifest_path)
    selected = selected_samples(manifest, split)
    if len(selected) > 4000:
        raise ResearchManifestError("reference split exceeds 4000 rows")
    jrm.require_version()

    def extract(batch):
        files = []
        for sample in batch:
            path = source / sample["path"]
            _regular(path)
            with path.open("rb") as stream:
                data = stream.read(MAX_IMAGE_BYTES + 1)
            if (
                len(data) > MAX_IMAGE_BYTES
                or len(data) != sample["size"]
                or hashlib.sha256(data).hexdigest() != sample["sha256"]
            ):
                raise ResearchManifestError("reference JPEG integrity failure")
            files.append(data)
        return jrm.jpeg_jrm_batch(files)

    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    started = time.monotonic()
    batches = [selected[i : i + jrm.MAX_BATCH] for i in range(0, len(selected), jrm.MAX_BATCH)]
    # The descriptor is created last. Interrupted/failed output is not a complete cache.
    with (
        (out / "features.f32").open("xb") as stream,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        # Queue only one group per worker; failures never drain thousands of jobs.
        for first in range(0, len(batches), workers):
            if time.monotonic() - started > MAX_EXTRACTION_SECONDS:
                raise ResearchManifestError("reference extraction deadline exceeded")
            for values in executor.map(extract, batches[first : first + workers]):
                raw = values.astype("<f4").tobytes()
                stream.write(raw)
                digest.update(raw)
        if time.monotonic() - started > MAX_EXTRACTION_SECONDS:
            raise ResearchManifestError("reference extraction deadline exceeded")
    descriptor = {
        "schema_version": "jrm-reference-cache-v1",
        "feature_version": jrm.FEATURE_VERSION,
        "dimensions": jrm.DIMENSIONS,
        "layout_sha256": jrm.LAYOUT_SHA256,
        "sealwatch_version": jrm.VERSION,
        "manifest_sha256": manifest_hash,
        "split": split,
        "dtype": "<f4",
        "data_sha256": digest.hexdigest(),
        "rows": [{k: s[k] for k in ("sha256", "lineage", "label")} for s in selected],
        "support_status": "experimental",
        "deployed": False,
    }
    write_json(out / "cache.json", descriptor)
    return descriptor


def load_cache(manifest_path: Path, cache: Path, *, checksum: str, split: str):
    manifest, digest = read_document(manifest_path)
    selected = selected_samples(manifest, split)
    descriptor, actual = read_document(cache)
    if (
        actual != checksum
        or descriptor.get("manifest_sha256") != digest
        or descriptor.get("schema_version") != "jrm-reference-cache-v1"
        or descriptor.get("feature_version") != jrm.FEATURE_VERSION
        or descriptor.get("dimensions") != jrm.DIMENSIONS
        or descriptor.get("layout_sha256") != jrm.LAYOUT_SHA256
        or descriptor.get("sealwatch_version") != jrm.VERSION
        or descriptor.get("split") != split
        or descriptor.get("dtype") != "<f4"
        or descriptor.get("rows")
        != [{k: s[k] for k in ("sha256", "lineage", "label")} for s in selected]
        or len(selected) > 4000
    ):
        raise ResearchManifestError("reference cache/manifest contract mismatch")
    path = cache.parent / "features.f32"
    _regular(path)
    size = len(selected) * jrm.OUTPUT_BYTES
    if size > MAX_CACHE_BYTES or path.stat().st_size != size:
        raise ResearchManifestError("reference cache byte limits exceeded")
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != descriptor.get("data_sha256"):
        raise ResearchManifestError("reference cache data checksum mismatch")
    features = np.frombuffer(data, dtype="<f4").reshape(len(selected), jrm.DIMENSIONS)
    if not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 1):
        raise ResearchManifestError("invalid reference cache values")
    return features, selected, descriptor


def paired_indices(samples):
    groups: dict[tuple[Any, ...], list[tuple[int, dict]]] = {}
    for index, sample in enumerate(samples):
        key = (sample.get("source_group"), sample["lineage"], sample.get("quality_factor"))
        groups.setdefault(key, []).append((index, sample))
    covers, stegos = [], []
    for members in groups.values():
        cover = [i for i, s in members if s["label"] == "cover"]
        hidden = [(i, s) for i, s in members if s["label"] == "stego"]
        if (
            len(cover) != 1
            or len(hidden) != 2
            or {s["method"] for _, s in hidden} != {"JUNIWARD", "UERD"}
        ):
            raise ResearchManifestError("reference requires complete matched-family JPEG pairs")
        for index, _ in sorted(hidden, key=lambda pair: pair[1]["method"]):
            covers.append(cover[0])
            stegos.append(index)
    return np.asarray(covers), np.asarray(stegos)


def load_model(path: Path, *, checksum: str) -> FLDReference:
    _regular(path)
    if path.stat().st_size > MAX_MODEL_BYTES:
        raise ResearchManifestError("reference model exceeds byte limit")
    with path.open("rb") as stream:
        data = stream.read(MAX_MODEL_BYTES + 1)
    if len(data) > MAX_MODEL_BYTES or hashlib.sha256(data).hexdigest() != checksum:
        raise ResearchManifestError("reference model checksum/byte limit failure")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (
            {m.filename for m in members} != {"subspaces.npy", "weights.npy", "biases.npy"}
            or len(members) != 3
            or sum(m.file_size for m in members) > MAX_MODEL_BYTES
        ):
            raise ResearchManifestError("reference model archive contract mismatch")
        shapes = {
            "subspaces.npy": (LEARNERS, SUBSPACE),
            "weights.npy": (LEARNERS, SUBSPACE),
            "biases.npy": (LEARNERS,),
        }
        # A tiny ZIP member may declare a gigantic NPY allocation. Check headers
        # before np.load, not after it has already allocated the declared array.
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(
                        stream, max_header_size=4096
                    )
                else:
                    raise ResearchManifestError("unsupported reference NPY header")
                if dtype.hasobject:
                    raise ResearchManifestError("Object arrays are forbidden in reference models")
                valid_dtype = (
                    dtype.kind in "iu" and dtype.itemsize <= 8
                    if member.filename == "subspaces.npy"
                    else dtype.kind == "f" and dtype.itemsize in (4, 8)
                )
                if (
                    shape != shapes[member.filename]
                    or not valid_dtype
                    or stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size
                ):
                    raise ResearchManifestError("reference NPY dimensions/dtype/size mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=4096) as model:
        return FLDReference(model["subspaces"], model["weights"], model["biases"])


def train_reference(manifest: Path, cache: Path, out: Path, *, cache_sha256: str) -> dict[str, Any]:
    _fresh(out)
    jrm.require_version()
    features, samples, descriptor = load_cache(
        manifest, cache, checksum=cache_sha256, split="train"
    )
    covers, stegos = paired_indices(samples)
    document, _ = read_document(manifest)
    origin_provenance: dict[str, Any] = {}
    _jpeg_source_weights(document, samples, origin_provenance)
    import sealwatch as sw

    trainer = sw.ensemble_classifier.FldEnsembleTrainer(
        np.ascontiguousarray(features[covers]),
        np.ascontiguousarray(features[stegos]),
        seed=SEED,
        seed_subspaces=SEED + 1,
        seed_bootstrap=SEED + 2,
        L=LEARNERS,
        d_sub=SUBSPACE,
        max_num_base_learners=LEARNERS,
        verbose=0,
    )
    model, records = trainer.train()
    reference = FLDReference(
        np.array([b.subspace for b in model.base_learners]),
        np.array([b.learner.w for b in model.base_learners]),
        np.array([b.learner.b for b in model.base_learners]),
    )
    # Independent numeric reconstruction must agree before saving weights.
    expected = (model.predict_confidence(features) + 1) / 2
    if not np.array_equal(reference.predict(features), expected):
        raise ResearchManifestError("upstream/numeric FLD vote mismatch")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "model.npz").open("xb") as stream:
        np.savez(
            stream,
            subspaces=reference.subspaces,
            weights=reference.weights,
            biases=reference.biases,
        )
    card = {
        "schema_version": "jrm-fld-reference-v1",
        "feature_version": jrm.FEATURE_VERSION,
        "manifest_sha256": descriptor["manifest_sha256"],
        "train_cache_sha256": cache_sha256,
        "model_sha256": hashlib.sha256((out / "model.npz").read_bytes()).hexdigest(),
        "seed": SEED,
        "learners": LEARNERS,
        "subspace": SUBSPACE,
        "paired_training_rows": len(covers),
        "training_records": records,
        "weighting": "unweighted complete paired rows; covers repeated for both families",
        "source_validation": origin_provenance["source_balance"],
        "score": "(mean sign(FLD margin) + 1) / 2; vote fraction, not calibrated probability",
        "threshold": 0.5,
        "ties": "score >= threshold; no upstream randomized tie decisions",
        "upstream_training_vote_parity": True,
        "support_status": "experimental",
        "calibrated": False,
        "deployed": False,
        "sealwatch_version": jrm.VERSION,
    }
    write_json(out / "model-card.json", card)
    return card


def predict_reference(
    manifest: Path, cache: Path, model_dir: Path, out: Path, *, cache_sha256: str, card_sha256: str
) -> dict[str, Any]:
    _fresh(out)
    features, samples, descriptor = load_cache(
        manifest, cache, checksum=cache_sha256, split="validation"
    )
    card, digest = read_document(model_dir / "model-card.json")
    if (
        digest != card_sha256
        or card.get("manifest_sha256") != descriptor["manifest_sha256"]
        or card.get("feature_version") != jrm.FEATURE_VERSION
        or card.get("schema_version") != "jrm-fld-reference-v1"
        or card.get("sealwatch_version") != jrm.VERSION
        or card.get("seed") != SEED
        or card.get("learners") != LEARNERS
        or card.get("subspace") != SUBSPACE
        or card.get("calibrated") is not False
        or card.get("deployed") is not False
    ):
        raise ResearchManifestError("reference model/validation contract mismatch")
    reference = load_model(model_dir / "model.npz", checksum=card["model_sha256"])
    scores = reference.predict(features)
    report = {
        "schema_version": "jrm-reference-predictions-v1",
        "manifest_sha256": descriptor["manifest_sha256"],
        "validation_cache_sha256": cache_sha256,
        "model_card_sha256": digest,
        "support_status": "experimental",
        "calibrated": False,
        "deployed": False,
        "predictions": [
            {**{k: s[k] for k in ("sha256", "lineage", "label", "method")}, "score": float(score)}
            for s, score in zip(samples, scores, strict=True)
        ],
    }
    write_json(out, report)
    return report


def run_reference(config_path: Path, out: Path) -> dict[str, Any]:
    """CLI adapter's declared stage; no automatic acquisition or installation."""
    config, _ = read_document(config_path)
    stage = config.get("stage")
    if stage == "features":
        result = extract_cache(
            Path(config["manifest"]),
            out,
            source=Path(config["source"]),
            split=config["split"],
            workers=config.get("workers", 1),
        )
        return {"rows": len(result["rows"]), "split": result["split"], "deployed": False}
    if stage == "train":
        return train_reference(
            Path(config["manifest"]),
            Path(config["cache"]),
            out,
            cache_sha256=config["cache_sha256"],
        )
    if stage == "predict":
        return predict_reference(
            Path(config["manifest"]),
            Path(config["cache"]),
            Path(config["model_dir"]),
            out,
            cache_sha256=config["cache_sha256"],
            card_sha256=config["card_sha256"],
        )
    raise ResearchManifestError("reference stage must be features, train or predict")
