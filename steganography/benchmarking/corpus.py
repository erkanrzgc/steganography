"""Generate a reproducible clean/stego evaluation corpus."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from core.payload import HEADER_OVERHEAD
from core.service import StegoService
from core.version import __version__
from registry import Registry

CORPUS_SCHEMA_VERSION = "1.0"
DEFAULT_CORPUS_SEED = 20260813
DEFAULT_DENSITIES: tuple[tuple[str, float], ...] = (
    ("low", 0.05),
    ("medium", 0.25),
    ("high", 0.60),
)
_MARKER_FILE = ".steganography-corpus"
_PRACTICAL_CAPACITY = 4096
_PLACEMENT_KEY = "steganography-benchmark-placement-key-v1"


@dataclass(frozen=True, slots=True)
class CorpusRecipe:
    id: str
    method: str
    extension: str
    cover_kind: str
    options: dict[str, Any] | None = None
    requires_dct: bool = False


@dataclass(frozen=True, slots=True)
class CorpusSample:
    id: str
    path: str
    label: Literal["clean", "stego"]
    method: str
    format: str
    density: str
    target_density: float
    actual_utilization: float
    payload_bytes: int
    carrier_capacity: int
    sha256: str
    pair_id: str


_RECIPES: tuple[CorpusRecipe, ...] = (
    CorpusRecipe("image_lsb_png", "image_lsb", ".png", "noise_rgb"),
    CorpusRecipe("image_lsb_bmp", "image_lsb", ".bmp", "noise_rgb"),
    CorpusRecipe(
        "image_lsb_scatter_png",
        "image_lsb_scatter",
        ".png",
        "noise_rgb",
        {"channels": "rgb"},
    ),
    CorpusRecipe("audio_wav", "audio_wav", ".wav", "wav_pcm16"),
    CorpusRecipe("text_whitespace", "text_whitespace", ".txt", "many_lines"),
    CorpusRecipe("text_zerowidth", "text_zerowidth", ".txt", "plain_text"),
    CorpusRecipe("filestruct_exif_jpeg", "filestruct_exif", ".jpg", "noise_rgb"),
    CorpusRecipe("image_jpeg_trailer", "image_jpeg", ".jpg", "noise_rgb"),
    CorpusRecipe("filestruct_trailer_pdf", "filestruct_trailer", ".pdf", "pdf"),
    CorpusRecipe("filestruct_trailer_gif", "filestruct_trailer", ".gif", "gif"),
    CorpusRecipe(
        "image_jpeg_dct",
        "image_jpeg_dct",
        ".jpg",
        "noise_rgb",
        requires_dct=True,
    ),
)

EXTENDED_RECIPES: tuple[CorpusRecipe, ...] = (
    *_RECIPES,
    CorpusRecipe("image_lsb_gradient_png", "image_lsb", ".png", "gradient_rgb"),
    CorpusRecipe("image_lsb_gradient_bmp", "image_lsb", ".bmp", "gradient_rgb"),
    CorpusRecipe(
        "image_lsb_scatter_gradient_png",
        "image_lsb_scatter",
        ".png",
        "gradient_rgb",
        {"channels": "rgb"},
    ),
)


def generate_corpus(
    output: Path,
    *,
    seed: int = DEFAULT_CORPUS_SEED,
    force: bool = False,
    include_dct: bool = True,
    methods: set[str] | None = None,
    densities: tuple[tuple[str, float], ...] = DEFAULT_DENSITIES,
    recipes: tuple[CorpusRecipe, ...] | None = None,
) -> dict[str, Any]:
    """Build a corpus atomically and return its manifest.

    Existing directories are replaced only with ``force=True`` and only when
    they contain this generator's marker file.
    """
    output = Path(output).resolve()
    _validate_output_target(output, force=force)
    _validate_densities(densities)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        manifest = _generate_into(
            staging,
            seed=seed,
            include_dct=include_dct,
            methods=methods,
            densities=densities,
            recipes=recipes,
        )
        (staging / _MARKER_FILE).write_text(
            f"schema={CORPUS_SCHEMA_VERSION}\nseed={seed}\n", encoding="utf-8"
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_manifest(corpus: Path, *, verify_files: bool = True) -> dict[str, Any]:
    corpus = Path(corpus).resolve()
    manifest_path = corpus / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported corpus schema: {data.get('schema_version')!r}"
        )
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("corpus manifest contains no samples")
    seen: set[str] = set()
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError(f"duplicate or empty sample id: {sample_id!r}")
        seen.add(sample_id)
        relative = Path(str(sample.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe sample path: {relative}")
        path = (corpus / relative).resolve()
        if corpus not in path.parents:
            raise ValueError(f"sample escapes corpus directory: {relative}")
        if verify_files:
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256_file(path)
            if actual != sample.get("sha256"):
                raise ValueError(f"sample checksum mismatch: {sample_id}")
    expected_digest = _corpus_digest(samples)
    if expected_digest != data.get("corpus_digest"):
        raise ValueError("corpus manifest digest mismatch")
    return data


def available_recipes(extended: bool = False) -> tuple[CorpusRecipe, ...]:
    return EXTENDED_RECIPES if extended else _RECIPES


def _generate_into(
    root: Path,
    *,
    seed: int,
    include_dct: bool,
    methods: set[str] | None,
    densities: tuple[tuple[str, float], ...],
    recipes: tuple[CorpusRecipe, ...] | None = None,
) -> dict[str, Any]:
    registry = Registry()
    registry.autodiscover()
    service = StegoService(registry)
    samples: list[CorpusSample] = []
    skipped: list[dict[str, str]] = []
    source_recipes = recipes if recipes is not None else _RECIPES
    selected = [
        recipe
        for recipe in source_recipes
        if methods is None or recipe.method in methods or recipe.id in methods
    ]
    if not selected:
        raise ValueError("no corpus recipes matched the requested methods")

    for recipe in selected:
        carrier = registry.get_carrier(recipe.method)
        if recipe.requires_dct and not include_dct:
            skipped.append({"recipe": recipe.id, "reason": "DCT excluded by request"})
            continue
        unavailable = getattr(carrier, "unavailable_reason", None)
        if unavailable:
            skipped.append({"recipe": recipe.id, "reason": str(unavailable)})
            continue
        cover = root / "_covers" / f"{recipe.id}{recipe.extension}"
        cover.parent.mkdir(parents=True, exist_ok=True)
        _create_cover(recipe, cover, _derived_seed(seed, recipe.id))
        capacity = carrier.capacity(cover)
        practical = min(capacity, _PRACTICAL_CAPACITY)
        if practical <= HEADER_OVERHEAD:
            skipped.append({"recipe": recipe.id, "reason": "insufficient capacity"})
            continue

        for density_name, density_value in densities:
            pair_id = f"{recipe.id}-{density_name}"
            sample_dir = root / "samples" / recipe.id / density_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            clean = sample_dir / f"clean{recipe.extension}"
            stego = sample_dir / f"stego{recipe.extension}"
            shutil.copyfile(cover, clean)
            target_envelope = max(
                HEADER_OVERHEAD + 1, round(practical * density_value)
            )
            payload_size = max(1, target_envelope - HEADER_OVERHEAD)
            payload = _deterministic_bytes(seed, pair_id, payload_size)
            payload_path = root / "_payload.bin"
            payload_path.write_bytes(payload)
            options = dict(recipe.options or {})
            if recipe.method in {"image_lsb_scatter", "image_jpeg_dct"}:
                options["placement_salt"] = hashlib.sha256(
                    f"{seed}:{pair_id}:salt".encode()
                ).digest()[:16]
            service.embed(
                payload_path,
                clean,
                stego,
                method=recipe.method,
                steg_key=_PLACEMENT_KEY,
                options=options,
                no_clobber=True,
            )
            recovered, _, _ = service.extract(
                stego,
                method=recipe.method,
                steg_key=_PLACEMENT_KEY,
            )
            if recovered != payload:
                raise RuntimeError(f"corpus round-trip failed: {pair_id}")
            utilization = (payload_size + HEADER_OVERHEAD) / max(capacity, 1)
            clean_id = f"{pair_id}-clean"
            stego_id = f"{pair_id}-stego"
            samples.extend(
                (
                    _sample(
                        root,
                        clean,
                        sample_id=clean_id,
                        label="clean",
                        recipe=recipe,
                        density_name=density_name,
                        density_value=density_value,
                        utilization=0.0,
                        payload_bytes=0,
                        capacity=capacity,
                        pair_id=pair_id,
                    ),
                    _sample(
                        root,
                        stego,
                        sample_id=stego_id,
                        label="stego",
                        recipe=recipe,
                        density_name=density_name,
                        density_value=density_value,
                        utilization=utilization,
                        payload_bytes=payload_size,
                        capacity=capacity,
                        pair_id=pair_id,
                    ),
                )
            )
    (root / "_payload.bin").unlink(missing_ok=True)
    shutil.rmtree(root / "_covers", ignore_errors=True)
    serialized = [asdict(sample) for sample in samples]
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "tool_version": __version__,
        "seed": seed,
        "density_targets": {name: value for name, value in densities},
        "sample_count": len(serialized),
        "corpus_digest": _corpus_digest(serialized),
        "samples": serialized,
        "skipped": skipped,
    }


def _sample(
    root: Path,
    path: Path,
    *,
    sample_id: str,
    label: Literal["clean", "stego"],
    recipe: CorpusRecipe,
    density_name: str,
    density_value: float,
    utilization: float,
    payload_bytes: int,
    capacity: int,
    pair_id: str,
) -> CorpusSample:
    return CorpusSample(
        id=sample_id,
        path=path.relative_to(root).as_posix(),
        label=label,
        method=recipe.method,
        format=recipe.extension.lstrip("."),
        density=density_name,
        target_density=density_value,
        actual_utilization=round(utilization, 8),
        payload_bytes=payload_bytes,
        carrier_capacity=capacity,
        sha256=_sha256_file(path),
        pair_id=pair_id,
    )


def _create_cover(recipe: CorpusRecipe, path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if recipe.cover_kind == "noise_rgb":
        array = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(array, "RGB")
        if recipe.extension == ".jpg":
            image.save(path, format="JPEG", quality=92, optimize=False, progressive=False)
        else:
            image.save(path, format=recipe.extension.lstrip(".").upper())
        return
    if recipe.cover_kind == "gradient_rgb":
        x = np.linspace(0, 0.5 * np.pi, 256)
        y = np.linspace(0, 0.5 * np.pi, 256)
        xx, yy = np.meshgrid(x, y)
        base = ((np.sin(xx) * np.cos(yy) + 1.0) * 110.0 + 15.0).astype(np.uint8)
        offset = seed % 17
        r = np.clip(base + offset, 0, 255).astype(np.uint8)
        g = np.clip(base + offset + 5, 0, 255).astype(np.uint8)
        b = np.clip(base + offset + 10, 0, 255).astype(np.uint8)
        image = Image.fromarray(np.stack([r, g, b], axis=2), "RGB")
        if recipe.extension == ".jpg":
            image.save(path, format="JPEG", quality=92, optimize=False, progressive=False)
        else:
            image.save(path, format=recipe.extension.lstrip(".").upper())
        return
    if recipe.cover_kind == "gif":
        array = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
        Image.fromarray(array, "L").save(path, format="GIF")
        return
    if recipe.cover_kind == "wav_pcm16":
        samples = rng.integers(-12000, 12000, size=48000, dtype=np.int16)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            output.writeframes(samples.tobytes())
        return
    if recipe.cover_kind == "many_lines":
        path.write_text(
            "".join(
                f"deterministic corpus line {index:05d} value {seed % 997}\n"
                for index in range(20000)
            ),
            encoding="utf-8",
        )
        return
    if recipe.cover_kind == "plain_text":
        path.write_text(
            "This is an ordinary deterministic UTF-8 document.\n" * 64,
            encoding="utf-8",
        )
        return
    if recipe.cover_kind == "pdf":
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
            b"xref\n0 2\n0000000000 65535 f \n0000000009 00000 n \n"
            b"trailer\n<< /Size 2 /Root 1 0 R >>\nstartxref\n49\n%%EOF"
        )
        return
    raise ValueError(f"unknown cover kind: {recipe.cover_kind}")


def _validate_output_target(output: Path, *, force: bool) -> None:
    if not output.exists():
        return
    if not force:
        raise FileExistsError(f"corpus output already exists: {output}")
    if not output.is_dir() or not (output / _MARKER_FILE).is_file():
        raise ValueError(
            "refusing to replace a directory not created by the corpus generator"
        )


def _validate_densities(densities: tuple[tuple[str, float], ...]) -> None:
    if not densities:
        raise ValueError("at least one density is required")
    names: set[str] = set()
    for name, value in densities:
        if not name or name in names:
            raise ValueError(f"duplicate or empty density name: {name!r}")
        if not 0 < value <= 1:
            raise ValueError(f"density must be in (0, 1]: {value}")
        names.add(name)


def _derived_seed(seed: int, context: str) -> int:
    digest = hashlib.sha256(f"{seed}:{context}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _deterministic_bytes(seed: int, context: str, size: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < size:
        output.extend(
            hashlib.sha256(f"{seed}:{context}:{counter}".encode()).digest()
        )
        counter += 1
    return bytes(output[:size])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_digest(samples: list[dict[str, Any]]) -> str:
    identity = [
        {
            "id": sample.get("id"),
            "path": sample.get("path"),
            "sha256": sample.get("sha256"),
            "label": sample.get("label"),
        }
        for sample in samples
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
