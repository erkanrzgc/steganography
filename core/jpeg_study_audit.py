"""Independent SciPy IDCT and native coefficient audit of prepared study data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core import jpeg_study_data
from core.jpeg_timing import read_bytes
from core.srnet_stream import document


def audit(root, manifest_sha256, *, deadline):
    import jpeglib
    from scipy.fft import idctn  # type: ignore[import-untyped]

    manifest = document(root / "manifest.json", manifest_sha256)
    jpeg_study_data.validate_manifest(manifest)
    counts: dict[str, int] = defaultdict(int)
    largest_difference = 0.0
    jpeglib.version.set("6b")
    for role in ("fit", "development"):
        with jpeg_study_data.Reader(root, manifest_sha256, role, deadline=deadline) as reader:
            for i in range(0, len(reader.samples), 2):
                deadline()
                tensors = reader.batch(np.asarray([i, i + 1], dtype=np.int64))
                previous = None
                for row, tensor in zip(reader.samples[i:i + 2], tensors, strict=True):
                    path = root / row["path"]
                    raw = read_bytes(path, row["sha256"], row["size"], maximum=256 * 1024)
                    # Native code consumes only our byte-verified private copy,
                    # never a corpus path which could be swapped after hashing.
                    with TemporaryDirectory(prefix="jpeg-independent-audit-") as folder:
                        private = Path(folder) / "input.jpg"
                        with private.open("xb") as stream:
                            stream.write(raw)
                        jpeg = jpeglib.read_dct(str(private))
                        jpeg.load()
                    if (jpeg.jpeg_color_space.name != "JCS_GRAYSCALE"
                        or (jpeg.width, jpeg.height) != (512, 512)):
                        raise ValueError("study audit requires native gray 512 JPEGs")
                    coefficients = jpeg.Y.copy()
                    qt = jpeg.get_component_qt(0).copy()
                    # No call to the preparation cosine-basis reconstruction.
                    blocks = coefficients[16:48, 16:48].astype(np.float64) * qt
                    decoded = idctn(blocks, axes=(-2, -1), norm="ortho") + 128
                    expected = decoded.transpose(0, 2, 1, 3).reshape(256, 256).astype("<f4")
                    difference = float(np.max(np.abs(tensor[0].astype(np.float64) - expected)))
                    largest_difference = max(largest_difference, difference)
                    if difference > 1e-4:
                        raise ValueError("study independent IDCT/cache comparison failed")
                    if previous is not None:
                        before, before_qt = previous
                        delta = coefficients.astype(np.int64) - before.astype(np.int64)
                        if (not np.array_equal(qt, before_qt) or not np.any(delta)
                            or np.max(np.abs(delta)) > 1 or not np.any(delta[16:48, 16:48])):
                            raise ValueError("study coefficient/quantization/signal audit failed")
                    previous = (coefficients.copy(), qt.copy())
                    counts[role] += 1
            reader.verify()
    return {"schema_version": "jpeg-study-independent-audit-v1", "status": "completed",
            "manifest_sha256": manifest_sha256, "idct_rows": dict(counts),
            "maximum_absolute_pixel_difference": largest_difference,
            "original_lineage_roles_disjoint": True, "optimizer_updates": 0,
            "accuracy_qualification": "unavailable", "deployed": False}
