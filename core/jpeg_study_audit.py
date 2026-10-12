"""Independent SciPy IDCT and native coefficient audit of prepared study data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core import jpeg_study_data, jpeg_study_views
from core.jpeg_timing import read_bytes
from core.srnet_stream import document


def audit(root, manifest_sha256, *, deadline):
    import jpeglib
    from scipy.fft import idctn  # type: ignore[import-untyped]

    manifest = document(root / "manifest.json", manifest_sha256)
    jpeg_study_data.validate_manifest(manifest)
    counts: dict[str, int] = defaultdict(int)
    identical_views = {role: [0] * 5 for role in ("fit", "development")}
    five = manifest["schema_version"] == jpeg_study_views.SCHEMA
    regions = jpeg_study_views.REGIONS if five else (jpeg_study_views.REGIONS[0],)
    largest_difference = 0.0
    jpeglib.version.set("6b")
    for role in ("fit", "development"):
        with jpeg_study_data.Reader(root, manifest_sha256, role, deadline=deadline) as reader:
            for i in range(0, len(reader.samples), 2):
                deadline()
                indices = np.asarray([i, i + 1], dtype=np.int64)
                previous = None
                for position, row in enumerate(reader.samples[i:i + 2]):
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
                    for view, (left, top, width, height) in enumerate(regions):
                        blocks = coefficients[top // 8:(top + height) // 8,
                                              left // 8:(left + width) // 8].astype(np.float64) * qt
                        decoded = idctn(blocks, axes=(-2, -1), norm="ortho") + 128
                        expected = decoded.transpose(0, 2, 1, 3).reshape(256, 256).astype("<f4")
                        tensors = reader.batch(indices, np.full(2, view, dtype=np.int64)) if five \
                            else reader.batch(indices)
                        difference = float(np.max(np.abs(
                            tensors[position, 0].astype(np.float64) - expected)))
                        largest_difference = max(largest_difference, difference)
                        if difference > 1e-4:
                            raise ValueError("study independent IDCT/cache comparison failed")
                    if previous is not None:
                        before, before_qt = previous
                        delta = coefficients.astype(np.int64) - before.astype(np.int64)
                        if (not np.array_equal(qt, before_qt) or not np.any(delta)
                            or np.max(np.abs(delta)) > 1
                            or (not five and not np.any(delta[16:48, 16:48]))):
                            raise ValueError("study coefficient/quantization/signal audit failed")
                        if five:
                            for view, (left, top, width, height) in enumerate(regions):
                                identical_views[role][view] += int(not np.any(delta[
                                    top // 8:(top + height) // 8, left // 8:(left + width) // 8]))
                    previous = (coefficients.copy(), qt.copy())
                    counts[role] += 1
            reader.verify()
    result = {"schema_version": "jpeg-study-independent-audit-v2" if five
              else "jpeg-study-independent-audit-v1", "status": "completed",
            "manifest_sha256": manifest_sha256, "idct_rows": dict(counts),
            "maximum_absolute_pixel_difference": largest_difference,
            "original_lineage_roles_disjoint": True, "optimizer_updates": 0,
            "accuracy_qualification": "unavailable", "deployed": False}
    if five:
        result.update(representation=jpeg_study_views.contract(),
                      idct_views={role: count * 5 for role, count in counts.items()},
                      identical_coefficient_view_pairs=identical_views)
    return result
