"""Build an offline, generated-only CPU process-QA notebook from reviewed sources."""

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import colab_artifacts as a  # noqa: E402


def build(root: Path, out: Path) -> dict:
    names = ("core/__init__.py", "core/colab_artifacts.py", "core/colab_execution.py", "LICENSE")
    records = {}
    for name in names:
        path = root / name
        a.safe(path)
        raw = a.read(path, 65536) if path.stat().st_size else b""
        records[name] = {"sha256": a.digest(raw), "utf8": raw.decode()}
    commit = "2f41bfe2e95febcaab80f80ab4875c72073c3a32"
    packet = {"schema": "colab-process-source-v1", "base_commit": commit, "files": records}
    raw = a.encode(packet)
    checksum = a.digest(raw)
    encoded = base64.b64encode(raw).decode()
    chunks = "\n".join(f"    {encoded[n:n + 88]!r}" for n in range(0, len(encoded), 88))
    bootstrap = f'''import base64
import hashlib
import json
import os
import sys
from pathlib import Path

SOURCE_SHA256 = {checksum!r}
BASE_COMMIT = {commit!r}
SOURCE_B64 = (
{chunks}
)


def install_sources(raw, root):
    allowed = set(
        {names!r}
    )
    if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("source package hash/size mismatch")
    value = json.loads(raw)
    if (set(value) != {{"schema", "base_commit", "files"}}
            or value["schema"] != "colab-process-source-v1"
            or value["base_commit"] != BASE_COMMIT or set(value["files"]) != allowed):
        raise ValueError("source package schema mismatch")
    prepared = {{}}
    for name, entry in value["files"].items():
        if set(entry) != {{"utf8", "sha256"}}:
            raise ValueError("source entry schema mismatch")
        content = entry["utf8"].encode("utf-8")
        if len(content) > 65536 or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("source file hash/size mismatch")
        prepared[name] = content
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("source directory symlink")
    root.mkdir(mode=0o700)
    (root / "core").mkdir(mode=0o700)
    for name, content in prepared.items():
        fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
    return root


if any(name == "core" or name.startswith("core.") for name in sys.modules):
    raise RuntimeError("use a fresh CPU kernel; a core module is already loaded")
source_root = install_sources(base64.b64decode(SOURCE_B64, validate=True),
                             Path("/content/spatial-colab-process-source-20261106"))
sys.path.insert(0, str(source_root))
print("Verified generated process QA sources:", SOURCE_SHA256)
'''
    run_cell = '''from core import colab_artifacts as artifacts
from core import colab_execution as service

protocol = source_root / "generated-process-contract.json"
output = source_root / "generated-process-result"
limits = service.Limits(10, 128 * artifacts.MIB, 8 * 1024**3,
                        8 * artifacts.MIB, artifacts.MIB,
                        address_bytes=256 * artifacts.MIB, closing_seconds=1)
service.freeze(source_root, protocol, "generated_qa", {"action": "success", "value": 0}, limits)
result = service.run(source_root, protocol, output)
service.audit(source_root, protocol, output)
print(json.dumps(result, sort_keys=True, indent=2))
'''
    text = (
        "# Colab süreç denetimi — CPU yazılım sınaması\n\n"
        "Python 3 / **CPU** / Runtime Version **2026.07** seçin.\n"
        "Bu notebook yalnızca üretilmiş, küçük bir alt süreç çalıştırır; Torch yüklemez.\n"
        "Yerel çalışma diski kullanılır. Paket kurulumu, ağ isteği ve Drive bağlantısı yoktur.\n"
        "Süre, veri bölümü ve dosya sınırları çekirdeğe uygulanır; RAM izlenir.\n"
        "**Kesin cgroup RAM/swap sınırı sağlanmaz.**\n"
        "Kernel tepe RAM değeri süreç açılışını içerir.\n"
        "Aynı çalışma dizininde deneme tekrarlanamaz; kesilmiş kayıtları koruyun.\n"
        "Bu yazılımda native ölçüm/eğitim komutu yoktur. Eski deneyleri yeniden çalıştırmaz.\n"
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}]
    for code in (bootstrap, run_cell):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": code.splitlines(True)})
    for index, cell in enumerate(cells):
        cell["id"] = f"colab-process-{index:02d}"
    name = "png_bmp_colab_process_qa_20261106.ipynb"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"colab": {"name": name},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    a.directory(out)
    a.put(out / "sources.json", raw, 262144)
    a.put(out / name, a.encode(notebook), a.MIB)
    return {"source_package_sha256": checksum, "base_commit": commit,
            "source_sha256": {key: value["sha256"] for key, value in records.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(a.encode(build(Path(__file__).parent.parent, args.out)).decode())
