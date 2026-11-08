"""Offline generated handoff tests; actual reserved/development pixels stay closed."""

import base64
import copy
import importlib.util
import json
import runpy
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core import colab_artifacts as a
from core import colab_execution as process
from core import spatial_colab_handoff as h
from core import spatial_native_data as data
from core import spatial_native_pilot as p


@pytest.fixture
def generated(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    return p.generated_cache(inputs, "RGB", 1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="optional Torch absent; isolated generated profile unavailable")
def test_source_delivery_independent_of_app_imports(tmp_path):
    root = Path(h.__file__).parent.parent
    packet = tmp_path / "sources.json"
    sha = h.source_package(root, packet)
    installed = tmp_path / "installed"
    info = h.install_sources(packet, sha, installed)
    assert info["source_package_sha256"] == sha
    assert (installed / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()
    with pytest.raises(FileExistsError):
        h.install_sources(packet, sha, installed)
    with pytest.raises(ValueError, match="hash"):
        h.install_sources(packet, "0" * 64, tmp_path / "bad")
    code = '''import sys
sys.path.insert(0,sys.argv[1])
class RejectApplication:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'steganography','registry','modules','config','api','report','ui'}:
   raise ImportError('application package forbidden in this isolated profile QA')
sys.meta_path.insert(0,RejectApplication())
from pathlib import Path
from core import spatial_colab_profile as profile
from core import spatial_native_pilot as old
root=Path(sys.argv[1]);inputs=root/'inputs';inputs.mkdir()
cache,split=old.generated_cache(inputs,'RGB',1)
result=profile.measure(cache,split,root/'generated-profile','b'*64,device='cpu',seconds=180)
assert result['generated_optimizer_updates']==6 and result['native_optimizer_updates']==0
assert not result['physical_cuda']
assert not any(n.split('.')[0] in {'steganography','modules','registry'} for n in sys.modules)
print('isolated_generated_profile_verified')
'''
    completed = subprocess.run(  # noqa: S603 -- owned isolated generated CPU fixture
        [sys.executable, "-I", "-B", "-c", code, str(installed)],
        capture_output=True, text=True, timeout=200, check=False,
        env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated_generated_profile_verified"


@pytest.mark.parametrize("fault", ["schema", "path", "entry", "extent", "hash", "base"])
def test_malformed_source_packet_rejected_before_directory(tmp_path, fault):
    value = {"schema": "colab-native-source-package-v1", "base_commit": h.BASE_COMMIT,
             "files": {"core/__init__.py": {"utf8": "", "sha256": a.digest(b"")},
                       "core/x.py": {"utf8": "x=1\n", "sha256": a.digest(b"x=1\n")},
                       "LICENSE": {"utf8": "MIT\n", "sha256": a.digest(b"MIT\n")}}}
    if fault == "schema":
        value["schema"] = "bad"
    elif fault == "path":
        value["files"]["../escape.py"] = value["files"].pop("core/x.py")
    elif fault == "entry":
        value["files"]["core/x.py"]["extra"] = "no"
    elif fault == "extent":
        value["files"]["core/x.py"]["utf8"] = "x" * (a.MIB + 1)
    elif fault == "hash":
        value["files"]["core/x.py"]["sha256"] = "0" * 64
    else:
        value["base_commit"] = "0" * 40
    path, out = tmp_path / "packet.json", tmp_path / "output"
    path.write_bytes(a.encode(value))
    with pytest.raises(ValueError):
        h.install_sources(path, a.digest(path.read_bytes()), out)
    assert not out.exists()


def test_train_only_copy_does_not_open_closed_pixel_symlinks(generated, tmp_path):
    cache, split = generated
    for source_id in (*split.development, *split.future_reserved):
        path = cache.root / (source_id + ".u8")
        path.rename(path.with_suffix(".closed"))
        path.symlink_to("/forbidden-closed-pixels")
    packet = tmp_path / "cache-packet.json"
    sha = h.cache_package(cache, split, packet)
    content = a.document(packet, h.CACHE_LIMIT)
    assert set(content["files"]) == set(split.train)
    copied = h.install_cache(packet, sha, split, tmp_path / "copied")
    assert {path.name for path in copied.root.iterdir()} == {"cache.json", split.train[0] + ".u8"}
    assert copied.rows == cache.rows and all(row.role == "unassigned" for row in copied.rows)
    left = data.batch(cache, "RGB", (0,), 5)
    right = data.batch(copied, "RGB", (0,), 5)
    assert all(np.array_equal(x, y) for x, y in zip(left, right, strict=True))
    with pytest.raises(FileExistsError):
        h.install_cache(packet, sha, split, copied.root)
    with pytest.raises(ValueError, match="hash"):
        h.install_cache(packet, "0" * 64, split, tmp_path / "bad")


@pytest.mark.parametrize("fault", ["extra", "order", "manifest", "entry", "base64", "hash",
                                   "extent"])
def test_cache_role_hash_and_resource_guards(generated, tmp_path, fault):
    cache, split = generated
    packet = tmp_path / "original.json"
    h.cache_package(cache, split, packet)
    value = copy.deepcopy(a.document(packet, h.CACHE_LIMIT))
    if fault == "extra":
        value["files"][split.future_reserved[0]] = value["files"][split.train[0]]
    elif fault == "order":
        value["train_ids"] = []
    elif fault == "manifest":
        value["manifest_utf8"] += " "
    elif fault == "entry":
        value["files"][split.train[0]]["extra"] = "no"
    elif fault == "base64":
        value["files"][split.train[0]]["base64"] = "!notbase64!"
    elif fault == "hash":
        value["files"][split.train[0]]["sha256"] = "0" * 64
    else:
        value["files"][split.train[0]]["base64"] = base64.b64encode(b"x").decode()
    path = tmp_path / "changed.json"
    path.write_bytes(a.encode(value))
    with pytest.raises(ValueError):
        h.install_cache(path, a.digest(path.read_bytes()), split, tmp_path / "output")


def test_changed_manifest_and_train_source_symlink_refused(generated, tmp_path):
    cache, split = generated
    path = cache.root / (split.train[0] + ".u8")
    path.rename(path.with_suffix(".held"))
    path.symlink_to(path.with_suffix(".held"))
    with pytest.raises(ValueError, match="symlink"):
        h.cache_package(cache, split, tmp_path / "bad.json")
    manifest = cache.root / "cache.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        h.cache_package(cache, split, tmp_path / "changed.json")


def test_private_evidence_archive_roundtrip_and_no_extraction(tmp_path):
    out = tmp_path / "output"
    (out / "work").mkdir(parents=True)
    (out / "work" / "generated.numeric").write_bytes(b"generated-numeric-fixture" * 100)
    (out / "worker-result.json").write_text('{"generated_only":true}\n')
    limits = process.Limits(10, 128 * a.MIB, 128 * a.MIB, 8 * a.MIB, a.MIB,
                            closing_seconds=1)
    sha = h.evidence_package(out, limits, "a" * 64)
    result = h.verify_evidence(out / "verified-profile.zip", sha, "a" * 64, a.MIB)
    assert set(result["files"]) == {"worker-result.json", "work/generated.numeric"}
    with pytest.raises(ValueError, match="hash"):
        h.verify_evidence(out / "verified-profile.zip", "0" * 64, "a" * 64)
    with pytest.raises(ValueError, match="binding"):
        h.verify_evidence(out / "verified-profile.zip", sha, "b" * 64)
    with pytest.raises(ValueError, match="byte limit"):
        h.stream_hash(out / "verified-profile.zip", 1)
    assert {p.name for p in out.iterdir()} == {"verified-profile.zip", "work", "worker-result.json"}


@pytest.mark.parametrize("fault", ["traversal", "symlink", "fifo", "compressed", "duplicate",
                                   "size", "hash", "inventory", "crc"])
def test_hostile_evidence_archive_is_bounded_and_never_extracted(tmp_path, fault):
    name, content = "generated.bin", b"fixture"
    manifest = {"schema": "colab-profile-evidence-v1", "contract_sha256": "a" * 64,
                "files": {name: {"bytes": len(content), "sha256": a.digest(content)}}}
    if fault == "size":
        manifest["files"][name]["bytes"] += 1
    elif fault == "hash":
        manifest["files"][name]["sha256"] = "0" * 64
    elif fault == "inventory":
        manifest["files"] = {}
    path = tmp_path / "hostile.zip"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as z:
        z.writestr("payload-manifest.json", a.encode(manifest))
        entry = zipfile.ZipInfo("../escape" if fault == "traversal" else name)
        if fault == "symlink":
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        if fault == "fifo":
            entry.external_attr = (stat.S_IFIFO | 0o600) << 16
        if fault == "compressed":
            entry.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(entry, content)
        if fault == "duplicate":
            with pytest.warns(UserWarning):
                z.writestr(name, content)
    if fault == "crc":
        path.write_bytes(path.read_bytes().replace(b"fixture", b"broken!", 1))
    with pytest.raises(ValueError):
        h.verify_evidence(path, h.stream_hash(path, a.MIB), "a" * 64)
    assert {p.name for p in tmp_path.iterdir()} == {"hostile.zip"}


def test_archive_reservation_refused_before_creation(tmp_path):
    (tmp_path / "generated.bin").write_bytes(b"x" * 65536)
    limits = process.Limits(10, 128 * a.MIB, 128 * a.MIB, 128 * 1024, 65536,
                            closing_seconds=1)
    with pytest.raises(ValueError, match="reservation"):
        h.evidence_package(tmp_path, limits, "a" * 64)
    assert not (tmp_path / "verified-profile.zip").exists()


def test_native_notebook_default_source_only_and_all_execution_switches_closed(tmp_path):
    root = Path(h.__file__).parent.parent
    builder = runpy.run_path(str(root / "scripts" / "build-colab-native-notebook.py"))["build"]
    delivery = tmp_path / "delivery"
    binding = builder(root, delivery)
    assert binding["permission_created"] is False and binding["data_or_gpu_execution"] is False
    notebook = a.document(delivery / "png_bmp_colab_native_profile_20261106.ipynb")
    assert "accelerator" not in notebook["metadata"]
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, code in enumerate(codes):
        compile(code, f"owned-native-cell-{index}", "exec")
    # Change only the owned fixture's /content destination; never execute a downloaded notebook.
    destination = tmp_path / "isolated-notebook-source"
    assert codes[0].count('Path("/content/spatial-colab-native-profile-source-20261106")') == 1
    codes[0] = codes[0].replace('Path("/content/spatial-colab-native-profile-source-20261106")',
                               f"Path({str(destination)!r})")
    path = tmp_path / "owned-cells.json"
    path.write_text(json.dumps(codes))
    code = '''import json,sys,types
from pathlib import Path
calls=[]
packet=Path(sys.argv[1]).read_bytes()
files=types.SimpleNamespace(upload=lambda: calls.append('source-only') or {'sources.json':packet})
google=types.ModuleType('google');colab=types.ModuleType('google.colab');colab.files=files
google.colab=colab;sys.modules['google']=google;sys.modules['google.colab']=colab
namespace={}
for source in json.loads(Path(sys.argv[2]).read_text()):exec(source,namespace)
assert calls==['source-only'] and 'torch' not in sys.modules
assert namespace['RUN_APPROVED_PROFILE'] is False
assert namespace['VERIFY_APPROVED_PERMISSION'] is False
assert namespace['DOWNLOAD_VERIFIED_EVIDENCE'] is False
assert not (namespace['source_root']/'.colab-process-attempts').exists()
print('default_notebook_no_media_no_gpu_no_attempt')
'''
    completed = subprocess.run(  # noqa: S603 -- owned generated source/metadata-only fixture
        [sys.executable, "-I", "-B", "-c", code, str(delivery / "sources.json"), str(path)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "default_notebook_no_media_no_gpu_no_attempt" in completed.stdout


def test_zip_directory_count_is_bounded_before_member_allocation(tmp_path, monkeypatch):
    path = tmp_path / "directory-bomb.zip"
    path.write_bytes(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 60000, 60000, 0, 0, 0))
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args: pytest.fail("unbounded directory parsed"))
    with pytest.raises(ValueError, match="bounded"):
        h.verify_evidence(path, h.stream_hash(path, a.MIB), "a" * 64)


def test_truncated_zip_is_unavailable(tmp_path):
    path = tmp_path / "cut.zip"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="trailer"):
        h.verify_evidence(path, h.stream_hash(path, a.MIB), "a" * 64)
