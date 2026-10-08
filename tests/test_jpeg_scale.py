import hashlib
import json

import pytest

from core import jpeg_scale as core
from steganography import research_jpeg_scale as scale


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(alaska_count=2, boss_count=2):
    alaska, boss = [], []
    for i in range(alaska_count):
        lineage = digest(f"a{i}:cover")
        for method in (None, "JMiPOD", *core.METHODS):
            alaska.append(
                {
                    "sha256": lineage if method is None else digest(f"a{i}:{method}"),
                    "lineage": lineage,
                    "split": "train" if i % 2 == 0 else "validation",
                    "method": method,
                    "label": "cover" if method is None else "stego",
                    "source_group": "ALASKA2",
                    "format": "JPEG",
                    "path": f"a/{i}-{method}.jpg",
                }
            )
    for i in range(boss_count):
        sha = digest(f"b{i}")
        boss.append(
            {
                "sha256": sha,
                "lineage": sha,
                "split": "train" if i % 2 == 0 else "validation",
                "method": None,
                "label": "cover",
                "source_group": "BOSSbase-1.01",
                "path": f"{i}.pgm",
            }
        )
    return alaska, boss


def test_full_deterministic_blocks_no_truncation_or_lineage_split():
    a, b = rows(997, 1000)
    plan = core.layout(a, b, set(), set())
    assert [len(p["lineages"]) for p in plan] == [128] * 7 + [101] + [128] * 7 + [104]
    assert plan == core.layout(a[::-1], b[::-1], set(), set())
    assert len({key for p in plan for key in p["lineages"]}) == 1997


@pytest.mark.parametrize(
    "fault",
    [
        "limit",
        "bad_identity",
        "unsafe",
        "backslash",
        "invalid_row",
        "source",
        "format",
        "split",
        "label",
        "overlap",
        "missing",
        "cover",
        "role",
        "cross_split",
        "quarantine",
        "unknown_quarantine",
        "duplicate_methods",
        "empty",
        "boss_row",
        "boss_source",
        "boss_duplicate",
        "boss_overlap",
        "boss_tail",
    ],
)
def test_layout_invalid_roles_limits_lineages_and_leakage(fault):
    a, b = rows()
    q, r = set(), set()
    if fault == "limit":
        a = []
    if fault == "bad_identity":
        a[0]["sha256"] = "bad"
    if fault == "unsafe":
        a[0]["path"] = "../escape.jpg"
    if fault == "backslash":
        a[0]["path"] = "a\\b.jpg"
    if fault == "invalid_row":
        a[0] = None
    if fault == "source":
        a[0]["source_group"] = "WIFD"
    if fault == "format":
        a[0]["format"] = "MPO"
    if fault == "split":
        a[0]["split"] = "test"
    if fault == "label":
        a[0]["label"] = "bad"
    if fault == "overlap":
        r.add(a[0]["sha256"])
    if fault == "missing":
        a.pop()
    if fault == "cover":
        a[0]["sha256"] = digest("wrong_cover")
    if fault == "role":
        a[1]["label"] = "cover"
    if fault == "cross_split":
        a[1]["split"] = "validation"
    if fault == "quarantine":
        a[1]["sha256"] = a[0]["sha256"]
    if fault == "unknown_quarantine":
        q.add(digest("unknown"))
    if fault == "duplicate_methods":
        a[1]["sha256"] = a[2]["sha256"]
    if fault == "empty":
        for i in (0, 4):
            a[i + 1]["sha256"] = a[i]["sha256"]
            q.add(a[i]["sha256"])
    if fault == "boss_row":
        b[0] = None
    if fault == "boss_source":
        b[0]["source_group"] = "WIFD"
    if fault == "boss_duplicate":
        b.append(b[0].copy())
    if fault == "boss_overlap":
        r.add(b[0]["sha256"])
    if fault == "boss_tail":
        a, b = rows(2, 129)
    with pytest.raises(ValueError):
        core.layout(a, b, q, r)


def test_quarantine_excludes_all_correlated_rows():
    a, b = rows()
    a[1]["sha256"] = a[0]["sha256"]
    plan = core.layout(a, b, {a[0]["lineage"]}, set())
    assert plan[0]["lineages"] == [a[4]["lineage"]]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    a, b = rows()
    alaska, boss = tmp_path / "alaska", tmp_path / "boss"
    alaska.mkdir()
    boss.mkdir()
    for row in a:
        body = ("fixture:" + row["sha256"]).encode()
        row["sha256"] = hashlib.sha256(body).hexdigest()
        target = alaska / row["path"]
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(body)
        row["size"] = len(body)
    for first in (0, 4):
        for r in a[first : first + 4]:
            r["lineage"] = a[first]["sha256"]
    ad = {
        "purpose": "development",
        "samples": a,
        "license": "fixture",
        "source_url": "https://example.org/fixture",
    }
    bd = {"purpose": "development covers only", "samples": b}
    wf = {"samples": [{"sha256": digest("wifd"), "lineage": digest("wifd")}]}
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": digest("reserved")}]}))
    wifd = tmp_path / "wifd.json"
    wifd.write_text(json.dumps(wf))
    (alaska / "source.json").write_text(json.dumps(ad))
    (boss / "source.json").write_text(json.dumps(bd))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "completed",
                "models_trained": False,
                "reserved_manifest_sha256": [sha(reserved)],
                "sources": [
                    {
                        "source_group": "ALASKA2",
                        "manifest_sha256": sha(alaska / "source.json"),
                        "quarantined_lineages": [],
                    },
                    {"source_group": "BOSSbase-1.01", "manifest_sha256": sha(boss / "source.json")},
                    {"source_group": "WIFD", "manifest_sha256": sha(wifd)},
                ],
            }
        )
    )
    monkeypatch.setattr(scale, "AUDIT_SHA", sha(audit))
    monkeypatch.setattr(scale, "WIFD_SHA", sha(wifd))

    def fake_boss(root, out, **kwargs):
        out.mkdir()
        result = []
        for sample in b:
            for quality in (75, 95):
                for method in (None, *core.METHODS):
                    body = f"{sample['sha256']}:{quality}:{method}".encode()
                    name = f"{sample['sha256']}-{quality}-{method}.jpg"
                    (out / name).write_bytes(body)
                    result.append(
                        {
                            **sample,
                            "path": name,
                            "sha256": hashlib.sha256(body).hexdigest(),
                            "size": len(body),
                            "format": "JPEG",
                            "quality_factor": quality,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                        }
                    )
        scale.write_json(out / "manifest.json", {"schema_version": "1.0", "samples": result})

    def fake_pixels(path, out, **kwargs):
        doc, _ = scale.read_document(path)
        selected = [r for r in doc["samples"] if r["split"] == kwargs["split"]]
        out.mkdir()
        d = {"rows": selected, "data_sha256": digest("tensor")}
        scale.write_json(out / "cache.json", d)
        return d

    monkeypatch.setattr(scale, "generate_boss", fake_boss)
    monkeypatch.setattr(scale, "extract_pixels", fake_pixels)
    return alaska, boss, audit, wifd, [reserved], tmp_path / "out"


def test_complete_service_blocks_contract_fresh_portable_and_roles(setup):
    progress = []
    result = scale.prepare(*setup, progress=progress.append)
    assert result["jpeg_rows"] == 18 and result["original_lineages"] == 4
    assert result["splits"] == {"train": 9, "validation": 9}
    assert not result["models_trained"] and not result["detection_measured"]
    assert len(progress) == 2 and str(setup[0].parent) not in json.dumps(result)
    assert all(set(e["caches"]) == {"train", "validation"} for e in result["blocks"])
    with pytest.raises(FileExistsError):
        scale.prepare(*setup)


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "audit",
        "wifd",
        "source",
        "purpose",
        "reserved",
        "empty_reserved",
        "bad_reserved",
        "changed_jpeg",
        "symlink",
        "deadline",
        "budget",
        "incomplete",
        "duplicate",
    ],
)
def test_service_fail_closed_no_complete_index(setup, monkeypatch, fault):
    alaska, boss, audit, wifd, reserved, out = setup
    if fault == "protocol":
        monkeypatch.setattr(scale, "PROTOCOL_SHA", "0" * 64)
    if fault == "audit":
        monkeypatch.setattr(scale, "AUDIT_SHA", "0" * 64)
    if fault == "wifd":
        monkeypatch.setattr(scale, "WIFD_SHA", "0" * 64)
    if fault in ("source", "purpose"):
        p = alaska / "source.json"
        d = json.loads(p.read_bytes())
        d["purpose"] = "evaluation"
        p.write_text(json.dumps(d))
        if fault == "purpose":
            doc = json.loads(audit.read_bytes())
            doc["sources"][0]["manifest_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            audit.write_text(json.dumps(doc))
            monkeypatch.setattr(scale, "AUDIT_SHA", hashlib.sha256(audit.read_bytes()).hexdigest())
    if fault == "reserved":
        reserved[0].write_text(json.dumps({"samples": [{"sha256": digest("other")}]}))
    if fault == "empty_reserved":
        reserved.clear()
    if fault == "bad_reserved":
        reserved[0].write_text('{"samples":[]}')
    if fault == "changed_jpeg":
        next((alaska / "a").iterdir()).write_bytes(b"changed")
    if fault == "symlink":
        p = next((alaska / "a").iterdir())
        p.unlink()
        p.symlink_to(alaska / "source.json")
    if fault == "deadline":
        monkeypatch.setattr(scale, "MAX_SECONDS", -1)
    if fault == "budget":
        monkeypatch.setattr(scale, "MAX_OUTPUT", 1)
    if fault in ("incomplete", "duplicate"):
        old = scale.native_block

        def native(*args):
            old(*args)
            p = args[1] / "manifest.json"
            d = json.loads(p.read_bytes())
            if fault == "incomplete":
                d["samples"].pop()
            else:
                d["samples"][1] = d["samples"][0].copy()
            p.write_text(json.dumps(d))

        monkeypatch.setattr(scale, "native_block", native)
    with pytest.raises(ValueError):
        scale.prepare(*setup)
    assert not (out / "index.json").exists()


def test_output_symlink_byte_member_bounds_and_cli_redaction(setup, monkeypatch, capsys):
    import resource

    *_, out = setup
    out.mkdir()
    (out / "blob").write_bytes(b"ab")
    assert scale.output_bytes(out) == 2
    monkeypatch.setattr(scale, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError):
        scale.output_bytes(out)
    monkeypatch.setattr(scale, "MAX_OUTPUT", 100)
    (out / "link").symlink_to(out / "blob")
    with pytest.raises(ValueError):
        scale.output_bytes(out)
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    monkeypatch.setattr(scale, "prepare", lambda *a, **k: {"jpeg_rows": 18})
    args = [
        "--alaska",
        "a",
        "--boss",
        "b",
        "--audit",
        "c",
        "--wifd-reserved",
        "w",
        "--reserved",
        "r",
        "--out",
        "o",
    ]
    assert scale.main(args) == 0 and len(calls) == 4

    def fail(*a, **k):
        raise ValueError("private/password=secret")

    monkeypatch.setattr(scale, "prepare", fail)
    assert scale.main(args) == 2 and "secret" not in capsys.readouterr().out
