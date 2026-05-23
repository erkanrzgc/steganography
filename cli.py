"""steganography CLI: embed | extract | analyze | scan | list-modules."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from core.crypto import KDF_SALT_LEN, decrypt, encrypt
from core.payload import pack, unpack
from registry import Registry
from report.report import write_html, write_json
from ui.banner import print_gradient_banner


def _build_registry() -> Registry:
    reg = Registry()
    reg.autodiscover()
    return reg


def _maybe_password() -> str | None:
    return os.environ.get("STEGANO_PASSWORD") or None


def cmd_embed(args: argparse.Namespace) -> int:
    reg = _build_registry()
    carrier = reg.select_carrier_for_embed(Path(args.carrier))
    raw = Path(args.input).read_bytes()
    password = args.password or _maybe_password()
    if password:
        salt = os.urandom(KDF_SALT_LEN)
        nonce, ct = encrypt(raw, password, salt)
        blob = pack(payload=ct, encrypted=True, salt=salt, nonce=nonce)
    else:
        blob = pack(payload=raw, encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)
    res = carrier.embed(Path(args.carrier), blob, Path(args.out))
    print(f"embedded {res.bytes_written} bytes via {res.carrier} → {res.out_path}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    reg = _build_registry()
    carriers = [c for c in reg.select_carriers(Path(args.input)) if c.can_extract]
    if not carriers:
        print(f"no extractable carrier for {args.input}", file=sys.stderr)
        return 2
    blob = carriers[0].extract(Path(args.input))
    parsed = unpack(blob)
    password = args.password or _maybe_password()
    if parsed.encrypted:
        if not password:
            print("payload encrypted; set --password or STEGANO_PASSWORD", file=sys.stderr)
            return 2
        data = decrypt(parsed.payload, password, parsed.salt, parsed.nonce)
    else:
        data = parsed.payload
    Path(args.out).write_bytes(data)
    print(f"extracted {len(data)} bytes → {args.out}")
    return 0


def _analyze_file(reg: Registry, path: Path) -> list:
    results = []
    for c in reg.select_carriers(path):
        results.append(c.analyze(path))
    for a in reg.all_analyzers():
        results.append(a.analyze(path))
    return results


def cmd_analyze(args: argparse.Namespace) -> int:
    reg = _build_registry()
    results = _analyze_file(reg, Path(args.input))
    if args.json:
        payload = [
            {
                "analyzer": r.analyzer,
                "suspicion": r.suspicion,
                "signals": [
                    {"name": s.name, "score": s.score, "detail": s.detail}
                    for s in r.signals
                ],
                "explanation": r.explanation,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            print(f"  [{r.suspicion:3d}] {r.analyzer}")
            for s in r.signals:
                print(f"        - {s.name}({s.score}): {s.detail}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    reg = _build_registry()
    root = Path(args.dir)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        for r in _analyze_file(reg, p):
            rows.append((p, r))
    out = Path(args.out)
    if args.report == "json":
        write_json(rows, out)
    else:
        write_html(rows, out)
    print(f"scanned {len(rows)} results → {out}")
    return 0


def cmd_list_modules(_: argparse.Namespace) -> int:
    reg = _build_registry()
    print("carriers:")
    for c in reg.all_carriers():
        print(f"  {c.name}  {c.extensions}")
    print("analyzers:")
    for a in reg.all_analyzers():
        print(f"  {a.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="steganography")
    p.add_argument("--quiet", action="store_true", help="suppress banner")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--in", dest="input", required=True)
    e.add_argument("--carrier", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--password")
    e.set_defaults(fn=cmd_embed)

    x = sub.add_parser("extract")
    x.add_argument("--in", dest="input", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--password")
    x.set_defaults(fn=cmd_extract)

    a = sub.add_parser("analyze")
    a.add_argument("--in", dest="input", required=True)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("scan")
    s.add_argument("--dir", required=True)
    s.add_argument("--report", choices=("json", "html"), default="html")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_scan)

    lm = sub.add_parser("list-modules")
    lm.set_defaults(fn=cmd_list_modules)

    return p


_MACHINE_READABLE_CMDS = frozenset({"list-modules", "extract"})


def _should_show_banner(args: argparse.Namespace) -> bool:
    if args.quiet:
        return False
    if args.cmd in _MACHINE_READABLE_CMDS:
        return False
    # analyze with --json prints structured output; suppress banner.
    return not (args.cmd == "analyze" and getattr(args, "json", False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _should_show_banner(args):
        print_gradient_banner()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
