#!/usr/bin/env python3
"""Generate a minimal CycloneDX SBOM from installed distribution metadata."""
from __future__ import annotations

import argparse
import json
import tomllib
from importlib import metadata
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        distribution = metadata.distribution("steganography-dfir")
        project_version = distribution.version
        requirements = distribution.requires or []
    except metadata.PackageNotFoundError:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        project_version = project["version"]
        requirements = project.get("dependencies", [])
    components = []
    for requirement in requirements:
        name = requirement.split(";", 1)[0].strip().split("[", 1)[0]
        for marker in (">=", "==", "~=", "!=", "<=", ">", "<"):
            name = name.split(marker, 1)[0].strip()
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name.lower()}@{version}",
            }
        )
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "steganography-dfir",
                "version": project_version,
            }
        },
        "components": sorted(components, key=lambda item: item["name"].lower()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
