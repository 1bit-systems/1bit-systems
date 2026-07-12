#!/usr/bin/env python3
"""Generate site/numbers.json from real artifacts.

Build-time only (see CLAUDE.md: zero Python at runtime).

Binary sizes are measured by stat()-ing the actual built binaries. Nothing is
hardcoded. If an artifact is absent, its entry is emitted as null and listed in
`_missing` -- the site hides null fields rather than showing a stale number.

Benchmark figures that require hardware (ROCm GPU / XDNA2 NPU) come from
benchmarks/latest.json, which tools/validate_claims.py refreshes and audits.

Usage:
    python3 tools/gen_numbers.py [--check]

    --check  exit 1 if site/numbers.json is out of date (for CI)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# label -> path of the artifact whose size we publish.
ARTIFACTS = {
    "fused": REPO / "engine/fusion/zig-out/bin/fused-engine",
    "decode": REPO / "build/bitnet_decode",
    "tui": REPO / "build/bitnet_tui",
}

# Artifacts we also report a stripped size for.
STRIPPABLE = {"decode", "tui"}


def stripped_size(path: Path) -> int | None:
    """Size after `strip`, measured on a copy. None if strip is unavailable."""
    import shutil
    import tempfile

    if not shutil.which("strip"):
        return None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / path.name
        shutil.copy2(path, tmp)
        try:
            subprocess.run(["strip", str(tmp)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return None
        return tmp.stat().st_size


def measure() -> tuple[dict, list[str]]:
    binary: dict[str, int | None] = {}
    missing: list[str] = []

    for label, path in ARTIFACTS.items():
        if not path.is_file():
            missing.append(f"{label}: {path.relative_to(REPO)}")
            binary[f"{label}_bytes"] = None
            binary[f"{label}_kb"] = None
            if label in STRIPPABLE:
                binary[f"{label}_stripped_kb"] = None
            continue

        size = path.stat().st_size
        binary[f"{label}_bytes"] = size
        binary[f"{label}_kb"] = round(size / 1024)

        if label in STRIPPABLE:
            s = stripped_size(path)
            binary[f"{label}_stripped_kb"] = round(s / 1024) if s else None

    return binary, missing


def build(bench_path: Path) -> dict:
    bench = json.loads(bench_path.read_text())
    binary, missing = measure()

    return {
        "_generated_by": "tools/gen_numbers.py",
        "_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_warning": "Generated file. Do not hand-edit; run tools/gen_numbers.py.",
        "_missing": missing,
        "binary": binary,
        "benchmarks": bench["benchmarks"],
        "_sources": bench["_sources"],
        "repo": bench["repo"],
        "site": bench["site"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if numbers.json is stale")
    args = ap.parse_args()

    out_path = REPO / "site/numbers.json"
    data = build(REPO / "benchmarks/latest.json")

    if args.check:
        if not out_path.is_file():
            print("site/numbers.json missing", file=sys.stderr)
            return 1
        old = json.loads(out_path.read_text())
        # Timestamps, binary sizes, and missing-artifact lists are runner-dependent.
        # Only validate the stable payload derived from benchmarks/latest.json.
        volatile = {"_generated_at", "_missing", "binary"}
        a = {k: v for k, v in old.items() if k not in volatile}
        b = {k: v for k, v in data.items() if k not in volatile}
        if a != b:
            print("site/numbers.json is STALE -- run tools/gen_numbers.py", file=sys.stderr)
            return 1
        print("site/numbers.json is up to date")
        return 0

    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")

    for label in ARTIFACTS:
        kb = data["binary"].get(f"{label}_kb")
        print(f"  {label:8} {str(kb) + ' KB' if kb else 'ABSENT (null)':>14}")

    if data["_missing"]:
        print("\nmissing artifacts (emitted as null, not guessed):", file=sys.stderr)
        for m in data["_missing"]:
            print(f"  - {m}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
