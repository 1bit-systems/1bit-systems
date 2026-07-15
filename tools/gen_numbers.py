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
    "zaya": REPO / "build/zaya_server",
    "fused": REPO / "engine/fusion/zig-out/bin/fused-engine",
    "decode": REPO / "build/bitnet_decode",
    "tui": REPO / "build/bitnet_tui",
}

# Artifacts we also report a stripped size for.
STRIPPABLE = {"zaya", "decode", "tui"}


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


def _managed_model_count() -> int | None:
    """Count managed models declared in the ZINC catalog.

    `site.models` used to be a hand-set literal ("73") that matched nothing in
    the tree. The catalog (`engine/gpu/src/model/catalog.zig`) is the real list,
    so count its entries directly. Returns None if the catalog can't be read
    (caller leaves the existing value untouched). (issue #188)
    """
    catalog = REPO / "engine/gpu/src/model/catalog.zig"
    if not catalog.is_file():
        return None
    import re
    return len(re.findall(r'\.id\s*=\s*"', catalog.read_text()))


def _engine_benchmarks(benchmarks_path: Path) -> dict:
    """Read site/benchmarks.json and map engine tok/s fields to the names the
    site JS expects (see site/index.html lines ~557-586). These are the
    hardware-validated performance numbers shown on the live site (fixes #49)."""
    data = json.loads(benchmarks_path.read_text())
    B = data.get("engines", {})
    S = data.get("system", {})

    # Field-name mapping from site/benchmarks.json engine keys to JS-expected names.
    engine_map = {
        "npu_fused_raw_tok_s":   B.get("npu_fused", {}).get("tok_s"),
        "npu_validated_tok_s":   B.get("npu_flm",    {}).get("tok_s"),
        "gpu_ternary_tok_s":     B.get("ternary",    {}).get("tok_s"),
        "gpu_1bit_llama_tok_s":  B.get("gpu_1bit",   {}).get("tok_s"),
        "gpu_f16_baseline_tok_s": B.get("gpu_zinc",  {}).get("tok_s"),
        # NOTE: the standalone "tflops" (55.7) is quarantined in
        # benchmarks/latest.json._unverified (tflops_55_7: "NO SOURCE; measured
        # ternary prefill peak is 35.57"). It is NOT published. The verified,
        # sourced TFLOPS figures are prefill_tflops_4h / prefill_tflops_i8apre,
        # which come straight from benchmarks/latest.json. (issue #107)
    }
    return {k: v for k, v in engine_map.items() if v is not None}


def build(bench_path: Path) -> dict:
    bench = json.loads(bench_path.read_text())
    binary, missing = measure()

    # Add engine-performance fields from site/benchmarks.json (for site/index.html JS)
    engine_path = REPO / "site/benchmarks.json"
    if engine_path.is_file():
        bench["benchmarks"].update(_engine_benchmarks(engine_path))

    # Quarantine gate (issue #107): benchmarks/latest.json._unverified lists
    # every claim that has NO reproducible source in this repo and "MUST NOT be
    # published". Strip any of those keys before they can reach numbers.json --
    # they were leaking in via _engine_benchmarks() overlaying site/
    # benchmarks.json. This is the structural fix, not a one-off scrub.
    unverified = set(bench.get("_unverified", {}))
    unverified.discard("_comment")
    dropped = [k for k in list(bench["benchmarks"]) if k in unverified]
    for k in dropped:
        bench["benchmarks"].pop(k, None)
    if dropped:
        print(f"gen_numbers: dropped unverified/quarantined keys: {sorted(dropped)}")

    # Display alias: site/index.html reads B.tflops for the hero "TFLOPS" spans.
    # Point it at the sourced prefill peak (prefill_tflops_i8apre) so those spans
    # can never drift from a measured value. Previously B.tflops was undefined,
    # so the JS silently left a stale hardcoded "38.5 TFLOPS" in the HTML.
    # (issue #184)
    if "prefill_tflops_i8apre" in bench["benchmarks"]:
        bench["benchmarks"]["tflops"] = bench["benchmarks"]["prefill_tflops_i8apre"]

    # site.models: derive from the ZINC catalog, not a hand-set literal. The old
    # value ("73") matched no list in the repo; the catalog has the real count.
    # (issue #188)
    managed = _managed_model_count()
    if managed is not None:
        bench.setdefault("site", {})["models"] = managed

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
        # Only the build timestamp and the missing-artifact list are truly
        # runner-dependent. Binary SIZES are runner-dependent only in
        # *presence*: a runner with no build artifacts reports null. So where
        # we could not measure (None), carry the committed value forward (no
        # constraint); where we DID measure, the size must match what's
        # committed. This catches real drift (e.g. a stale 25 MB artifact)
        # without false-failing on CI runners that don't build. (issue #186)
        volatile = {"_generated_at", "_missing"}
        committed_binary = old.get("binary", {})
        fresh_binary = data["binary"]
        merged_binary = {
            k: (fresh_binary[k] if fresh_binary.get(k) is not None
                else committed_binary.get(k))
            for k in (set(committed_binary) | set(fresh_binary))
        }
        a = {k: v for k, v in old.items() if k not in volatile}
        a["binary"] = committed_binary
        b = {k: v for k, v in data.items() if k not in volatile}
        b["binary"] = merged_binary
        if a != b:
            drifted = sorted(
                k for k in merged_binary
                if merged_binary.get(k) is not None
                and merged_binary.get(k) != committed_binary.get(k)
            )
            print("site/numbers.json is STALE -- run tools/gen_numbers.py",
                  file=sys.stderr)
            for k in drifted:
                print(f"  binary {k}: committed {committed_binary.get(k)} "
                      f"vs measured {merged_binary.get(k)}", file=sys.stderr)
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
