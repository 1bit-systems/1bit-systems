#!/usr/bin/env python3
"""Re-measure every published benchmark claim against real hardware.

Designed to run unattended every 24h (see .github/workflows/validate-claims.yml).

Exit codes:
    0  every published claim re-measured within tolerance
    1  a published claim drifted beyond tolerance, or a benchmark crashed
    2  hardware unavailable -- nothing was measured (not a failure of the claims)

Usage:
    python3 tools/validate_claims.py --build-dir 1bit/build
    python3 tools/validate_claims.py --build-dir 1bit/build --update   # rewrite measured values
    python3 tools/validate_claims.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmarks/latest.json"

GPU_ENV = {
    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
    "HSA_ENABLE_SDMA": "0",
}


def have_gpu() -> bool:
    return Path("/dev/kfd").exists()


def have_npu() -> bool:
    return any(Path("/dev/accel").glob("accel*")) if Path("/dev/accel").is_dir() else False


def run(argv: list[str], cwd: Path, timeout: int = 600) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(GPU_ENV)
    env["LD_LIBRARY_PATH"] = f"{cwd}:/opt/rocm/lib:" + env.get("LD_LIBRARY_PATH", "")
    try:
        p = subprocess.run(
            argv, cwd=cwd, env=env, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except FileNotFoundError:
        return 127, f"not found: {argv[0]}"
    return p.returncode, p.stdout + p.stderr


# ---- parsers: benchmark stdout -> {key: measured value} -------------------

def parse_prefill(out: str) -> dict[str, float]:
    """Pin to *named* variants.

    `Fastest variant:` is not a stable metric -- which variant wins varies
    run to run, so the number it reports is not comparable across runs.
    """
    got: dict[str, float] = {}
    # "4i-I8-APRE (I8 A+ tiled B)              2.5468      35.57"
    for label, key in (("4i-I8-APRE", "prefill_tflops_i8apre"),
                       ("4h", "prefill_tflops_4h")):
        m = re.search(rf"^\s*{re.escape(label)}\s+\(.*?\)\s+([\d.]+)\s+([\d.]+)\s*$", out, re.M)
        if m:
            got[key] = float(m.group(2))
    return got


def parse_sherry(out: str) -> dict[str, float]:
    got: dict[str, float] = {}
    # "[bench_sherry] sherry     : 18.58 us/call  148.8 GB/s"
    for kernel, key in (("sherry", "sherry_gemv_gbps"),
                        ("tq1", "tq1_gemv_gbps"),
                        ("halo", "halo_gemv_gbps")):
        m = re.search(rf"^\[bench_sherry\]\s+{kernel}\s*:.*?([\d.]+)\s*GB/s", out, re.M)
        if m:
            got[key] = float(m.group(1))
    return got


PARSERS = {
    "bench_prefill_variants": parse_prefill,
    "bench_sherry": parse_sherry,
}


# README benchmark-table engines whose tok/s figures live in
# benchmarks/latest.json _unverified (i.e. have NO reproducible source in this
# repo). If the README marks any of these as measured/validated/reported, that
# is a false claim -- the honest status is one of the hedged markers
# (⚙️ raw / ❌ broken / 🔶 unresolved / ❓ unsourced). (issue #185, #190)
README_ENGINES_WITHOUT_SOURCE = {
    "GPU 1-bit": "gpu_1bit_llama_tok_s",
    "NPU FLM": "npu_validated_tok_s",
    "GPU ternary": "gpu_ternary_tok_s",
    "GPU ROCm HIP": "rocm_decode_tok_s",
    "NPU fused": "npu_fused_raw_tok_s",
}


def check_readme_consistency() -> list[str]:
    """Fail if README.md stamps a tok/s figure as measured/validated/reported
    for an engine that benchmarks/latest.json has quarantined in _unverified.

    Hardware-free: safe to run on any CI runner (see --check-readme)."""
    readme = REPO / "README.md"
    if not readme.is_file():
        return ["README.md missing"]
    spec = json.loads(BENCH.read_text())
    unverified = set(spec.get("_unverified", {})) - {"_comment"}

    violations: list[str] = []
    # Data rows end with: | **<num>** | <status> |
    row = re.compile(r"\|\s*\*\*([0-9.]+)\*\*\s*\|\s*([^|]+?)\s*\|\s*$")
    for line in readme.read_text().splitlines():
        for engine, key in README_ENGINES_WITHOUT_SOURCE.items():
            if engine not in line:
                continue
            if key not in unverified:
                continue
            m = row.search(line)
            if not m:
                continue
            num, status = m.group(1), m.group(2).strip()
            positive = any(p in status for p in ("measured", "validated", "reported"))
            if positive:
                violations.append(
                    f'ReADME marks "{engine}" {num} tok/s as "{status}", but '
                    f"{key} is quarantined in _unverified (no reproducible source). "
                    "Use a hedged status instead."
                )
    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="1bit/build")
    ap.add_argument("--update", action="store_true", help="rewrite measured values into latest.json")
    ap.add_argument("--json", help="write a machine-readable report here")
    ap.add_argument("--repeat", type=int, default=3,
                    help="runs per benchmark; the median is used (default 3)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded warmup runs per benchmark (default 1)")
    ap.add_argument("--check-readme", action="store_true",
                    help="hardware-free: fail if README marks an _unverified "
                         "engine as measured/validated/reported (issue #185)")
    args = ap.parse_args()

    if args.check_readme:
        violations = check_readme_consistency()
        if violations:
            print("README/_unverified consistency check FAILED:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1
        print("README benchmark statuses are consistent with _unverified")
        return 0

    build_dir = (REPO / args.build_dir).resolve()
    spec = json.loads(BENCH.read_text())
    claims = spec["benchmarks"]
    verify = spec["_verify"]
    tol = verify["tolerance"]

    print(f"validate_claims: gpu={have_gpu()} npu={have_npu()} build_dir={build_dir}")

    if not have_gpu():
        print("no /dev/kfd -- GPU absent, nothing measured", file=sys.stderr)
        return 2
    if not build_dir.is_dir():
        print(f"build dir missing: {build_dir}", file=sys.stderr)
        return 2

    measured: dict[str, float] = {}
    spread: dict[str, tuple[float, float]] = {}
    failures: list[str] = []

    crashes: list[str] = []

    for name, cfg in verify["commands"].items():
        samples: dict[str, list[float]] = {}

        # Discard a cold run: the GPU clocks up over the first invocation, and a
        # cold sample reads ~13% low (observed 35.6 vs 41.0 TFlops on 4i-I8-APRE).
        if args.warmup:
            run(cfg["argv"], cwd=build_dir)

        for i in range(args.repeat):
            rc, out = run(cfg["argv"], cwd=build_dir)
            if rc != 0:
                # bench_prefill_variants segfaults intermittently under
                # back-to-back load. Retry once so one flake cannot set a claim.
                crashes.append(f"{name}: exit {rc} on run {i + 1} (retrying)")
                print(f"  CRASH {name}: exit {rc} (run {i + 1}) -- retrying")
                rc, out = run(cfg["argv"], cwd=build_dir)
                if rc != 0:
                    failures.append(f"{name}: exited {rc} twice on run {i + 1}")
                    print(f"  FAIL {name}: exit {rc} twice")
                    break
            for k, v in PARSERS[name](out).items():
                samples.setdefault(k, []).append(v)

        for k in cfg["keys"]:
            vals = samples.get(k, [])
            if not vals:
                failures.append(f"{name}: could not parse `{k}` from output")
                print(f"  FAIL {name}: no `{k}` in output")
                continue
            measured[k] = round(statistics.median(vals), 2)
            spread[k] = (min(vals), max(vals))

    # Compare
    print(f"\n{'claim':<26} {'published':>10} {'median':>9} {'drift':>7}  {'spread(min-max)':>17}  status")
    print("-" * 84)
    drifted: list[str] = []
    for key, published in claims.items():
        if key not in measured:
            continue
        got = measured[key]
        lo, hi = spread[key]
        drift = abs(got - published) / published if published else 0.0
        ok = drift <= tol
        if not ok:
            drifted.append(f"{key}: published {published}, median {got} ({drift:.1%} drift)")
        rng = f"{lo:.2f}-{hi:.2f}"
        print(f"{key:<26} {published:>10} {got:>9} {drift:>6.1%}  {rng:>17}  {'ok' if ok else 'DRIFT'}")

    unmeasured = [k for k in claims if k not in measured]
    if unmeasured:
        print(f"\nnot re-measured this run: {', '.join(unmeasured)}")

    n_unver = len([k for k in spec["_unverified"] if not k.startswith("_")])
    print(f"\n{n_unver} claim(s) quarantined in _unverified (not published) -- see benchmarks/latest.json")

    if args.update and measured:
        for k, v in measured.items():
            spec["benchmarks"][k] = v
        BENCH.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"\nupdated {BENCH.relative_to(REPO)} with {len(measured)} measured values")

    if crashes:
        print("\nintermittent crashes (retried, did not set claims):")
        for c in crashes:
            print(f"  ! {c}")

    exit_code = 1 if (failures or drifted) else 0
    report = {
        "exit_code": exit_code,
        "gpu": have_gpu(), "npu": have_npu(),
        "measured": measured, "spread": {k: list(v) for k, v in spread.items()},
        "drifted": drifted, "failures": failures,
        "crashes": crashes, "unmeasured": unmeasured,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")

    if failures or drifted:
        print("\nFAILED:", file=sys.stderr)
        for f in failures + drifted:
            print(f"  - {f}", file=sys.stderr)
        return exit_code

    print("\nall published claims re-measured within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
