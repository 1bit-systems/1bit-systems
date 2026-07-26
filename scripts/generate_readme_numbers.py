#!/usr/bin/env python3
"""generate_readme_numbers.py — Generate README badge tables and site badges from site/numbers.json.

Usage:
    python3 scripts/generate_readme_numbers.py           # print markdown table for README
    python3 scripts/generate_readme_numbers.py --badges   # regenerate site/badge_*.json
    python3 scripts/generate_readme_numbers.py --check    # verify README matches JSON (CI)
"""

import json, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBERS_PATH = os.path.join(ROOT, "site", "numbers.json")
README_PATH = os.path.join(ROOT, "README.md")

def load():
    with open(NUMBERS_PATH) as f:
        return json.load(f)

def gen_kernel_table(d):
    """Generate the kernel-level microbenchmark markdown table."""
    b = d.get("benchmarks", {})
    e = d.get("engines", {})
    lines = [
        "| Benchmark | Value | Backend |",
        "|-----------|:-----:|---------|",
    ]
    # Kernel table entries (from benchmarks.json engines)
    kernel_entries = [
        ("q1_gemv", "Q1 GEMV", "ROCm HIP (fused kernel)"),
        ("fused_tq2", "Fused TQ2", "ROCm HIP (QKV+GU fused)"),
        ("ternary", "GPU ternary", "Vulkan ZINC"),
        ("tq2_gemv", "TQ2 GEMV", "ROCm HIP"),
        ("npu_v12", "NPU v12", "XDNA 2 (32 tiles)"),
    ]
    for key, name, backend in kernel_entries:
        if key in e:
            val = e[key].get("tok_s", "?")
            lines.append(f"| {name} | **{val} tok/s** | {backend} |")
    # Prefill
    tflops = b.get("prefill_tflops_i8apre", b.get("tflops", "?"))
    lines.append(f"| Prefill | **{tflops} TFLOPS** | INT8 WMMA |")
    # ROCm HIP fallback
    rocm = e.get("rocm_hip_fallback", {}).get("tok_s", "64")
    lines.append(f"| ROCm HIP | **{rocm} tok/s** | ROCm HIP (kernels) |")
    return "\n".join(lines)

def gen_e2e_table(d):
    """Generate the end-to-end inference markdown table."""
    e = d.get("engines", {})
    lines = [
        "| Benchmark | Value | Backend | Notes |",
        "|-----------|:-----:|---------|-------|",
    ]
    e2e_entries = [
        ("blackmamba_1_5b", "BlackMamba 1.5B"),
        ("blackmamba_2_8b", "BlackMamba 2.8B"),
    ]
    for key, name in e2e_entries:
        if key in e:
            val = e[key].get("tok_s", "?")
            backend = e[key].get("backend", "Mamba1 HIP (Strix Halo)")
            notes = e[key].get("notes", "Full decode")
            lines.append(f"| {name} | **{val} tok/s** | {backend} | {notes} |")
    return "\n".join(lines)

def gen_badges(d):
    """Regenerate site/badge_*.json files from numbers.json."""
    b = d.get("benchmarks", {})
    tflops = b.get("prefill_tflops_i8apre", b.get("tflops", "?"))
    # Prefill badge
    with open(os.path.join(ROOT, "site", "badge_prefill.json"), "w") as f:
        json.dump({
            "schemaVersion": 1,
            "label": "prefill (INT8 WMMA)",
            "message": f"{tflops} TFLOPS",
            "color": "informational",
            "cacheSeconds": 86400
        }, f, indent=2)
    print(f"  badge_prefill.json: {tflops} TFLOPS")

def check_readme(d):
    """Verify README.md contains the same numbers as numbers.json."""
    with open(README_PATH) as f:
        readme = f.read()
    e = d.get("engines", {})
    errors = []
    for key, name in [("blackmamba_1_5b", "79.4"), ("blackmamba_2_8b", "46.0")]:
        if key in e:
            expected = str(e[key].get("tok_s", ""))
            if expected and expected not in readme:
                errors.append(f"README missing {name}={expected} tok/s")
    b = d.get("benchmarks", {})
    tflops = str(b.get("prefill_tflops_i8apre", b.get("tflops", "")))
    if tflops and tflops not in readme:
        errors.append(f"README missing prefill={tflops} TFLOPS")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print("OK: README numbers match numbers.json")

if __name__ == "__main__":
    d = load()
    if "--badges" in sys.argv:
        gen_badges(d)
    elif "--check" in sys.argv:
        check_readme(d)
    else:
        print("=== Kernel-Level Microbenchmarks ===")
        print()
        print(gen_kernel_table(d))
        print()
        print("=== End-to-End Inference ===")
        print()
        print(gen_e2e_table(d))
