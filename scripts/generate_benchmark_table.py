#!/usr/bin/env python3
"""
Generate the README.md performance table and status footnotes from
site/benchmarks.json — the single source of truth.

Usage:
    python3 scripts/generate_benchmark_table.py           # print to stdout
    python3 scripts/generate_benchmark_table.py --readme   # in-place update README.md
    python3 scripts/generate_benchmark_table.py --check    # exit 1 if README is stale
"""

import json, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_PATH = REPO_ROOT / "site" / "benchmarks.json"
README_PATH = REPO_ROOT / "README.md"

STATUS_EMOJI = {
    "validated": "✅ validated",
    "measured": "✅ measured",
    "projected": "📐 projected",
    "raw": "⚙️ raw",
    "reported": "📋 reported",
    "disproven": "❌ disproven",
    "unresolved": "🔶 unresolved",
    "broken": "❌ broken",
}

ENGINE_META = {
    "gpu_1bit": {"engine": "**GPU 1-bit** (llama.cpp ROCm) 🏆", "backend": "ROCm HIP", "hw": "Radeon 8060S"},
    "ternary":  {"engine": "**GPU ternary** (Vulkan)",           "backend": "Vulkan GLSL",   "hw": "Radeon 8060S"},
    "npu_v12":  {"engine": "**NPU v12**",                        "backend": "XDNA 2 xclbin", "hw": "XDNA 2 · 32 tiles"},
    "npu_flm":  {"engine": "**NPU FLM** (production)",           "backend": "XDNA 2 xclbin", "hw": "XDNA 2 · 32 tiles"},
    "all_5":    {"engine": "**C++ all-5** (auto-detect)",        "backend": "Q4NX header parse", "hw": "XDNA 2 · 32 tiles"},
    "gpu_zinc": {"engine": "**GPU ZINC** (Vulkan)",              "backend": "Vulkan GLSL",   "hw": "Radeon 8060S"},
    "rocm_hip": {"engine": "**GPU ROCm HIP** (kernels)",         "backend": "ROCm HIP",      "hw": "Radeon 8060S"},
    "npu_fused":{"engine": "**NPU fused**",                      "backend": "XDNA 2 xclbin", "hw": "XDNA 2 · 32 tiles"},
    "zaya":     {"engine": "**GPU Zaya** (ROCm HIP)",            "backend": "HIP kernels",   "hw": "Radeon 8060S"},
    "dspark":   {"engine": "**DSpark** (spec-decode)",           "backend": "Speculative draft", "hw": "XDNA 2 · 32 tiles"},
}


def load_benchmarks():
    with open(BENCHMARKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def fmt_tok_s(val):
    return str(int(val)) if val == int(val) else f"{val:.1f}"


def fmt_row(key, entry):
    meta = ENGINE_META.get(key, {"engine": key, "backend": "?", "hw": "?"})
    status = entry["status"]
    emoji = STATUS_EMOJI.get(status, f"❓ {status}")
    note_marker = " (see note)" if entry.get("note") else ""
    return (
        f"| {meta['engine']} | {meta['backend']} | {meta['hw']} "
        f"| **{fmt_tok_s(entry['tok_s'])}** | {emoji}{note_marker} |"
    )


def generate(benchmarks):
    lines = [
        "## Hardware-Validated Performance",
        "",
        "Measured on **AMD Strix Halo** (Ryzen AI Max+ 395) — 32 XDNA 2 NPU tiles + Radeon 8060S (gfx1151) GPU, 128 GB unified LPDDR5X memory.",
        "",
        "| Engine | Backend | Hardware | tok/s | Status |",
        "|--------|---------|----------|:-----:|--------|",
    ]

    order = benchmarks.get("order", [])
    engines = benchmarks["engines"]
    notes = []

    for key in order:
        entry = engines.get(key)
        if entry is None:
            continue
        lines.append(fmt_row(key, entry))
        note = entry.get("note", "").strip()
        if note:
            label = re.sub(r"\*\*", "", ENGINE_META.get(key, {}).get("engine", key))
            notes.append(f"> **{label}:** {note}")

    lines.append("")
    if notes:
        lines.append("\n\n".join(notes))
        lines.append("")
    lines.append((
        f"*Table auto-generated from [`site/benchmarks.json`](site/benchmarks.json)"
        f" — last updated {benchmarks['updated']}*"
    ))
    lines.append("")
    lines.append("See [full benchmark data](benchmarks/RESULTS-stack-2026-04-28.md).")

    return "\n".join(lines)


def replace_in_readme(markdown):
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    start = content.find("## Hardware-Validated Performance")
    if start == -1:
        print("ERROR: Section header not found in README.md", file=sys.stderr)
        sys.exit(1)

    # Find the line after "See [full benchmark data]" to use as boundary
    marker = "See [full benchmark data]"
    end = content.find(marker, start)
    if end >= 0:
        # Advance to end of that line
        end = content.find("\n", end) + 1
    else:
        # Fall back to next section
        end = content.find("\n## ", start + 10)
        if end < 0:
            end = len(content)

    new_content = content[:start] + markdown + "\n" + content[end:]

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Updated {README_PATH}", file=sys.stderr)


def check_stale():
    """--check: verify README matches generated output, exit 1 if not."""
    with open(README_PATH, encoding="utf-8") as f:
        current = f.read()
    generated = generate(load_benchmarks())
    start = current.find("## Hardware-Validated Performance")
    marker = "See [full benchmark data]"
    end = current.find(marker, start)
    if end >= 0:
        end = current.find("\n", end) + 1
    else:
        end = current.find("\n## ", start + 10)
    section = current[start:end]
    if section.strip() != generated.strip():
        print("❌ README is stale. Run `python3 scripts/generate_benchmark_table.py --readme`", file=sys.stderr)
        sys.exit(1)
    print("✅ README is up to date with benchmarks.json", file=sys.stderr)


def main():
    benchmarks = load_benchmarks()
    markdown = generate(benchmarks)

    if "--check" in sys.argv:
        check_stale()
    elif "--readme" in sys.argv:
        replace_in_readme(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
