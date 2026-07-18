#!/usr/bin/env python3
"""
Generate the README.md "## Benchmarks" section from site/benchmarks.json —
the single source of truth. Two tables: kernel-level microbenchmarks and
end-to-end inference, matching the split introduced in issue #294 (the old
single-table format conflated the two, which was misleading).

Usage:
    python3 scripts/generate_benchmark_table.py           # print to stdout
    python3 scripts/generate_benchmark_table.py --readme   # in-place update README.md
    python3 scripts/generate_benchmark_table.py --check    # exit 1 if README is stale
"""

import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_PATH = REPO_ROOT / "site" / "benchmarks.json"
README_PATH = REPO_ROOT / "README.md"

SECTION_HEADER = "## Benchmarks"

WARNING_CALLOUT = (
    "> ⚠️ **The table below mixes kernel-level synthetic microbenchmarks with "
    "end-to-end inference.** Rows in the first table are kernel-level only — "
    "they exclude KV-cache attention, softmax, RoPE, FFN non-GEMM ops, sampler, "
    "tokenizer, and host↔device transfers. **Real end-to-end throughput is "
    "substantially lower** — see the second table. See "
    "[issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235) "
    "for discussion."
)


def load_benchmarks():
    with open(BENCHMARKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def fmt_num(val):
    return str(int(val)) if val == int(val) else f"{val:.2f}"


def fmt_value(entry):
    if entry.get("tflops"):
        return f"**{fmt_num(entry['tflops'])} TFLOPS**"
    return f"**{fmt_num(entry['tok_s'])} tok/s**"


def kernel_row(key, engines):
    e = engines[key]
    return f"| {e['display_name']} | {fmt_value(e)} | {e['backend']} |"


def end_to_end_row(key, engines):
    e = engines[key]
    notes = e.get("notes", "")
    return f"| {e['display_name']} | {fmt_value(e)} | {e['backend']} | {notes} |"


def generate(benchmarks):
    engines = benchmarks["engines"]
    kernel_keys = benchmarks.get("table_kernel", [])
    e2e_keys = benchmarks.get("table_end_to_end", [])

    lines = [
        SECTION_HEADER,
        "",
        "*Numbers auto-update from [`site/benchmarks.json`](site/benchmarks.json) on every push.*",
        "",
        WARNING_CALLOUT,
        "",
        "### 🧪 Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)",
        "",
        "| Benchmark | Value | Backend |",
        "|-----------|:-----:|---------|",
    ]
    lines += [kernel_row(k, engines) for k in kernel_keys if k in engines]

    lines += [
        "",
        "### 🏁 End-to-End Inference (real model, real prompts)",
        "",
        "| Benchmark | Value | Backend | Notes |",
        "|-----------|:-----:|---------|-------|",
    ]
    lines += [end_to_end_row(k, engines) for k in e2e_keys if k in engines]

    return "\n".join(lines)


def _section_bounds(content):
    """Bounds of the generated content only — up to (not including) the
    "\\n\\n---\\n\\n" divider that separates this section from the next.
    replace_in_readme() re-adds that divider itself when writing."""
    start = content.find(SECTION_HEADER)
    if start == -1:
        return None, None
    divider = content.find("\n\n---\n\n", start + len(SECTION_HEADER))
    if divider >= 0:
        end = divider
    else:
        next_header = content.find("\n## ", start + len(SECTION_HEADER))
        end = next_header if next_header >= 0 else len(content)
    return start, end


def replace_in_readme(markdown):
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    start, end = _section_bounds(content)
    if start is None:
        print(f"ERROR: '{SECTION_HEADER}' not found in README.md", file=sys.stderr)
        sys.exit(1)

    # content[end:] already starts with the "\n\n---\n\n" divider — _section_bounds
    # stops right before it, so nothing more needs inserting here.
    new_content = content[:start] + markdown + content[end:]

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Updated {README_PATH}", file=sys.stderr)


def check_stale():
    """--check: verify README matches generated output, exit 1 if not."""
    with open(README_PATH, encoding="utf-8") as f:
        current = f.read()
    generated = generate(load_benchmarks())
    start, end = _section_bounds(current)
    if start is None:
        print(f"ERROR: '{SECTION_HEADER}' not found in README.md", file=sys.stderr)
        sys.exit(1)
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
