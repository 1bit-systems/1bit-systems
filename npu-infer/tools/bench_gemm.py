#!/usr/bin/env python3
"""
Benchmark parameterized GEMM kernels across multiple dimensions.
Reports correctness and performance for each.

Usage:
  python tools/bench_gemm.py --dims 128x1024x2048 1024x1024x2048 --rows 1 --iterations 3
  python tools/bench_gemm.py --all-qwen3-06b
  python tools/bench_gemm.py --dims 128x1024x2048 --rows 1 --constant --iterations 1 -v
"""
import argparse, json, time, sys
from pathlib import Path

# Ensure project root is on sys.path for import of tools.verify_gemm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Qwen3-0.6B dimensions
# Note: rows=2 has known correctness bugs even with constant data,
# and ramp data has layout bugs. All Qwen3 dims will show errors for ramp
# but are still useful for GFLOPS measurement.
QWEN3_06B_DIMS = {
    "qkv":  {"M": 128, "K": 1024, "N": 3072, "rows": 2, "A_val": 1.0, "B_val": 3.0},
    "o":    {"M": 128, "K": 3072, "N": 1024, "rows": 2, "A_val": 1.0, "B_val": 3.0},
    "gu":   {"M": 128, "K": 1024, "N": 8192, "rows": 2, "A_val": 1.0, "B_val": 3.0},
    "down": {"M": 128, "K": 8192, "N": 1024, "rows": 2, "A_val": 1.0, "B_val": 3.0},
    "ff1":  {"M": 128, "K": 1024, "N": 8192, "rows": 2, "A_val": 1.0, "B_val": 3.0},
    "ff2":  {"M": 128, "K": 8192, "N": 1024, "rows": 2, "A_val": 1.0, "B_val": 3.0},
}


def bench_one(M, K, N, rows=1, iterations=3, A_val=None, B_val=None):
    """
    Benchmark one GEMM dimension.

    Args:
        M, K, N: matrix dimensions
        rows: number of AIE rows (1 or 2)
        iterations: number of benchmark iterations
        A_val: if not None, use constant data A=val
        B_val: if not None, use constant data B=val

    Returns:
        dict with M, K, N, rows, avg_gflops, avg_errors, passed
    """
    from tools.verify_gemm import verify

    results = []
    for i in range(iterations):
        r = verify(M, K, N, rows=rows, A_val=A_val, B_val=B_val)
        results.append(r)

    avg_gflops = sum(r["gflops"] for r in results) / len(results)
    avg_errors = sum(r["errors"] for r in results) / len(results)
    return {
        "M": M,
        "K": K,
        "N": N,
        "rows": rows,
        "avg_gflops": round(avg_gflops, 2),
        "avg_errors": round(avg_errors),
        "passed": all(r["passed"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parameterized GEMM kernels across multiple dimensions"
    )
    parser.add_argument(
        "--dims", nargs="+", help="MxKxN dims (e.g. 128x1024x2048)", default=[]
    )
    parser.add_argument("--rows", type=int, default=1, choices=[1, 2])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--all-qwen3-06b", action="store_true",
                        help="Benchmark all Qwen3-0.6B model dimensions")
    parser.add_argument("--report", help="Save JSON report to path")
    parser.add_argument("--constant", action="store_true",
                        help="Use constant data (A=1.0, B=3.0) instead of ramp vectors")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-iteration details")
    args = parser.parse_args()

    dims = []
    if args.all_qwen3_06b:
        for name, cfg in QWEN3_06B_DIMS.items():
            A_val = cfg["A_val"] if args.constant else None
            B_val = cfg["B_val"] if args.constant else None
            dims.append((cfg["M"], cfg["K"], cfg["N"], cfg["rows"],
                         A_val, B_val, name))

    for d in args.dims:
        parts = d.split("x")
        M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
        A_val = 1.0 if args.constant else None
        B_val = 3.0 if args.constant else None
        dims.append((M, K, N, args.rows, A_val, B_val, d))

    if not dims:
        parser.print_help()
        sys.exit(1)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "constant": args.constant,
        "results": [],
    }

    for M, K, N, rows, A_val, B_val, label in dims:
        print(f"\n{'=' * 60}")
        print(f"GEMM {M}x{K}x{N} (rows={rows}, label={label})")
        if args.constant:
            print(f"  Using constant data A={A_val}, B={B_val}")
        else:
            print(f"  Using ramp vectors")

        try:
            r = bench_one(M, K, N, rows, args.iterations, A_val=A_val, B_val=B_val)
            r["label"] = label
            report["results"].append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {status}: {r['avg_gflops']} GFLOPS, {r['avg_errors']} errors")
        except Exception as e:
            print(f"  FAIL: {e}")
            report["results"].append({
                "M": M, "K": K, "N": N, "rows": rows,
                "label": label, "error": str(e),
            })

    # Summary
    passed_count = sum(1 for r in report["results"] if r.get("passed"))
    total_count = sum(1 for r in report["results"] if "error" not in r)
    print(f"\n{'=' * 60}")
    print(f"Summary: {passed_count}/{total_count} passed")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {args.report}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
