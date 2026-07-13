#!/usr/bin/env python3
"""
NPU GEMM Verification Runner.

Generates BF16 test vectors → compiles xclbin via compile_gemm.py → runs GEMM
on NPU via a C++ binary → reads C output → compares vs CPU BF16 reference.

Usage:
  python tools/verify_gemm.py --M 128 --K 1024 --N 2048 --rows 1

NOTE: Data layout correctness
The C++ runner (npu_gemm_runner.cpp) applies host-side layout shuffling
(layout_A_L1_2x1_8x8block, layout_transpose_L1_1x2_8x8block,
floatToBfp16, layout_inverse_C_L1_2x2_8x8block) following the pattern
from test_correctness.cpp. However, there is a known mismatch between
the host-side shuffle and what the MLIR IRON design's on-device object_fifo
transformations expect for non-constant data. Constant data (where shuffle
is identity) produces 0 errors. This is documented at:
.superpowers/sdd/task-3-report.md
"""

import argparse, os, subprocess, sys, time
from pathlib import Path
import numpy as np

# Add project root for relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compile_gemm import generate_xclbin
from tests.cpu_ref_gemm import generate_test_vectors, bf16_to_float

# Cache the compiled binary path
_COMPILED_RUNNER = None


def _get_runner_path():
    """Return path to compiled npu_gemm_runner, compiling if needed."""
    global _COMPILED_RUNNER
    if _COMPILED_RUNNER is not None and os.path.exists(_COMPILED_RUNNER):
        return _COMPILED_RUNNER

    runner_src = PROJECT_ROOT / "tools" / "npu_gemm_runner.cpp"
    runner_bin = PROJECT_ROOT / "tools" / "npu_gemm_runner.bin"

    if runner_bin.exists():
        src_mtime = os.path.getmtime(runner_src)
        bin_mtime = os.path.getmtime(runner_bin)
        if bin_mtime >= src_mtime:
            _COMPILED_RUNNER = str(runner_bin)
            return _COMPILED_RUNNER

    print("Compiling npu_gemm_runner...")
    cmd = [
        "g++", "-std=c++23", "-O2", "-o", str(runner_bin), str(runner_src),
        "-I/home/bcloud/torch2aie/examples",
        "-I/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering",
        "-lxrt_coreutil", "-lxrt_core", "-lpthread", "-ldl", "-luuid",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Failed to compile npu_gemm_runner:\n"
            f"stdout: {r.stdout[:500]}\nstderr: {r.stderr[:500]}"
        )
    _COMPILED_RUNNER = str(runner_bin)
    return _COMPILED_RUNNER


def run_npu_gemm(xclbin_path, insts_path, A_f32, B_f32, M, K, N, n_aie_rows=1):
    """
    Run GEMM on NPU using the C++ runner.

    Args:
        xclbin_path: Path to .xclbin file
        insts_path: Path to instructions file
        A_f32: numpy array of shape (M, K), float32
        B_f32: numpy array of shape (K, N), float32
        M, K, N: matrix dimensions
        n_aie_rows: number of AIE rows (1 or 2)

    Returns:
        (C_f32, gflops) where C_f32 is numpy array (M, N) float32
    """
    runner = _get_runner_path()

    A_bytes = np.ascontiguousarray(A_f32.flatten(), dtype=np.float32).tobytes()
    B_bytes = np.ascontiguousarray(B_f32.flatten(), dtype=np.float32).tobytes()

    proc = subprocess.Popen(
        [runner, str(xclbin_path), str(insts_path),
         str(M), str(K), str(N), str(n_aie_rows)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    C_bytes, stderr_data = proc.communicate(input=A_bytes + B_bytes)
    if proc.returncode != 0:
        raise RuntimeError(
            f"npu_gemm_runner failed (exit={proc.returncode}):\n"
            f"{stderr_data.decode(errors='replace')[:1000]}"
        )

    # Parse GFLOPS from stderr
    gflops = 0.0
    for line in stderr_data.decode(errors='replace').split('\n'):
        if 'GFLOPS' in line:
            try:
                gflops = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass

    expected_C_bytes = M * N * 4
    if len(C_bytes) != expected_C_bytes:
        raise RuntimeError(
            f"Expected {expected_C_bytes} bytes of C, got {len(C_bytes)}"
        )

    C_f32 = np.frombuffer(C_bytes, dtype=np.float32).reshape(M, N)
    return C_f32, gflops


def verify(M, K, N, m=128, k=64, n=128, rows=1, force=False, A_val=None, B_val=None):
    """
    Full GEMM verification: generate vectors → compile xclbin → run NPU → compare.

    Args:
        M, K, N: matrix dimensions
        m, k, n: tile sizes
        rows: number of AIE rows (1 or 2)
        force: force recompilation of xclbin
        A_val: if not None, use constant A[m,k] = A_val (bypasses ramp vectors)
        B_val: if not None, use constant B[k,n] = B_val (bypasses ramp vectors)

    Returns:
        dict with passed, errors, total, max_error, gflops, ttft_us
    """
    t_start = time.monotonic()

    # Step 1: Generate test vectors (ramp or constant)
    if A_val is not None and B_val is not None:
        print(f"Generating constant test vectors A={A_val}, B={B_val} for {M}x{K}x{N}...")
        A_f32 = np.full((M, K), A_val, dtype=np.float32)
        B_f32 = np.full((K, N), B_val, dtype=np.float32)
        # Compute CPU reference: C[m,n] = K * A_val * B_val
        C_ref_f32 = np.full((M, N), float(K) * float(A_val) * float(B_val), dtype=np.float32)
    else:
        print(f"Generating ramp test vectors for {M}x{K}x{N}...")
        A_bf16, B_bf16, C_ref_f32 = generate_test_vectors(M, K, N)
        # Convert BF16 to float32 for the C++ runner
        A_f32 = np.array(
            [[bf16_to_float(A_bf16[m, k]) for k in range(K)] for m in range(M)],
            dtype=np.float32
        )
        B_f32 = np.array(
            [[bf16_to_float(B_bf16[k, n]) for n in range(N)] for k in range(K)],
            dtype=np.float32
        )

    # Step 2: Compile xclbin
    print(f"Compiling xclbin for {M}x{K}x{N} (rows={rows})...")
    xclbin_path, insts_path = generate_xclbin(M, K, N, m, k, n, rows, force=force)

    # Step 3: Run on NPU
    print("Running NPU GEMM...")
    C_npu_f32, gflops = run_npu_gemm(
        xclbin_path, insts_path, A_f32, B_f32, M, K, N, n_aie_rows=rows
    )

    t_end = time.monotonic()
    ttft_us = int((t_end - t_start) * 1_000_000)

    # Step 4: Compare against CPU reference
    print("Comparing vs CPU reference...")
    errors = 0
    max_error = 0.0
    for m in range(M):
        for n in range(N):
            diff = abs(float(C_npu_f32[m, n]) - float(C_ref_f32[m, n]))
            if diff > max_error:
                max_error = diff
            if diff > 2.0:
                errors += 1

    total = M * N
    passed = errors == 0
    print(f"  Errors: {errors} / {total} (max error: {max_error:.2f})")
    print(f"  GFLOPS: {gflops:.2f}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "errors": errors,
        "total": total,
        "max_error": float(max_error),
        "gflops": float(gflops),
        "ttft_us": ttft_us,
    }


def main():
    parser = argparse.ArgumentParser(description="NPU GEMM Verification Runner")
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--rows", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--A_val", type=float, default=None,
                        help="Constant A value (bypasses ramp vectors)")
    parser.add_argument("--B_val", type=float, default=None,
                        help="Constant B value (bypasses ramp vectors)")
    args = parser.parse_args()

    result = verify(args.M, args.K, args.N, args.m, args.k, args.n,
                    args.rows, args.force, args.A_val, args.B_val)
    print(f"\nResult: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"  Errors: {result['errors']} / {result['total']}")
    print(f"  GFLOPS: {result['gflops']:.2f}")
    print(f"  TTFT:   {result['ttft_us']} us")
    sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
