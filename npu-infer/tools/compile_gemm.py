#!/usr/bin/env python3
"""
Parameterized GEMM xclbin compiler wrapper.
Generates MLIR via IRON Python dialect, compiles via aiecc.

Usage:
  python compile_gemm.py --M 128 --K 1024 --N 2048 --rows 1
"""

import argparse, os, shutil, subprocess, sys
from pathlib import Path

SRCDIR = Path("/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1")
BUILD_DIR = SRCDIR / "build"
AIECC = Path("/home/bcloud/mlir-aie/build/bin/aiecc")
PYTHON = Path("/home/bcloud/torch2aie/.venv/bin/python")
AIE_PYTHONPATH = "/home/bcloud/torch2aie/toolchain/mlir_aie/python"
AIE_TOOLCHAIN_BIN = "/home/bcloud/torch2aie/toolchain/bin"
AIE_TOOLS = "/home/bcloud/torch2aie/toolchain/aietools"
VALID_ROWS = {1: "n1_core_bf16", 2: "n2_core_placed", 4: "n32_core_placed"}
# AIE column count is always 8 in this config
N_AIE_COLS = 8


def generate_xclbin(M, K, N, m=128, k=64, n=128, rows=1, force=False):
    """Generate xclbin + insts for given GEMM dims. Returns (xclbin_path, insts_path)."""
    if M % m != 0:
        raise ValueError(f"M={M} not divisible by tile m={m}")
    if K % k != 0:
        raise ValueError(f"K={K} not divisible by tile k={k}")
    if N % n != 0:
        raise ValueError(f"N={N} not divisible by tile n={n}")
    if rows not in VALID_ROWS:
        raise ValueError(f"rows must be in {list(VALID_ROWS.keys())}")

    n_aie_rows = rows
    num_row_tile = M // m // n_aie_rows
    num_col_tile = N // n // N_AIE_COLS
    num_groups = num_row_tile * num_col_tile
    if num_groups < 1:
        raise ValueError(
            f"(M={M}, m={m}, rows={rows}) => M//m//rows={num_row_tile}, "
            f"(N={N}, n={n}) => N//n//{N_AIE_COLS}={num_col_tile}, "
            f"need at least 1 group, got {num_groups}. "
            f"Minimum: M >= {m * n_aie_rows}, N >= {n * N_AIE_COLS}"
        )

    mlir_name = f"aie_{M}x{K}x{N}_{m}x{k}x{n}.mlir"
    xclbin_name = f"final_{M}x{K}x{N}_{m}x{k}x{n}.xclbin"
    insts_name = f"insts_{M}x{K}x{N}_{m}x{k}x{n}.txt"
    mlir_path, xclbin_path, insts_path = (
        BUILD_DIR / mlir_name, BUILD_DIR / xclbin_name, BUILD_DIR / insts_name
    )
    if not force and xclbin_path.exists() and insts_path.exists():
        return (xclbin_path, insts_path)

    os.makedirs(BUILD_DIR, exist_ok=True)
    py_src = SRCDIR / f"{VALID_ROWS[rows]}.py"

    # Generate MLIR (needs PYTHONPATH for aie package)
    mlir_env = os.environ.copy()
    mlir_env["PYTHONPATH"] = f"{AIE_PYTHONPATH}:{mlir_env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        [PYTHON, str(py_src), "-m", str(m), "-k", str(k), "-n", str(n),
         "-M", str(M), "-K", str(K), "-N", str(N)],
        capture_output=True, text=True, cwd=SRCDIR, env=mlir_env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"MLIR gen failed: {r.stderr[:1000]}")
    with open(mlir_path, "w") as f:
        f.write(r.stdout)
    print(f"  [OK] MLIR: {mlir_name}")

    # Ensure chess_intrinsic_wrapper.ll exists where aiecc expects it
    chess_wrapper_src = Path("/home/bcloud/torch2aie/toolchain/mlir_aie/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll")
    chess_wrapper_dst = Path("/home/bcloud/mlir-aie/build/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll")
    if not chess_wrapper_dst.exists():
        chess_wrapper_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chess_wrapper_src, chess_wrapper_dst)
        print(f"  [FIX] Copied {chess_wrapper_src} -> {chess_wrapper_dst}")

    # Compile via aiecc
    aiecc_env = os.environ.copy()
    aiecc_env["PATH"] = f"{AIE_TOOLCHAIN_BIN}:/home/bcloud/mlir-aie/build/bin:{aiecc_env.get('PATH', '')}"
    aiecc_env["PYTHONPATH"] = f"{AIE_PYTHONPATH}:{aiecc_env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        [AIECC, f"--aietools={AIE_TOOLS}",
         "--alloc-scheme=basic-sequential", "--aie-generate-xclbin",
         "--no-compile-host", f"--xclbin-name={xclbin_name}",
         "--unified", "--dynamic-objFifos", "--aie-generate-npu-insts",
         f"--npu-insts-name={insts_name}", str(mlir_path)],
        capture_output=True, text=True, cwd=BUILD_DIR, env=aiecc_env,
    )
    if r.returncode != 0:
        raise RuntimeError(f"aiecc failed:\n{r.stdout[-1000:]}\n{r.stderr[-1000:]}")
    if not xclbin_path.exists() or not insts_path.exists():
        raise RuntimeError("xclbin/insts not produced")
    print(f"  [OK] xclbin ({xclbin_path.stat().st_size} B), insts ({insts_path.stat().st_size} B)")
    return (xclbin_path, insts_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--rows", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    xclbin, insts = generate_xclbin(args.M, args.K, args.N, args.m, args.k, args.n, args.rows, args.force)
    print(f"xclbin: {xclbin}\ninsts:  {insts}")

if __name__ == "__main__":
    main()
