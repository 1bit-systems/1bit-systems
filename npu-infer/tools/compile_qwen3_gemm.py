#!/usr/bin/env python3
"""
Adapt torch2aie config1 GEMM design for Qwen3-0.6B dimensions.
Uses the Python MLIR-AIE dialect to generate proper MLIR, then compiles via aiecc.

Qwen3-0.6B GEMM requirements:
  Q/K/V/O projections: M=1024, K=1024, N=64 (batched prefill)
  gate/up: M=3072, K=1024
  down: M=1024, K=3072

We compile one xclbin that can handle all these shapes by picking good tile sizes.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# ---- Path setup ----
TORCH2AIE_ROOT = Path("/home/bcloud/torch2aie")
TOOLCHAIN = TORCH2AIE_ROOT / "toolchain"
AIETOOLS = TOOLCHAIN / "aietools"
MLIR_AIE_DIR = TOOLCHAIN / "mlir_aie"
XRT_DIR = TOOLCHAIN / "xrt"
SYSROOT = TOOLCHAIN / "sysroot"
SRCDIR = TORCH2AIE_ROOT / "examples/gemm_asymmetric_tile_buffering/config1"

# ---- Environment ----
os.environ.setdefault("TORCH2AIE_TOOLCHAIN", str(TOOLCHAIN))
os.environ.setdefault("AIETOOLS_DIR", str(AIETOOLS))
os.environ.setdefault("MLIR_AIE_DIR", str(MLIR_AIE_DIR))
os.environ.setdefault("XRT_DIR", str(XRT_DIR))
os.environ.setdefault("TORCH2AIE_SYSROOT", str(SYSROOT))

for var, paths in [
    ("PATH", [TOOLCHAIN/"bin", MLIR_AIE_DIR/"bin", XRT_DIR/"bin", AIETOOLS/"bin"]),
    ("LD_LIBRARY_PATH", [SYSROOT/"usr/lib64", TOOLCHAIN/"mlir_aie.libs",
                         MLIR_AIE_DIR/"lib", AIETOOLS/"lib", AIETOOLS/"lib/lnx64.o",
                         XRT_DIR/"lib64"]),
    ("PYTHONPATH", [str(TORCH2AIE_ROOT/"examples"),
                    str(XRT_DIR/"python"), str(MLIR_AIE_DIR/"python"),
                    str(MLIR_AIE_DIR/"src/python")]),
]:
    existing = os.environ.get(var, "")
    vals = ":".join(str(p) for p in paths if p.exists())
    os.environ[var] = f"{vals}:{existing}" if existing else vals

# Now import MLIR-AIE
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

BUILD_DIR = Path("/home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm")


def ceildiv(a, b):
    return (a + b - 1) // b


def gen_qwen3_gemm_mlir(M, K, N, m_tile, k_tile, n_tile, n_aie_cols, n_aie_rows):
    """
    Generate MLIR for a GEMM design using the Python AIE dialect.
    Modelled after n32_core_placed.py but adapted for qwen3-0.6B sizes.

    Uses the pre-compiled mm_128x64x128.o Chess kernel.
    """
    # Validate tile sizes match the Chess kernel
    if m_tile != 128 or k_tile != 64 or n_tile != 128:
        print(f"WARNING: tile sizes {m_tile}x{k_tile}x{n_tile} != 128x64x128")
        print("The Chess kernel mm_128x64x128.o is compiled for 128x64x128 tiles.")
        print("Proceeding but may fail.")

    dtype_in = bfloat16
    dtype_out = bfloat16

    # Use the BFP16 mixed kernel for better efficiency (weights are BFP16)
    use_bfp16_weights = True

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def device_body():
            # L2 tile types
            A_l2_ty = np.ndarray[(m_tile, 512), np.dtype[dtype_in]]
            C_l2_ty = np.ndarray[(n_aie_rows * m_tile, n_tile), np.dtype[dtype_out]]

            # L1 tile types
            A_l1_ty = np.ndarray[(m_tile // 4, k_tile), np.dtype[dtype_in]]
            C_l1_ty = np.ndarray[(m_tile, n_tile), np.dtype[dtype_out]]

            if use_bfp16_weights:
                B_l2_ty = np.ndarray[(k_tile, n_tile // 8), np.dtype[v8bfp16ebs8]]
                B_l1_ty = np.ndarray[(k_tile, n_tile // 8), np.dtype[v8bfp16ebs8]]
                matmul_func = "matmul_vectorized_different_datatypes"
            else:
                B_l2_ty = np.ndarray[(k_tile, n_tile), np.dtype[dtype_in]]
                B_l1_ty = np.ndarray[(k_tile, n_tile), np.dtype[dtype_in]]
                matmul_func = "matmul_vectorized_bf16"

            # External kernel functions - link against the compiled Chess kernel
            zero = external_func(
                "zero_kernel_bf16",
                inputs=[C_l1_ty],
                link_with=f"mm_{m_tile}x{k_tile}x{n_tile}.o",
            )
            matmul = external_func(
                matmul_func,
                inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
                link_with=f"mm_{m_tile}x{k_tile}x{n_tile}.o",
            )

            # Tile grid: rows 0-1 are shim/mem, rows 2+ are compute
            tiles = [
                [tile(col, row) for col in range(0, n_aie_cols)]
                for row in range(0, 2 + n_aie_rows)
            ]
            shim_tiles = tiles[0]
            mem_tiles = tiles[1]
            core_tiles = tiles[2:]

            # ObjectFIFOs for A (activation): L3→L2→L1
            A_l3l2_fifos = [None] * n_aie_rows
            A_l2l1_fifos = [None] * n_aie_rows

            for row in range(n_aie_rows):
                shim_col = (row * 2) % n_aie_cols
                A_l3l2_fifos[row] = object_fifo(
                    f"A_L3L2_{row}",
                    shim_tiles[shim_col],
                    mem_tiles[shim_col],
                    2,
                    A_l2_ty,
                    None,
                    [[(m_tile, k_tile), (512 // k_tile, m_tile * k_tile), (k_tile, 1)]],
                )
                A_l2l1_fifos[row] = object_fifo(
                    f"A_L2L1_{row}",
                    mem_tiles[shim_col],
                    core_tiles[row][0:n_aie_cols],
                    2,
                    A_l1_ty,
                    [(512 // k_tile * 4, m_tile * k_tile // 4), (k_tile // 8, 8),
                     (m_tile // 4, k_tile), (8, 1)],
                    [[(k_tile // 8, 2 * 8), (m_tile // 4 // 2, 2 * k_tile), (2 * 8, 1)]
                     for _ in range(n_aie_cols)],
                )
                object_fifo_link(A_l3l2_fifos[row], A_l2l1_fifos[row])

            # ObjectFIFOs for B (weights): L3→L2→L1
            B_l3l2_fifos = [None] * n_aie_cols
            B_l2l1_fifos = [None] * n_aie_cols

            for col in range(n_aie_cols):
                B_l3l2_fifos[col] = object_fifo(
                    f"B_L3L2_{col}",
                    shim_tiles[col],
                    mem_tiles[col],
                    2,
                    B_l2_ty,
                )
                B_l2l1_fifos[col] = object_fifo(
                    f"B_L2L1_{col}",
                    mem_tiles[col],
                    [core_tiles[j][col] for j in range(n_aie_rows)],
                    2,
                    B_l1_ty,
                )
                object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

            # ObjectFIFOs for C (output): L1→L2→L3
            C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
            C_l2l3_fifos = [None] * n_aie_cols

            for col in range(n_aie_cols):
                for row in range(n_aie_rows):
                    C_l1l2_fifos[row][col] = object_fifo(
                        f"C_L1L2_{col}_{row}",
                        core_tiles[row][col],
                        mem_tiles[col],
                        1,
                        C_l1_ty,
                    )
                C_l2l3_fifos[col] = object_fifo(
                    f"C_L2L3_{col}",
                    mem_tiles[col],
                    shim_tiles[col],
                    2,
                    C_l2_ty,
                    [(m_tile // 2, 2 * n_tile), (2, 8), (n_tile // 8, 2 * 8), (8, 1)],
                )
                of_offsets = [m_tile * n_tile * i for i in range(n_aie_rows)]
                object_fifo_link(
                    [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],
                    C_l2l3_fifos[col],
                    of_offsets,
                )

            # Compute tile code
            for row in range(n_aie_rows):
                for col in range(n_aie_cols):
                    @core(core_tiles[row][col], stack_size=0xD00)
                    def core_body():
                        for _ in range_(0xFFFFFFFF):
                            for _ in range_(
                                (M // m_tile) * (N // n_tile) // (n_aie_cols * n_aie_rows)
                            ):
                                elem_out = C_l1l2_fifos[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                zero(elem_out)
                                for _ in range_(K // k_tile):
                                    elem_in_b = B_l2l1_fifos[col].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    for i in range(4):
                                        elem_in_a = A_l2l1_fifos[row].acquire(
                                            ObjectFifoPort.Consume, 1
                                        )
                                        matmul(elem_in_a, elem_in_b, elem_out)
                                        A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                    B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                                C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)

            # Runtime sequence for DMA
            if use_bfp16_weights:
                @runtime_sequence(
                    np.ndarray[(M * K,), np.dtype[dtype_in]],
                    np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]],
                    np.ndarray[(M * N,), np.dtype[dtype_out]],
                )
                def sequence(A, B, C):
                    A_taps = TensorTiler2D.group_tiler((M, K), (m_tile, 512), (1, K // 512))
                    B_taps = TensorTiler2D.group_tiler(
                        (1, N * K // 8), (1, n_tile * K // 8), (1, 1)
                    )
                    C_taps = TensorTiler2D.group_tiler(
                        (M, N), (n_aie_rows * m_tile, n_tile), (1, 1)
                    )

                    num_row_tile = M // m_tile // n_aie_rows
                    num_col_tile = N // n_tile // n_aie_cols
                    num_groups = max(num_row_tile * num_col_tile, 1)
                    tb_max_n_rows = 4
                    input_task_groups = [[] for _ in range(tb_max_n_rows)]
                    output_task_groups = [[] for _ in range(tb_max_n_rows)]

                    for group_idx in range(num_groups):
                        a_base_idx = (group_idx // max(num_col_tile, 1)) * n_aie_rows
                        for row in range(n_aie_rows):
                            a_task = shim_dma_single_bd_task(
                                A_l3l2_fifos[row], A,
                                tap=A_taps[a_base_idx + row],
                                issue_token=False,
                            )
                            dma_start_task(a_task)
                            input_task_groups[group_idx % tb_max_n_rows].append(a_task)

                        b_base_idx = (group_idx % max(num_col_tile, 1)) * n_aie_cols
                        for col in range(n_aie_cols):
                            b_task = shim_dma_single_bd_task(
                                B_l3l2_fifos[col], B,
                                tap=B_taps[b_base_idx + col],
                                issue_token=False,
                            )
                            dma_start_task(b_task)
                            input_task_groups[group_idx % tb_max_n_rows].append(b_task)

                        c_base_idx = group_idx * n_aie_cols
                        for col in range(n_aie_cols):
                            c_task = shim_dma_single_bd_task(
                                C_l2l3_fifos[col], C,
                                tap=C_taps[c_base_idx + col],
                                issue_token=True,
                            )
                            dma_start_task(c_task)
                            output_task_groups[group_idx % tb_max_n_rows].append(c_task)

                        # Drain and free tasks periodically
                        if group_idx % tb_max_n_rows == 1 and group_idx != 1:
                            for g in [2, 3]:
                                if output_task_groups[g]:
                                    dma_await_task(*output_task_groups[g])
                                    output_task_groups[g] = []
                                if input_task_groups[g]:
                                    dma_free_task(*input_task_groups[g])
                                    input_task_groups[g] = []
                        if group_idx % tb_max_n_rows == 3:
                            for g in [0, 1]:
                                if output_task_groups[g]:
                                    dma_await_task(*output_task_groups[g])
                                    output_task_groups[g] = []
                                if input_task_groups[g]:
                                    dma_free_task(*input_task_groups[g])
                                    input_task_groups[g] = []

                    # Drain remaining
                    for g in range(tb_max_n_rows):
                        if output_task_groups[g]:
                            dma_await_task(*output_task_groups[g])
                        if input_task_groups[g]:
                            dma_free_task(*input_task_groups[g])

        # Print generated MLIR
        mlir_text = str(ctx.module)
        return mlir_text


def compile_design(mlir_text, xclbin_path, insts_path):
    """Compile the MLIR design through aiecc."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    mlir_path = BUILD_DIR / "design.mlir"
    mlir_path.write_text(mlir_text)

    # Copy the pre-compiled Chess kernel
    kernel_obj = SRCDIR / "build/mm_128x64x128.o"
    if kernel_obj.exists():
        shutil.copy2(kernel_obj, BUILD_DIR / "mm_128x64x128.o")
        print(f"Copied {kernel_obj}")

    # Compile with aiecc
    aiecc = TOOLCHAIN / "bin/aiecc"
    cmd = [
        str(aiecc), "-v", "-j4",
        f"--aietools={AIETOOLS}",
        "--no-compile-host",
        "--alloc-scheme=basic-sequential",
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print(f"Running aiecc...")
    print(f"  {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, cwd=str(BUILD_DIR))
    if result.returncode != 0:
        print("AIECC FAILED!")
        return False
    print(f"  xclbin: {xclbin_path}")
    print(f"  insts:  {insts_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--name", type=str, default="qwen3_gemm")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["decode_128x1024x2048", "decode_128x1024x1024",
                                 "decode_128x1024x3072", "decode_128x2048x1024",
                                 "decode_128x3072x1024", "prefill_1024x1024x2048"])
    args = parser.parse_args()

    # Apply presets
    if args.preset == "decode_128x1024x2048":
        args.M, args.K, args.N = 128, 1024, 2048; args.name = "decode_128x1024x2048"
    elif args.preset == "decode_128x1024x1024":
        args.M, args.K, args.N = 128, 1024, 1024; args.name = "decode_128x1024x1024"
    elif args.preset == "decode_128x1024x3072":
        args.M, args.K, args.N = 128, 1024, 3072; args.name = "decode_128x1024x3072"
    elif args.preset == "decode_128x2048x1024":
        args.M, args.K, args.N = 128, 2048, 1024; args.name = "decode_128x2048x1024"
    elif args.preset == "decode_128x3072x1024":
        args.M, args.K, args.N = 128, 3072, 1024; args.name = "decode_128x3072x1024"
    elif args.preset == "prefill_1024x1024x2048":
        args.M, args.K, args.N = 1024, 1024, 2048; args.name = "prefill_1024x1024x2048"

    m_tile, k_tile, n_tile = 128, 64, 128
    n_cols, n_rows = 4, 4  # 16 cores

    # For M=128 decode (single token padded), use 1 AIE row design
    if args.M <= 128:
        n_rows = 1
        print(f"NOTE: Using n_aie_rows=1 (8-core design) for M={args.M}")

    print(f"=== Qwen3-0.6B GEMM xclbin compilation ===")
    print(f"  M,N,K = {args.M},{args.N},{args.K}")
    print(f"  Tiles = {m_tile}x{k_tile}x{n_tile}")
    print(f"  Cores = {n_cols}x{n_rows} = {n_cols*n_rows}")

    # Generate MLIR
    print("Generating MLIR...")
    mlir_text = gen_qwen3_gemm_mlir(args.M, args.K, args.N,
                                     m_tile, k_tile, n_tile,
                                     n_cols, n_rows)
    print(f"  Generated {len(mlir_text)} chars of MLIR")

    if args.build_only:
        xclbin_path = BUILD_DIR / f"{args.name}.xclbin"
        insts_path = BUILD_DIR / f"{args.name}.insts"
        success = compile_design(mlir_text, xclbin_path, insts_path)
        if success:
            print(f"SUCCESS: {xclbin_path}")
            print(f"  Usage: point engine at {xclbin_path}")
            print(f"  Instrs: {insts_path}")
        return 0 if success else 1

    print("\n#0 FUSION: Build custom M=128 decode xclbins:")
    for p in ["decode_128x1024x1024", "decode_128x1024x2048", "decode_128x1024x3072",
              "decode_128x2048x1024", "decode_128x3072x1024"]:
        print(f"  python tools/compile_qwen3_gemm.py --preset {p} --build-only")
    print()
    print("Or use the existing built xclbins at:")
    print("  /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
