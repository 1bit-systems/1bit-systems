#!/usr/bin/env python3
"""
Native ternary 8-core MLIR generator — object_fifo dataflow for NPU xclbin.

Uses the native mm_ternary_32x64x128.o kernel (2-bit packed ternary decode +
BF16 MAC) across an 8-column AIE core grid. Same proven object_fifo +
shim_dma_single_bd_task pattern as n1_core_ternary.py.

Architecture:
  8 columns × 1 row = 8 AIE cores
  Each core: M/8 weight rows, K*4 ternary values, produces M/8 bf16 scalars

Dataflow (shim → mem → core → mem → shim via object_fifo):
  Input:  flat i32 buffer per core = [weights | scales | activations]
  Output: M bf16 scalars gathered across 8 columns

The native kernel takes (flat_input_ptr, output_ptr) — weights, scales, and
activations packed into one contiguous buffer, not separate A/B streams.

Kernel buffer layout (per core, all in i32 array):
  [0..M/8*K_packed bytes)          : packed uint8 weights (4 ternary/byte)
  [..M/8*K_packed + M/8*2 bytes)  : per-row bf16 scales
  [.. + K*4*2 bytes)               : bf16 activation vector (K ternary values)

Usage:
    python3 n1_core_native_ternary_8core.py -M 32 -K 64 > ternary_8core.mlir
    # Build: aiecc.py ternary_8core.mlir ... → ternary_8core.xclbin
    # Test: ./test_ternary_npu ternary_8core/design.xclbin ...
"""

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser(
        description="Native ternary 8-core NPU xclbin generator"
    )
    parser.add_argument("-M", type=int, default=32,
                        help="Total output rows (must be multiple of 8)")
    parser.add_argument("-K", type=int, default=64,
                        help="Packed K bytes per tile (K*4 = ternary values)")
    parser.add_argument("--dump", action="store_true",
                        help="Print buffer layout calculations")
    args = parser.parse_args()

    assert args.M % 8 == 0, f"M={args.M} must be a multiple of 8"

    with mlir_mod_ctx() as ctx:
        dump_info = args.dump
        mlir = my_native_ternary_8core(args.M, args.K, dump_info)
        if mlir:
            print(mlir)
        else:
            print(ctx.module)


def my_native_ternary_8core(M, K_packed, dump=False):
    """Generate 8-core native ternary MLIR.

    Args:
        M: total output rows (split across 8 cores)
        K_packed: packed weight bytes per tile → K_packed*4 ternary values
    """
    n_cols = 8
    n_rows = 1
    m_per_core = M // n_cols
    k_ternary = K_packed * 4  # e.g. 64*4 = 256 ternary values

    # ── Buffer sizes (per core) ─────────────────────────
    weight_bytes = m_per_core * K_packed    # M/8 * 64
    scale_bytes = m_per_core * 2            # M/8 bf16 = M/8 * 2
    act_bytes = k_ternary * 2               # 256 bf16 = 512 bytes
    in_bytes = weight_bytes + scale_bytes + act_bytes
    in_dwords = (in_bytes + 3) // 4

    out_elems = m_per_core                  # M/8 bf16 outputs per core
    out_dwords = (out_elems * 2 + 3) // 4

    # Total across all columns
    total_in_dwords = in_dwords * n_cols
    total_out_elems = M
    total_out_dwords = out_dwords * n_cols

    if dump:
        print(f"// M={M} K_packed={K_packed} → {k_ternary} ternary values",
              file=__import__('sys').stderr)
        print(f"// m_per_core={m_per_core} in_bytes={in_bytes} in_dwords={in_dwords}",
              file=__import__('sys').stderr)
        print(f"// out_elems={out_elems} out_dwords={out_dwords}",
              file=__import__('sys').stderr)
        print(f"// total_in={total_in_dwords}dw total_out={total_out_dwords}dw",
              file=__import__('sys').stderr)

    # Kernel entry
    kernel_entry = "mm_ternary_32x64x128"
    kernel_o = f"{kernel_entry}.o"

    # Types for object_fifos: flat i32 input, bf16 output
    dtype_in = np.int32
    dtype_out = np.float32  # bf16 not in np; use f32 for shape, kernel handles it

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(out_elems,), np.dtype[dtype_out]]
        C_l2_ty = C_l1_ty

        native_ternary = external_func(
            kernel_entry,
            inputs=[A_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=kernel_o,
        )

        tiles = [
            [tile(col, row) for col in range(n_cols)]
            for row in range(3)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2]

        # ── Input object_fifos: shim→mem→core per column ──
        A_l3l2 = [None] * n_cols
        A_l2l1 = [None] * n_cols
        for col in range(n_cols):
            A_l3l2[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col], mem_tiles[col],
                2, A_l2_ty,
            )
            A_l2l1[col] = object_fifo(
                f"A_L2L1_{col}",
                mem_tiles[col], [core_tiles[0][col]],
                2, A_l1_ty,
            )
            object_fifo_link(A_l3l2[col], A_l2l1[col])

        # ── Output object_fifos: core→mem→shim per column ──
        C_l1l2 = [None] * n_cols
        C_l2l3 = [None] * n_cols
        for col in range(n_cols):
            C_l1l2[col] = object_fifo(
                f"C_L1L2_{col}",
                core_tiles[0][col], mem_tiles[col],
                1, C_l1_ty,
            )
            C_l2l3[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col], shim_tiles[col],
                2, C_l2_ty,
            )
            object_fifo_link(C_l1l2[col], C_l2l3[col])

        # ── Core bodies ─────────────────────────────────
        for col in range(n_cols):
            @core(core_tiles[0][col], stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    A = A_l2l1[col].acquire(ObjectFifoPort.Consume, 1)
                    C = C_l1l2[col].acquire(ObjectFifoPort.Produce, 1)
                    native_ternary(A, C, 0, m_per_core)
                    A_l2l1[col].release(ObjectFifoPort.Consume, 1)
                    C_l1l2[col].release(ObjectFifoPort.Produce, 1)

        # ── Runtime sequence ────────────────────────────
        @runtime_sequence(
            np.ndarray[(total_in_dwords,), np.dtype[dtype_in]],
            np.ndarray[(total_out_dwords,), np.dtype[dtype_in]],
        )
        def sequence(A_flat, C_flat):
            # Tile the flat input buffer across 8 columns.
            # A_flat = [col0_buf][col1_buf]...[col7_buf]
            # Each col_buf = [weights | scales | activations]
            A_taps = TensorTiler2D.group_tiler(
                (total_in_dwords, 1),
                (in_dwords, 1),
                (n_cols, 1),
            )
            # Output: each column writes its slice
            C_taps = TensorTiler2D.group_tiler(
                (total_out_dwords, 1),
                (out_dwords, 1),
                (n_cols, 1),
            )

            # Issue all input DMA tasks
            a_tasks = []
            for col in range(n_cols):
                a_task = shim_dma_single_bd_task(
                    A_l3l2[col], A_flat,
                    tap=A_taps[col],
                    issue_token=False,
                )
                dma_start_task(a_task)
                a_tasks.append(a_task)

            # Issue all output DMA tasks (with tokens for synchronization)
            c_tasks = []
            for col in range(n_cols):
                c_task = shim_dma_single_bd_task(
                    C_l2l3[col], C_flat,
                    tap=C_taps[col],
                    issue_token=True,
                )
                dma_start_task(c_task)
                c_tasks.append(c_task)

            # Wait for output, then free input
            if c_tasks:
                dma_await_task(*c_tasks)
            if a_tasks:
                dma_free_task(*a_tasks)


main()
