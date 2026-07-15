#!/usr/bin/env python3
"""
Native ternary 32-core (4×8) MLIR generator — per-column DMA, same-column fan-out.

FIXED: NPU2 AIE interconnect does NOT support cross-column mem→core broadcast.
Each column's mem tile can only reach cores in its own column.

Architecture (per column, col 0..7):
  shim[col] → mem[col] → fan-out to core[0..3][col]  (same column, 4 rows)

Per-column flat buffer:
  [core0_weights|scales] [core1_w|s] [core2_w|s] [core3_w|s] [activations(shared)]

Each core gets the full column buffer, picks its slice via row_start/num_rows.
Kernel compiled with DIM_M = 4 * M_PER_CORE (total rows per column).

Usage:
    python3 n1_core_native_ternary_32core.py -M 512 -K 512 > ternary_32core.mlir
"""

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128, help="Total output rows (multiple of 32)")
    parser.add_argument("-K", type=int, default=512, help="Packed K bytes (K*4 = ternary values; block scale layout)")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()
    assert args.M % 32 == 0, f"M={args.M} must be a multiple of 32"

    with mlir_mod_ctx() as ctx:
        my_native_ternary_32core(args.M, args.K, args.dump)
        print(ctx.module)


def my_native_ternary_32core(M, K_packed, dump=False):
    n_cols = 8
    n_rows = 4
    m_per_core = M // (n_rows * n_cols)  # M/32 rows per core
    m_per_col = m_per_core * n_rows       # M/8 rows per column buffer
    k_ternary = K_packed * 4
    n_blocks = k_ternary // 256              # blocks of 256 ternary values

    # Per-column buffer sizes (contains all 4 cores' data)
    col_weight_bytes = m_per_col * K_packed   # 4*M_PR*512
    col_scale_bytes = m_per_col * n_blocks * 2  # per-block scale (n_blocks scales per row)
    col_act_bytes = k_ternary * 2
    col_in_bytes = col_weight_bytes + col_scale_bytes + col_act_bytes
    col_in_dwords = (col_in_bytes + 3) // 4

    out_elems = m_per_core                       # M/32 bf16 per core
    col_out_elems = out_elems * n_rows            # 4*M/32 per column
    col_out_dwords = (col_out_elems * 4 + 3) // 4  # using f32 type

    total_in_dwords = col_in_dwords * n_cols
    total_out_dwords = col_out_dwords * n_cols

    if dump:
        import sys
        print(f"// M={M} m_per_core={m_per_core} m_per_col={m_per_col}", file=sys.stderr)
        print(f"// col_in_dwords={col_in_dwords} col_out_dwords={col_out_dwords}", file=sys.stderr)
        print(f"// total_in={total_in_dwords}dw total_out={total_out_dwords}dw", file=sys.stderr)

    kernel_entry = "mm_ternary_32x64x128"
    kernel_o = f"{kernel_entry}.o"
    dtype_in = np.int32
    dtype_out = np.float32  # maps to bf16 in AIE dialect

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(col_in_dwords,), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(col_in_dwords,), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(out_elems,), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(col_out_elems,), np.dtype[dtype_out]]

        native_ternary = external_func(
            kernel_entry,
            inputs=[A_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=kernel_o,
        )

        tiles = [
            [tile(col, row) for col in range(n_cols)]
            for row in range(6)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]  # rows 2,3,4,5

        # ── Per-column input: shim→mem→4 cores (same column only) ──
        A_l3l2 = [None] * n_cols
        A_l2l1 = [None] * n_cols
        for col in range(n_cols):
            A_l3l2[col] = object_fifo(
                f"A_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, A_l2_ty,
            )
            # Fan-out to all 4 rows in THIS column (same-column = works)
            A_l2l1[col] = object_fifo(
                f"A_L2L1_{col}",
                mem_tiles[col],
                [core_tiles[r][col] for r in range(n_rows)],
                2, A_l1_ty,
            )
            object_fifo_link(A_l3l2[col], A_l2l1[col])

        # ── Output: per-core → gather per column ───────────
        C_l1l2 = [[None] * n_cols for _ in range(n_rows)]
        C_l2l3 = [None] * n_cols
        for col in range(n_cols):
            for row in range(n_rows):
                C_l1l2[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col], mem_tiles[col],
                    1, C_l1_ty,
                )
            C_l2l3[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col], shim_tiles[col],
                2, C_l2_ty,
            )
            object_fifo_link(
                [C_l1l2[r][col] for r in range(n_rows)],
                C_l2l3[col],
                [out_elems * r for r in range(n_rows)],
            )

        # ── Core bodies ───────────────────────────────────
        for col in range(n_cols):
            for row in range(n_rows):
                core_row_start = row * m_per_core
                core_num_rows = m_per_core

                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        A = A_l2l1[col].acquire(ObjectFifoPort.Consume, 1)
                        C = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                        native_ternary(A, C, core_row_start, core_num_rows)
                        A_l2l1[col].release(ObjectFifoPort.Consume, 1)
                        C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        # ── Runtime sequence ──────────────────────────────
        @runtime_sequence(
            np.ndarray[(total_in_dwords,), np.dtype[dtype_in]],
            np.ndarray[(total_out_dwords,), np.dtype[dtype_in]],
        )
        def sequence(A_flat, C_flat):
            A_taps = TensorTiler2D.group_tiler(
                (total_in_dwords, 1), (col_in_dwords, 1), (1, 1),
            )
            C_taps = TensorTiler2D.group_tiler(
                (total_out_dwords, 1), (col_out_dwords, 1), (1, 1),
            )

            a_tasks = []
            for col in range(n_cols):
                a_task = shim_dma_single_bd_task(
                    A_l3l2[col], A_flat, tap=A_taps[col], issue_token=False,
                )
                dma_start_task(a_task)
                a_tasks.append(a_task)

            c_tasks = []
            for col in range(n_cols):
                c_task = shim_dma_single_bd_task(
                    C_l2l3[col], C_flat, tap=C_taps[col], issue_token=True,
                )
                dma_start_task(c_task)
                c_tasks.append(c_task)

            if c_tasks:
                dma_await_task(*c_tasks)
            if a_tasks:
                dma_free_task(*a_tasks)


main()
