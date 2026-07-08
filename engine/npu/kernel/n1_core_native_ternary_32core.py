#!/usr/bin/env python3
"""
Native ternary 32-core (4×8) MLIR generator — row+column tiling for NPU xclbin.

Uses mm_ternary_32x64x128.o with the row_start/num_rows kernel API to tile
across 4 rows × 8 columns = 32 AIE cores.

Architecture:
  4 rows × 8 columns = 32 cores
  Each row broadcasts one flat buffer to all 8 cores in that row.
  Each core picks its slice via row_start/num_rows kernel params.

Dataflow per row:
  shim[row_tile] → mem[row_tile] → broadcast to 8 core[row][0..7]
  Each core: row_start = row*(M/4) + col*(M/32), num_rows = M/32

  C-path (output): 32 per-core outputs gathered per-column via mem→shim.

Kernel buffer layout (broadcast per row, all M/4 rows for this row):
  [M/4 * K_packed bytes weights] [M/4 * 2 bytes scales] [K*4 * 2 bytes acts]

Usage:
    python3 n1_core_native_ternary_32core.py -M 128 -K 64 > ternary_32core.mlir
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
        description="Native ternary 32-core NPU xclbin generator"
    )
    parser.add_argument("-M", type=int, default=128,
                        help="Total output rows (must be multiple of 32)")
    parser.add_argument("-K", type=int, default=64,
                        help="Packed K bytes per tile (K*4 = ternary values)")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    assert args.M % 32 == 0, f"M={args.M} must be a multiple of 32"

    with mlir_mod_ctx() as ctx:
        if args.dump:
            my_native_ternary_32core(args.M, args.K, True)
        else:
            my_native_ternary_32core(args.M, args.K, False)
        print(ctx.module)


def my_native_ternary_32core(M, K_packed, dump=False):
    n_cols = 8
    n_rows = 4
    m_per_core = M // (n_rows * n_cols)  # M/32
    m_per_row = M // n_rows               # M/4 rows per row broadcast
    k_ternary = K_packed * 4

    # Buffer sizes: sized for m_per_row rows (all cores in a row share)
    weight_bytes = m_per_row * K_packed
    scale_bytes = m_per_row * 2
    act_bytes = k_ternary * 2
    in_bytes = weight_bytes + scale_bytes + act_bytes
    in_dwords = (in_bytes + 3) // 4

    out_elems = m_per_core
    # Per-column output: n_rows of per-core outputs, each out_elems f32
    col_out_elems = out_elems * n_rows
    out_dwords = (out_elems * 4 + 3) // 4   # f32 = 4 bytes each
    col_out_dwords = (col_out_elems * 4 + 3) // 4

    total_in_dwords = in_dwords * n_rows
    total_out_elems = M
    total_out_dwords = col_out_dwords * n_cols

    if dump:
        import sys
        print(f"// M={M} m_per_core={m_per_core} m_per_row={m_per_row}",
              file=sys.stderr)
        print(f"// in_dwords={in_dwords} out_elems={out_elems}",
              file=sys.stderr)

    kernel_entry = "mm_ternary_32x64x128"
    kernel_o = f"{kernel_entry}.o"

    dtype_in = np.int32
    dtype_out = np.float32  # maps to bf16 in AIE dialect

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(out_elems,), np.dtype[dtype_out]]
        # Column output buffer: gathers n_rows of per-core outputs
        C_l2_ty = np.ndarray[(out_elems * n_rows,), np.dtype[dtype_out]]

        native_ternary = external_func(
            kernel_entry,
            inputs=[A_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=kernel_o,
        )

        tiles = [
            [tile(col, row) for col in range(n_cols)]
            for row in range(6)  # 0=shim, 1=mem, 2-5=core
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]  # rows 2,3,4,5 = core rows 0,1,2,3

        # ── Input: per-row broadcast (4 rows, 8 cols each) ──
        # Row r uses shim col [r*2] to avoid tile conflicts
        A_l3l2 = [None] * n_rows
        A_l2l1 = [None] * n_rows
        for row in range(n_rows):
            shim_col = row * 2  # 0, 2, 4, 6
            A_l3l2[row] = object_fifo(
                f"A_L3L2_{row}", shim_tiles[shim_col], mem_tiles[shim_col],
                2, A_l2_ty,
            )
            # Broadcast to all 8 cores in this row
            A_l2l1[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[shim_col], core_tiles[row][0:n_cols],
                2, A_l1_ty,
            )
            object_fifo_link(A_l3l2[row], A_l2l1[row])

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
            # Gather 4 rows into one column output stream
            # Each row produces out_elems f32; n_rows combined
            object_fifo_link(
                [C_l1l2[j][col] for j in range(n_rows)],
                C_l2l3[col],
                [out_elems * j for j in range(n_rows)],
            )

        # ── Core bodies ───────────────────────────────────
        for row in range(n_rows):
            for col in range(n_cols):
                # Each core processes its own slice of the row broadcast
                core_row_start = row * m_per_row + col * m_per_core
                core_num_rows = m_per_core

                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        A = A_l2l1[row].acquire(ObjectFifoPort.Consume, 1)
                        C = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                        native_ternary(A, C, core_row_start, core_num_rows)
                        A_l2l1[row].release(ObjectFifoPort.Consume, 1)
                        C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        # ── Runtime sequence ──────────────────────────────
        @runtime_sequence(
            np.ndarray[(total_in_dwords,), np.dtype[dtype_in]],
            np.ndarray[(total_out_dwords,), np.dtype[dtype_in]],
        )
        def sequence(A_flat, C_flat):
            # Input tiling: 4 rows, each gets in_dwords
            # Use group=(1,1) for individual tiles, 4 total
            A_taps = TensorTiler2D.group_tiler(
                (total_in_dwords, 1),
                (in_dwords, 1),
                (1, 1),
            )
            # Output tiling: 8 columns, each collects n_rows worth
            C_taps = TensorTiler2D.group_tiler(
                (total_out_dwords, 1),
                (col_out_dwords, 1),
                (1, 1),
            )

            # Input DMA: one per row (broadcast), 4 rows
            a_tasks = []
            for row in range(n_rows):
                a_task = shim_dma_single_bd_task(
                    A_l3l2[row], A_flat,
                    tap=A_taps[row],
                    issue_token=False,
                )
                dma_start_task(a_task)
                a_tasks.append(a_task)

            # Output DMA: one per column (gathers 4 rows), 8 columns
            c_tasks = []
            for col in range(n_cols):
                c_task = shim_dma_single_bd_task(
                    C_l2l3[col], C_flat,
                    tap=C_taps[col],
                    issue_token=True,
                )
                dma_start_task(c_task)
                c_tasks.append(c_task)

            # Await output, then free input
            if c_tasks:
                dma_await_task(*c_tasks)
            if a_tasks:
                dma_free_task(*a_tasks)


main()
