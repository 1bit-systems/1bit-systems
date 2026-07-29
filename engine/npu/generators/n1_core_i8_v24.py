#!/usr/bin/env python3
#
# INT8 MLIR generator v24 — BD descriptor pipelining for K-iteration batching
#
# Fixes Issue #1075: NPU decode at 16.3s/tok because each K-iteration
# issues separate DMA start/await cycles (flat BD descriptors).
#
# v24 batches K-iterations in groups of BATCH_SIZE (default 6):
#   Instead of: for each K-iteration { DMA A; DMA B; wait; compute }
#   Do: for batch of 6 K-iterations { DMA all 6 A BDs; DMA all 6 B BDs;
#                                       wait once; compute all 6 }
#
# Key changes from v2:
#   - BATCH_SIZE = 6 (configurable via --batch-size)
#   - mtk  = BATCH_SIZE * k = 384 (L2 tile holds exactly one batch)
#   - A_l2l1 depth = BATCH_SIZE + 1 = 7  (holds batch + 1 in-flight)
#   - B_l2l1 depth = BATCH_SIZE + 1 = 7
#   - Core loop: acquire batch → compute batch → release batch
#   - Within 16-BD-per-tile limit: 7 A + 7 B + 1 C = 15 BDs ✓
#   - Memory tile 512KB: 7×2048 + 7×8192 + 1×8192 = 78KB ✓
#
# Toolchain: MLIR-AIE aiecc (Peano compiler) v0.3.x
#   python3 n1_core_i8_v24.py -M 128 -K 1024 -N 4096 > mm_32x64x128.mlir
#   aiecc mm_32x64x128.mlir ... (see build_xclbins.sh)
#
# Kernel: mm_32x64x128.o (same matmul_scalar_i8_i16 as v2)
#   DIM_M=32, DIM_K=64, DIM_N=128

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

# ── Configuration ─────────────────────────────────────────────────────

BATCH_SIZE = 6        # K-iterations per DMA batch (default, overridable)
MTK        = 384      # BATCH_SIZE * k_tile = 6 * 64 = 384

def main():
    parser = argparse.ArgumentParser(
        description="n1_core_i8_v24 — BD pipelined INT8 MLIR generator")
    parser.add_argument("-M", type=int, default=128,
                        help="Output rows (M dimension)")
    parser.add_argument("-K", type=int, default=1024,
                        help="Inner reduction dimension")
    parser.add_argument("-N", type=int, default=4096,
                        help="Output columns (N dimension)")
    parser.add_argument("-m", type=int, default=32,
                        help="M tile per core (default: 32)")
    parser.add_argument("-k", type=int, default=64,
                        help="K tile (default: 64)")
    parser.add_argument("-n", type=int, default=128,
                        help="N tile (default: 128)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"K-iterations per DMA batch (default: {BATCH_SIZE})")
    parser.add_argument("--mtk", type=int, default=MTK,
                        help=f"L2 K dimension per tile (default: {MTK})")
    args = parser.parse_args()

    batch_size = args.batch_size
    mtk = args.mtk

    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n,
                  batch_size, mtk)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, batch_size, mtk):
    """INT8 GEMM with BD descriptor pipelining.

    Batching: instead of issuing one DMA start/wait per K-iteration,
    the core acquires BATCH_SIZE A and B buffers at once, computes them
    all, then releases them all. This amortizes DMA setup overhead and
    keeps the NPU shim DMA engine fully utilized.

    BD budget per memory tile (16 max):
      7 A_l2l1 BDs  (depth = batch_size + 1)
      7 B_l2l1 BDs  (depth = batch_size + 1)
      1 C_l1l2  BD  (depth = 1)
      Total: 15 BDs  (within 16-BD limit)  ✓

    Memory budget per memory tile (512KB):
      7 A buffers: 7 × 2048 = 14KB
      7 B buffers: 7 × 8192 = 56KB
      1 C buffer:  1 × 8192  = 8KB
      Total: 78KB  (well within 512KB)  ✓
    """
    n_aie_cols = 8
    n_aie_rows = 1
    dtype_in = np.int8
    dtype_out = np.int16

    # ── Tile type definitions ─────────────────────────────────────────

    @device(AIEDevice.npu2)
    def device_body():
        # L2 types — each L2 buffer holds BATCH_SIZE K-iterations
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]

        # L1 types (single K-iteration)
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]   # m×k  (32×64)
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]   # k×n  (64×128)
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # m×n  (32×128)

        # Kernel (same as v2 — vectorized matmul_scalar_i8_i16)
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i16", inputs=[C_l1_ty],
                             link_with=kernel_o)
        matmul = external_func("matmul_i8_i16",
                               inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
                               link_with=kernel_o)

        # ── Tile layout ───────────────────────────────────────────────────
        # NPU2: 6 rows × 8 cols
        #   Row 0: Shim tiles (DDR DMA)
        #   Row 1: Memory tiles (L2 scratchpad)
        #   Rows 2-5: Compute tiles (AIE cores)
        tiles = [[tile(col, row)
                  for col in range(0, n_aie_cols)]
                 for row in range(0, 3)]
        shim_tiles = tiles[0]
        mem_tiles  = tiles[1]
        core_tiles = tiles[2:]   # row 2 onward (1 row for v24)

        # ── FIFO declarations ─────────────────────────────────────────────

        A_l3l2 = [None] * n_aie_rows
        A_l2l1 = [None] * n_aie_rows
        B_l3l2 = [None] * n_aie_cols
        B_l2l1 = [None] * n_aie_cols
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols

        # AIE row for compute (single row, row 2 = CT00-CT07)
        row = 0

        # ── A path: DDR→L2→L1 with BD descriptor pipelining ──────────────
        #
        # Producer (L3→L2): walks the DDR buffer in (m, mtk) tiles
        # Consumer (L2→L1): fans out to each col's compute tile
        #
        # Depth = batch_size + 1:
        #   batch_size buffers for the current batch
        #   1 in-flight refill from L2→L1 while compute runs
        #
        A_l3l2[row] = object_fifo(
            f"A_L3L2_{row}",
            shim_tiles[row], mem_tiles[row],
            2,                                                  # depth (L3→L2)
            A_l2_ty, None,
            [[(m, mtk), (mtk, 1)]]                              # row-major
        )

        A_l2l1[row] = object_fifo(
            f"A_L2L1_{row}",
            mem_tiles[row],
            core_tiles[row][0:n_aie_cols],
            batch_size + 1,                                     # depth (L2→L1)
            A_l1_ty,
            [(m, mtk), (k, 1)],                                 # producer stride
            [[(m, k), (k, 1)] for _ in range(n_aie_cols)]       # consumer stride per col
        )
        object_fifo_link(A_l3l2[row], A_l2l1[row])

        # ── B path: DDR→L2→L1 (broadcast to all rows) ────────────────────
        for col in range(n_aie_cols):
            B_l3l2[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col], mem_tiles[col],
                2,                                              # depth (L3→L2)
                B_l2_ty
            )
            B_l2l1[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [core_tiles[j][col] for j in range(n_aie_rows)],
                batch_size + 1,                                 # depth (L2→L1)
                B_l1_ty
            )
            object_fifo_link(B_l3l2[col], B_l2l1[col])

        # ── C path: L1→L2→DDR (output) ──────────────────────────────────
        for col in range(n_aie_cols):
            C_l1l2[row][col] = object_fifo(
                f"C_L1L2_{col}_{row}",
                core_tiles[row][col], mem_tiles[col],
                1,                                              # depth (L1→L2)
                C_l1_ty
            )
            C_l2l3[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col], shim_tiles[col],
                2,                                              # depth (L2→DDR)
                C_l2_ty
            )
            # Reassemble row contributions for C output
            object_fifo_link(
                [C_l1l2[j][col] for j in range(n_aie_rows)],
                C_l2l3[col],
                [m * n * j for j in range(n_aie_rows)]
            )

        # ── Core body: batched K-iteration processing ────────────────────
        #
        # v24 change:
        #   Instead of: for each K-iteration { acquire A; acquire B;
        #                                       compute; release A; release B }
        #   Do: for batch of batch_size K-iterations {
        #         acquire all batch_size A buffers
        #         acquire all batch_size B buffers
        #         compute all batch_size
        #         release all A and B buffers
        #       }
        #
        # This lets the object_fifo hardware pipeline all L2→L1 DMAs
        # in one shot (using batch_size BD descriptors per FIFO), then
        # the core churns through the results with zero DMA wait.
        #
        num_k_tiles = K // k                     # total K-iterations
        num_batches = num_k_tiles // batch_size  # full batches
        remainder   = num_k_tiles % batch_size   # leftover K-iterations

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    # Outer loop: infinite token sequence
                    for _ in range_(0xFFFFFFFF):
                        # M-tile groups (M//m × N//n total groups)
                        for _ in range(
                            (M // m) * (N // n) // (n_aie_cols * n_aie_rows)
                        ):
                            # Acquire C output buffer
                            C = C_l1l2[row][col].acquire(
                                ObjectFifoPort.Produce, 1)
                            zero(C)

                            # ── Batch: full batches of batch_size ────────
                            for batch in range_(num_batches):
                                # Acquire ALL batch_size A and B buffers
                                # This triggers all L2→L1 DMA BDs at once
                                A_bufs = []
                                B_bufs = []
                                for i in range_(batch_size):
                                    B_buf = B_l2l1[col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    A_buf = A_l2l1[row].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    A_bufs.append(A_buf)
                                    B_bufs.append(B_buf)

                                # Compute all batch_size iterations
                                # (A and B data has been pipelined in)
                                for i in range_(batch_size):
                                    matmul(A_bufs[i], B_bufs[i], C)

                                # Release all buffers
                                for i in range_(batch_size):
                                    A_l2l1[row].release(
                                        ObjectFifoPort.Consume, 1)
                                    B_l2l1[col].release(
                                        ObjectFifoPort.Consume, 1)

                            # ── Remainder: process leftover K-iterations ──
                            if remainder > 0:
                                A_bufs_r = []
                                B_bufs_r = []
                                for i in range_(remainder):
                                    B_buf = B_l2l1[col].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    A_buf = A_l2l1[row].acquire(
                                        ObjectFifoPort.Consume, 1)
                                    A_bufs_r.append(A_buf)
                                    B_bufs_r.append(B_buf)
                                for i in range_(remainder):
                                    matmul(A_bufs_r[i], B_bufs_r[i], C)
                                for i in range_(remainder):
                                    A_l2l1[row].release(
                                        ObjectFifoPort.Consume, 1)
                                    B_l2l1[col].release(
                                        ObjectFifoPort.Consume, 1)

                            # Release C output
                            C_l1l2[row][col].release(
                                ObjectFifoPort.Produce, 1)

        # ── Runtime sequence: host-side DMA orchestration ─────────────────
        #
        # Controls L3→DDR DMA. Each M-tile group:
        #   1. Start all A BDs (1 per row × n_aie_rows rows)
        #   2. Start all B BDs (1 per col × n_aie_cols cols)
        #   3. Start all C BDs (1 per col)
        #   4. Await C (compute complete), free A+B buffers
        #
        # The K-iteration batching happens INSIDE the core, not here.
        #
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # Tilers
            # A: (M, K) → (m, mtk) tiles, (1, K // mtk) repeats in K
            # B: (1, N*K) → (1, n*K) tiles — same B tap for all K-iterations
            # C: (M, N) → (n_aie_rows * m, n) tiles
            A_taps = TensorTiler2D.group_tiler(
                (M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler(
                (1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler(
                (M, N), (n_aie_rows * m, n), (1, 1))

            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile

            # Sequential per-group: start A+B+C, await C, free A+B.
            for group_idx in range(num_groups):
                # ── A: start DMA for all rows ──
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                a_tasks = []
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(
                        A_l3l2[row], A,
                        tap=A_taps[a_base_idx + row],
                        issue_token=False)
                    dma_start_task(a_task)
                    a_tasks.append(a_task)

                # ── B: start DMA for all columns ──
                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                b_tasks = []
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(
                        B_l3l2[col], B,
                        tap=B_taps[b_base_idx + col],
                        issue_token=False)
                    dma_start_task(b_task)
                    b_tasks.append(b_task)

                # ── C: start DMA for output ──
                c_base_idx = group_idx * n_aie_cols
                c_tasks = []
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(
                        C_l2l3[col], C,
                        tap=C_taps[c_base_idx + col],
                        issue_token=True)
                    dma_start_task(c_task)
                    c_tasks.append(c_task)

                # Wait for compute to finish (C drains), then free A/B
                if c_tasks:
                    dma_await_task(*c_tasks)
                if a_tasks:
                    dma_free_task(*a_tasks)
                if b_tasks:
                    dma_free_task(*b_tasks)


if __name__ == "__main__":
    main()
