#!/usr/bin/env python3
#
# INT8 MLIR generator v2 — Correct approach
# - One A acquire per K-tile iteration (no 4x inner loop)
# - M-tiling handled by runtime sequence
# - A_l1 = memref<m x k> (full M tile)
# - Uses matmul_scalar_i8_i16 for correctness
# - Kernel DIM_M=32, DIM_K=64, DIM_N=128
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)  # M tile per core = 32
    parser.add_argument("-k", type=int, default=64)  # K tile = 64
    parser.add_argument("-n", type=int, default=128) # N tile = 128
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    n_aie_rows = 1
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = 512  # L2 K dimension (how much K per L2 tile)
    
    @device(AIEDevice.npu2)
    def device_body():
        # L2 types
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]

        # L1 types — flat, no 4x unrolling
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]   # m x k (32x64)
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]   # k x n (64x128)
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]  # m x n (32x128)

        # Kernel: vectorized with DIM_M=32, mac_8x8_8x8
        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 3)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # A fifos
        A_l3l2 = [None] * n_aie_rows
        A_l2l1 = [None] * n_aie_rows
        B_l3l2 = [None] * n_aie_cols
        B_l2l1 = [None] * n_aie_cols
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols

        for row in range(n_aie_rows):
            # Simple row-major: read (m, mtk) with row stride; no interleaving
            A_l3l2[row] = object_fifo(f"A_L3L2_{row}", shim_tiles[row], mem_tiles[row], 2, A_l2_ty, None,
                [[(m, mtk), (mtk, 1)]])
            # Producer walks L2 buffer: 32 M-rows at stride mtk=512, 64 K-cols contiguous
            # Consumer gets L1 buffer: 32 M-rows at stride k=64 (row-major in sub-view)
            A_l2l1[row] = object_fifo(f"A_L2L1_{row}", mem_tiles[row], core_tiles[row][0:n_aie_cols], 2, A_l1_ty,
                [(m, mtk), (k, 1)], [[(m, k), (k, 1)] for _ in range(n_aie_cols)])
            object_fifo_link(A_l3l2[row], A_l2l1[row])

        for col in range(n_aie_cols):
            B_l3l2[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            B_l2l1[col] = object_fifo(f"B_L2L1_{col}", mem_tiles[col], [core_tiles[j][col] for j in range(n_aie_rows)], 2, B_l1_ty)
            object_fifo_link(B_l3l2[col], B_l2l1[col])

        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2[row][col] = object_fifo(f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 1, C_l1_ty)
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link([C_l1l2[j][col] for j in range(n_aie_rows)], C_l2l3[col], [m * n * j for j in range(n_aie_rows)])

        # Core body: zero C, then K-loop with single A+B acquire per iteration
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M // m) * (N // n) // (n_aie_cols * n_aie_rows)):
                            C = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(C)
                            for _ in range_(K // k):
                                B = B_l2l1[col].acquire(ObjectFifoPort.Consume, 1)
                                A = A_l2l1[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul(A, B, C)  # C += A x B
                                A_l2l1[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1[col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # A: (M, K) → (m, mtk) tiles, repeat (1, K//mtk) in K
            # B: (1, N*K) → (1, n*K) tiles — same B tap is used for ALL K-iterations
            # The K-loop happens INSIDE the core, the runtime feeds the base tap
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (n_aie_rows * m, n), (1, 1))
            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile

            # Sequential per-group: start A+B+C, await C, free A+B.
            # This prevents A data overwrite in the shared depth-2 fifo
            # when M > m (multiple row groups).
            for group_idx in range(num_groups):
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                a_tasks = []
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(A_l3l2[row], A, tap=A_taps[a_base_idx + row], issue_token=False)
                    dma_start_task(a_task)
                    a_tasks.append(a_task)

                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                b_tasks = []
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(B_l3l2[col], B, tap=B_taps[b_base_idx + col], issue_token=False)
                    dma_start_task(b_task)
                    b_tasks.append(b_task)

                c_base_idx = group_idx * n_aie_cols
                c_tasks = []
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(C_l2l3[col], C, tap=C_taps[c_base_idx + col], issue_token=True)
                    dma_start_task(c_task)
                    c_tasks.append(c_task)

                # Wait for compute to finish (C drains) before starting next group
                if c_tasks: dma_await_task(*c_tasks)
                if a_tasks: dma_free_task(*a_tasks)
                if b_tasks: dma_free_task(*b_tasks)


main()
