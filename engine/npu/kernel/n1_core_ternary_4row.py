#!/usr/bin/env python3
"""
Ternary 4-row (32-core) MLIR generator — for large model xclbins.
Adapted from n1_core_i8_4row.py, using i8→i32 kernel.

4 rows × 8 cols = 32 AIE cores. Each core: m=32, k=64, n=128.
Total per invocation: M=128 (4×32), K, N as specified.
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
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    n_aie_rows = 4
    m_total = n_aie_rows * m  # 4 × 32 = 128 total M rows
    r = 8; s = 8; t = 8
    mtk = 512
    dtype_in = np.int8
    dtype_out = np.int32

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m_total, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(m_total, n), np.dtype[dtype_out]]

        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(6)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # A fifos: one per row, broadcast to all 8 cols
        A_l3l2 = [None] * n_aie_rows
        A_l2l1 = [None] * n_aie_rows
        for row in range(n_aie_rows):
            shim_col = 2 * row if 2 * row < n_aie_cols else row % n_aie_cols
            A_l3l2[row] = object_fifo(f"A_L3L2_{row}", shim_tiles[shim_col], mem_tiles[shim_col], 2, A_l2_ty, None,
                [[(m_total, mtk), (mtk, 1)]])
            A_l2l1[row] = object_fifo(f"A_L2L1_{row}",
                mem_tiles[shim_col], core_tiles[row][0:n_aie_cols], 2, A_l1_ty,
                [(mtk // k * 4, m_total * k // 4), (k // s, s), (m, k), (s, 1)],
                [[(k // s, r * s), (m // r, r * k), (r * s, 1)] for _ in range(n_aie_cols)])
            object_fifo_link(A_l3l2[row], A_l2l1[row])

        # B fifos: one per column, broadcast to all 4 rows
        B_l3l2 = [None] * n_aie_cols
        B_l2l1 = [None] * n_aie_cols
        for col in range(n_aie_cols):
            B_l3l2[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            B_l2l1[col] = object_fifo(f"B_L2L1_{col}", mem_tiles[col],
                [core_tiles[j][col] for j in range(n_aie_rows)], 2, B_l1_ty)
            object_fifo_link(B_l3l2[col], B_l2l1[col])

        # C fifos: per-core → gather per column
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2[row][col] = object_fifo(f"C_L1L2_{col}_{row}",
                    core_tiles[row][col], mem_tiles[col], 1, C_l1_ty)
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col],
                2, C_l2_ty)
            object_fifo_link([C_l1l2[j][col] for j in range(n_aie_rows)], C_l2l3[col], [m * n * j for j in range(n_aie_rows)])

        # Core body: 4 inner A-acquires per B-acquire (one per M-sub-tile)
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M // m_total) * (N // n) // (n_aie_cols * n_aie_rows)):
                            C = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(C)
                            for _ in range_(K // k):
                                B = B_l2l1[col].acquire(ObjectFifoPort.Consume, 1)
                                for i in range(4):
                                    A = A_l2l1[row].acquire(ObjectFifoPort.Consume, 1)
                                    matmul(A, B, C)
                                    A_l2l1[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1[col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m_total, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m_total, n), (1, 1))
            num_row_tile = max(M // m_total, 1)
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            tb_max_n_rows = 4
            input_task_groups = [[] for _ in range(tb_max_n_rows)]
            output_task_groups = [[] for _ in range(tb_max_n_rows)]

            for group_idx in range(num_groups):
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(A_l3l2[row], A, tap=A_taps[0], issue_token=False)
                    dma_start_task(a_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(a_task)

                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(B_l3l2[col], B, tap=B_taps[b_base_idx + col], issue_token=False)
                    dma_start_task(b_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(b_task)

                c_base_idx = group_idx * n_aie_cols
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(C_l2l3[col], C, tap=C_taps[c_base_idx + col], issue_token=True)
                    dma_start_task(c_task)
                    output_task_groups[group_idx % tb_max_n_rows].append(c_task)

                if (group_idx % tb_max_n_rows == 1) and (group_idx != 1):
                    for g in [2,3]:
                        if output_task_groups[g]: dma_await_task(*output_task_groups[g]); output_task_groups[g]=[]
                        if input_task_groups[g]: dma_free_task(*input_task_groups[g]); input_task_groups[g]=[]
                if group_idx % tb_max_n_rows == 3:
                    for g in [0,1]:
                        if output_task_groups[g]: dma_await_task(*output_task_groups[g]); output_task_groups[g]=[]
                        if input_task_groups[g]: dma_free_task(*input_task_groups[g]); input_task_groups[g]=[]

            for g in [0,1,2,3]:
                if output_task_groups[g]: dma_await_task(*output_task_groups[g])
                if input_task_groups[g]: dma_free_task(*input_task_groups[g])

main()
