import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=128)
    p.add_argument("-K", type=int, default=1024)
    p.add_argument("-N", type=int, default=4096)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=128)
    args = p.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8; n_aie_rows = 4; mtk = 512
    r = 8; s = 8; t = 8
    assert m % (2 * r) == 0 and k % s == 0 and n % (2 * t) == 0
    dtype_in = np.int8; dtype_out = np.int32

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        zero = external_func("zero_i32", inputs=[C_l1_ty], link_with="mm_32x64x128_i32_vec.o")
        matmul = external_func("matmul_i8_i32", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with="mm_32x64x128_i32_vec.o")

        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
        shim_tiles = tiles[0]; mem_tiles = tiles[1]; core_tiles = tiles[2:]

        A_l3l2 = [None] * n_aie_rows; A_l2l1 = [None] * n_aie_rows
        B_l3l2 = [None] * n_aie_cols; B_l2l1 = [None] * n_aie_cols
        C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3 = [None] * n_aie_cols

        # V10: A FIFO hierarchy with 1D dimsFromStream [(512,1)] to avoid BD stride constraint
        for row in range(n_aie_rows):
            sc = 2 * row if 2 * row < n_aie_cols else row % n_aie_cols
            A_l3l2[row] = object_fifo(f"A_L3L2_{row}", shim_tiles[sc], mem_tiles[sc], 2, A_l2_ty, None,
                                      [[(m, mtk), (mtk, 1)]])
            # dimsToStream: micro-tile reorder from (32,512)
            # dimsFromStream: 1D contiguous write to L1 (avoids 32b BD stride constraint)
            a_to_stream = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
            a_from_stream = [[(m * k // 4, 1)] for _ in range(n_aie_cols)]
            A_l2l1[row] = object_fifo(f"A_L2L1_{row}", mem_tiles[sc],
                core_tiles[row][0:n_aie_cols], 2, A_l1_ty,
                a_to_stream, a_from_stream)
            object_fifo_link(A_l3l2[row], A_l2l1[row])

        # B: same as scalar (row-major, no reorder needed)
        for col in range(n_aie_cols):
            B_l3l2[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            B_l2l1[col] = object_fifo(f"B_L2L1_{col}", mem_tiles[col],
                [core_tiles[j][col] for j in range(n_aie_rows)], 2, B_l1_ty,
                [(k, n), (n, 1)])
            object_fifo_link(B_l3l2[col], B_l2l1[col])

        # C: same as scalar (row-major output, C_l2l3 reassembles to DDR)
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2[row][col] = object_fifo(f"C_L1L2_{col}_{row}",
                    core_tiles[row][col], mem_tiles[col], 1, C_l1_ty)
            C_l2l3[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            object_fifo_link([C_l1l2[j][col] for j in range(n_aie_rows)], C_l2l3[col],
                             [m * n * j for j in range(n_aie_rows)])

        # Core body
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M // m) * (N // n) // (n_aie_cols * n_aie_rows)):
                            elem_out = C_l1l2[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(K // k):
                                elem_in_b = B_l2l1[col].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_a = A_l2l1[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul(elem_in_a, elem_in_b, elem_out)
                                A_l2l1[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1[col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2[row][col].release(ObjectFifoPort.Produce, 1)

        # Runtime sequence (same as V6)
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (n_aie_rows * m, n), (1, 1))
            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            tb_max_n_rows = 4
            input_task_groups = [[] for _ in range(tb_max_n_rows)]
            output_task_groups = [[] for _ in range(tb_max_n_rows)]

            for group_idx in range(num_groups):
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(A_l3l2[row], A, tap=A_taps[a_base_idx + row], issue_token=False)
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
                    if output_task_groups[2]: dma_await_task(*output_task_groups[2]); output_task_groups[2] = []
                    if input_task_groups[2]: dma_free_task(*input_task_groups[2]); input_task_groups[2] = []
                    if output_task_groups[3]: dma_await_task(*output_task_groups[3]); output_task_groups[3] = []
                    if input_task_groups[3]: dma_free_task(*input_task_groups[3]); input_task_groups[3] = []
                if group_idx % tb_max_n_rows == 3:
                    if output_task_groups[0]: dma_await_task(*output_task_groups[0]); output_task_groups[0] = []
                    if input_task_groups[0]: dma_free_task(*input_task_groups[0]); input_task_groups[0] = []
                    if output_task_groups[1]: dma_await_task(*output_task_groups[1]); output_task_groups[1] = []
                    if input_task_groups[1]: dma_free_task(*input_task_groups[1]); input_task_groups[1] = []
            if output_task_groups[2]: dma_await_task(*output_task_groups[2]); dma_free_task(*input_task_groups[2])
            if output_task_groups[3]: dma_await_task(*output_task_groups[3]); dma_free_task(*input_task_groups[3])

main()
