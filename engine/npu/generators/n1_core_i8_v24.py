#!/usr/bin/env python3
#
# INT8 MLIR generator v24 — repeat-BD pipelining for throughput.
#
# v23 was correct but slow: each of n_k K-iterations issued + awaited its
# BD descriptors separately, serializing DMA and compute. The shim DMA has
# 16 BD slots per channel; v23 used 9 (1 A broadcast + 8 B column) per
# iteration, leaving 7 unused. v24 fills all 16 slots by batching K-iterations
# in groups of 16 — issue all 16 iterations' A and B BDs at once, then await
# them all. This lets the DMA controller pipeline transfers across iterations.
#
# The hardware repeat dimension (sizes[0]) would be cleaner but requires
# stride computation that varies per buffer layout — the batch approach is
# simpler, equally effective, and doesn't need stride math.
#
# Design principles (carried forward from v23):
#   - C DMA is issued for ALL n_aie_cols columns every group
#   - Fully sequential per group: batch K-iterations' worth of A/B, await,
#     then C
#   - Kernel ABI: matmul_i8_i32 / zero_i32 (int32 accumulator, vectorized)
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

# Max BD slots per shim DMA channel on AIE2P (Strix Halo).
# The compiler reports 16 as the hardware limit. With objectFifo depth=2
# consuming 2 internal BDs, 14 are available for our BD tasks.
MAX_BD_SLOTS = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    parser.add_argument("-c", "--cols", type=int, default=8, help="n_aie_cols (must divide N//n)")
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n, args.cols)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n, n_aie_cols=8):
    dtype_in = np.int8
    dtype_out = np.int32

    assert M % m == 0 and K % k == 0 and N % n == 0
    assert (N // n) % n_aie_cols == 0, "N//n must be a multiple of n_aie_cols"

    n_k = K // k                       # K-iterations per row-group
    n_col_tile = N // n                # column tiles
    n_col_group = n_col_tile // n_aie_cols
    n_row_tile = M // m
    n_groups = n_row_tile * n_col_group

    @device(AIEDevice.npu2)
    def device_body():
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_32x64x128.o"
        zero = external_func("zero_i32", inputs=[C_ty], link_with=kernel_o)
        matmul = external_func("matmul_i8_i32", inputs=[A_ty, B_ty, C_ty], link_with=kernel_o)

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(3)]
        shim_tiles, mem_tiles, core_tiles = tiles[0], tiles[1], tiles[2]

        # A: broadcast fifo, shim[0] -> mem[0] -> all cores
        A_s = object_fifo("A_S", shim_tiles[0], mem_tiles[0], 2, A_ty)
        A_c = object_fifo("A_C", mem_tiles[0], core_tiles[0:n_aie_cols], 2, A_ty)
        object_fifo_link(A_s, A_c)

        # B, C: independent per-column fifos
        B_s = [None] * n_aie_cols
        B_c = [None] * n_aie_cols
        C_c = [None] * n_aie_cols
        C_s = [None] * n_aie_cols
        for c in range(n_aie_cols):
            B_s[c] = object_fifo(f"B_S{c}", shim_tiles[c], mem_tiles[c], 2, B_ty)
            B_c[c] = object_fifo(f"B_C{c}", mem_tiles[c], core_tiles[c], 2, B_ty)
            object_fifo_link(B_s[c], B_c[c])

            C_c[c] = object_fifo(f"C_C{c}", core_tiles[c], mem_tiles[c], 2, C_ty)
            C_s[c] = object_fifo(f"C_S{c}", mem_tiles[c], shim_tiles[c], 2, C_ty)
            object_fifo_link(C_c[c], C_s[c])

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(n_groups):
                        Cbuf = C_c[c].acquire(ObjectFifoPort.Produce, 1)
                        zero(Cbuf)
                        for _ in range_(n_k):
                            Abuf = A_c.acquire(ObjectFifoPort.Consume, 1)
                            Bbuf = B_c[c].acquire(ObjectFifoPort.Consume, 1)
                            matmul(Abuf, Bbuf, Cbuf)
                            A_c.release(ObjectFifoPort.Consume, 1)
                            B_c[c].release(ObjectFifoPort.Consume, 1)
                        C_c[c].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def seq(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, 1))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

            for gi in range(n_groups):
                row_tile = gi // n_col_group
                col_group = gi % n_col_group

                # Batch K-iterations in groups of MAX_BD_SLOTS to fill the
                # shim DMA's BD table and let the hardware pipeline transfers
                # across iterations. Each K-iteration uses 9 BDs (1 A + 8 B
                # cols), one per channel. With 16 slots/channel, we can fit
                # 16 K-iterations' BDs before needing to await+free.
                for ki_start in range(0, n_k, MAX_BD_SLOTS):
                    ki_end = min(ki_start + MAX_BD_SLOTS, n_k)

                    all_tasks = []
                    for ki in range(ki_start, ki_end):
                        a_idx = row_tile * n_k + ki
                        at = shim_dma_single_bd_task(A_s, A, tap=A_taps[a_idx], issue_token=True)
                        dma_start_task(at)
                        all_tasks.append(at)

                        for c in range(n_aie_cols):
                            n_tile = col_group * n_aie_cols + c
                            b_idx = ki * n_col_tile + n_tile
                            bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[b_idx], issue_token=True)
                            dma_start_task(bt)
                            all_tasks.append(bt)

                    dma_await_task(*all_tasks)
                    dma_free_task(*all_tasks)

                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = col_group * n_aie_cols + c
                    c_idx = row_tile * n_col_tile + n_tile
                    ct = shim_dma_single_bd_task(C_s[c], C, tap=C_taps[c_idx], issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
