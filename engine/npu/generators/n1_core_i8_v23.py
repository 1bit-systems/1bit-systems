#!/usr/bin/env python3
#
# INT8 MLIR generator v23 — written from scratch 2026-07-28, correctness-first.
#
# Informed by 22 prior falsified attempts (v2-v22, v2_gu) in this same directory.
# Design principles, each chosen specifically to avoid a documented prior failure:
#   - Issue exactly ONE DMA transfer per FIFO acquire the core actually performs.
#     No "dimensionsFromStream"/"dimensionsFromTransform" repeat-descriptor tricks
#     (v11, v19, v21, v22, v2_gu all used these and all hang, including v2_gu with
#     n_aie_cols=1 where broadcast isn't even in play — the repeat-descriptor math
#     itself is the suspect, not multi-core broadcast).
#   - A is a genuine broadcast objectFifo (shim -> mem -> all 8 cores), fed with
#     plain (m,k) flat tiles, nothing fancier.
#   - B and C are per-column fifos, one flat (k,n) / (m,n) tile per transfer.
#   - C DMA is issued for ALL n_aie_cols columns every group (v6/v7/v8/v9 only
#     issued it for column 0 — a real, previously-undocumented bug found today).
#   - Fully sequential per group: issue all of a group's A/B, then all of its C
#     (with issue_token=True), await ALL of them, THEN free, THEN move to the next
#     group. No multi-buffer pipelining (the "tb=4 rotation" every prior attempt
#     used) — sacrifices throughput for a shot at actually being correct first.
#   - Kernel ABI matches the real, currently-used mm.cc build: matmul_i8_i32 /
#     zero_i32 (int32 accumulator), NOT the i16 scalar entry points most prior
#     versions mistakenly targeted (a second, separate bug that made most of them
#     untestable against our actual kernel object at all).
#
# Usage: python3 n1_core_i8_v23_fromscratch.py -M 128 -K 1024 -N 4096 > design.mlir
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

        # A: single broadcast fifo, shim[0] -> mem[0] -> all 8 cores. Plain flat
        # (m,k) element, no repeat dimensions — one DMA delivers exactly one tile.
        A_s = object_fifo("A_S", shim_tiles[0], mem_tiles[0], 2, A_ty)
        A_c = object_fifo("A_C", mem_tiles[0], core_tiles[0:n_aie_cols], 2, A_ty)
        object_fifo_link(A_s, A_c)

        # B, C: independent per-column fifos.
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

        num_row_tile = M // m
        num_col_group = N // n // n_aie_cols
        num_groups = num_row_tile * num_col_group
        n_k = K // k

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0x2000)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(num_groups):
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
            # Row-major (1,1)-grouped taps: exactly one (m,k)/(k,n)/(m,n) tile per
            # index, no internal repeat — plain, unambiguous indexing.
            A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, 1))
            B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

            for gi in range(num_groups):
                row_tile = gi // num_col_group
                col_group = gi % num_col_group

                # Serialize DMA issuance per K-iteration: this tile's hardware BD
                # queue only supports 16 simultaneously active descriptors, and a
                # batched issue-everything-then-wait approach (like the first draft
                # of this file, and like several of the 22 prior falsified attempts)
                # blows past that for any K large enough. Await+free each
                # K-iteration's A/B tasks before issuing the next.
                for ki in range(n_k):
                    a_idx = row_tile * n_k + ki
                    at = shim_dma_single_bd_task(A_s, A, tap=A_taps[a_idx], issue_token=True)
                    dma_start_task(at)

                    bt_list = []
                    for c in range(n_aie_cols):
                        n_tile = col_group * n_aie_cols + c
                        b_idx = ki * (N // n) + n_tile
                        bt = shim_dma_single_bd_task(B_s[c], B, tap=B_taps[b_idx], issue_token=True)
                        dma_start_task(bt)
                        bt_list.append(bt)

                    dma_await_task(at, *bt_list)
                    dma_free_task(at, *bt_list)

                c_tasks = []
                for c in range(n_aie_cols):
                    n_tile = col_group * n_aie_cols + c
                    c_idx = row_tile * (N // n) + n_tile
                    ct = shim_dma_single_bd_task(C_s[c], C, tap=C_taps[c_idx], issue_token=True)
                    dma_start_task(ct)
                    c_tasks.append(ct)

                dma_await_task(*c_tasks)
                dma_free_task(*c_tasks)


main()
