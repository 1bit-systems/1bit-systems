#!/usr/bin/env python3
"""
Single-shot native ternary MLIR generator — same as n1_core_native_ternary.py
but with a bounded loop (1 iteration) instead of 0xFFFFFFFF infinite loop.

Each kernel dispatch processes exactly one input → produces one output → exits.
Host tiles M and K dimensions across multiple dispatches.

Usage:
    python3 n1_core_native_ternary_oneshot.py -M 32 -K 64 > ternary_oneshot.mlir
"""

import argparse
import sys
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=32)
    parser.add_argument("-K", type=int, default=64)
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:
        my_oneshot(args.M, args.K, args.dump)
        print(ctx.module)


def my_oneshot(M, K_packed, dump=False):
    k_ternary = K_packed * 4

    weight_bytes = M * K_packed
    scale_bytes = M * 2
    act_bytes = k_ternary * 2
    in_bytes = weight_bytes + scale_bytes + act_bytes
    in_dwords = (in_bytes + 3) // 4
    out_elems = M
    out_dwords = (out_elems * 2 + 3) // 4

    if dump:
        print(f"// M={M} K_packed={K_packed} → {k_ternary} ternary", file=sys.stderr)
        print(f"// in_bytes={in_bytes} in_dwords={in_dwords}", file=sys.stderr)
        print(f"// out_elems={out_elems} out_dwords={out_dwords}", file=sys.stderr)

    kernel_entry = "mm_ternary_32x64x128"
    kernel_o = f"{kernel_entry}.o"
    dtype_in = np.int32

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(out_elems,), np.dtype[dtype_in]]
        C_l2_ty = C_l1_ty

        native_ternary = external_func(
            kernel_entry,
            inputs=[A_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=kernel_o,
        )

        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        core_tile = tile(0, 2)

        A_l3l2 = object_fifo("A_L3L2", shim_tile, mem_tile, 2, A_l2_ty)
        A_l2l1 = object_fifo("A_L2L1", mem_tile, core_tile, 2, A_l1_ty)
        object_fifo_link(A_l3l2, A_l2l1)

        C_l1l2 = object_fifo("C_L1L2", core_tile, mem_tile, 1, C_l1_ty)
        C_l2l3 = object_fifo("C_L2L3", mem_tile, shim_tile, 2, C_l2_ty)
        object_fifo_link(C_l1l2, C_l2l3)

        # KEY CHANGE: single-shot (1 iteration) instead of 0xFFFFFFFF
        @core(core_tile, stack_size=0xD00)
        def core_body():
            for _ in range_(1):
                A = A_l2l1.acquire(ObjectFifoPort.Consume, 1)
                C = C_l1l2.acquire(ObjectFifoPort.Produce, 1)
                native_ternary(A, C, 0, M)
                A_l2l1.release(ObjectFifoPort.Consume, 1)
                C_l1l2.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(in_dwords,), np.dtype[dtype_in]],
            np.ndarray[(out_dwords,), np.dtype[dtype_in]],
        )
        def sequence(A_flat, C_flat):
            A_taps = TensorTiler2D.group_tiler((in_dwords, 1), (in_dwords, 1), (1, 1))
            C_taps = TensorTiler2D.group_tiler((out_dwords, 1), (out_dwords, 1), (1, 1))

            a_task = shim_dma_single_bd_task(A_l3l2, A_flat, tap=A_taps[0], issue_token=False)
            dma_start_task(a_task)

            c_task = shim_dma_single_bd_task(C_l2l3, C_flat, tap=C_taps[0], issue_token=True)
            dma_start_task(c_task)

            dma_await_task(c_task)
            dma_free_task(a_task)


main()
