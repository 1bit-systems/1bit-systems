#!/usr/bin/env python3
"""
Native ternary single-core MLIR generator — flat buffer interface.

Uses mm_ternary_32x64x128.o (2-bit packed ternary, BF16 MAC) with the
object_fifo + shim_dma_single_bd_task pattern.

Single core: processes M weight rows × K*4 ternary values, produces M bf16
output scalars.  This is the building block for multi-core tiling.

The kernel takes (flat_input_ptr, output_ptr):
  flat_input: [M*K_packed bytes weights (uint8)] [M*2 bytes scales (bf16)]
              [K*4*2 bytes activations (bf16)]
  output:     M bf16 scalars

Usage:
    python3 n1_core_native_ternary.py -M 32 -K 64 > ternary.mlir
    # Build: aiecc.py ternary.mlir ... → ternary.xclbin
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
        description="Native ternary single-core NPU xclbin generator"
    )
    parser.add_argument("-M", type=int, default=32,
                        help="Output rows (per core)")
    parser.add_argument("-K", type=int, default=64,
                        help="Packed K bytes (K*4 = ternary values)")
    parser.add_argument("--dump", action="store_true",
                        help="Print buffer layout calculations")
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:
        dump_info = args.dump
        my_native_ternary(args.M, args.K, dump_info)
        print(ctx.module)


def my_native_ternary(M, K_packed, dump=False):
    """Generate single-core native ternary MLIR.

    Args:
        M: output rows
        K_packed: packed weight bytes → K_packed*4 ternary values
    """
    k_ternary = K_packed * 4  # e.g. 64*4 = 256

    # ── Buffer sizes ───────────────────────────────────
    weight_bytes = M * K_packed       # M * 64
    scale_bytes = M * 2               # M bf16
    act_bytes = k_ternary * 2         # 256 bf16
    in_bytes = weight_bytes + scale_bytes + act_bytes
    in_dwords = (in_bytes + 3) // 4

    out_elems = M                     # M bf16 outputs
    out_dwords = (out_elems * 2 + 3) // 4

    if dump:
        print(f"// M={M} K_packed={K_packed} → {k_ternary} ternary",
              file=__import__('sys').stderr)
        print(f"// in_bytes={in_bytes} in_dwords={in_dwords}",
              file=__import__('sys').stderr)
        print(f"// out_elems={out_elems} out_dwords={out_dwords}",
              file=__import__('sys').stderr)

    # Kernel entry
    kernel_entry = "mm_ternary_32x64x128"
    kernel_o = f"{kernel_entry}.o"

    dtype_in = np.int32

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(in_dwords,), np.dtype[dtype_in]]
        # Output: bf16 values; MLIR handles the type mapping
        C_l1_ty = np.ndarray[(out_elems,), np.dtype[dtype_in]]
        C_l2_ty = C_l1_ty

        native_ternary = external_func(
            kernel_entry,
            inputs=[A_l1_ty, C_l1_ty, np.int32, np.int32],
            link_with=kernel_o,
        )

        # Single column: shim(0,0) ↔ mem(0,1) ↔ core_tile(0,2)
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        core_tile = tile(0, 2)

        A_l3l2 = object_fifo("A_L3L2", shim_tile, mem_tile, 2, A_l2_ty)
        A_l2l1 = object_fifo("A_L2L1", mem_tile, core_tile, 2, A_l1_ty)
        object_fifo_link(A_l3l2, A_l2l1)

        C_l1l2 = object_fifo("C_L1L2", core_tile, mem_tile, 1, C_l1_ty)
        C_l2l3 = object_fifo("C_L2L3", mem_tile, shim_tile, 2, C_l2_ty)
        object_fifo_link(C_l1l2, C_l2l3)

        @core(core_tile, stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
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
            # Single tile tap for full 1D buffer
            A_taps = TensorTiler2D.group_tiler((in_dwords, 1), (in_dwords, 1), (1, 1))
            C_taps = TensorTiler2D.group_tiler((out_dwords, 1), (out_dwords, 1), (1, 1))

            a_task = shim_dma_single_bd_task(
                A_l3l2, A_flat, tap=A_taps[0], issue_token=False,
            )
            dma_start_task(a_task)

            c_task = shim_dma_single_bd_task(
                C_l2l3, C_flat, tap=C_taps[0], issue_token=True,
            )
            dma_start_task(c_task)

            dma_await_task(c_task)
            dma_free_task(a_task)


main()
