#
# ternary_gemv_i8_i32.py
#
# MLIR-AIE design for the v0 ternary GEMV on AMD XDNA2 NPU2 (Strix Halo).
# Adapted from
#   mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector/
#       matrix_vector.py
# but with the A operand carrying packed ternary weights (4 trits per byte,
# uint8 storage). The kernel side performs the unpack + i8 multiply + i32
# accumulate; see ternary_gemv_i8_i32.cc for the contract.
#
# Single-core, single-tile v0:
#   M = 64, K = 64, K_PACK = 16
# i.e. one 64-row × 16-byte tile for A and one 64-element vector for b.
# Result c is a 64-element i32 vector (the host applies scales + fp16 cast).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2026 Daniel <d1r7yman@gmail.com>
#
import argparse
import sys

import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def my_design(dev, M, K):
    assert K % 4 == 0, "K must be divisible by 4 (4 trits per byte)"
    K_PACK = K // 4

    # v0: single tile == single core. The kernel object is compiled with
    # DIM_M=M, DIM_K_PACK=K_PACK so the inner loops are constant-bounded.
    m = M
    k_pack = K_PACK

    # No internal blocking on K — single matvec call per output. K_div_kp == 1.
    K_div_kp = K_PACK // k_pack
    M_div_m = M // m

    A_sz = M * K_PACK   # bytes
    B_sz = K            # i8 elements
    C_sz = M            # i32 elements

    # Per-tile shapes seen by the AIE core.
    inA_shape = (m, k_pack)         # uint8 packed weights
    inB_shape = (K,)                # i8 activations (entire vector goes to core)
    outC_shape = (m,)               # i32 accumulator

    # MLIR types.
    a_dtype_np = np.dtype[np.uint8]
    b_dtype_np = np.dtype[np.int8]
    c_dtype_np = np.dtype[np.int32]

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            inA_ty = np.ndarray[inA_shape, a_dtype_np]
            inB_ty = np.ndarray[inB_shape, b_dtype_np]
            outC_ty = np.ndarray[outC_shape, c_dtype_np]

            zero_fn = external_func(
                "zero_scalar_i32_ternary",
                inputs=[outC_ty],
                link_with="ternary_gemv_kernel.o",
            )
            tgemv_fn = external_func(
                "ternary_gemv_scalar_u8packed_i8_i32",
                inputs=[inA_ty, inB_ty, outC_ty],
                link_with="ternary_gemv_kernel.o",
            )

            # Tile placement: one shim, one mem tile, one compute tile.
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)

            # Object FIFO: A goes shim -> mem -> core, sized for one tile.
            memA_fifo = object_fifo("memA", ShimTile, MemTile, 2, inA_ty)
            inA_fifo = object_fifo("inA", MemTile, ComputeTile, 2, inA_ty)
            object_fifo_link(memA_fifo, inA_fifo)

            # B goes shim -> core directly (small, single broadcast).
            inB_fifo = object_fifo("inB", ShimTile, [ComputeTile], 2, inB_ty)

            # C goes core -> shim.
            outC_fifo = object_fifo("outC", ComputeTile, ShimTile, 2, outC_ty)

            @core(ComputeTile, stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_out = outC_fifo.acquire(ObjectFifoPort.Produce, 1)
                    zero_fn(elem_out)

                    # Single-tile K loop (K_div_kp == 1 in v0). Kept as a loop
                    # so future K-blocking is a one-line change.
                    for _ in (
                        range_(K_div_kp) if K_div_kp > 1 else range(1)
                    ):
                        elem_a = inA_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                        tgemv_fn(elem_a, elem_b, elem_out)
                        inA_fifo.release(ObjectFifoPort.Consume, 1)
                        inB_fifo.release(ObjectFifoPort.Consume, 1)

                    outC_fifo.release(ObjectFifoPort.Produce, 1)

            # Runtime DMA. Outer loop over row-tiles of M; one b broadcast.
            A_taps = TensorTiler2D.group_tiler(
                (M, K_PACK), (m, k_pack), (M_div_m, K_div_kp), prune_step=False
            )
            b_tap = TensorTiler2D.simple_tiler(
                (1, K), pattern_repeat=M_div_m, prune_step=False
            )[0]

            @runtime_sequence(
                np.ndarray[(A_sz,), a_dtype_np],
                np.ndarray[(B_sz,), b_dtype_np],
                np.ndarray[(C_sz,), c_dtype_np],
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(metadata=inB_fifo, bd_id=2, mem=B, tap=b_tap)
                for i, a_tap in enumerate(A_taps):
                    C_offset = i * m
                    npu_dma_memcpy_nd(
                        metadata=memA_fifo, bd_id=1, mem=A, tap=a_tap
                    )
                    npu_dma_memcpy_nd(
                        metadata=outC_fifo,
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, m],
                    )
                dma_wait(outC_fifo)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ternary_gemv_i8_i32")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu2")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=64)
    args, _ = p.parse_known_args()
    my_design(args.dev, args.M, args.K)
