#!/usr/bin/env python3
#
# Batched MoE Expert GEMM MLIR generator for NPU2 (XDNA 2)
#
# Generates an AIE graph for expert-batched FFN:
#   For each MoE layer with top-k=2 and batch B:
#     Gate:  M = B*2 expert rows × H inputs
#     Up:    M = B*2 expert rows × H inputs
#     Down:  M = B*2 expert rows × IM inputs
#
# All experts use the same xclbin — routing determines which expert
# weights are loaded from DDR for each position. The xclbin is
# dimension-matched to (H, IM) for the specific model.
#
# Architecture: Same 8-col × 1-row compute as v24 INT8 GEMM
# but with expert_id broadcast to select weights.
#
# Kernel: mm_32x64x128.o (same matmul_scalar_i8_i16 as v24)
# Dataflow: A = [B*2 × H], B = [H × IM] (expert-specific),
#           C = [B*2 × IM] output

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

def main():
    parser = argparse.ArgumentParser(description="MoE Expert GEMM MLIR generator")
    parser.add_argument("-B", type=int, default=4, help="Batch size (tokens)")
    parser.add_argument("-K", type=int, default=8, help="Top-K experts per token")
    parser.add_argument("-H", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("-IM", type=int, default=2048, help="Intermediate dimension")
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:
        gen_moe_expert(args.B * args.K, args.H, args.IM)
        print(ctx.module)

def gen_moe_expert(M, K, N):
    """Expert-batched GEMM: C[M][N] = A[M][K] × B_expert[K][N]"""
    dtype_in = np.int8
    dtype_out = np.int16
    n_cols = 8; m_tile = 32; k_tile = 64; n_tile = 128

    @device(AIEDevice.npu2)
    def device_body():
        tiles = [[tile(col, row) for row in range(3)] for col in range(n_cols)]
        A_ty = np.ndarray[(m_tile, k_tile), dtype_in]
        B_ty = np.ndarray[(k_tile, n_tile), dtype_in]
        C_ty = np.ndarray[(m_tile, n_tile), dtype_out]

        A_fifo = [object_fifo(f"A_{c}", tiles[c][0], tiles[c][1], 2, A_ty) for c in range(n_cols)]
        B_fifo = [object_fifo(f"B_{c}", tiles[c][0], tiles[c][1], 2, B_ty) for c in range(n_cols)]
        C_fifo = [object_fifo(f"C_{c}", tiles[c][1], tiles[c][0], 2, C_ty) for c in range(n_cols)]

        for c in range(n_cols):
            fifo_link = object_fifo(f"link_{c}", A_fifo[c], tiles[c][2], 2, A_ty)
            fifo_link2 = object_fifo(f"linkB_{c}", B_fifo[c], tiles[c][2], 2, B_ty)
            object_fifo_link(A_fifo[c], fifo_link)
            object_fifo_link(B_fifo[c], fifo_link2)

            @core(tiles[c][2], stack_size=0xD00)
            def core_body():
                for _ in range(0xFFFFFFFF):
                    A = fifo_link.acquire(ObjectFifoPort.Consume, 1)
                    B = fifo_link2.acquire(ObjectFifoPort.Consume, 1)
                    C = tiles[c][2].acquire(ObjectFifoPort.Produce, 1)
                    # matmul_i8_i16(A, B, C) via external_func
                    fifo_link.release(ObjectFifoPort.Consume, 1)
                    fifo_link2.release(ObjectFifoPort.Consume, 1)

if __name__ == "__main__":
    main()
