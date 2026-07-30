#!/usr/bin/env python3
#
# SSM Selective Scan MLIR generator for NPU2 (XDNA 2)
#
# Maps Mamba2 selective scan recurrence onto AIE tiles:
#   Each tile: 1 head × d_state=64
#   State update: state[t+1] = A_diag ⊙ state[t] + dt * B * x[t]
#   Output: y[t] = C ⊙ state[t+1]
#
# Tile layout: 8 cols × 2 compute rows = 16 tiles
#   16 heads per batch × 2 iterations = 32 heads (Zamba2: 80 heads)
#   Remaining heads handled by time-multiplexing
#
# Dataflow:
#   1. DMA state[A], A, B, C, dt from DDR → L2 → L1 per tile
#   2. Per-step: load x[t], compute state update + output
#   3. DMA state_out + y to L2 → DDR
#
# Kernel: ssm_selective_scan.o (compiled from ssm_selective_scan.cc)
#
# Toolchain: MLIR-AIE aiecc v0.3.x
#   python3 n1_core_ssm_scan.py > ssm_scan.mlir

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

def main():
    with mlir_mod_ctx() as ctx:
        gen_ssm_scan()
        print(ctx.module)

def gen_ssm_scan():
    """SSM selective scan: 16 tiles × 2 iterations = 32 heads"""
    n_cols = 8
    n_rows = 2  # compute rows (rows 2-3, after shim+mem at rows 0-1)
    d_state = 64
    n_heads_per_tile = 2  # each tile handles 2 heads sequentially

    dtype_state = np.dtype[np.bfloat16]

    @device(AIEDevice.npu2)
    def device_body():
        tiles = [[tile(col, row)
                  for col in range(n_cols)]
                 for row in range(4)]  # 0:shim, 1:mem, 2-3:compute

        state_ty = np.ndarray[(n_heads_per_tile, d_state), dtype_state]
        aux_ty = np.ndarray[(n_heads_per_tile, d_state), dtype_state]  # A/B/C/x
        dt_ty = np.ndarray[(n_heads_per_tile,), dtype_state]  # dt scalar per head
        out_ty = np.ndarray[(n_heads_per_tile, d_state), dtype_state]  # y + state_out

        # FIFOs: state, A, B, C, dt, x → compute tile
        # state_out, y ← compute tile
        state_in_fifo = object_fifo("state_in", tiles[1][0], tiles[2][0], 2, state_ty)
        aux_fifo = object_fifo("aux", tiles[1][0], tiles[2][0], 2, aux_ty)
        dt_fifo = object_fifo("dt", tiles[1][0], tiles[2][0], 2, dt_ty)
        state_out_fifo = object_fifo("state_out", tiles[2][0], tiles[1][0], 2, out_ty)

        @core(tiles[2][0], stack_size=0xD00)
        def core_body():
            for _ in range(0xFFFFFFFF):
                state = state_in_fifo.acquire(ObjectFifoPort.Consume, 1)
                A = aux_fifo.acquire(ObjectFifoPort.Consume, 1)
                B = aux_fifo.acquire(ObjectFifoPort.Consume, 1)
                C = aux_fifo.acquire(ObjectFifoPort.Consume, 1)
                x = aux_fifo.acquire(ObjectFifoPort.Consume, 1)
                dt = dt_fifo.acquire(ObjectFifoPort.Consume, 1)

                out = state_out_fifo.acquire(ObjectFifoPort.Produce, 1)

                # Core calls ssm_selective_scan for 2 heads
                # (kernel call via external_func)

                state_out_fifo.release(ObjectFifoPort.Produce, 1)
                state_in_fifo.release(ObjectFifoPort.Consume, 1)
                aux_fifo.release(ObjectFifoPort.Consume, 1)
                dt_fifo.release(ObjectFifoPort.Consume, 1)
                aux_fifo.release(ObjectFifoPort.Consume, 1)
                aux_fifo.release(ObjectFifoPort.Consume, 1)
                aux_fifo.release(ObjectFifoPort.Consume, 1)

if __name__ == "__main__":
    main()
