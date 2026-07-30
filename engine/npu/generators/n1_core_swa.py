#!/usr/bin/env python3
#
# Sliding Window Attention MLIR generator for NPU2 (XDNA 2)
#
# Generates an AIE graph for SWA (Sliding Window Attention):
#   O = softmax(Q × K^T[W] / sqrt(d)) × V[W]
#   Where W = window size, K[W] = K within last W positions
#
# Compared to full attention, SWA loads only W K/V positions instead
# of the full sequence length — ~16× less compute at W=4096 vs seq=64K.
#
# Architecture: 8 columns × 3 AIE rows
#   Row 0: Shim tiles (DDR DMA)
#   Row 1: Memory tiles (L2 scratchpad — Q, K_window, V_window)
#   Rows 2: Compute tiles (Q×K^T matmul + softmax + score×V matmul)
#
# Flow per token:
#   1. DMA Q (1×d) from DDR → L2 → L1 (broadcast to all cols)
#   2. DMA K_window (W×d) from DDR → L2 → L1 (striped across cols)
#   3. Compute QK^T: each col computes scores for W/8 positions
#   4. Softmax over W scores (cross-col sync via mem tile)
#   5. DMA V_window (W×d) from DDR → L2 → L1
#   6. Compute score×V: each col accumulates W/8 positions
#   7. DMA output (1×d) from L1 → L2 → DDR
#
# Toolchain: MLIR-AIE aiecc v0.3.x
#   python3 n1_core_swa.py -H 4096 -W 4096 > swa.mlir
#   aiecc swa.mlir ... (see build_xclbins.sh)
#
# Kernel: attn_swi32.o (SWA-specific: i32 QK^T + exp + i32×i8 V)

import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D

def main():
    parser = argparse.ArgumentParser(description="SWA MLIR generator")
    parser.add_argument("-H", type=int, default=4096, help="Hidden dim")
    parser.add_argument("-d", type=int, default=128, help="Head dim")
    parser.add_argument("-W", type=int, default=4096, help="Window size")
    parser.add_argument("--n-query-heads", type=int, default=32, help="Query heads")
    parser.add_argument("--n-kv-heads", type=int, default=8, help="KV heads")
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:
        gen_swa(args.H, args.d, args.W, args.n_query_heads, args.n_kv_heads)
        print(ctx.module)

def gen_swa(H, d, W, nh, nkv):
    n_aie_cols = 8
    n_aie_rows = 3  # shim + mem + compute

    dtype_score = np.int32
    dtype_out = np.int16

    @device(AIEDevice.npu2)
    def device_body():
        # Q tile: 1 head × head_dim (bf16 → i8)
        Q_ty = np.ndarray[(1, d), np.dtype[np.int8]]
        # KV window tile: W/n cols × d per column
        W_per_col = (W + n_aie_cols - 1) // n_aie_cols
        KV_ty = np.ndarray[(W_per_col, d), np.dtype[np.int8]]
        # Score tile: 1 × W_per_col (int32)
        S_ty = np.ndarray[(1, W_per_col), np.dtype[dtype_score]]
        # Output tile: 1 × d (bf16)
        O_ty = np.ndarray[(1, d), np.dtype[dtype_out]]

        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(n_aie_rows)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2]

        # FIFOs: Q broadcast, KV striped, scores local, output reduced
        Q_fifo = object_fifo("Q", shim_tiles[0], mem_tiles[0], 2, Q_ty)
        Q_core = [object_fifo(f"Q_{c}", mem_tiles[0], core_tiles[c], 2, Q_ty) for c in range(n_aie_cols)]
        object_fifo_link(Q_fifo, Q_core, [0]*n_aie_cols)

        KV_fifo = [object_fifo(f"KV_{c}", shim_tiles[c], mem_tiles[c], 2, KV_ty) for c in range(n_aie_cols)]
        KV_core = [object_fifo(f"KVc_{c}", mem_tiles[c], core_tiles[c], 2, KV_ty) for c in range(n_aie_cols)]
        for c in range(n_aie_cols):
            object_fifo_link(KV_fifo[c], KV_core[c])

        # Score accumulation (reduced across cols via mem tile)
        S_local = [object_fifo(f"S_{c}", core_tiles[c], mem_tiles[c], 2, S_ty) for c in range(n_aie_cols)]
        S_global = object_fifo("S_global", mem_tiles[0], shim_tiles[0], 2,
                               np.ndarray[(1, W), np.dtype[dtype_score]])
        object_fifo_link(S_local, S_global, [W_per_col*c for c in range(n_aie_cols)])

        O_fifo = object_fifo("O", mem_tiles[0], shim_tiles[0], 2, O_ty)

        for c in range(n_aie_cols):
            @core(core_tiles[c], stack_size=0xD00)
            def core_body():
                for _ in range(0xFFFFFFFF):
                    Q = Q_core[c].acquire(ObjectFifoPort.Consume, 1)
                    KV = KV_core[c].acquire(ObjectFifoPort.Consume, 1)

                    # Q × K^T → scores (int32)
                    S = S_local[c].acquire(ObjectFifoPort.Produce, 1)
                    # Core calls: swa_qk(Q, KV, S, d, W_per_col)
                    # ... (kernel call via external_func)

                    S_local[c].release(ObjectFifoPort.Produce, 1)
                    Q_core[c].release(ObjectFifoPort.Consume, 1)
                    KV_core[c].release(ObjectFifoPort.Consume, 1)

        # Softmax + score×V (second pass, uses same tiles)
        @core(core_tiles[0], stack_size=0xD00)
        def softmax_core():
            for _ in range(0xFFFFFFFF):
                S_in = S_global.acquire(ObjectFifoPort.Produce, 1)
                # softmax over full W scores, then × V
                # ... (kernel call)
                O_out = O_fifo.acquire(ObjectFifoPort.Produce, 1)
                O_fifo.release(ObjectFifoPort.Produce, 1)
                S_global.release(ObjectFifoPort.Produce, 1)

if __name__ == "__main__":
    main()
