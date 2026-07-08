#!/usr/bin/env python3
"""
generate_multitile_ternary.py — Multi-tile native ternary NPU MLIR generator

Generates MLIR for N tiles, each on a separate shim column (avoids routing conflicts).
Each tile runs bitnet_ternary_layer_scheduler independently.

Usage:
    python3 generate_multitile_ternary.py [N_tiles] > multitile.mlir
"""

import sys

N_TILES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SHIM_ROW = 0
CORE_ROW = 2
COLS = [2, 3, 4, 5][:N_TILES]  # use columns 2-5

# Buffer dimensions per tile
ROWS_PER_TILE = 32
CHUNKS = 10
WT_DWORDS = (CHUNKS * 128) // 4    # 320
ACT_DWORDS = (CHUNKS * 512) // 4   # 1280
RECORD_DWORDS = 28 * 17            # 476
BUF_DWORDS = max(WT_DWORDS, ACT_DWORDS, RECORD_DWORDS)

KERNEL_OBJECT = "bitnet_ternary_main16.o"
KERNEL_ENTRY = "bitnet_ternary_layer_scheduler"


def gen():
    lines = []
    w = lines.append

    w(f"// Multi-tile native ternary NPU — {N_TILES} tiles")
    w(f"// bitnet_ternary_layer_scheduler, {BUF_DWORDS} dwords/tile")
    w("")
    w("module {")
    w("  aie.device(npu2) {")

    # Tiles
    for col in COLS:
        w(f"    %shim_{col} = aie.tile({col}, {SHIM_ROW})")
        w(f"    %core_{col} = aie.tile({col}, {CORE_ROW})")

    # Flows — one per column, no conflicts
    for col in COLS:
        w(f"    aie.flow(%shim_{col}, DMA : 0, %core_{col}, DMA : 0)")
        w(f"    aie.flow(%core_{col}, DMA : 1, %shim_{col}, DMA : 0)")

    # External kernel
    w("")
    w(f"    func.func private @{KERNEL_ENTRY}(")
    w(f"      memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>,")
    w(f"      memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>,")
    w(f"      memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>,")
    w( "      i32, i32, i32, i32")
    w(f"    ) attributes {{link_with = \"{KERNEL_OBJECT}\"}}")

    # Per-tile bodies
    for i, col in enumerate(COLS):
        w("")
        # Buffers
        w(f"    %buf_{col}_wp  = aie.buffer(%core_{col}) {{sym_name = \"wp_{col}\"}}  : memref<{BUF_DWORDS}xi32>")
        w(f"    %buf_{col}_wq  = aie.buffer(%core_{col}) {{sym_name = \"wq_{col}\"}}  : memref<{BUF_DWORDS}xi32>")
        w(f"    %buf_{col}_ap  = aie.buffer(%core_{col}) {{sym_name = \"ap_{col}\"}}  : memref<{BUF_DWORDS}xi32>")
        w(f"    %buf_{col}_aq  = aie.buffer(%core_{col}) {{sym_name = \"aq_{col}\"}}  : memref<{BUF_DWORDS}xi32>")
        w(f"    %buf_{col}_rp  = aie.buffer(%core_{col}) {{sym_name = \"rp_{col}\"}}  : memref<{BUF_DWORDS}xi32>")
        w(f"    %buf_{col}_rq  = aie.buffer(%core_{col}) {{sym_name = \"rq_{col}\"}}  : memref<{BUF_DWORDS}xi32>")

        # Locks
        w(f"    %lk_{col}_ie = aie.lock(%core_{col}, 0) {{init = 1 : i32}}")
        w(f"    %lk_{col}_if = aie.lock(%core_{col}, 1) {{init = 0 : i32}}")
        w(f"    %lk_{col}_oe = aie.lock(%core_{col}, 2) {{init = 1 : i32}}")
        w(f"    %lk_{col}_of = aie.lock(%core_{col}, 3) {{init = 0 : i32}}")

        # Core
        w(f"    %core_{col}_body = aie.core(%core_{col}) {{")
        w(f"      aie.use_lock(%lk_{col}_if, AcquireGreaterEqual, 1)")
        w(f"      aie.use_lock(%lk_{col}_oe, AcquireGreaterEqual, 1)")
        w(f"      %g_{col} = arith.constant {i} : i32")
        w(f"      %r_{col} = arith.constant 0 : i32")
        w(f"      %m_{col} = arith.constant {ROWS_PER_TILE} : i32")
        w(f"      %p_{col} = arith.constant 7 : i32")
        w(f"      func.call @{KERNEL_ENTRY}(%buf_{col}_wp, %buf_{col}_wq, %buf_{col}_ap, %buf_{col}_aq, %buf_{col}_rp, %buf_{col}_rq, %g_{col}, %r_{col}, %m_{col}, %p_{col})")
        w(f"        : (memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>, memref<{BUF_DWORDS}xi32>, i32, i32, i32, i32) -> ()")
        w(f"      aie.use_lock(%lk_{col}_of, Release, 1)")
        w(f"      aie.use_lock(%lk_{col}_ie, Release, 1)")
        w(f"      aie.end")
        w(f"    }}")

        # DMA
        w(f"    %core_{col}_mem = aie.mem(%core_{col}) {{")
        w(f"      %dma_{col}_in = aie.dma_start(S2MM, 0, ^in_{col}, ^out_{col})")
        w(f"    ^in_{col}:")
        w(f"      aie.use_lock(%lk_{col}_ie, AcquireGreaterEqual, 1)")
        w(f"      aie.dma_bd(%buf_{col}_wp : memref<{BUF_DWORDS}xi32>, 0, {BUF_DWORDS}) {{bd_id = 0 : i32}}")
        w(f"      aie.use_lock(%lk_{col}_if, Release, 1)")
        w(f"      aie.next_bd ^in_{col}")
        w(f"    ^out_{col}:")
        w(f"      %dma_{col}_out = aie.dma_start(MM2S, 0, ^outl_{col}, ^end_{col})")
        w(f"    ^outl_{col}:")
        w(f"      aie.use_lock(%lk_{col}_of, AcquireGreaterEqual, 1)")
        w(f"      aie.dma_bd(%buf_{col}_rp : memref<{BUF_DWORDS}xi32>, 0, {RECORD_DWORDS}) {{bd_id = 1 : i32}}")
        w(f"      aie.use_lock(%lk_{col}_oe, Release, 1)")
        w(f"      aie.next_bd ^outl_{col}")
        w(f"    ^end_{col}:")
        w(f"      aie.end")
        w(f"    }}")

    # Runtime sequence
    w("")
    w(f"    aie.runtime_sequence(")
    args = []
    for col in COLS:
        args.append(f"%arg_{col}_in: memref<{BUF_DWORDS}xi32>")
        args.append(f"%arg_{col}_out: memref<{RECORD_DWORDS}xi32>")
    w("      " + ", ".join(args))
    w("    ) {")

    for i, col in enumerate(COLS):
        in_idx = i * 2
        out_idx = i * 2 + 1
        base = 0x4001D004 + i * 0x40

        w(f"      // Tile col={col}")
        w(f"      aiex.npu.writebd {{bd_id = {out_idx} : i32, buffer_length = {RECORD_DWORDS} : i32, buffer_offset = 0 : i32,")
        w(f"        burst_length = 64 : i32, column = {col} : i32,")
        w( "        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,")
        w( "        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,")
        w( "        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,")
        w( "        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,")
        w( "        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,")
        w( "        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,")
        w( "        next_bd = 0 : i32, out_of_order_id = 0 : i32,")
        w( "        packet_id = 0 : i32, packet_type = 0 : i32,")
        w(f"        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}")
        w(f"      aiex.npu.address_patch {{addr = {hex(base)} : ui32, arg_idx = {out_idx} : i32, arg_plus = 0 : i32}}")

        w(f"      aiex.npu.writebd {{bd_id = {in_idx} : i32, buffer_length = {BUF_DWORDS} : i32, buffer_offset = 0 : i32,")
        w(f"        burst_length = 64 : i32, column = {col} : i32,")
        w( "        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,")
        w( "        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,")
        w( "        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,")
        w( "        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,")
        w( "        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,")
        w( "        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,")
        w( "        next_bd = 0 : i32, out_of_order_id = 0 : i32,")
        w( "        packet_id = 0 : i32, packet_type = 0 : i32,")
        w(f"        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}")
        w(f"      aiex.npu.address_patch {{addr = {hex(base + 0x20)} : ui32, arg_idx = {in_idx} : i32, arg_plus = 0 : i32}}")

    w("")
    for col in COLS:
        w(f"      aiex.npu.push_queue({col}, 0, S2MM : 0) {{bd_id = {COLS.index(col) * 2} : i32, issue_token = true, repeat_count = 0 : i32}}")
        w(f"      aiex.npu.push_queue({col}, 0, MM2S : 0) {{bd_id = {COLS.index(col) * 2 + 1} : i32, issue_token = true, repeat_count = 0 : i32}}")

    w(f"      aiex.npu.sync {{channel = 0 : i32, column = {COLS[0]} : i32, column_num = {N_TILES} : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}")
    w("    }")
    w("  }")
    w("}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(gen())
