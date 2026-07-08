#!/usr/bin/env python3
"""
generate_objfifo_ternary.py — Object-fifo based single-tile native ternary NPU MLIR

Uses the proven object_fifo + dma_configure_task_for pattern that works reliably
on NPU, replacing the broken flow + writebd pattern.

Links against mm_ternary_32x64x128.o for native 2-bit ternary decode.
"""

import sys

M = 32
K_PACKED = 64      # 64 packed bytes → 256 ternary
K_TERNARY = 256
IN_DWORDS = 656    # (2048 + 64 + 512) / 4
OUT_BF16 = 32      # M bf16 values

KERNEL_OBJ = "mm_ternary_32x64x128.o"
KERNEL_ENTRY = "mm_ternary_32x64x128"

mlir = f"""// Auto-generated object_fifo MLIR for native ternary NPU kernel
// Single tile: M={M}, K_PACKED={K_PACKED} ({K_TERNARY} ternary)
// Pattern: shim → mem → core (object_fifo) + dma_configure_task_for

module {{
  aie.device(npu2) {{
    func.func private @{KERNEL_ENTRY}(
      memref<{IN_DWORDS}xi32>,
      memref<{OUT_BF16}xbf16>
    ) attributes {{link_with = "{KERNEL_OBJ}"}}

    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %core = aie.tile(0, 2)

    // Input object_fifo: shim → mem → core
    aie.objectfifo @in_L3L2(%shim, {{%mem}}, 2 : i32) : !aie.objectfifo<memref<{IN_DWORDS}xi32>>
    aie.objectfifo @in_L2L1(%mem, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{IN_DWORDS}xi32>>
    aie.objectfifo.link [@in_L3L2] -> [@in_L2L1]([] [])

    // Output object_fifo: core → mem → shim
    aie.objectfifo @out_L1L2(%core, {{%mem}}, 1 : i32) : !aie.objectfifo<memref<{OUT_BF16}xbf16>>
    aie.objectfifo @out_L2L3(%mem, {{%shim}}, 2 : i32) : !aie.objectfifo<memref<{OUT_BF16}xbf16>>
    aie.objectfifo.link [@out_L1L2] -> [@out_L2L3]([] [])

    // Core body
    %core_body = aie.core(%core) {{
      %in_sv  = aie.objectfifo.acquire @in_L2L1(Consume, 1) : !aie.objectfifosubview<memref<{IN_DWORDS}xi32>>
      %out_sv = aie.objectfifo.acquire @out_L1L2(Produce, 1) : !aie.objectfifosubview<memref<{OUT_BF16}xbf16>>
      %in_buf  = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{IN_DWORDS}xi32>> -> memref<{IN_DWORDS}xi32>
      %out_buf = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{OUT_BF16}xbf16>> -> memref<{OUT_BF16}xbf16>
      func.call @{KERNEL_ENTRY}(%in_buf, %out_buf) : (memref<{IN_DWORDS}xi32>, memref<{OUT_BF16}xbf16>) -> ()
      aie.objectfifo.release @in_L2L1(Consume, 1)
      aie.objectfifo.release @out_L1L2(Produce, 1)
      aie.end
    }}

    // Runtime sequence
    aie.runtime_sequence(%arg0: memref<{IN_DWORDS}xi32>, %arg1: memref<{OUT_BF16}xbf16>) {{
      %0 = aiex.dma_configure_task_for @in_L3L2 {{
        aie.dma_bd(%arg0 : memref<{IN_DWORDS}xi32>, 0, {IN_DWORDS}, [<size = {IN_DWORDS}, stride = 1>])
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out_L2L3 {{
        aie.dma_bd(%arg1 : memref<{OUT_BF16}xbf16>, 0, {OUT_BF16}, [<size = {OUT_BF16}, stride = 1>])
        aie.end
      }}
      aiex.dma_start_task(%1)
    }}
  }}
}}
"""
print(mlir)
