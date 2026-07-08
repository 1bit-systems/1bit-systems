// Auto-generated MLIR — native ternary GEMM
// mm_ternary_32x64x128: M=32, K=64 (256 ternary), N=128

module {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %core = aie.tile(2, 2)

    // case marker ternary-kernel-32x64x128

    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 1, %shim, DMA : 0)

    func.func private @mm_ternary_32x64x128(
      memref<656xi32>,
      memref<32xbf16>
    ) attributes {link_with = "mm_ternary_32x64x128.o"}

    %buf_in  = aie.buffer(%core) {sym_name = "buf_in"}  : memref<656xi32>
    %buf_out = aie.buffer(%core) {sym_name = "buf_out"} : memref<32xbf16>

    %core_in_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "core_in_empty"}
    %core_in_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "core_in_full"}
    %core_out_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "core_out_empty"}
    %core_out_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "core_out_full"}

    %core_body = aie.core(%core) {
      aie.use_lock(%core_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%core_out_empty, AcquireGreaterEqual, 1)
      func.call @mm_ternary_32x64x128(%buf_in, %buf_out) : (memref<656xi32>, memref<32xbf16>) -> ()
      aie.use_lock(%core_out_full, Release, 1)
      aie.use_lock(%core_in_empty, Release, 1)
      aie.end
    }

    %mem = aie.mem(%core) {
      %dma_in = aie.dma_start(S2MM, 0, ^in_loop, ^out_start)
    ^in_loop:
      aie.use_lock(%core_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<656xi32>, 0, 656) {bd_id = 0 : i32}
      aie.use_lock(%core_in_full, Release, 1)
      aie.next_bd ^in_loop
    ^out_start:
      %dma_out = aie.dma_start(MM2S, 0, ^out_loop, ^end)
    ^out_loop:
      aie.use_lock(%core_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<32xbf16>, 0, 32) {bd_id = 1 : i32}
      aie.use_lock(%core_out_empty, Release, 1)
      aie.next_bd ^out_loop
    ^end:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<656xi32>, %arg1: memref<16xi32>) {
      aiex.npu.push_queue(2, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.push_queue(2, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
