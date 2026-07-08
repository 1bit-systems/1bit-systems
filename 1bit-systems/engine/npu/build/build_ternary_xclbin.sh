#!/bin/bash
# build_ternary_xclbin.sh — One-command build: chess compile → MLIR → xclbin
#
# Builds native ternary AIE kernel(s) into NPU xclbins for Strix Halo.
#
# Three kernel types:
#   mm_ternary       — single-tile native ternary GEMM (M×K_packed scalar output)
#   bitnet_micro     — scheduler microbenchmark (32-row BF16 output)
#   bitnet_scheduler — full layer scheduler (multi-phase, ping-pong buffers)
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_ternary_xclbin.sh [name] [type] [M] [K] [N]
#
# Defaults: name=ternary, type=mm_ternary, M=32, K=64, N=128

set -euo pipefail

NAME="${1:-ternary}"
KERNEL_TYPE="${2:-mm_ternary}"
DIM_M="${3:-32}"
DIM_K="${4:-64}"
DIM_N="${5:-128}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/../kernel" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
OUT_DIR="$BUILD_DIR/$NAME"

# Source environment if not already done
if [ -z "${TOOLCHAIN_DIR:-}" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

# Toolchain
CC="${TOOLCHAIN_DIR}/bin/xchesscc_wrapper"
AIECC="${TOOLCHAIN_DIR}/mlir_aie/bin/aiecc.py"
AIECC_PYTHON="${TORCH2AIE_ROOT}/.venv/bin/python3"
AIECC_PYTHONPATH="${MLIR_AIE_DIR}/python"

K_TERNARY=$(( DIM_K * 4 ))
SHIM_COL=2
CORE_COL=2

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Build helper: common aiecc invocation ─────────────────

run_aiecc() {
    local mlir_file="$1"
    local xclbin="$2"
    local insts="$3"
    local out_dir="$4"

    echo ""
    echo "[3/3] Compiling MLIR → xclbin (aiecc)..."
    echo "  This may take several minutes..."

    cd "$out_dir"
    PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
        ${AIECC_PYTHON} \
        ${AIECC} \
        -v \
        -j4 \
        --aietools="$AIETOOLS_DIR" \
        --no-compile-host \
        --alloc-scheme=basic-sequential \
        --aie-generate-xclbin \
        --xclbin-name="$xclbin" \
        --xclbin-kernel-name=MLIR_AIE \
        --aie-generate-npu-insts \
        --npu-insts-name="$insts" \
        "$mlir_file"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " ✅ Build complete!"
    echo "   xclbin : $xclbin"
    echo "   insts  : $insts"
    echo "   kernel : $OBJ_FILE"
    echo "   type   : $KERNEL_TYPE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Compile chess kernel ──────────────────────────────────

compile_chess() {
    local src="$1"
    local obj="$2"

    echo ""
    echo "[1/3] Compiling kernel with Chess C++ (xchesscc)..."
    echo "  Source: $src"
    echo "  Object: $obj"

    $CC aie2p \
        -I"$AIETOOLS_DIR/include" \
        -I"$MLIR_AIE_DIR/include" \
        -I"$MLIR_AIE_DIR/include/aie_kernels" \
        -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
        -DDIM_M="$DIM_M" \
        -DDIM_K_PACKED="$DIM_K" \
        -DDIM_N="$DIM_N" \
        -c "$src" \
        -o "$obj"

    echo "  ✅ Kernel object built: $(du -h "$obj" | cut -f1)"
}

# ════════════════════════════════════════════════════════════
#  mm_ternary — Native ternary GEMM (M=32, K=64→256 ternary)
#  Produces M scalar outputs.
# ════════════════════════════════════════════════════════════

build_mm_ternary() {
    local weight_bytes=$(( DIM_M * DIM_K ))
    local scale_bytes=$(( DIM_M * 2 ))
    local act_bytes=$(( K_TERNARY * 2 ))
    local output_bytes=$(( DIM_M * 2 ))

    local in_bytes=$(( weight_bytes + scale_bytes + act_bytes ))
    local in_dwords=$(( (in_bytes + 3) / 4 ))
    local out_dwords=$(( (output_bytes + 3) / 4 ))
    local out_elems=$DIM_M

    local KERNEL_ENTRY="mm_ternary_${DIM_M}x${DIM_K}x${DIM_N}"
    OBJ_FILE="$OUT_DIR/${KERNEL_ENTRY}.o"

    echo "   Type    : Native ternary GEMM (mm_ternary)"
    echo "   Dims    : M=$DIM_M  K=$DIM_K (→ $K_TERNARY ternary)"
    echo "   Input   : $in_bytes bytes ($in_dwords dwords)"
    echo "   Output  : $output_bytes bytes ($out_dwords dwords)"

    mkdir -p "$OUT_DIR"
    compile_chess "$KERNEL_DIR/mm_ternary_32x64x128.cpp" "$OBJ_FILE"

    local MLIR_FILE="$OUT_DIR/design.mlir"
    echo ""
    echo "[2/3] Generating MLIR design..."

    cat > "$MLIR_FILE" << MLIREOF
// Auto-generated MLIR — native ternary GEMM
// ${KERNEL_ENTRY}: M=${DIM_M}, K=${DIM_K} (${K_TERNARY} ternary), N=${DIM_N}

module {
  aie.device(npu2) {
    %shim = aie.tile(${SHIM_COL}, 0)
    %core = aie.tile(${CORE_COL}, 2)

    // case marker ${NAME}-kernel-${DIM_M}x${DIM_K}x${DIM_N}

    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 1, %shim, DMA : 0)

    func.func private @${KERNEL_ENTRY}(
      memref<${in_dwords}xi32>,
      memref<${out_elems}xbf16>
    ) attributes {link_with = "${KERNEL_ENTRY}.o"}

    %buf_in  = aie.buffer(%core) {sym_name = "buf_in"}  : memref<${in_dwords}xi32>
    %buf_out = aie.buffer(%core) {sym_name = "buf_out"} : memref<${out_elems}xbf16>

    %core_in_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "core_in_empty"}
    %core_in_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "core_in_full"}
    %core_out_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "core_out_empty"}
    %core_out_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "core_out_full"}

    %core_body = aie.core(%core) {
      aie.use_lock(%core_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%core_out_empty, AcquireGreaterEqual, 1)
      func.call @${KERNEL_ENTRY}(%buf_in, %buf_out) : (memref<${in_dwords}xi32>, memref<${out_elems}xbf16>) -> ()
      aie.use_lock(%core_out_full, Release, 1)
      aie.use_lock(%core_in_empty, Release, 1)
      aie.end
    }

    %mem = aie.mem(%core) {
      %dma_in = aie.dma_start(S2MM, 0, ^in_loop, ^out_start)
    ^in_loop:
      aie.use_lock(%core_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<${in_dwords}xi32>, 0, ${in_dwords}) {bd_id = 0 : i32}
      aie.use_lock(%core_in_full, Release, 1)
      aie.next_bd ^in_loop
    ^out_start:
      %dma_out = aie.dma_start(MM2S, 0, ^out_loop, ^end)
    ^out_loop:
      aie.use_lock(%core_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<${out_elems}xbf16>, 0, ${out_elems}) {bd_id = 1 : i32}
      aie.use_lock(%core_out_empty, Release, 1)
      aie.next_bd ^out_loop
    ^end:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<${in_dwords}xi32>, %arg1: memref<${out_dwords}xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = ${out_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = ${in_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D024 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = ${SHIM_COL} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
MLIREOF

    echo "  ✅ MLIR written: $MLIR_FILE"
    run_aiecc "$MLIR_FILE" "$OUT_DIR/design.xclbin" "$OUT_DIR/design.insts" "$OUT_DIR"
}

# ════════════════════════════════════════════════════════════
#  bitnet_micro — Scheduler microbenchmark (32-row output)
#  Calls bitnet_ternary_micro: 1 chunk, 32×BF16 output.
# ════════════════════════════════════════════════════════════

build_bitnet_micro() {
    # bitnet_ternary_micro: 64B weights + 64B scales + 512B acts = 640 bytes
    local in_bytes=640
    local in_dwords=$(( (in_bytes + 3) / 4 ))
    local out_bytes=$(( DIM_M * 2 ))
    local out_elems=$DIM_M
    local out_dwords=$(( (out_bytes + 3) / 4 ))

    KERNEL_ENTRY="bitnet_ternary_micro"
    OBJ_FILE="$OUT_DIR/bitnet_ternary_scheduler.o"

    echo "   Type    : BitNet ternary microbenchmark"
    echo "   Dims    : M=$DIM_M  K=$DIM_K (→ $K_TERNARY ternary)"
    echo "   Input   : $in_bytes bytes ($in_dwords dwords)"
    echo "   Output  : $out_bytes bytes ($out_dwords dwords)"

    mkdir -p "$OUT_DIR"
    compile_chess "$KERNEL_DIR/bitnet_ternary_scheduler.cpp" "$OBJ_FILE"

    local MLIR_FILE="$OUT_DIR/design.mlir"
    echo ""
    echo "[2/3] Generating MLIR design..."

    cat > "$MLIR_FILE" << MLIREOF
// Auto-generated MLIR — BitNet ternary microbenchmark
// bitnet_ternary_micro: 32 rows, K=256 ternary per chunk

module {
  aie.device(npu2) {
    %shim = aie.tile(${SHIM_COL}, 0)
    %core = aie.tile(${CORE_COL}, 2)

    // case marker ${NAME}-bitnet-micro

    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 1, %shim, DMA : 0)

    func.func private @${KERNEL_ENTRY}(
      memref<${in_dwords}xi32>,
      memref<${out_elems}xbf16>
    ) attributes {link_with = "bitnet_ternary_scheduler.o"}

    %buf_in  = aie.buffer(%core) {sym_name = "buf_in"}  : memref<${in_dwords}xi32>
    %buf_out = aie.buffer(%core) {sym_name = "buf_out"} : memref<${out_elems}xbf16>

    %core_in_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "core_in_empty"}
    %core_in_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "core_in_full"}
    %core_out_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "core_out_empty"}
    %core_out_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "core_out_full"}

    %core_body = aie.core(%core) {
      aie.use_lock(%core_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%core_out_empty, AcquireGreaterEqual, 1)
      func.call @${KERNEL_ENTRY}(%buf_in, %buf_out) : (memref<${in_dwords}xi32>, memref<${out_elems}xbf16>) -> ()
      aie.use_lock(%core_out_full, Release, 1)
      aie.use_lock(%core_in_empty, Release, 1)
      aie.end
    }

    %mem = aie.mem(%core) {
      %dma_in = aie.dma_start(S2MM, 0, ^in_loop, ^out_start)
    ^in_loop:
      aie.use_lock(%core_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<${in_dwords}xi32>, 0, ${in_dwords}) {bd_id = 0 : i32}
      aie.use_lock(%core_in_full, Release, 1)
      aie.next_bd ^in_loop
    ^out_start:
      %dma_out = aie.dma_start(MM2S, 0, ^out_loop, ^end)
    ^out_loop:
      aie.use_lock(%core_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<${out_elems}xbf16>, 0, ${out_elems}) {bd_id = 1 : i32}
      aie.use_lock(%core_out_empty, Release, 1)
      aie.next_bd ^out_loop
    ^end:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<${in_dwords}xi32>, %arg1: memref<${out_dwords}xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = ${out_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = ${in_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D024 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = ${SHIM_COL} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
MLIREOF

    echo "  ✅ MLIR written: $MLIR_FILE"
    run_aiecc "$MLIR_FILE" "$OUT_DIR/design.xclbin" "$OUT_DIR/design.insts" "$OUT_DIR"
}

# ════════════════════════════════════════════════════════════
#  bitnet_scheduler — Full layer scheduler (single tile test)
#  6 ping-pong buffers + scalar config args
# ════════════════════════════════════════════════════════════

build_bitnet_scheduler() {
    # Buffer sizing: per-buffer = 10 chunks × (64B ternary + 64B scale + 256 bf16 act)
    # Weight buffer: 10 × 128 = 1280 bytes (320 dwords)
    # Act buffer:    10 × 512 = 5120 bytes (1280 dwords)
    # Record buffer: 28 records × 17 dwords = 476 dwords (1904 bytes)
    local CHUNKS=10
    local RECORDS=28
    local wt_dwords=$(( CHUNKS * 128 / 4 ))        # 320
    local act_dwords=$(( CHUNKS * 512 / 4 ))       # 1280
    local rec_dwords=$(( RECORDS * 17 ))           # 476

    # Use largest buffer size for uniformity (act = 1280 dwords)
    local buf_dwords=$act_dwords
    local in_dwords=$buf_dwords   # largest single-buffer size
    local out_dwords=$rec_dwords  # record output size

    KERNEL_ENTRY="bitnet_ternary_layer_scheduler"
    OBJ_FILE="$OUT_DIR/bitnet_ternary_scheduler.o"

    echo "   Type    : BitNet full layer scheduler"
    echo "   Dims    : M=$DIM_M  K=$DIM_K (→ $K_TERNARY ternary)"
    echo "   Chunks  : $CHUNKS  Records: $RECORDS"
    echo "   Buf sz  : $buf_dwords dwords ($(( buf_dwords * 4 )) bytes)"
    echo "   Out dir : $OUT_DIR"

    mkdir -p "$OUT_DIR"
    compile_chess "$KERNEL_DIR/bitnet_ternary_scheduler.cpp" "$OBJ_FILE"

    local MLIR_FILE="$OUT_DIR/design.mlir"
    echo ""
    echo "[2/3] Generating MLIR design..."

    cat > "$MLIR_FILE" << MLIREOF
// Auto-generated MLIR — BitNet full layer scheduler (single-tile test)
// bitnet_ternary_layer_scheduler: 6 ping-pong buffers + config

module {
  aie.device(npu2) {
    %shim = aie.tile(${SHIM_COL}, 0)
    %core = aie.tile(${CORE_COL}, 2)

    // case marker ${NAME}-bitnet-scheduler

    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 1, %shim, DMA : 0)

    func.func private @${KERNEL_ENTRY}(
      memref<${buf_dwords}xi32>,
      memref<${buf_dwords}xi32>,
      memref<${buf_dwords}xi32>,
      memref<${buf_dwords}xi32>,
      memref<${buf_dwords}xi32>,
      memref<${buf_dwords}xi32>,
      i32, i32, i32, i32
    ) attributes {link_with = "bitnet_ternary_scheduler.o"}

    // ── 6 ping-pong buffers ─────────────────────────────
    %wt_ping  = aie.buffer(%core) {sym_name = "wt_ping"}  : memref<${buf_dwords}xi32>
    %wt_pong  = aie.buffer(%core) {sym_name = "wt_pong"}  : memref<${buf_dwords}xi32>
    %act_ping = aie.buffer(%core) {sym_name = "act_ping"} : memref<${buf_dwords}xi32>
    %act_pong = aie.buffer(%core) {sym_name = "act_pong"} : memref<${buf_dwords}xi32>
    %rec_ping = aie.buffer(%core) {sym_name = "rec_ping"} : memref<${buf_dwords}xi32>
    %rec_pong = aie.buffer(%core) {sym_name = "rec_pong"} : memref<${buf_dwords}xi32>

    // ── Locks ──────────────────────────────────────────
    %in_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "in_empty"}
    %in_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "in_full"}
    %out_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "out_empty"}
    %out_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "out_full"}

    // ── Core body ──────────────────────────────────────
    %core_body = aie.core(%core) {
      aie.use_lock(%in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%out_empty, AcquireGreaterEqual, 1)

      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c7 = arith.constant 7 : i32

      func.call @${KERNEL_ENTRY}(%wt_ping, %wt_pong, %act_ping, %act_pong, %rec_ping, %rec_pong, %c0, %c0, %c1, %c7)
        : (memref<${buf_dwords}xi32>, memref<${buf_dwords}xi32>, memref<${buf_dwords}xi32>, memref<${buf_dwords}xi32>, memref<${buf_dwords}xi32>, memref<${buf_dwords}xi32>, i32, i32, i32, i32) -> ()

      aie.use_lock(%out_full, Release, 1)
      aie.use_lock(%in_empty, Release, 1)
      aie.end
    }

    // ── DMA engines ────────────────────────────────────
    %mem = aie.mem(%core) {
      %dma_in = aie.dma_start(S2MM, 0, ^in_loop, ^out_start)
    ^in_loop:
      aie.use_lock(%in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%wt_ping : memref<${buf_dwords}xi32>, 0, ${buf_dwords}) {bd_id = 0 : i32}
      aie.use_lock(%in_full, Release, 1)
      aie.next_bd ^in_loop
    ^out_start:
      %dma_out = aie.dma_start(MM2S, 0, ^out_loop, ^end)
    ^out_loop:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%rec_ping : memref<${buf_dwords}xi32>, 0, ${out_dwords}) {bd_id = 1 : i32}
      aie.use_lock(%out_empty, Release, 1)
      aie.next_bd ^out_loop
    ^end:
      aie.end
    }

    // ── Runtime sequence ───────────────────────────────
    aie.runtime_sequence(%arg0: memref<${buf_dwords}xi32>, %arg1: memref<${out_dwords}xi32>) {
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = ${out_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D004 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = ${buf_dwords} : i32, buffer_offset = 0 : i32,
        burst_length = 64 : i32, column = ${SHIM_COL} : i32,
        d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32,
        d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32,
        d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32,
        enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32,
        lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32,
        lock_rel_id = 0 : i32, lock_rel_val = 0 : i32,
        next_bd = 0 : i32, out_of_order_id = 0 : i32,
        packet_id = 0 : i32, packet_type = 0 : i32,
        row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 0x4001D024 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.sync {channel = 0 : i32, column = ${SHIM_COL} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
MLIREOF

    echo "  ✅ MLIR written: $MLIR_FILE"
    run_aiecc "$MLIR_FILE" "$OUT_DIR/design.xclbin" "$OUT_DIR/design.insts" "$OUT_DIR"
}

# ════════════════════════════════════════════════════════════
#  Dispatch
# ════════════════════════════════════════════════════════════

case "$KERNEL_TYPE" in
    mm_ternary)       build_mm_ternary ;;
    bitnet_micro)     build_bitnet_micro ;;
    bitnet_scheduler) build_bitnet_scheduler ;;
    *)
        echo "ERROR: Unknown kernel type '$KERNEL_TYPE'"
        echo "Valid types: mm_ternary, bitnet_micro, bitnet_scheduler"
        exit 1
        ;;
esac
