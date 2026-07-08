#!/bin/bash
# build_ternary_xclbin.sh — One-command build: chess compile → MLIR → xclbin
#
# Builds the native ternary AIE kernel into an NPU xclbin for Strix Halo.
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_ternary_xclbin.sh [name] [M] [K] [N]
#
# Defaults: name=ternary, M=32, K=64, N=128

set -euo pipefail

NAME="${1:-ternary}"
DIM_M="${2:-32}"
DIM_K="${3:-64}"
DIM_N="${4:-128}"

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
AIECC="${TOOLCHAIN_DIR}/bin/aiecc"

# Derived dimensions
K_TERNARY=$(( DIM_K * 4 ))              # 64*4 = 256 ternary values

# Buffer sizes in bytes
WEIGHT_BYTES=$(( DIM_M * DIM_K ))        # M × K packed bytes
SCALE_BYTES=$(( DIM_M * 2 ))             # M bf16 = M*2 bytes
ACT_BYTES=$(( K_TERNARY * 2 ))           # 256 bf16 = 512 bytes
OUTPUT_BYTES=$(( DIM_M * 2 ))            # M bf16 = M*2 bytes
OUTPUT_ELEMS=$(( DIM_M ))                 # M bf16 values

# Total input bytes and dwords
INPUT_BYTES=$(( WEIGHT_BYTES + SCALE_BYTES + ACT_BYTES ))
INPUT_DWORDS=$(( (INPUT_BYTES + 3) / 4 ))
OUTPUT_DWORDS=$(( (OUTPUT_BYTES + 3) / 4 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Building ternary NPU kernel"
echo "   Name    : $NAME"
echo "   Dims    : M=$DIM_M  K=$DIM_K (→ $K_TERNARY ternary)  N=$DIM_N"
echo "   Input   : $INPUT_BYTES bytes ($INPUT_DWORDS dwords)"
echo "   Output  : $OUTPUT_BYTES bytes ($OUTPUT_DWORDS dwords)"
echo "   Out dir : $OUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$OUT_DIR"

# ── Step 1: Compile the Chess C++ kernel ────────────────────────
KERNEL_SRC="$KERNEL_DIR/mm_ternary_32x64x128.cpp"
OBJ_FILE="$OUT_DIR/mm_ternary_${DIM_M}x${DIM_K}x${DIM_N}.o"

echo ""
echo "[1/3] Compiling kernel with Chess C++ (xchesscc)..."
echo "  Source: $KERNEL_SRC"
echo "  Object: $OBJ_FILE"

$CC aie2p \
    -I"$AIETOOLS_DIR/include" \
    -I"$MLIR_AIE_DIR/include" \
    -I"$MLIR_AIE_DIR/include/aie_kernels" \
    -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
    -DDIM_M="$DIM_M" \
    -DDIM_K_PACKED="$DIM_K" \
    -DDIM_N="$DIM_N" \
    -c "$KERNEL_SRC" \
    -o "$OBJ_FILE"

echo "  ✅ Kernel object built: $(du -h "$OBJ_FILE" | cut -f1)"

# ── Step 2: Generate MLIR design ────────────────────────────────
MLIR_FILE="$OUT_DIR/design.mlir"

echo ""
echo "[2/3] Generating MLIR design..."

SHIM_COL=2
CORE_COL=2

cat > "$MLIR_FILE" << MLIREOF
// Auto-generated MLIR for ternary NPU kernel (single-tile microbenchmark)
// mm_ternary_${DIM_M}x${DIM_K}x${DIM_N} — native ternary AIE2P matrix multiply

module {
  aie.device(npu2) {
    %shim = aie.tile(${SHIM_COL}, 0)
    %core = aie.tile(${CORE_COL}, 2)

    // case marker ternary-kernel-${DIM_M}x${DIM_K}x${DIM_N}

    // Data flows: shim→core (input), core→shim (output)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    aie.flow(%core, DMA : 1, %shim, DMA : 0)

    // External kernel — links the Chess-compiled object
    func.func private @mm_ternary_${DIM_M}x${DIM_K}x${DIM_N}(
      memref<${INPUT_DWORDS}xi32>,
      memref<${OUTPUT_ELEMS}xbf16>
    ) attributes {link_with = "mm_ternary_${DIM_M}x${DIM_K}x${DIM_N}.o"}

    // ── Core tile buffers ─────────────────────────────────
    %buf_in = aie.buffer(%core) {sym_name = "buf_in"} : memref<${INPUT_DWORDS}xi32>
    %buf_out = aie.buffer(%core) {sym_name = "buf_out"} : memref<${OUTPUT_ELEMS}xbf16>

    // Locks for ping-pong (single-shot: use count=1)
    %core_in_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "core_in_empty"}
    %core_in_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "core_in_full"}
    %core_out_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "core_out_empty"}
    %core_out_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "core_out_full"}

    // ── Core body ────────────────────────────────────────
    %core_body = aie.core(%core) {
      aie.use_lock(%core_in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%core_out_empty, AcquireGreaterEqual, 1)

      func.call @mm_ternary_${DIM_M}x${DIM_K}x${DIM_N}(%buf_in, %buf_out)
        : (memref<${INPUT_DWORDS}xi32>, memref<${OUTPUT_ELEMS}xbf16>) -> ()

      aie.use_lock(%core_out_full, Release, 1)
      aie.use_lock(%core_in_empty, Release, 1)

      aie.end
    }

    // ── Core DMA engines ─────────────────────────────────
    %mem = aie.mem(%core) {
      // DMA channel 0: input (S2MM from shim)
      %dma_in = aie.dma_start(S2MM, 0, ^in_loop, ^out_start)
    ^in_loop:
      aie.use_lock(%core_in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_in : memref<${INPUT_DWORDS}xi32>, 0, ${INPUT_DWORDS}) {bd_id = 0 : i32}
      aie.use_lock(%core_in_full, Release, 1)
      aie.next_bd ^in_loop

    ^out_start:
      // DMA channel 1: output (MM2S to shim)
      %dma_out = aie.dma_start(MM2S, 0, ^out_loop, ^end)
    ^out_loop:
      aie.use_lock(%core_out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_out : memref<${OUTPUT_ELEMS}xbf16>, 0, ${OUTPUT_ELEMS}) {bd_id = 1 : i32}
      aie.use_lock(%core_out_empty, Release, 1)
      aie.next_bd ^out_loop

    ^end:
      aie.end
    }

    // ── Runtime sequence (host → NPU shim DMA setup) ─────
    aie.runtime_sequence(%arg0: memref<${INPUT_DWORDS}xi32>, %arg1: memref<${OUTPUT_DWORDS}xi32>) {
      // BD 0: output (S2MM direction: core→shim, shim receives)
      aiex.npu.writebd {bd_id = 0 : i32, buffer_length = ${OUTPUT_DWORDS} : i32, buffer_offset = 0 : i32,
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

      // BD 1: input (MM2S direction: shim→core, shim sends)
      aiex.npu.writebd {bd_id = 1 : i32, buffer_length = ${INPUT_DWORDS} : i32, buffer_offset = 0 : i32,
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

      // Push queues: output (S2MM), then input (MM2S)
      aiex.npu.push_queue(${SHIM_COL}, 0, S2MM : 0) {bd_id = 0 : i32, issue_token = true, repeat_count = 0 : i32}
      aiex.npu.push_queue(${SHIM_COL}, 0, MM2S : 0) {bd_id = 1 : i32, issue_token = true, repeat_count = 0 : i32}

      // Sync: wait for completion
      aiex.npu.sync {channel = 0 : i32, column = ${SHIM_COL} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
  }
}
MLIREOF

echo "  ✅ MLIR written: $MLIR_FILE"

# ── Step 3: Compile MLIR to xclbin ──────────────────────────────
XCLBIN="$OUT_DIR/design.xclbin"
INSTS="$OUT_DIR/design.insts"

echo ""
echo "[3/3] Compiling MLIR → xclbin (aiecc)..."
echo "  This may take several minutes..."

# Run aiecc from OUT_DIR so relative link_with paths resolve
cd "$OUT_DIR"

$AIECC \
    -v \
    -j4 \
    --aietools="$AIETOOLS_DIR" \
    --no-compile-host \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --xclbin-name="$XCLBIN" \
    --xclbin-kernel-name=MLIR_AIE \
    --aie-generate-npu-insts \
    --npu-insts-name="$INSTS" \
    "$MLIR_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ Build complete!"
echo "   xclbin : $XCLBIN"
echo "   insts  : $INSTS"
echo "   kernel : $OBJ_FILE"
echo ""
echo " Usage (standalone test):"
echo "   ./npu_engine_universal ternary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
