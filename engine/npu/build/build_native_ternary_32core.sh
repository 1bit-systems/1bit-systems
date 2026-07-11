#!/bin/bash
# build_native_ternary_32core.sh — Build 32-core native ternary NPU xclbin
#
# 4 rows × 8 columns = 32 AIE cores. Each row broadcasts one flat buffer
# (containing M/4 rows of weights) to all 8 cores in that row.
# Each core picks its slice via row_start/num_rows kernel params.
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_native_ternary_32core.sh [M] [K] [name]
#
# Defaults: M=128, K=64, name=ternary_32core (4×8 grid; K=512 for oneshot)

set -euo pipefail

TOTAL_M="${1:-128}"
DIM_K="${2:-64}"
NAME="${3:-ternary_32core}"

if [ $(( TOTAL_M % 32 )) -ne 0 ]; then
    echo "ERROR: TOTAL_M=$TOTAL_M must be a multiple of 32"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/../kernel" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
OUT_DIR="$BUILD_DIR/$NAME"

if [ -z "${TOOLCHAIN:-}" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

# Toolchain (matches env.sh exported vars)
CC="${TOOLCHAIN}/bin/xchesscc_wrapper"
# Use locally-built aiecc (toolchain's aie-opt lacks contiguous shim-DMA optimization)
LOCAL_AIECC="/home/bcloud/mlir-aie/build/bin/aiecc"
if [ -x "$LOCAL_AIECC" ]; then
  AIECC="$LOCAL_AIECC"
else
  AIECC="${TOOLCHAIN}/mlir_aie/bin/aiecc.py"
fi
AIECC_PYTHON="${TOOLCHAIN}/../.venv/bin/python3"
AIECC_PYTHONPATH="${MLIR_AIE_DIR}/python"
GEN_PYTHON="${AIECC_PYTHON}"

K_TERNARY=$(( DIM_K * 4 ))
KERNEL_ENTRY="mm_ternary_32x64x128"
OBJ_FILE="$OUT_DIR/${KERNEL_ENTRY}.o"

PER_CORE_M=$(( TOTAL_M / 32 ))   # e.g. 128/32 = 4
PER_COL_M=$(( TOTAL_M / 8 ))     # e.g. 128/8 = 16 rows per column buffer

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Native Ternary 32-Core xclbin Builder"
echo "  total M=$TOTAL_M  per_core M=$PER_CORE_M  per_col M=$PER_COL_M"
echo "  K_packed=$DIM_K  (K_ternary=$K_TERNARY)"
echo "  Cores: 4 rows × 8 columns = 32"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$OUT_DIR"

# ── Step 1: Compile Chess kernel ────────────────────────
# Kernel is compiled with DIM_M=per_row_M so the buffer holds all rows
# for the broadcast. Each core picks its slice via row_start/num_rows.
echo ""
echo "[1/3] Compiling mm_ternary kernel (DIM_M=$PER_COL_M)..."

$CC aie2p \
    -I"$AIETOOLS_DIR/include" \
    -I"$MLIR_AIE_DIR/include" \
    -I"$MLIR_AIE_DIR/include/aie_kernels" \
    -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
    -DDIM_M="$PER_COL_M" \
    -DDIM_K_PACKED="$DIM_K" \
    -DDIM_N=128 \
    -c "$SCRIPT_DIR/../../../1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp" \
    -o "$OBJ_FILE"

echo "  ✅ Kernel built: $(du -h "$OBJ_FILE" | cut -f1)"

# ── Step 2: Generate 32-core MLIR ───────────────────────
echo ""
echo "[2/3] Generating 32-core MLIR (4×8 grid, row broadcast)..."

MLIR_FILE="$OUT_DIR/design.mlir"

PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
    $GEN_PYTHON \
    "$KERNEL_DIR/n1_core_native_ternary_32core.py" \
    -M "$TOTAL_M" -K "$DIM_K" \
    > "$MLIR_FILE"

echo "  ✅ MLIR: $(wc -l < "$MLIR_FILE") lines"

# ── Step 3: Build xclbin ────────────────────────────────
echo ""
echo "[3/3] Compiling MLIR → xclbin (aiecc)..."
echo "  This may take several minutes..."

mkdir -p "$OUT_DIR/design.mlir.prj"
cp "$OBJ_FILE" "$OUT_DIR/design.mlir.prj/"

cd "$OUT_DIR"
PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
    $AIECC_PYTHON \
    $AIECC \
    -v \
    -j4 \
    --aietools="$AIETOOLS_DIR" \
    --no-compile-host \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin \
    --xclbin-name="$NAME.xclbin" \
    --xclbin-kernel-name=MLIR_AIE \
    --aie-generate-npu-insts \
    --npu-insts-name="insts_${NAME}.txt" \
    "$MLIR_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅ 32-Core Native Ternary Build Complete!"
echo "   xclbin : $OUT_DIR/$NAME.xclbin"
echo "   insts  : $OUT_DIR/insts_${NAME}.txt"
echo "   kernel : $OBJ_FILE"
echo "   cores  : 32 (4 rows × 8 columns)"
echo "   M      : $TOTAL_M rows ($PER_CORE_M per core)"
echo ""
echo "  To test:"
echo "    ./engine/npu/tests/test_ternary_npu \\"
echo "      $OUT_DIR/$NAME.xclbin \\"
echo "      $OUT_DIR/insts_${NAME}.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
