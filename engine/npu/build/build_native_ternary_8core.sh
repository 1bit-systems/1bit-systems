#!/bin/bash
# build_native_ternary_8core.sh — Build 8-core native ternary NPU xclbin
#
# Uses n1_core_native_ternary_8core.py (Python MLIR generator with object_fifo
# dataflow) + Chess C++ kernel (mm_ternary_32x64x128.cpp) → xclbin.
#
# This is the multi-tile version: 8 AIE cores in a 1×8 grid.
# Each core processes M/8 weight rows against the full activation vector,
# producing M/8 bf16 scalars. Total output: M bf16 values.
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_native_ternary_8core.sh [M] [K] [name]
#
# Defaults: M=32, K=64, name=ternary_8core

set -euo pipefail

TOTAL_M="${1:-32}"
DIM_K="${2:-64}"
NAME="${3:-ternary_8core}"

# 8 cores → each core processes TOTAL_M/8 rows
PER_CORE_M=$(( TOTAL_M / 8 ))

if [ $(( TOTAL_M % 8 )) -ne 0 ]; then
    echo "ERROR: TOTAL_M=$TOTAL_M must be a multiple of 8"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/../kernel" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
OUT_DIR="$BUILD_DIR/$NAME"

# Source environment if not already done
if [ -z "${TOOLCHAIN:-}" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

# Toolchain (matches env.sh exported vars)
CC="${TOOLCHAIN}/bin/xchesscc_wrapper"
AIECC="${TOOLCHAIN}/mlir_aie/bin/aiecc.py"
AIECC_PYTHON="${TOOLCHAIN}/../.venv/bin/python3"
AIECC_PYTHONPATH="${MLIR_AIE_DIR}/python"
GEN_PYTHON="${AIECC_PYTHON}"

K_TERNARY=$(( DIM_K * 4 ))
# Kernel function name is hardcoded in mm_ternary_32x64x128.cpp
KERNEL_ENTRY="mm_ternary_32x64x128"
OBJ_FILE="$OUT_DIR/${KERNEL_ENTRY}.o"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Native Ternary 8-Core xclbin Builder"
echo "  total M=$TOTAL_M  per_core M=$PER_CORE_M  K_packed=$DIM_K  (K_ternary=$K_TERNARY)"
echo "  Name: $NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "$OUT_DIR"

# ── Step 1: Compile Chess kernel ────────────────────────
echo ""
echo "[1/3] Compiling mm_ternary kernel with Chess C++..."

$CC aie2p \
    -I"$AIETOOLS_DIR/include" \
    -I"$MLIR_AIE_DIR/include" \
    -I"$MLIR_AIE_DIR/include/aie_kernels" \
    -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
    -DDIM_M="$PER_CORE_M" \
    -DDIM_K_PACKED="$DIM_K" \
    -DDIM_N=128 \
    -c "$SCRIPT_DIR/../../../1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp" \
    -o "$OBJ_FILE"

echo "  ✅ Kernel built: $(du -h "$OBJ_FILE" | cut -f1)"

# ── Step 2: Generate MLIR via Python ────────────────────
echo ""
echo "[2/3] Generating 8-core MLIR (object_fifo dataflow)..."

MLIR_FILE="$OUT_DIR/design.mlir"

PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
    $GEN_PYTHON \
    "$KERNEL_DIR/n1_core_native_ternary_8core.py" \
    -M "$TOTAL_M" -K "$DIM_K" \
    > "$MLIR_FILE"

echo "  ✅ MLIR: $(wc -l < "$MLIR_FILE") lines"

# ── Step 3: Build xclbin via aiecc ─────────────────────
echo ""
echo "[3/3] Compiling MLIR → xclbin (aiecc)..."
echo "  This may take several minutes..."

# Pre-stage kernel .o for aiecc
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
echo " ✅ 8-Core Native Ternary Build Complete!"
echo "   xclbin : $OUT_DIR/$NAME.xclbin  ($(stat -c%s "$OUT_DIR/$NAME.xclbin" 2>/dev/null || echo '?'))"
echo "   insts  : $OUT_DIR/insts_${NAME}.txt"
echo "   kernel : $OBJ_FILE"
echo "   cores  : 8 (1 row × 8 columns)"
echo ""
echo "  To test (on NPU hardware):"
echo "    ./engine/npu/tests/test_ternary_npu \\"
echo "      $OUT_DIR/$NAME.xclbin \\"
echo "      $OUT_DIR/insts_${NAME}.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
