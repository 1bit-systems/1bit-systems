#!/bin/bash
# build_oneshot_xclbin.sh — Build single-shot native ternary xclbin
#
# Single AIE core, single-shot kernel (exits after one input).
# Host tiles M and K dimensions by dispatching multiple kernel calls.
# Each dispatch uses a fresh hw_context → no multi-dispatch crash.
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_oneshot_xclbin.sh

set -euo pipefail

M="${1:-32}"
K="${2:-64}"
NAME="${3:-ternary_oneshot}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/../kernel" && pwd)"
OUT_DIR="$SCRIPT_DIR/$NAME"

if [ -z "${TOOLCHAIN:-}" ]; then
    source "$SCRIPT_DIR/env.sh"
fi

CC="${TOOLCHAIN}/bin/xchesscc_wrapper"
AIECC="${TOOLCHAIN}/mlir_aie/bin/aiecc.py"
AIECC_PYTHON="${TOOLCHAIN}/../.venv/bin/python3"
AIECC_PYTHONPATH="${MLIR_AIE_DIR}/python"

KERNEL_SRC="$SCRIPT_DIR/../../../1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp"
[ ! -f "$KERNEL_SRC" ] && KERNEL_SRC="$KERNEL_DIR/mm_ternary_32x64x128.cpp"

KERNEL_ENTRY="mm_ternary_32x64x128"
OBJ_FILE="$OUT_DIR/${KERNEL_ENTRY}.o"

echo "=== Single-Shot Native Ternary xclbin (M=$M K_packed=$K) ==="
mkdir -p "$OUT_DIR"

echo "[1/3] Compiling kernel..."
$CC aie2p \
    -I"$AIETOOLS_DIR/include" -I"$MLIR_AIE_DIR/include" \
    -I"$MLIR_AIE_DIR/include/aie_kernels" -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
    -DDIM_M="$M" -DDIM_K_PACKED="$K" -DDIM_N=128 \
    -c "$KERNEL_SRC" -o "$OBJ_FILE" 2>&1 | tail -2
echo "  ✅ $(du -h "$OBJ_FILE" | cut -f1)"

echo "[2/3] Generating oneshot MLIR..."
MLIR_FILE="$OUT_DIR/design.mlir"
PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
    $AIECC_PYTHON "$KERNEL_DIR/n1_core_native_ternary_oneshot.py" \
    -M "$M" -K "$K" > "$MLIR_FILE"
echo "  ✅ $(wc -l < "$MLIR_FILE") lines"

echo "[3/3] aiecc → xclbin..."
mkdir -p "$OUT_DIR/design.mlir.prj"
cp "$OBJ_FILE" "$OUT_DIR/design.mlir.prj/"
cd "$OUT_DIR"
PYTHONPATH="${AIECC_PYTHONPATH}:${PYTHONPATH:-}" \
    $AIECC_PYTHON $AIECC -v -j4 \
    --aietools="$AIETOOLS_DIR" --no-compile-host \
    --alloc-scheme=basic-sequential --aie-generate-xclbin \
    --xclbin-name="${NAME}.xclbin" --xclbin-kernel-name=MLIR_AIE \
    --aie-generate-npu-insts --npu-insts-name="insts_${NAME}.txt" \
    "$MLIR_FILE" 2>&1 | tail -5

cp "${NAME}.xclbin" "$OUT_DIR/${NAME}.xclbin" 2>/dev/null || true
cp "insts_${NAME}.txt" "$OUT_DIR/insts_${NAME}.txt" 2>/dev/null || true

echo ""
echo "✅ Done: $OUT_DIR/${NAME}.xclbin ($(stat -c%s "$OUT_DIR/${NAME}.xclbin" 2>/dev/null || echo ?) bytes)"
echo "   insts: $OUT_DIR/insts_${NAME}.txt"
