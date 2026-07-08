#!/bin/bash
# build_32c_oneshot.sh — Build single-shot 32-core native ternary xclbin
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env.sh"

M="${1:-128}"
K="${2:-64}"
NAME="${3:-ternary_32c_oneshot}"
OUT="$SCRIPT_DIR/$NAME"

PER_COL_M=$(( M / 8 ))
KERNEL_SRC="$SCRIPT_DIR/../../../1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp"
MLIR_GEN="$SCRIPT_DIR/../kernel/n1_core_native_ternary_32core_oneshot.py"

CC="${TOOLCHAIN}/bin/xchesscc_wrapper"
AIECC="${TOOLCHAIN}/mlir_aie/bin/aiecc.py"
GEN_PYTHON="${TOOLCHAIN}/../.venv/bin/python3"

echo "=== 32-Core Single-Shot xclbin (M=$M K_packed=$K) ==="
mkdir -p "$OUT"

echo "[1/3] Compiling kernel (DIM_M=$PER_COL_M)..."
$CC aie2p -I"$AIETOOLS_DIR/include" -I"$MLIR_AIE_DIR/include" \
  -I"$MLIR_AIE_DIR/include/aie_kernels" -I"$MLIR_AIE_DIR/include/aie_kernels/aie2p" \
  -DDIM_M="$PER_COL_M" -DDIM_K_PACKED="$K" -DDIM_N=128 \
  -c "$KERNEL_SRC" -o "$OUT/mm_ternary_32x64x128.o" 2>&1 | tail -2
echo "  ✅ $(du -h $OUT/mm_ternary_32x64x128.o | cut -f1)"

echo "[2/3] Generating single-shot MLIR..."
PYTHONPATH="${MLIR_AIE_DIR}/python" $GEN_PYTHON "$MLIR_GEN" -M $M -K $K > "$OUT/design.mlir"
echo "  ✅ $(wc -l < $OUT/design.mlir) lines"

echo "[3/3] aiecc → xclbin (this takes minutes)..."
mkdir -p "$OUT/design.mlir.prj"
cp "$OUT/mm_ternary_32x64x128.o" "$OUT/design.mlir.prj/"
cd "$OUT"
PYTHONPATH="${MLIR_AIE_DIR}/python" $GEN_PYTHON $AIECC -v -j4 \
  --aietools="$AIETOOLS_DIR" --no-compile-host --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --xclbin-name="${NAME}.xclbin" --xclbin-kernel-name=MLIR_AIE \
  --aie-generate-npu-insts --npu-insts-name="insts_${NAME}.txt" \
  design.mlir 2>&1 | tail -5

echo ""
echo "✅ Done: $OUT/${NAME}.xclbin ($(stat -c%s $OUT/${NAME}.xclbin) bytes)"
echo "   insts: $OUT/insts_${NAME}.txt"
