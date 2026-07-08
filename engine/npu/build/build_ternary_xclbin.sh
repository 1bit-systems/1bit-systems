#!/bin/bash
# Build ternary (1.58-bit) xclbin for NPU inference
#
# Uses the proven INT8 GEMM pipeline — ternary weights {-1,0,+1} are
# dequantized to INT8 by tools/q2_0_to_q4nx.py. The NPU sees int8 weights
# and runs the same bit-exact GEMM.
#
# Prerequisites:
#   source engine/npu/build/env.sh
#
# Usage:
#   ./build_ternary_xclbin.sh [model] [M] [K] [N]

set -e
MODEL=${1:-ternary}
M=${2:-128}
K=${3:-1024}
N=${4:-4096}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$SCRIPT_DIR/ternary_${MODEL}"
mkdir -p "$BUILD"

# ---- toolchain paths ----
: ${AIETOOLS_DIR:=/home/bcloud/torch2aie/toolchain/aietools}
: ${MLIR_AIE_DIR:=/home/bcloud/torch2aie/toolchain/mlir_aie}
: ${TOOLCHAIN_BIN:=/home/bcloud/torch2aie/toolchain/bin}
AIECC=/home/bcloud/mlir-aie/build/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
KERNEL_SRC="${MLIR_AIE_DIR}/include/aie_kernels/aie2p/mm.cc"
GEN_PYTHON=/home/bcloud/torch2aie/.venv/bin/python

export PATH="${TOOLCHAIN_BIN}:${AIETOOLS_DIR}/bin:$PATH"
export PYTHONPATH="${MLIR_AIE_DIR}/python"

echo "=== Step 1: Compile i8→i32 kernel (DIM_M=32, DIM_K=64, DIM_N=128) ==="
xchesscc_wrapper aie2p -c \
  -I "${AIETOOLS_DIR}/include" \
  -I "${MLIR_AIE_DIR}/include" \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 \
  -I"${MLIR_AIE_DIR}/include/aie_kernels" \
  -Di8_i32_ONLY \
  "${KERNEL_SRC}" \
  -o "$BUILD/mm_32x64x128.o" 2>&1 | tail -2

echo "  Kernel: $(stat -c%s "$BUILD/mm_32x64x128.o") bytes"
echo "  Symbols: $(nm "$BUILD/mm_32x64x128.o" | grep ' T ' | awk '{print $3}' | tr '\n' ' ')"

echo ""
echo "=== Step 2: Generate ternary MLIR (${M}x${K}x${N}) ==="
"$GEN_PYTHON" \
  "$ENGINE_DIR/kernel/n1_core_ternary.py" \
  -M "$M" -K "$K" -N "$N" > "$BUILD/${MODEL}.mlir" 2>/dev/null

echo "  MLIR: $(wc -l < "$BUILD/${MODEL}.mlir") lines"

echo ""
echo "=== Step 3: Build xclbin ==="

# aiecc auto-creates ${MODEL}.mlir.prj/ and copies .o files from the
# working directory into it. Pre-stage the kernel .o there.
mkdir -p "$BUILD/${MODEL}.mlir.prj"
cp "$BUILD/mm_32x64x128.o" "$BUILD/${MODEL}.mlir.prj/"

cd "$BUILD"
"$AIECC" \
  --aietools="$AIETOOLS_DIR" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name="${MODEL}.xclbin" \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name="insts_${MODEL}.txt" \
  "${MODEL}.mlir" 2>&1 | tail -5

echo ""
echo "=== Done ==="
echo "  xclbin: $BUILD/${MODEL}.xclbin  ($(stat -c%s "$BUILD/${MODEL}.xclbin") bytes)"
echo "  insts:  $BUILD/insts_${MODEL}.txt"
echo ""
echo "To run with a ternary model:"
echo "  python3 tools/q2_0_to_q4nx.py model.gguf model.q4nx"
echo "  ./npu_engine_universal model.q4nx"
