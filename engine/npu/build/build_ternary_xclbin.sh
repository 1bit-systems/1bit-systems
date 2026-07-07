#!/bin/bash
# Build ternary (1.58-bit) xclbin for NPU inference
# Uses the same INT8 GEMM pipeline — ternary weights are dequantized to INT8
#
# Prerequisites:
#   source /opt/toolchain/env.sh  # MLIR-AIE + Vitis + Chess tools
#
# Usage:
#   ./build_ternary_xclbin.sh <model> <M> <K> <N>

set -e
MODEL=${1:-ternary}
M=${2:-128}
K=${3:-1024}
N=${4:-4096}

DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$DIR/build/ternary"
mkdir -p "$BUILD"

echo "=== Generating ternary MLIR ==="
python3 "$DIR/n1_core_ternary.py" -M "$M" -K "$K" -N "$N" > "$BUILD/ternary.mlir"

echo "=== Compiling xclbin ==="
# Same compilation pipeline as INT8 — the MLIR references the same kernel
cd "$BUILD"
aiecc.py --aie-targets=aiec autoregression \
    -I/opt/toolchain/include \
    ternary.mlir \
    "$DIR/build/int8/mm_32x64x128.o" \
    -o ternary.xclbin 2>&1 | tee build.log

echo "=== Generating insts ==="
python3 "$DIR/gen_mlir_v2.py" insts-only "$M" "$K" "$N" > insts_ternary.txt

echo ""
echo "=== Done ==="
echo "  xclbin: $BUILD/ternary.xclbin"
echo "  insts:  $BUILD/insts_ternary.txt"
echo ""
echo "To verify:"
echo "  python3 tools/q2_0_to_q4nx.py model.gguf model.q4nx"
echo "  ./npu_engine_universal model.q4nx"
