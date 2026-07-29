#!/bin/bash
# build_npu.sh — Build all model variants of the universal NPU engine
set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$SRCDIR/build"
SRC="$SRCDIR/src/npu_engine_universal.cpp"
DEQUANT="$SRCDIR/src/dequant_q4nx.cpp"
DEQUANT_O="$BUILDDIR/dequant_q4nx.o"
INSTR_GEN="$SRCDIR/src/gemm_npu_instructions.cpp"
INSTR_GEN_O="$BUILDDIR/gemm_npu_instructions.o"

# XRT headers at /usr/include, libs at system default path
XRT_INC="/usr/include"

# One-time: compile dequantizer
if [ ! -f "$DEQUANT_O" ] || [ "$DEQUANT" -nt "$DEQUANT_O" ]; then
    echo "gcc -c -O3 -o $DEQUANT_O $DEQUANT"
    gcc -c -O3 -o "$DEQUANT_O" "$DEQUANT"
fi

# One-time: compile NPU instruction generator
if [ ! -f "$INSTR_GEN_O" ] || [ "$INSTR_GEN" -nt "$INSTR_GEN_O" ]; then
    echo "g++ -c -std=c++23 -O3 -o $INSTR_GEN_O $INSTR_GEN"
    g++ -c -std=c++23 -O3 -fopenmp -I"$SRCDIR"/src -I"$SRCDIR"/include -I$XRT_INC \
        -o "$INSTR_GEN_O" "$INSTR_GEN"
fi

# Models to build
MODELS=(
    "qwen3_0_6b"
    "qwen3_8b"
    "qwen3_vl_4b"
    "llama"
    "gemma4_e2b"
    "qwen3_6_moe_35b"
    "qwen3_5_4b"
    "gemma4_e4b"
    "phi4_mini_4b"
    "nanbeige4_1_3b"
    "zr1"
)

CXX="${CXX:-g++}"
# XRT uses shared libs (must come AFTER source on command line)
LIBS=(-lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl)
CXXFLAGS=(-std=c++23 -O3 -fopenmp -I"$SRCDIR/src" -I"$SRCDIR/include" -I"$XRT_INC")

echo "=== Building NPU engine variants ==="
mkdir -p "$BUILDDIR"

for model in "${MODELS[@]}"; do
    binary="$BUILDDIR/npu_engine_$model"
    echo ""
    echo "--- $model -> $binary ---"
    $CXX "-DMODEL_$model" "${CXXFLAGS[@]}" -o "$binary" "$SRC" "$DEQUANT_O" "$INSTR_GEN_O" "${LIBS[@]}"
    ls -lh "$binary"
done

# Also build a default (qwen3_0_6b) as npu_engine for backward compat
echo ""
echo "--- default (qwen3_0_6b) -> $BUILDDIR/npu_engine ---"
$CXX -DMODEL_qwen3_0_6b "${CXXFLAGS[@]}" -o "$BUILDDIR/npu_engine" "$SRC" "$DEQUANT_O" "$INSTR_GEN_O" "${LIBS[@]}"
ls -lh "$BUILDDIR/npu_engine"

echo ""
echo "=== All builds complete ==="
ls -lh "$BUILDDIR"/npu_engine*
