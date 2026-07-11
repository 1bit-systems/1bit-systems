#!/bin/bash
# build_npu.sh — Build all model variants of the universal NPU engine
set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$SRCDIR/build"
SRC="$SRCDIR/src/npu_engine_universal.cpp"
DEQUANT="$SRCDIR/src/dequant_q4nx.c"
DEQUANT_O="$BUILDDIR/dequant_q4nx.o"

# One-time: compile dequantizer
if [ ! -f "$DEQUANT_O" ] || [ "$DEQUANT" -nt "$DEQUANT_O" ]; then
    echo "gcc -c -O3 -o $DEQUANT_O $DEQUANT"
    gcc -c -O3 -o "$DEQUANT_O" "$DEQUANT"
fi

# XRT headers at /usr/include, libs at system default path
XRT_INC="/usr/include"

# Models to build
MODELS=(
    "qwen3_0_6b"
    "qwen3_8b"
    "qwen3_vl_4b"
    "llama"
    "gemma4_e2b"
)

CXX="${CXX:-g++}"
# XRT uses shared libs (must come AFTER source on command line)
LIBS="-lxrt_coreutil -lxrt_core -luuid -lm -ldl"
CXXFLAGS="-std=c++23 -O3 -I$SRCDIR/src -I$XRT_INC"

echo "=== Building NPU engine variants ==="
mkdir -p "$BUILDDIR"

for model in "${MODELS[@]}"; do
    binary="$BUILDDIR/npu_engine_$model"
    echo ""
    echo "--- $model -> $binary ---"
    $CXX -DMODEL_$model $CXXFLAGS -o "$binary" "$SRC" "$DEQUANT_O" $LIBS
    ls -lh "$binary"
done

# Also build a default (qwen3_0_6b) as npu_engine for backward compat
echo ""
echo "--- default (qwen3_0_6b) -> $BUILDDIR/npu_engine ---"
$CXX -DMODEL_qwen3_0_6b $CXXFLAGS -o "$BUILDDIR/npu_engine" "$SRC" "$DEQUANT_O" $LIBS
ls -lh "$BUILDDIR/npu_engine"

echo ""
echo "=== All builds complete ==="
ls -lh "$BUILDDIR"/npu_engine*
