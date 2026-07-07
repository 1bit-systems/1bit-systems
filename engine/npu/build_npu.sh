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

# Fused engine (self-contained, no model variants)
FUSED_SRC="$SRCDIR/src/npu_engine_fused.cpp"
FUSED_BIN="$BUILDDIR/npu_engine_fused"
echo ""
echo "--- fused engine -> $FUSED_BIN ---"
$CXX $CXXFLAGS -o "$FUSED_BIN" "$FUSED_SRC" $LIBS
ls -lh "$FUSED_BIN"

# Split engine (server mode with component xclbins)
SPLIT_SRC="$SRCDIR/src/npu_engine_split.cpp"
SPLIT_BIN="$BUILDDIR/npu_engine_split"
echo ""
echo "--- split engine -> $SPLIT_BIN ---"
$CXX $CXXFLAGS -o "$SPLIT_BIN" "$SPLIT_SRC" "$DEQUANT_O" $LIBS
ls -lh "$SPLIT_BIN"

# Fused pipeline test (QKV→Attention→FFN integration test)
PIPE_SRC="$SRCDIR/src/npu_fused_pipeline.cpp"
PIPE_BIN="$BUILDDIR/npu_fused_pipeline"
echo ""
echo "--- fused pipeline test -> $PIPE_BIN ---"
$CXX $CXXFLAGS -o "$PIPE_BIN" "$PIPE_SRC" $LIBS
ls -lh "$PIPE_BIN"

echo ""
echo "=== All builds complete ==="
ls -lh "$BUILDDIR"/npu_engine*
echo ""
echo "Windows build: cmake -B build -S . -DXRT_DIR=\"C:/Program Files/AMD/XRT\""
