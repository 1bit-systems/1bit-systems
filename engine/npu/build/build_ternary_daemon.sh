#!/bin/bash
# build_ternary_daemon.sh — Build the native ternary NPU daemon
#
# Usage:
#   source engine/npu/build/env.sh  (optional, for XRT paths)
#   bash engine/npu/build/build_ternary_daemon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR}"
SRC="$ENGINE_DIR/src/npu_ternaryd.cpp"
BIN="$BUILD_DIR/npu_ternaryd"

# XRT paths
XRT_INC="${XRT_INC:-/usr/include}"
XRT_LIB="${XRT_LIB:-/usr/lib}"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++23 -O3 -I$XRT_INC -I$ENGINE_DIR/.. -I$ENGINE_DIR/../../spec-decode/engine"
LDFLAGS="-L$XRT_LIB -lxrt_coreutil -lxrt_core -luuid -lm"

echo "=== Building native ternary NPU daemon ==="
echo "  Source: $SRC"
echo "  Binary: $BIN"
echo ""

$CXX $CXXFLAGS -o "$BIN" "$SRC" $LDFLAGS

echo ""
echo "✅ Build complete: $BIN"
ls -lh "$BIN"
echo ""
echo "Usage:"
echo "  $BIN model.ternary/ xclbin_dir/"
echo ""
echo "Protocol:"
echo "  echo '{\"tokens\":[1],\"max_new_tokens\":16}' | $BIN model.ternary/ xclbin_dir/"
