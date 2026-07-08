#!/bin/bash
# env.sh — Source this to activate the Chess/MLIR toolchain for NPU kernel builds.
#
# Usage: source engine/npu/build/env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TORCH2AIE_ROOT="${TORCH2AIE_ROOT:-$HOME/torch2aie}"
TOOLCHAIN_DIR="${TOOLCHAIN_DIR:-$TORCH2AIE_ROOT/toolchain}"
AIETOOLS_DIR="${AIETOOLS_DIR:-$TOOLCHAIN_DIR/aietools}"
MLIR_AIE_DIR="${MLIR_AIE_DIR:-$TOOLCHAIN_DIR/mlir_aie}"

# Verify toolchain exists
if [ ! -d "$TOOLCHAIN_DIR" ]; then
    echo "ERROR: Toolchain not found at $TOOLCHAIN_DIR" >&2
    echo "Set TORCH2AIE_ROOT or TOOLCHAIN_DIR to the correct path." >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -f "$TOOLCHAIN_DIR/bin/xchesscc_wrapper" ]; then
    echo "ERROR: xchesscc_wrapper not found at $TOOLCHAIN_DIR/bin/xchesscc_wrapper" >&2
    return 1 2>/dev/null || exit 1
fi

# PATH
export PATH="$TOOLCHAIN_DIR/bin:$MLIR_AIE_DIR/bin:$PATH"

# Include paths
export CPLUS_INCLUDE_PATH="$AIETOOLS_DIR/include:$MLIR_AIE_DIR/include:$MLIR_AIE_DIR/include/aie_kernels:$MLIR_AIE_DIR/include/aie_kernels/aie2p${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# Library paths
export LD_LIBRARY_PATH="$TOOLCHAIN_DIR/sysroot/usr/lib64:$TOOLCHAIN_DIR/mlir_aie.libs:$MLIR_AIE_DIR/lib:$AIETOOLS_DIR/lib:$LD_LIBRARY_PATH"

# Export for other scripts
export TORCH2AIE_ROOT TOOLCHAIN_DIR AIETOOLS_DIR MLIR_AIE_DIR SCRIPT_DIR REPO_ROOT

echo "✅ NPU toolchain activated"
echo "   Toolchain : $TOOLCHAIN_DIR"
echo "   AIETOOLS  : $AIETOOLS_DIR"
echo "   MLIR-AIE  : $MLIR_AIE_DIR"
echo "   Compiler  : $(which xchesscc_wrapper)"
