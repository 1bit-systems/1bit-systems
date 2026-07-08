#!/bin/bash
# build_all_ternary.sh — Build all native ternary NPU kernel variants
#
# Three kernel types, all at 32×64×128 (microbenchmark dimensions):
#   mm_ternary       — native ternary GEMM, scalar output per row
#   bitnet_micro     — scheduler microbenchmark, 32-row BF16 output
#   bitnet_scheduler — full layer scheduler, multi-phase with ping-pong
#
# For real layer sizes (K≫64, M≫32), use the multi-tile Python MLIR
# generator in torch2aie/examples/bitnet-decode-layer/.
#
# Usage:
#   source engine/npu/build/env.sh
#   bash engine/npu/build/build_all_ternary.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build_ternary_xclbin.sh"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Building ALL Native Ternary NPU Kernels                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

FAILED=()

build_one() {
    local name="$1"
    local type="$2"

    echo ""
    echo "─── $name ($type) ───"
    if bash "$BUILD_SCRIPT" "$name" "$type" 32 64 128; then
        echo "   ✅ $name OK"
    else
        echo "   ❌ $name FAILED"
        FAILED+=("$name")
    fi
}

# ── All three kernel types ────────────────────────────────

build_one "ternary"          "mm_ternary"
build_one "bitnet_micro"     "bitnet_micro"
build_one "scheduler"        "bitnet_scheduler"

# ── Summary ──────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "║   ✅ ALL KERNELS BUILT SUCCESSFULLY                         ║"
else
    echo "║   ❌ FAILED: ${FAILED[*]}"
fi
echo "║                                                            ║"
echo "║   Outputs in: engine/npu/build/build/                       ║"
echo "║                                                            ║"
echo "║   To test (on NPU hardware):                                ║"
echo "║     ./engine/npu/tests/test_ternary_npu                     ║"
echo "║       ternary/design.xclbin ternary/design.insts            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
