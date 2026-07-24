#!/bin/bash
# build_npu_ternary.sh — Compile NPU ternary microkernels (TQ2/TQ1/Q1_0)
#
# Compiles the on-tile LUT-decode Chess microkernel for the given format.
# Xclbins are pre-built and checked into engine/npu/xclbins/ — this script
# copies them (or symlinks) rather than rebuilding, because the aiecc MLIR
# toolchain has a pre-existing version mismatch that prevents clean builds.
#
# To rebuild xclbins from source, fix the toolchain first (see docs/npu-ternary-roadmap.md)
# then use: make -C ~/torch2aie/examples/.../config1 M=128 K=... N=...
#
# Usage:
#   ./build_npu_ternary.sh <fmt> <tag> <H> <NH> <NKV> <HD> <IM> [M=128]
#
# Examples:
#   ./build_npu_ternary.sh tq2 qwen3_0_6b 1024 16 8 128 3072    # compile + copy xclbins
#   ./build_npu_ternary.sh tq2 --compile-only                    # just compile kernel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="${SCRIPT_DIR}/kernel"
XCLBIN_DIR="${SCRIPT_DIR}/xclbins"
TORCH2AIE="${HOME}/torch2aie"

export PATH="${TORCH2AIE}/toolchain/bin:$PATH"
export AIETOOLS_DIR="${TORCH2AIE}/toolchain"
export MLIR_AIE_DIR="${TORCH2AIE}/toolchain/mlir_aie"

AIE_INCLUDES="-I${TORCH2AIE}/toolchain/mlir_aie/include"
AIE_INCLUDES="${AIE_INCLUDES} -I${TORCH2AIE}/toolchain/mlir_aie/include/aie_kernels"
AIE_INCLUDES="${AIE_INCLUDES} -I${TORCH2AIE}/toolchain/mlir_aie/include/aie_kernels/aie2p"

# ─── Format table ────────────────────────────────────────────────
# name    kernel_src         prefix       bits_desc
FORMATS="tq2:mm_ternary_tq2.cc:ternary_tq2:TQ2 (2-bit)
tq1:mm_ternary_tq1.cc:ternary_tq1:TQ1 (1.58-bit)
q1:mm_binary_q1.cc:binary_q1:Q1_0 (1-bit)"

compile_kernel() {
    local src="$1"
    local src_path="${KERNEL_DIR}/${src}"
    if [ ! -f "$src_path" ]; then
        echo "  ❌ Kernel not found: $src_path"
        return 1
    fi
    local obj_path="/tmp/${src/.cc/.o}"
    if [ -f "$obj_path" ] && [ "$src_path" -ot "$obj_path" ]; then
        echo "  ✓ ${src} (cached: $(stat -c%s "$obj_path") bytes)"
        return 0
    fi
    echo "  Compiling ${src}..."
    xchesscc_wrapper aie2p -c "$src_path" ${AIE_INCLUDES} -o "$obj_path"
    echo "  ✓ ${src}: $(stat -c%s "$obj_path") bytes"
}

copy_xclbins() {
    local prefix="$1" tag="$2"
    local count=0
    for xclbin in "${XCLBIN_DIR}"/${prefix}_*_${tag}.xclbin; do
        if [ -f "$xclbin" ]; then count=$((count+1)); fi
    done
    if [ "$count" -ge 4 ]; then
        echo "  ✓ ${count} xclbins available for ${prefix}_${tag}"
    else
        echo "  ⚠ Only ${count}/4 xclbins found for ${prefix}_${tag}. Use pre-built from repo."
    fi
}

# ─── Parse args ──────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fmt:tq2|tq1|q1> <tag> <H> <NH> <NKV> <HD> <IM> [M=128]"
    echo "       $0 all                            # compile all kernels"
    echo "       $0 <fmt> --compile-only             # compile one kernel"
    echo ""
    echo "Kernels:"
    echo "$FORMATS" | while IFS=: read name src prefix desc; do
        echo "  $name  →  ${src}  (${desc})"
    done
    exit 1
fi

if [ "$1" = "all" ]; then
    echo "=== Compiling all NPU ternary kernels ==="
    echo "$FORMATS" | while IFS=: read name src prefix desc; do
        echo "  [$name] ${desc}"
        compile_kernel "$src"
    done
    echo "=== Done ==="
    exit 0
fi

FMT="$1"
shift

# Find format
FMT_LINE=$(echo "$FORMATS" | grep "^${FMT}:" || true)
if [ -z "$FMT_LINE" ]; then
    echo "Unknown format: $FMT"
    echo "Available: tq2, tq1, q1"
    exit 1
fi

KERNEL_SRC=$(echo "$FMT_LINE" | cut -d: -f2)
PREFIX=$(echo "$FMT_LINE" | cut -d: -f3)
FMT_NAME=$(echo "$FMT_LINE" | cut -d: -f4)

# Compile kernel
compile_kernel "$KERNEL_SRC"

# If --compile-only, stop here
if [ "${1:-}" = "--compile-only" ]; then
    exit 0
fi

# Otherwise, copy xclbins
TAG="$1" H="$2" NH="$3" NKV="$4" HD="$5" IM="$6" M="${7:-128}"
QKV_N=$(( NH * HD + 2 * NKV * HD ))
O_K=$(( NH * HD ))
O_N=$H
GU_N=$(( 2 * IM ))
D_K=$IM
D_N=$H

echo ""
echo "=== ${FMT_NAME} xclbins: ${TAG} (H=$H) M=$M ==="
copy_xclbins "$PREFIX" "$TAG"

echo ""
echo "To use: export NPU_XCLBIN_DIR=${XCLBIN_DIR}"
echo "         export NPU_MODEL_PATH=/path/to/model.q4nx"
