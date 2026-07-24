#!/bin/bash
# build_npu_ternary.sh — Build NPU ternary xclbins for TQ2/TQ1/Q1_0 formats
#
# Compiles the on-tile LUT-decode microkernel, then builds xclbin bitstreams.
# All three formats use uint8_t weight types with on-tile decode:
#   TQ2: 2-bit, 4 codes/byte, LUT[256]  → mm_ternary_tq2.cc
#   TQ1: 1.58-bit base-3, 5 codes/byte, LUT[243] → mm_ternary_tq1.cc
#   Q1_0: 1-bit, 8 bits/byte, 64-bit mask → mm_binary_q1.cc
#
# Usage:
#   ./build_npu_ternary.sh <fmt> <tag> <H> <NH> <NKV> <HD> <IM> [M=128]
#
# Examples:
#   ./build_npu_ternary.sh tq2 qwen3_0_6b 1024 16 8 128 3072
#   ./build_npu_ternary.sh tq1 qwen3_8b 4096 32 8 128 12288
#   ./build_npu_ternary.sh q1  bonsai_27b 8192 64 8 128 28672

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TORCH2AIE="${HOME}/torch2aie"
CFG1="${TORCH2AIE}/examples/gemm_asymmetric_tile_buffering/config1"
OUT_DIR="${SCRIPT_DIR}/xclbins"

export PATH="${TORCH2AIE}/toolchain/bin:$PATH"
export PYTHONPATH="${TORCH2AIE}/toolchain/mlir_aie/python"
export AIETOOLS_DIR="${TORCH2AIE}/toolchain"
export MLIR_AIE_DIR="${TORCH2AIE}/toolchain/mlir_aie"

if [ $# -lt 7 ]; then
    echo "Usage: $0 <fmt:tq2|tq1|q1> <tag> <H> <NH> <NKV> <HD> <IM> [M=128]"
    exit 1
fi

FMT="$1" TAG="$2" H="$3" NH="$4" NKV="$5" HD="$6" IM="$7" M="${8:-128}"

# Set format-specific kernel, design, and prefix
case "$FMT" in
    tq2)
        KERNEL_SRC="mm_ternary_tq2.cc"
        DESIGN_PY="n1_core_tq2_placed.py"
        PREFIX="ternary_tq2"
        FMT_NAME="TQ2 (2-bit)"
        ;;
    tq1)
        KERNEL_SRC="mm_ternary_tq1.cc"
        DESIGN_PY="n1_core_tq1_placed.py"
        PREFIX="ternary_tq1"
        FMT_NAME="TQ1 (1.58-bit)"
        ;;
    q1)
        KERNEL_SRC="mm_binary_q1.cc"
        DESIGN_PY="n1_core_q1_placed.py"
        PREFIX="binary_q1"
        FMT_NAME="Q1_0 (1-bit)"
        ;;
    *)
        echo "Unknown format: $FMT (use tq2, tq1, or q1)"
        exit 1
        ;;
esac

QKV_N=$(( NH * HD + 2 * NKV * HD ))
O_K=$(( NH * HD ))
O_N=$H
GU_N=$(( 2 * IM ))
D_K=$IM
D_N=$H

mkdir -p "$OUT_DIR"

build_xclbin() {
    local label="$1" M="$2" K="$3" N="$4"
    local xclbin="${OUT_DIR}/${PREFIX}_${label}_${TAG}.xclbin"
    local insts="${OUT_DIR}/insts_${PREFIX}_${label}_${TAG}.txt"

    if [ -f "$xclbin" ] && [ -f "$insts" ]; then
        echo "  ✓ $label ($(stat -c%s "$xclbin") bytes)"
        return 0
    fi

    echo "  Building $FMT_NAME xclbin $label (${M}×${K}×${N})..."
    cd "$CFG1"

    # Copy kernel source to config1 directory
    cp "${SCRIPT_DIR}/kernel/${KERNEL_SRC}" "${CFG1}/${KERNEL_SRC}"

    # Compile Chess microkernel
    local KERNEL_OBJ="${KERNEL_SRC/.cc/.o}"
    if [ ! -f "build/${KERNEL_OBJ}" ]; then
        echo "    Compiling $KERNEL_SRC..."
        xchesscc_wrapper aie2p "${CFG1}/${KERNEL_SRC}" \
            -I"${TORCH2AIE}/toolchain/mlir_aie/include" \
            -I"${TORCH2AIE}/toolchain/mlir_aie/include/aie_kernels" \
            -I"${TORCH2AIE}/toolchain/mlir_aie/include/aie_kernels/aie2p" \
            -o "build/${KERNEL_OBJ}"
        echo "    Compiled: $(stat -c%s "build/${KERNEL_OBJ}") bytes"
    fi

    # Build xclbin via Makefile
    # Note: aiecc --unified has a pre-existing issue with clean builds
    # (same for original INT8 design). xclbins from cached builds work.
    make "M=$M" "K=$K" "N=$N" m=128 k=64 n=128 \
        use_placed=1 targetname=n1_core \
        kernelsrc="${KERNEL_SRC}" \
        aie_py_src="${DESIGN_PY}" \
        "build/final_${M}x${K}x${N}_128x64x128.xclbin" 2>&1 | tail -5

    cp "build/final_${M}x${K}x${N}_128x64x128.xclbin" "$xclbin" 2>/dev/null || \
        echo "    ⚠ xclbin build failed (pre-existing toolchain issue). Use cached: $(ls -la $xclbin 2>/dev/null || echo 'N/A')"
    cp "build/insts_${M}x${K}x${N}_128x64x128.txt" "$insts" 2>/dev/null || true

    if [ -f "$xclbin" ]; then
        echo "  Done: $(stat -c%s "$xclbin") bytes"
    fi
}

echo "=== Building $FMT_NAME xclbins: $TAG (H=$H) M=$M ==="
build_xclbin "QKV" "$M" "$H" "$QKV_N"
build_xclbin "O"   "$M" "$O_K" "$O_N"
build_xclbin "GU"  "$M" "$H" "$GU_N"
build_xclbin "D"   "$M" "$D_K" "$D_N"
echo "=== Done: $OUT_DIR ==="
echo ""
echo "Available xclbins:"
ls -1 ${OUT_DIR}/${PREFIX}_*_${TAG}.* 2>/dev/null | head -10
