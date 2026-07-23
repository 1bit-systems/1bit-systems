#!/usr/bin/env bash
# build_06b_kernels.sh — Compile Qwen3-0.6B NPU AIE kernels
# Uses AMD AIE compiler toolchain (aiecc/xchesscc) from torch2aie.
#
# Prerequisites:
#   TORCH2AIE_ROOT=/home/bcloud/torch2aie  (or set env)
#   source $TORCH2AIE_ROOT/setup_env.sh    (sets PATH, LD_LIBRARY_PATH)
set -euo pipefail

TORCH2AIE_ROOT="${TORCH2AIE_ROOT:-/home/bcloud/torch2aie}"
SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="${SRCDIR}/../build"
KERNELDIR="${SRCDIR}"
OUTDIR="${BUILDDIR}/qwen3_0_6b_kernels"
mkdir -p "$OUTDIR"

# Toolchain paths
AIETOOLS="${TORCH2AIE_ROOT}/toolchain/aietools"
MLIR_AIE="${TORCH2AIE_ROOT}/toolchain/mlir_aie"
XCHESSCC="${AIETOOLS}/bin/xchesscc_wrapper"

# Include paths for AIE kernel compilation
INCLUDES=(
  -I"${KERNELDIR}"
  -I"${AIETOOLS}/include"
  -I"${MLIR_AIE}/include"
  -I"${MLIR_AIE}/include/aie_kernels"
  -I"${MLIR_AIE}/include/aie_kernels/aie2p"
)

echo "=== Building Qwen3-0.6B NPU AIE kernels ==="
echo "Output: ${OUTDIR}"
echo ""

compile_kernel() {
    local src="$1"
    local out="$2"
    local extra_defs="${3:-}"
    echo "  Compiling: $(basename "$src") -> $(basename "$out")"
    $XCHESSCC aie2p \
        ${extra_defs} \
        "${INCLUDES[@]}" \
        -c "$src" \
        -o "$out"
}

# 1. main16 Q4NX GEMM/dequant kernel (06b variant)
compile_kernel \
    "${KERNELDIR}/qwen3_decode_kernels_06b.cc" \
    "${OUTDIR}/qwen3_decode_kernels_06b.o"

# 2. Edge attention kernel (06b: kHeads=4 per worker)
compile_kernel \
    "${KERNELDIR}/edge_attention.cc" \
    "${OUTDIR}/edge_attention_06b.o" \
    "-DMODEL_QWEN3_0_6B"

# 3. Post-process QKV (shared source, uses qwen3_constants_06b.h)
#    Q=2048bf16 output (NH×HD=16×128) from 1024bf16 input (H)
#    K=V=1024bf16 output (NKV×HD=8×128) from 1024bf16 input (H)
compile_kernel \
    "${KERNELDIR}/postprocess_qkv_06b.cc" \
    "${OUTDIR}/postprocess_qkv_06b.o"

# 4. Full vector station (residual add + RMSNorm, H=1024)
compile_kernel \
    "${KERNELDIR}/full_vector_station_06b.cc" \
    "${OUTDIR}/full_vector_station_06b.o"

# 5. SwiGLU (IM-independent, same for all models)
compile_kernel \
    "${KERNELDIR}/swiglu_06b.cc" \
    "${OUTDIR}/swiglu_06b.o"

echo ""
echo "=== All 5 kernels compiled successfully ==="
ls -lh "${OUTDIR}/"*.o
