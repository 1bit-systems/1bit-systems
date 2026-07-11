#!/bin/bash
# Run GEMM bench with TheRock rocBLAS vs system rocBLAS
set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
THEROCK=$HOME/therock/build
OUT=$HOME/Desktop/strixhalo-output/gemm-bench-results.txt

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0

echo "=== Compiling bench_gemm ==="
hipcc -O3 --offload-arch=gfx1151 \
    -I/opt/rocm/include \
    -L/opt/rocm/lib -lrocblas -lamdhip64 \
    "$SCRIPT_DIR/bench_gemm.cpp" -o "$SCRIPT_DIR/bench_gemm"

echo ""
echo "=== System rocBLAS ===" | tee $OUT
echo "Date: $(date)" | tee -a $OUT
echo "" | tee -a $OUT
"$SCRIPT_DIR/bench_gemm" 2>&1 | tee -a $OUT

echo "" | tee -a $OUT
echo "=== TheRock rocBLAS (gfx1151 native Tensile) ===" | tee -a $OUT
echo "" | tee -a $OUT
export LD_LIBRARY_PATH=$THEROCK/math-libs/BLAS/rocBLAS/dist/lib:$THEROCK/math-libs/BLAS/hipBLASLt/dist/lib:$THEROCK/core/clr/dist/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=$THEROCK/math-libs/BLAS/rocBLAS/dist/lib/rocblas/library
"$SCRIPT_DIR/bench_gemm" 2>&1 | tee -a $OUT

echo "" | tee -a $OUT
echo "=== Complete ===" | tee -a $OUT
