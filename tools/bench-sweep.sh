#!/usr/bin/env bash
# bench-sweep.sh — full prefill kernel sweep on Strix Halo
# Builds rocm-cpp, runs all prefill variants on the three FFN shapes that matter,
# then reports the winner. Requires ROCm/HIP environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Default to repo root (parent of tools/), since CMakeLists.txt lives there
ROCM_CPP_DIR="${ROCM_CPP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$ROCM_CPP_DIR"

echo "=== Building rocm-cpp with prefill tuner ==="
cmake -B build -G Ninja 2>&1 | tail -3
ninja -C build bench_prefill_variants 2>&1 | tail -3

echo ""
echo "=== Decode GEMV: phase5 variant sweep ==="
echo "  (phase5_halo uses sudot4 — expected winner for decode)"
echo "  (phase5_dot4 is the baseline dot4 path)"
echo "  To bench: ./build/ternary_gemv_bench or re-run bitnet_decode with --tune-prefill"

echo ""
echo "=== Prefill GEMM: shape sweep ==="
SHAPES=(
    "2560 6912 2560"   # BitNet FFN up
    "2560 2560 6912"   # BitNet FFN down
    "4096 4096 4096"   # Square (rocBLAS comparison)
)

for shape in "${SHAPES[@]}"; do
    echo ""
    echo "--- Shape: $shape ---"
    read -r M N K <<< "$shape"
    ./build/bench_prefill_variants "$M" "$N" "$K" 5 20
done

echo ""
echo "=== Recommendation ==="
echo "For BitNet FFN shapes: expect 4k (LDS ping-pong) or 4f (2x2 wave) to win."
echo "For square shapes: expect FP16-B to close the rocBLAS gap."
echo ""
echo "To use the best variant in production:"
echo "  onebit --model model.h1b --tune-prefill"
echo "  (auto-tunes at startup, caches result for subsequent runs)"
