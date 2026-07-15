#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build all compute shaders from source
# Requires: glslc (Vulkan SDK), nvcc (CUDA Toolkit), metal (Xcode)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VULKAN_DIR="$SCRIPT_DIR/vulkan"
CUDA_DIR="$SCRIPT_DIR/cuda"
METAL_DIR="$SCRIPT_DIR/metal"
OUT_DIR="$SCRIPT_DIR/../zig-out/share/zinc/shaders"
mkdir -p "$OUT_DIR"

echo "=== Building Vulkan SPIR-V shaders ==="
GLSLC="${GLSLC:-glslc}"
for glsl in "$VULKAN_DIR"/*.glsl; do
    name="$(basename "$glsl" .glsl)"
    out="$OUT_DIR/${name}.spv"
    echo "  $name => $out"
    "$GLSLC" "$glsl" -o "$out"
done

echo "=== Building CUDA fatbin ==="
NVCC="${NVCC:-nvcc}"
if command -v "$NVCC" &>/dev/null; then
    "$NVCC" -fatbin --fatbin -o "$OUT_DIR/cuda_kernels.fatbin" "$CUDA_DIR/kernels.cu"
    echo "  CUDA kernels => $OUT_DIR/cuda_kernels.fatbin"
else
    echo "  nvcc not found — skipping CUDA"
fi

echo "=== Building Metal shaders ==="
METAL="${METAL:-xcrun -sdk macosx metal}"
if command -v xcrun &>/dev/null; then
    $METAL -c "$METAL_DIR/kernels.metal" -o "$OUT_DIR/kernels.air" 2>/dev/null || echo "  Metal not available — skipping"
    echo "  Metal => $OUT_DIR/kernels.air"
else
    echo "  xcrun not found — skipping Metal"
fi

echo "=== Done ==="
ls -la "$OUT_DIR"/*.spv 2>/dev/null || echo "  No .spv files produced (Vulkan SDK not installed?)"
echo "Shaders built at: $OUT_DIR"
