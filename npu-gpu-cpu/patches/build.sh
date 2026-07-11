#!/bin/bash
# Build script for amdgpu NPU IP block
#
# Prerequisites:
#   - Linux kernel source tree with amdgpu enabled
#   - GCC, GNU Make, kernel build dependencies
#
# Usage:
#   ./build.sh /path/to/linux
#
# This script:
#   1. Applies patches to the kernel tree
#   2. Builds the amdgpu module
#   3. Installs it

set -euo pipefail

KERNEL_DIR="${1:-}"
if [ -z "$KERNEL_DIR" ]; then
    echo "Usage: $0 /path/to/linux"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR"

echo "=== Applying patches to $KERNEL_DIR ==="

# Copy new source files
cp "$PATCH_DIR/amdgpu_npu.c"     "$KERNEL_DIR/drivers/gpu/drm/amd/amdgpu/"
cp "$PATCH_DIR/amdgpu_npu.h"     "$KERNEL_DIR/drivers/gpu/drm/amd/amdgpu/"
cp "$PATCH_DIR/amdgpu_npu_mgr.c" "$KERNEL_DIR/drivers/gpu/drm/amd/amdgpu/"
cp "$PATCH_DIR/amdgpu_npu_sched.c" "$KERNEL_DIR/drivers/gpu/drm/amd/amdgpu/"

# Apply integration patch
git -C "$KERNEL_DIR" am "$PATCH_DIR/0002-add-npu-ip-block.patch" || \
    echo "Warning: patch may have already been applied"

echo "=== Building amdgpu module ==="
cd "$KERNEL_DIR"
make -C "$KERNEL_DIR" M=drivers/gpu/drm/amd/amdgpu modules -j$(nproc)

echo "=== Build complete ==="
echo "Install with: sudo make -C $KERNEL_DIR M=drivers/gpu/drm/amd/amdgpu modules_install"
echo "Then: sudo modprobe amdgpu"
