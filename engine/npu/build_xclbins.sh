#!/bin/bash
# build_xclbins.sh — List/verify pre-built NPU xclbins
#
# Xclbins are pre-built and checked into engine/npu/xclbins/.
# To rebuild from source, see docs/research/npu-ternary-roadmap.md —
# the aiecc toolchain has a pre-existing version mismatch.
#
# Usage:
#   ./build_xclbins.sh              # list all available xclbins
#   ./build_xclbins.sh <tag>        # list xclbins for a model tag

set -euo pipefail
XCLBIN_DIR="$(cd "$(dirname "$0")" && pwd)/xclbins"

if [ $# -ge 1 ]; then
    TAG="$1"
    echo "=== xclbins for ${TAG} ==="
    ls -1 "${XCLBIN_DIR}"/*"${TAG}"*.xclbin 2>/dev/null | while read f; do
        name=$(basename "$f")
        size=$(stat -c%s "$f")
        echo "  ${name}  ($(numfmt --to=iec $size))"
    done
else
    echo "=== Pre-built NPU xclbins ==="
    for f in "${XCLBIN_DIR}"/*.xclbin; do
        name=$(basename "$f")
        size=$(stat -c%s "$f")
        echo "  ${name}  ($(numfmt --to=iec $size))"
    done
    echo ""
    count=$(ls "${XCLBIN_DIR}"/*.xclbin 2>/dev/null | wc -l)
    echo "Total: ${count} xclbins"
fi
